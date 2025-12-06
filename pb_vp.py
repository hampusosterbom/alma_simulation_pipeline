
"""
H. Österbom

pb_vp.py

Primary-beam (PB) and voltage-pattern (VP) utilities for heterogeneous arrays.

This module provides:
  * Analytic PB models (ALMA Gaussian, Airy disk, custom Gaussian)
  * Generation of per-diameter PB images on a common grid
  * Construction of per-antenna VP tables for AW-projection
  * Inspection/debugging utilities for VP tables
  * Basic λ/D beam calculations for logging and sanity checks

Used by the simulation pipeline to supply accurate PB models to tclean
when imaging heterogeneous arrays.
"""


import os
import math
import logging
import numpy as np
from qol_pipeline import _read_antenna_table
from casatools import image as iatool, coordsys, vpmanager, table

from utils import safe_rm_tree


# CASA tool instances
ia = iatool()
cs = coordsys()
vp = vpmanager()
tb = table()

# Physical / geometric constants
C = 299792458.0                # m/s
ARCSEC_PER_RAD = 206264.806    # arcsec / rad
# -----------------------------------------------------------------------------
# Primary-beam widths and measurement
# -----------------------------------------------------------------------------

def _pb_fwhm_arcsec_alma(freq_ghz, dish_m, k_hpbw=1.13):
    """
    ALMA / CASA-style primary-beam FWHM in arcsec.

    HPBW ≈ k_hpbw * (λ / D)  [radians]
    Default k_hpbw = 1.13 for ALMA (CASA convention).
    """
    lam = C / (freq_ghz * 1e9)          # wavelength [m]
    hpbw_rad = k_hpbw * lam / float(dish_m)
    return hpbw_rad * ARCSEC_PER_RAD

# -----------------------------------------------------------------------------
# PB image creation
# -----------------------------------------------------------------------------

def _makePBImage(
    outfile,
    dish_m,
    freq_ghz,
    imsize,
    cell_arcsec,
    ra_pc,
    dec_pc,
    pb_model="alma",
    gauss_fwhm_arcsec=None,
):
    """
    Create a primary-beam image suitable for vp.setpbimage.
    """

    safe_rm_tree(outfile)


    stokes_n = 4
    chans = 1

    ia.fromshape(outfile, [imsize, imsize, stokes_n, chans], type='f')

    # --- Coordinate system (direction + spectral + stokes) ---
    csf = cs.newcoordsys(direction=True, spectral=True,
                         stokes=["XX", "XY", "YX", "YY"])

    refpix = (imsize - 1) / 2.0
    inc_str = f"{-cell_arcsec}arcsec {cell_arcsec}arcsec"

    csf.setdirection(
        refcode="J2000",
        proj="SIN",
        refpix=[refpix, refpix],
        refval=[ra_pc, dec_pc],
        incr=inc_str,
    )

    csf.setreferencevalue(freq_ghz * 1e9, type='spectral')
    csf.setreferencepixel([0.0], type='spectral')
    csf.setincrement("1Hz", type='spectral')

    ia.setcoordsys(csf.torecord())

    # --- Radial coordinate on the sky ---
    pix = np.zeros((imsize, imsize, stokes_n, chans), float)
    half = 0.5 * imsize * cell_arcsec
    y_arc, x_arc = np.meshgrid(
        np.linspace(-half, half, imsize),
        np.linspace(-half, half, imsize),
    )
    r_arc = np.hypot(x_arc, y_arc)
    r_rad = r_arc / ARCSEC_PER_RAD

    lam = C / (freq_ghz * 1e9)

    # --- Choose PB model: define POWER PB first ---
    if pb_model == "airy":
        from scipy.special import j1 as bessel_j1  
        # Uniform circular aperture → Airy *power* pattern
        x = math.pi * dish_m * r_rad / lam
        pb_power = np.ones_like(x, dtype=float)
        mask = np.abs(x) > 1e-6
        pb_power[mask] = (2.0 * bessel_j1(x[mask]) / x[mask]) ** 2
        fwhm_arcsec = np.nan  # hard to define analytically; we can measure from pb_power
    else:
        if pb_model == "alma":
            fwhm_arcsec = _pb_fwhm_arcsec_alma(freq_ghz, dish_m, k_hpbw=1.13)
        elif pb_model == "gauss":
            if gauss_fwhm_arcsec is None:
                raise ValueError("pb_model='gauss' requires gauss_fwhm_arcsec")
            fwhm_arcsec = float(gauss_fwhm_arcsec)
        else:
            raise ValueError(f"Unknown pb_model='{pb_model}'")

        sigma_rad = (fwhm_arcsec / ARCSEC_PER_RAD) / math.sqrt(8.0 * math.log(2.0))
        pb_power = np.exp(-0.5 * (r_rad / sigma_rad) ** 2)

    # Normalise POWER PB
    pb_power = np.clip(pb_power, 0.0, 1.0)
    if pb_power.max() > 0:
        pb_power /= pb_power.max()

    # Voltage pattern for VP table
    pb = np.sqrt(pb_power)

    # Guard ring at edges
    guard = 8
    pb[:guard, :] = 0
    pb[-guard:, :] = 0
    pb[:, :guard] = 0
    pb[:, -guard:] = 0

    # XX and YY get PB (voltage), XY/YX = 0
    pix[:, :, 0, 0] = pb  # XX
    pix[:, :, 3, 0] = pb  # YY

    ia.putchunk(pix)
    ia.done()

    return outfile


def calc_ang(freq_ghz, dia_m):
    """
    Approximate primary-beam FWHM in arcmin from λ/D (simple λ/D scaling).
    Kept here for quick sanity checks / logging.
    """
    return ((3e8 / (freq_ghz * 1e9)) / dia_m) * (180.0 / math.pi) * 60.0



def _get_antennas_by_diameter(msname):
    """
    Return a mapping {rounded_diameter_m : [antenna_name, ...]} for an MS.
    """
    diams, names = _read_antenna_table(msname)
    ants_by_d = {}
    for name, d in zip(names, diams):
        d_round = float(round(d))
        ants_by_d.setdefault(d_round, []).append(str(name))
    return ants_by_d


def build_per_antenna_pb_vptable(args, msname, vptab_path, imsize, cell_arcsec):
    """
    Build a VP table for a heterogeneous array using:
      1. Per-diameter PB images (Airy / ALMA-style, etc.)
      2. makeVPTable_fromPBs() to attach PBs to actual antenna names.
    """
    # Extract reference frequency in GHz
    nu_ghz = float(str(args.center_freq).lower().replace("ghz", "").strip())

    # choose PB model from args, defaulting to 'alma' if missing
    pb_model = getattr(args, "pb_model", "alma")
    gauss_fwhm_arcsec = getattr(args, "pb_gauss_fwhm_arcsec", None)

    # Group antennas by rounded diameter
    ants_by_d = _get_antennas_by_diameter(msname)
    logging.info("[PB] Building per-antenna PB → VP table from %s", msname)
    logging.info("[PB] Dish diameters present: %s", sorted(ants_by_d.keys()))

    for d_m, ants in ants_by_d.items():
        logging.info("[PB] Diameter %.1f m → antennas %s", d_m, ants)

    vp_dir = os.path.dirname(vptab_path) or "."
    os.makedirs(vp_dir, exist_ok=True)

    # STEP 1: Create PB images per dish diameter
    pb_images = {}  # {diameter_m : "PB_xx.im"}

    for d_m in ants_by_d.keys():
        pb_path = os.path.join(vp_dir, f"PB_{int(d_m)}m.im")
        safe_rm_tree(pb_path)

        _makePBImage(
            outfile=pb_path,
            dish_m=d_m,
            freq_ghz=nu_ghz,
            imsize=imsize,
            cell_arcsec=cell_arcsec,
            ra_pc=args.phasecenter_ra,
            dec_pc=args.declinations[0],
            pb_model=pb_model,
            gauss_fwhm_arcsec=gauss_fwhm_arcsec,
        )
        logging.info("[PB] Using PB grid: imsize=%d, cell=%.4f\"", imsize, cell_arcsec)
        logging.info(f"[PB] Created PB using model: {pb_model} for %.1f m → %s", d_m, pb_path)

        pb_images[d_m] = pb_path

    # STEP 2: Build VP table from PBs
    safe_rm_tree(vptab_path)
    logging.info("[VP] Creating VP table using makeVPTable_fromPBs()")
    vptab = _makeVPTable_fromPBs(msname, vptab_path, pb_images)

    logging.info("[VP] Saved VP table to: %s", vptab)
    return vptab


def _makeVPTable_fromPBs(msname, vptab, pb_images):
    """
    Build a VP table for a heterogeneous array using custom PB images.

    Parameters
    ----------
    msname : str
        Measurement Set path (for antenna names).
    vptab : str
        Path to output VP table.
    pb_images : dict
        Mapping {dish_diameter_m : pb_image_path}.
        Example: {12.0:"PB_12m.im", 7.0:"PB_7m.im"}
    """
    ants_by_d = _get_antennas_by_diameter(msname)

    safe_rm_tree(vptab)
    vp.reset()

    # Attach PB images per antenna group
    for d_m, antlist in ants_by_d.items():
        if d_m not in pb_images:
            raise ValueError(f"No PB image for dish {d_m} m")

        pb_path = pb_images[d_m]
        print(f"[VP] Diameter {d_m} m → antnames={antlist}")
        print(f"[VP]   PB image: {pb_path}")

        vp.setpbimage(
            telescope="ALMA",
            realimage=pb_path,
            antnames=antlist,
        )

    vp.summarizevps()
    vp.saveastable(vptab)

    # Optional: dump a quick summary of table structure
    tb.open(vptab)
    print("[VP] Columns:", tb.colnames())
    print("[VP] Nrows:", tb.nrows())
    tb.close()

    print(f"[VP] Saved table: {vptab}")
    return vptab


def inspect_vptable(vptab):
    """
    Print the full contents of a CASA VP table row-by-row.
    Useful for debugging what setpbimage() actually stored.
    """
    import pprint

    tb.open(vptab)
    try:
        print("\n=== VP TABLE CONTENTS ===")
        print("Columns:", tb.colnames())
        print("Nrows:", tb.nrows())

        for row in range(tb.nrows()):
            print(f"\n--- ROW {row} ---")
            for col in tb.colnames():
                try:
                    val = tb.getcell(col, row)
                except Exception:
                    val = "<cannot read>"
                print(f"{col}:")
                pprint.pprint(val)
    finally:
        tb.close()

    print("=== END VP TABLE CONTENTS ===\n")
