"""
H. Österbom

qol_pipeline.py

Low-level CASA and numerical utilities used throughout the simulation pipeline.

This module provides:
  • Measurement Set construction for heterogeneous arrays
  • Sky model prediction and noise injection (ATM + (manual?))
  • Baseline-group tools (A/B/cross), SEFD estimators, and RMS utilities
  • Helper routines for imaging geometry, PB/beam info, and CASA image stats
  • Configuration, hour-angle, and pointing utilities

These are building blocks consumed by the high-level simulation pipeline.
"""
import math
import os
import json
import shutil
from pathlib import Path
import logging
import numpy as np
from math import pi
import subprocess
from collections import Counter
from casatools import image, table, simulator, coordsys, measures, quanta, ctsys, ms
from casatasks import tclean, imhead, imregrid, flagdata, imstat, mstransform
from casatasks.private import simutil
from utils import next_smooth

#======For hetero arrays (helper functions)=======
# Instantiate all the required tools
sm = simulator()
ia = image()
tb = table()
cs = coordsys()
me = measures()
qa = quanta()
mysu = simutil.simutil()
myms = ms()

# === CONFIGURATION & PATHS ===
SCRIPT_DIR = Path(__file__).resolve().parent
START_DIR = Path.cwd()

# Physical / geometric constants
c = 299792458.0                # m/s

# --- CONFIG & GEOMETRY HELPERS ---------------------------------------------

def find_config_path(cfg, casa_bin=None, extra_dirs=None):
    """
    Search for an ALMA config file in likely directories.
    """
    candidates = []
    simmos_dir = Path.home() / '.casa' / 'data' / 'alma' / 'simmos'
    candidates.append(simmos_dir / cfg)
    candidates.append(Path.home() / '.casa' / cfg)
    if casa_bin:
        casa_root = Path(casa_bin).parent.parent
        candidates.append(casa_root / 'data' / 'alma' / 'simmos' / cfg)
    if extra_dirs:
        for d in extra_dirs:
            candidates.append(Path(d) / cfg)
    candidates.append(Path.cwd() / cfg)
    for p in candidates:
        if p.exists():
            logging.info(f"Found config file at {p}")
            return p
    raise FileNotFoundError(f"Config {cfg!r} not found in candidate paths: {candidates}")



def load_antenna_positions(cfg_path):
    """
    Load antenna (x,y,z) positions from a CASA .cfg file (or similar).

    Parameters
    ----------
    cfg_path : str or Path
        Path to configuration file.

    Returns
    -------
    list[tuple]
        List of (x, y, z) antenna positions.
    """
    cfg_path = Path(cfg_path)
    ants = []
    with cfg_path.open('r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            x, y, z = map(float, parts[:3])
            ants.append((x, y, z))
    if not ants:
        raise ValueError("No valid antenna positions found in the file.")
    return ants



def _parse_diameters_from_cfg(cfg_path: Path):
    """Return list of dish diameters [m] parsed from a CASA .cfg file."""
    diams = []
    try:
        with open(cfg_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                floats = []
                for p in parts:
                    try:
                        floats.append(float(p))
                    except ValueError:
                        continue
                if not floats:
                    continue
                d = floats[-1]
                if d > 0:
                    diams.append(round(d, 3))
    except Exception as e:
        logging.warning(f"[cfg] Could not parse diameters from {cfg_path}: {e}")
    return diams



def compute_baselines(ants):
    """
    Compute all pairwise baseline lengths from antenna positions.

    Parameters
    ----------
    ants : list[tuple]
        Antenna positions (x, y, z) in meters.

    Returns
    -------
    np.ndarray
        Sorted array of baseline lengths in meters.
    """
    baselines = []
    n = len(ants)
    for i in range(n):
        for j in range(i + 1, n):
            dx = ants[i][0] - ants[j][0]
            dy = ants[i][1] - ants[j][1]
            dz = ants[i][2] - ants[j][2]
            d = np.sqrt(dx*dx + dy*dy + dz*dz)
            if d > 0:
                baselines.append(d)

    if not baselines:
        raise ValueError("No valid baselines computed.")

    return np.sort(baselines)



def get_L80(baselines):
    """
    Compute the 80th percentile projected baseline length (L80).
    
    Parameters:
    baselines (np.array): Sorted baseline lengths.
    
    Returns:
    float: L80 in meters.
    """
    return np.percentile(baselines, 80)

def compute_lambda(freq_ghz):
    return c / (freq_ghz * 1e9)


def compute_beam_fwhm(lam, L80):
    theta_rad = 0.574 * lam / L80
    return theta_rad * (180 / pi) * 3600



def compute_imaging_geometry(args, cfg_path, config_tag):
    """
    Beam FWHM, cell size, imsize for a given config,
    PLUS automatically detect whether the array is heterogeneous.
    """
    # --- basic beam / cell / imsize ---
    ants = load_antenna_positions(cfg_path)
    lam = compute_lambda(float(args.center_freq.rstrip('GHz')))
    baselines = compute_baselines(ants)
    L80 = get_L80(baselines)
    beam = compute_beam_fwhm(lam, L80)
    logging.info(f"[{config_tag}] Beam FWHM: {beam:.6f} arcsec")

    # --- manual override? ---
    if getattr(args, "cell_arcsec", None) is not None and getattr(args, "imsize_manual", None) is not None:
        pixsize = float(args.cell_arcsec)
        dyn_cell = f"{pixsize}arcsec"
        imsize = int(args.imsize_manual)
        logging.info(f"[{config_tag}] Using manual cell={dyn_cell}, imsize={imsize}")
    else:
        # auto geometry
        dyn_cell = f"{beam / args.sampling_factor}arcsec"
        logging.info(f"[{config_tag}] Cell size set to {dyn_cell}")
        pixsize = beam / args.sampling_factor
        imsize_raw = int(np.ceil(args.fov_arcsec / pixsize))
        if imsize_raw % 2 == 1:
            imsize_raw += 1
        imsize = next_smooth(imsize_raw)
        logging.info(f"[{config_tag}] Adjusted imsize from {imsize_raw} to efficient {imsize}")

    # --- cfg-based hetero detection ---
    diams = _parse_diameters_from_cfg(cfg_path)
    if diams:
        uniq = sorted(set(diams))
        counts = Counter(diams)

        logging.info(f"[{config_tag}] Dish diameters present (m): {uniq}")
        for d in uniq:
            logging.info(f"[{config_tag}]   {d:.3f} m : {counts[d]} antennas")

        is_hetero = len(uniq) > 1

        if is_hetero:
            logging.info(f"[{config_tag}] HETEROGENEOUS array (cfg has multiple diameters)")
        else:
            logging.info(f"[{config_tag}] HOMOGENEOUS array (single diameter)")
    else:
        logging.info(f"[{config_tag}] Could not parse diameters from cfg {cfg_path}")
        is_hetero = False

    return beam, dyn_cell, pixsize, imsize, is_hetero



# -----------------------------------------------------------------------------
# MS creation & sky-model prediction
# -----------------------------------------------------------------------------

def makeMSFrame(
    msn='sim_data',
    tel='ALMA',
    cfg_path=None,
    pointings=None,
    spwname='SIM',
    freq='90GHz',
    deltafreq='2GHz',
    freqresolution='1MHz',
    nchannels=1,
    stokes='XX YY',
    integrationtime='10s',
    usehourangle=True,
    referencetime='2025/01/01/00:00:00',
    scans=None,
    overwrite=True,
):
    """
    Create an empty Measurement Set for simulations by loading a CASA cfg
    file, defining fields and SPWs, and generating scan timing/pointings.
    """
    if cfg_path is None:
        raise ValueError("makeMSFrame: cfg_path is required for heterogeneous simulation")
    if not pointings:
        raise ValueError("makeMSFrame: non-empty 'pointings' list is required")

    msname = f"{msn}_{tel}.ms"

    # Remove any pre-existing MS
    if os.path.exists(msname):
        if overwrite:
            shutil.rmtree(msname)
        else:
            raise FileExistsError(f"{msname} already exists; set overwrite=True to replace")

    ant_res = mysu.readantenna(cfg_path)

    an = None
    obspos = None

    if len(ant_res) == 4:
        # Simple cfg: x, y, z, diam
        stnx, stny, stnz, std = ant_res
    elif len(ant_res) >= 8:
        # Rich cfg: positions, diameters, names, ..., COFA
        stnx, stny, stnz, std, an, _an2, _telname_cfg, obspos = ant_res
    else:
        raise ValueError(
            f"readantenna({cfg_path!r}) returned {len(ant_res)} values; expected 4 or 8"
        )

    # Observatory (COFA)
    if obspos is None:
        obspos = me.observatory('ALMA')
        if 'ALMA' not in str(tel).upper():
            print(
                f"WARNING: cfg '{cfg_path}' has no observatory/COFA; "
                f"using ALMA COFA for telescope '{tel}'. "
                "Provide a COFA for accurate simulations."
            )

    # Diameters and antenna names
    std = np.asarray(std, dtype=float)
    nant = std.size

    if an is not None:
        try:
            ant_names = [str(x) for x in an]
        except Exception:
            ant_names = [f"ANT{i:03d}" for i in range(nant)]
    else:
        ant_names = [f"ANT{i:03d}" for i in range(nant)]

    mounts = ['alt-az'] * nant
    telname = str(tel)

    sm.open(ms=msname)
    sm.setconfig(
        telescopename=telname,
        x=stnx,
        y=stny,
        z=stnz,
        dishdiameter=std,
        mount=mounts,
        antname=ant_names,
        coordsystem='global',
        referencelocation=obspos,
    )

    # --- 2) Feed & SPW ---
    sm.setfeed(mode='perfect X Y', pol=[''])
    sm.setspwindow(
        spwname=spwname,
        freq=freq,
        deltafreq=deltafreq,
        freqresolution=freqresolution,
        nchannels=nchannels,
        stokes=stokes,
    )

    # --- 3) Fields ---
    for p in pointings:
        sm.setfield(
            sourcename=p['name'],
            sourcedirection=p['dir'],
        )

    sm.setauto(autocorrwt=0.0)
    sm.settimes(
        integrationtime=integrationtime,
        usehourangle=usehourangle,
        referencetime=me.epoch('UTC', referencetime),
    )

    # --- 4) Scans / observe ---
    if not scans:
        # Fallback: one scan per pointing, symmetric around transit
        for p in pointings:
            sm.observe(
                sourcename=p['name'],
                spwname=spwname,
                starttime='-0.5h',
                stoptime='+0.5h',
            )
    else:
        for s in scans:
            if isinstance(s, dict):
                name  = s.get('name')
                start = s.get('start')
                stop  = s.get('stop')
            else:
                try:
                    name, start, stop = s
                except Exception:
                    raise ValueError(f"scan entry has unexpected format: {s!r}")

            sm.observe(
                sourcename=name,
                spwname=spwname,
                starttime=start,
                stoptime=stop,
            )

    sm.close()

    # Ensure everything is unflagged initially
    flagdata(vis=msname, mode='unflag')

    return msname


def predictImager(
    msname,
    imname_true,
    gridder='mosaic',
    tel='ALMA',
    vptable='',
):
    """
    Write MODEL_DATA by predicting visibilities from a sky image using
    tclean in niter=0 (predict-only) mode.

    The function reads image dimensions and pixel scale from the model and
    runs tclean with the chosen gridder and optional VP table.

    Parameters
    ----------
    msname : str
        Target Measurement Set.
    imname_true : str
        Input sky model (CASA image).
    gridder : str
        tclean gridder ('mosaic', 'standard', etc.).
    vptable : str
        Optional VP table for AW-projection.
    """
    ia.open(imname_true)
    shape = ia.shape()              # [nx, ny, ...]
    csys  = ia.coordsys()
    incr  = csys.increment()['numeric']   # radians per pixel
    ia.close()

    nx = int(shape[0])
    ny = int(shape[1])

    # RA increment is typically negative; use absolute value
    cell_rad = abs(float(incr[0]))
    cell_as  = cell_rad * 206265.0       # rad => arcsec
    cell_str = f"{cell_as:.6f}arcsec"

    # Clean up any old sim_predict.* products
    os.system('rm -rf sim_predict.*')

    tclean(
        vis=msname,
        imagename='sim_predict',
        startmodel=imname_true,
        vptable=vptable,
        imsize=[nx, ny],
        cell=[cell_str, cell_str],
        specmode='cube',
        nchan=-1,                 # use all MS channels
        interpolation='nearest',
        gridder=gridder,
        normtype='flatsky',
        wbawp=True,
        pblimit=0.1,
        niter=0,                  # predict only
        savemodel='modelcolumn',
        calcres=True,
        calcpsf=True,
        datacolumn='data',
    )


def make_startmodel_match(vis, in_im, out_im):
    """Regrid 'in_im' onto itself (no-op in practice) and write 'out_im'."""
    imregrid(
        imagename=in_im,
        template=in_im,
        output=out_im,
        overwrite=True,
    )
    return out_im


def _parse_hours(s: str) -> float:
    """Parse strings like '1h', '30min', '600s' into hours (float)."""
    s = str(s).strip().lower()
    if s.endswith('h'):   return float(s[:-1])
    if s.endswith('min') or s.endswith('m'): return float(s.rstrip('minm'))/60.0
    if s.endswith('s'):   return float(s[:-1])/3600.0
    return float(s)  # assume already in hours


def build_ha_scans(pointings, total_time_str, ncycles=1, debug=True):
    """
    Build (name, ha_start, ha_stop) tuples for sm.observe(), splitting
    total observing time equally among all pointings and cycles, centred
    on transit (HA=0).
    """
    n_pt = len(pointings)
    if n_pt == 0:
        raise ValueError("build_ha_scans: 'pointings' must be non-empty")

    H = _parse_hours(total_time_str)        # total hours requested
    ncycles = max(int(ncycles), 1)          # guard against 0 / negatives

    per_cycle_h = H / ncycles               # hours per cycle
    t_per_pt    = per_cycle_h / n_pt        # hours per pointing in a cycle
    half_pt     = 0.5 * t_per_pt

    # HA window is [-per_cycle_h/2, +per_cycle_h/2]; centres evenly spaced
    centers = np.linspace(
        -per_cycle_h / 2 + half_pt,
        +per_cycle_h / 2 - half_pt,
        n_pt,
    )

    scans = [
        (p["name"], f"{(c - half_pt):+.6f}h", f"{(c + half_pt):+.6f}h")
        for cyc in range(ncycles)
        for p, c in zip(pointings, centers)
    ]

    if debug:
        t_start_h, t_end_h = -per_cycle_h / 2, +per_cycle_h / 2
        print(
            f"[build_ha_scans] Mosaic: {n_pt} pointings × {ncycles} cycles "
            f"= {len(scans)} scans; total={H:.2f}h; "
            f"HA window=[{t_start_h:+.2f}h,{t_end_h:+.2f}h]; "
            f"~{t_per_pt * 60:.1f} min/pointing"
        )

    return scans


# -----------------------------------------------------------------------------
# Noise injection & efficiency scaling
# -----------------------------------------------------------------------------

def addNoiseSim(
    msname: str,
    pwv_mm: float = 1.262,
    t_ground_K: float = 270.0,
    altitude_m: float = 5000.0,
    relhum: float = 20.0,
    pground: float = 650.0,
    eta_A: float = 0.63,
    eta_B: float = 0.63,
    spillefficiency: float = 0.96,
    trx_K: float = 72.0,
    correfficiency: float = 0.88,
    waterheight: float = 2000.0,
    tau1: float = 0.224,
) -> None:
    """
    Apply CASA's tsys-atm noise model to an MS, then adjust for
    per-diameter aperture efficiencies.

    Parameters
    ----------
    msname : str
        Measurement Set path.
    pwv_mm : float
        Precipitable water vapour (mm).
    t_ground_K : float
        Ground/atmospheric temperature (K).
    altitude_m : float
        Observatory altitude (m).
    relhum : float
        Relative humidity (%).
    pground : float
        Ground pressure (mbar).
    eta_12m, eta_21m : float
        Aperture efficiencies for 12m and 7m (or 21m) dishes.
    spillefficiency : float
        Spillover efficiency.
    trx_K : float
        Receiver temperature (K).
    correfficiency : float
        Correlator efficiency.
    waterheight : float
        Water vapour scale height (m).
    tau1 : float
        Zenith opacity at reference frequency.
    """
    # --- read antenna diameters from our helper ---
    diams, _names = _read_antenna_table(msname)
    uniq = sorted(set(np.round(diams, 3)))

    # --- build automatic eta_map ---
    eta_map = {}
    if len(uniq) == 1:
        # Homogeneous array (e.g., pure 12m)
        eta_map[uniq[0]] = eta_A
    else:
        smallest = uniq[0]
        largest = uniq[-1]
        eta_map[largest] = eta_A
        eta_map[smallest] = eta_B

        if len(uniq) > 2:
            print(
                f"WARNING: {msname} contains multiple dish diameters {uniq}. "
                f"Mapping largest={largest}m→eta_A={eta_A}, "
                f"smallest={smallest}m→eta_B={eta_B}. "
                "Other diameters will use eta_ref."
            )

    # use lowest efficiency as reference (scaling \leq 1 afterwards)
    eta_ref = min(eta_map.values())

    # --- apply CASA atmospheric noise model ---
    sm.openfromms(msname)
    sm.setnoise(
        mode='tsys-atm',
        pwv=f'{pwv_mm}mm',
        tatmos=t_ground_K,
        altitude=f'{altitude_m}m',
        relhum=relhum,
        pground=f'{pground}mbar',
        antefficiency=eta_ref,
        spillefficiency=spillefficiency,
        trx=trx_K,
        correfficiency=correfficiency,
        waterheight=f'{waterheight}m',
        tau=tau1,
    )
    sm.corrupt()
    sm.close()

    # --- apply per-diameter scaling (if actually needed) ---
    if any(abs(eta - eta_ref) > 1e-3 for eta in eta_map.values()):
        _apply_efficiency_scaling_tsystatm(
            msname,
            eta_ref=eta_ref,
            eta_by_diam=eta_map,
        )


def _apply_efficiency_scaling_tsystatm(msname, eta_ref, eta_by_diam):
    """
    After tsys-atm has added noise, adjust DATA, SIGMA, WEIGHT so that
    different dish diameters effectively have different aperture efficiencies.

    Parameters
    ----------
    msname : str
        Measurement set path.
    eta_ref : float
        Reference efficiency used in sm.setnoise(antefficiency=eta_ref).
        Must be >= all values in eta_by_diam.
    eta_by_diam : dict
        Maps dish diameter [m] -> desired efficiency (0 < η <= eta_ref).

    Behaviour
    ---------
    For each row (baseline i-j):

      - Let current noise from tsys-atm be σ_ref (from SIGMA).
      - Target factor f = eta_ref / sqrt(η_i * η_j).

      - If f > 1:
            σ_true  = f * σ_ref
            σ_extra = σ_ref * sqrt(f² - 1)
        add complex Gaussian noise with stddev σ_extra to DATA,
        update SIGMA and WEIGHT accordingly.

      - If f < 1:
        scale DATA by f and update SIGMA, WEIGHT.
    """
    from casatools import table as table_tool

    # --- antenna diameters -> efficiencies per antenna ---
    diams, _names = _read_antenna_table(msname)          
    diams = np.asarray(diams, dtype=float)

    # round keys a bit to avoid tiny floating diffs
    eta_map = {float(round(k, 3)): float(v) for k, v in eta_by_diam.items()}

    def eta_for_ant(aid: int) -> float:
        d = float(round(diams[aid], 3))
        return eta_map.get(d, eta_ref)

    # --- load full table ---
    tb.open(msname, nomodify=False)
    try:
        cols = set(tb.colnames())
        needed = {"DATA", "SIGMA", "WEIGHT", "ANTENNA1", "ANTENNA2"}
        if not needed.issubset(cols):
            print(f"[eff-tsys] {msname}: missing one of {needed}, skipping.")
            return

        ant1 = tb.getcol("ANTENNA1")  # (nrow,)
        ant2 = tb.getcol("ANTENNA2")  # (nrow,)
        sigma = tb.getcol("SIGMA")    # (npol, nrow) or (nrow,)
        weight = tb.getcol("WEIGHT")  # same shape as SIGMA
        data = tb.getcol("DATA")      # (npol, nchan, nrow)

        if sigma.ndim == 1:
            sigma = sigma[np.newaxis, :]
            weight = weight[np.newaxis, :]

        npol, nrow = sigma.shape
        _, nchan, _ = data.shape

        for i in range(nrow):
            eta_i = eta_for_ant(int(ant1[i]))
            eta_j = eta_for_ant(int(ant2[i]))

            f = eta_ref / math.sqrt(eta_i * eta_j)

            if abs(f - 1.0) < 1e-6:
                continue

            sigma_ref = float(sigma[0, i])
            if sigma_ref <= 0:
                continue

            if f > 1.0:
                # Need to ADD extra noise
                sigma_true  = f * sigma_ref
                sigma_extra = sigma_ref * math.sqrt(f*f - 1.0)

                sigma[:, i]  = sigma_true
                weight[:, i] = 1.0 / (sigma_true * sigma_true)

                shape = (npol, nchan)
                noise_re = np.random.normal(0.0, sigma_extra, size=shape)
                noise_im = np.random.normal(0.0, sigma_extra, size=shape)
                data[:, :, i] += noise_re + 1j * noise_im
            else:
                # Need to REDUCE noise: scale DATA by f, update σ
                sigma_true = f * sigma_ref
                data[:, :, i] *= f
                sigma[:, i]  = sigma_true
                weight[:, i] = 1.0 / (sigma_true * sigma_true)

        tb.putcol("SIGMA", sigma)
        tb.putcol("WEIGHT", weight)
        tb.putcol("DATA", data)

        print(f"[eff-tsys] Applied per-diameter η scaling to DATA,SIGMA,WEIGHT in {msname}")
    finally:
        tb.close()


# --- ANTENNA & BASELINE GROUPING ------------------------------------------

def _read_antenna_table(vis):
    """
    Return (diameters_m, antenna_names) from the ANTENNA table of an MS.
    """
    tb.open(vis + "/ANTENNA")
    try:
        diams = np.array(tb.getcol("DISH_DIAMETER"), dtype=float)
        names = tb.getcol("NAME")
    finally:
        tb.close()
    return diams, names


def build_baseline_group_selectors_by_diameter(vis, tol=0.01):
    """
    Build CASA antenna-selection strings for baseline groups:

        'A'      : A–A (largest-diameter dishes only)
        'B'      : B–B (second-largest diameter)
        'cross'  : A–B
        'all'    : all baselines (*)

    Parameters
    ----------
    vis : str
        Measurement Set path.
    tol : float
        Diameter matching tolerance in meters.

    Returns
    -------
    dict
        {'A': 'ant1,ant2,...&',
         'B': '...',
         'cross': 'A_ants&B_ants',
         'all': '*'}
    """

    diams, names = _read_antenna_table(vis)

    uniq = sorted({round(float(d), 3) for d in diams if d == d}, reverse=True)
    if not uniq:
        return {'all': '*'}

    D_A = uniq[0]
    D_B = uniq[1] if len(uniq) > 1 else None

    A_ants = [names[i] for i, d in enumerate(diams) if abs(d - D_A) < tol]
    B_ants = [names[i] for i, d in enumerate(diams) if (D_B is not None and abs(d - D_B) < tol)]

    print(f"[groups] D_A={D_A} m → A_ants={A_ants}")
    print(f"[groups] D_B={D_B} m → B_ants={B_ants}")

    def _within_group_selector(lst):
        if len(lst) < 2:
            return ""
        return ",".join(lst) + "&"

    def _cross_selector(lst1, lst2):
        if not lst1 or not lst2:
            return ""
        s1 = ",".join(lst1)
        s2 = ",".join(lst2)
        return f"{s1}&{s2}"

    antsels = {'all': '*'}

    sel_A = _within_group_selector(A_ants)
    sel_B = _within_group_selector(B_ants)
    sel_cross = _cross_selector(A_ants, B_ants)

    if sel_A:
        antsels['A'] = sel_A
    if sel_B:
        antsels['B'] = sel_B
    if sel_cross:
        antsels['cross'] = sel_cross

    return antsels


def build_group_ms(vis, antsels, suffix):
    """
    From a full MS and a dict of antenna selectors, build A/B/cross sub-MSs
    via mstransform. Returns {key: ms_path}.
    """
    group_ms = {}
    for key in ("A", "B", "cross"):
        antsel = antsels.get(key)
        if not antsel:
            continue

        outvis = vis.rstrip("/") + f"_{suffix}_{key}.ms"
        if os.path.exists(outvis):
            logging.info("[group_ms] Removing existing %s", outvis)
            shutil.rmtree(outvis)

        logging.info(
            "[group_ms] mstransform → %s (group=%s, antenna='%s')",
            outvis, key, antsel,
        )
        mstransform(
            vis=vis,
            outputvis=outvis,
            antenna=antsel,
            datacolumn="data",
            keepflags=True,
            regridms=False,
            chanaverage=False,
            timeaverage=False,
        )
        group_ms[key] = outvis
    return group_ms

# --- MS / VIS STATS & SEFD -------------------------------------------------

def _corr_type_names(corr_type_ids):
    # https://casacore.github.io/casacore/Stokes_8h_source.html
    m = {
        1: "I",   2: "Q",   3: "U",   4: "V",
        5: "RR",  6: "RL",  7: "LR",  8: "LL",
        9: "XX", 10: "XY", 11: "YX", 12: "YY",
    }
    return [m.get(int(x), str(int(x))) for x in corr_type_ids]



def checkvals(vis, datacol='DATA', combine_conj=True, max_baselines=None):
    """
    Compute RMS of complex visibilities per (antenna1, antenna2) baseline.

    Parameters
    ----------
    vis : str
        Path to the Measurement Set.
    datacol : str
        Column to inspect ('DATA', 'CORRECTED_DATA', etc.).
    combine_conj : bool
        If True, (i,j) and (j,i) are treated as the same baseline.
    max_baselines : int or None
        If set, only the first N baselines are printed (return still has all).

    Returns
    -------
    baseline_stats : list of dict
        Each dict has: ant1, ant2, blkey, D1, D2, npts, rms_re, rms_im, rms_abs.
    """

    tb.open(vis)
    try:
        cols = set(tb.colnames())
        if datacol not in cols:
            raise RuntimeError(f"{vis}: column {datacol!r} not found in MS.")
        ant1 = tb.getcol('ANTENNA1')
        ant2 = tb.getcol('ANTENNA2')
        data = tb.getcol(datacol)           # (npol, nchan, nrow)
        flag = tb.getcol('FLAG') if 'FLAG' in cols else None
    finally:
        tb.close()

    # Antenna diameters
    diams, names = _read_antenna_table(vis)

    npol, nchan, nrow = data.shape
    accum = {}  # (i,j) -> [sum_re2, sum_im2, sum_abs2, n]

    for r in range(nrow):
        i = int(ant1[r])
        j = int(ant2[r])

        if combine_conj and j < i:
            i, j = j, i

        key = (i, j)
        if key not in accum:
            accum[key] = [0.0, 0.0, 0.0, 0]

        slice_ij = data[:, :, r]   # (npol, nchan)
        if flag is not None:
            m = ~flag[:, :, r]
            vals = slice_ij[m]
        else:
            vals = slice_ij.ravel()

        if vals.size == 0:
            continue

        re = vals.real
        im = vals.imag
        abs2 = re**2 + im**2

        accum[key][0] += float((re**2).sum())
        accum[key][1] += float((im**2).sum())
        accum[key][2] += float(abs2.sum())
        accum[key][3] += int(vals.size)

    baseline_stats = []
    for (i, j), (sre2, sim2, sabs2, n) in sorted(accum.items()):
        if n == 0:
            continue
        rms_re  = math.sqrt(sre2 / n)
        rms_im  = math.sqrt(sim2 / n)
        rms_abs = math.sqrt(sabs2 / n)
        D1 = float(diams[i]) if i < len(diams) else float('nan')
        D2 = float(diams[j]) if j < len(diams) else float('nan')
        bl = {
            'ant1': i,
            'ant2': j,
            'name1': names[i] if i < len(names) else f'A{i}',
            'name2': names[j] if j < len(names) else f'A{j}',
            'D1_m': D1,
            'D2_m': D2,
            'blkey': f"{i:02d}-{j:02d}",
            'npts': n,
            'rms_re_Jy': rms_re,
            'rms_im_Jy': rms_im,
            'rms_abs_Jy': rms_abs,
        }
        baseline_stats.append(bl)

    #print("Calculated PB size for type A (dia=%2.2f) : %3.5f arcmin"%(D_A, calc_ang(freq,D_A)))
    #print("Calculated PB size for type B (dia=%2.2f) : %3.5f arcmin"%(D_B, calc_ang(freq,D_B)))
    _print_baseline_type_summary(baseline_stats)
    return baseline_stats

def _print_baseline_type_summary(baseline_stats):
    """
    Summarize per-baseline RMS into baseline-type groups like '12m-12m', '12m-18m', etc.

    baseline_stats : list of dict
        Output from checkvals(), each with keys:
        'D1_m', 'D2_m', 'npts', 'rms_abs_Jy', ...
    """
    import math
    from collections import defaultdict

    # Group by *dish size pairs*, smallest diameter first, e.g. 12-12, 12-18, 18-18
    groups = defaultdict(lambda: {'sum_abs2': 0.0, 'npts': 0, 'nbl': 0})

    for bl in baseline_stats:
        d1 = float(bl['D1_m'])
        d2 = float(bl['D2_m'])
        n  = int(bl['npts'])
        sig = float(bl['rms_abs_Jy'])

        # Skip weird entries
        if not (n > 0 and sig == sig):
            continue

        # (12,18) and (18,12) -> '12m-18m'
        if d2 < d1:
            d1, d2 = d2, d1
        label = f"{d1:.0f}m-{d2:.0f}m"

        g = groups[label]
        g['sum_abs2'] += (sig**2) * n   # accumulate variance × N
        g['npts']     += n
        g['nbl']      += 1

    # Also accumulate "All"
    total = {'sum_abs2': 0.0, 'npts': 0, 'nbl': 0}
    for g in groups.values():
        total['sum_abs2'] += g['sum_abs2']
        total['npts']     += g['npts']
        total['nbl']      += g['nbl']

    print("\n[Baseline-type noise summary]")
    hdr = ("Baseline", "n_bl", "Npts_tot", "σ_vis(sim) [Jy]", "σ_im≈σ_vis/√N [Jy]")
    print(" " + " | ".join(f"{h:>14}" for h in hdr))
    print("-" * 80)

    # Per-type rows (12m-12m, 12m-18m, 18m-18m, ...)
    for label in sorted(groups.keys()):
        g    = groups[label]
        npts = g['npts']
        nbl  = g['nbl']
        if npts <= 0:
            continue
        sigma_vis = math.sqrt(g['sum_abs2'] / npts)
        sigma_im  = sigma_vis / math.sqrt(npts)

        print(" " + " | ".join([
            f"{label:>14}",
            f"{nbl:14d}",
            f"{npts:14d}",
            f"{sigma_vis:14.4e}",
            f"{sigma_im:14.4e}",
        ]))

    # "All" row
    if total['npts'] > 0 and len(groups) > 1:
        sigma_vis = math.sqrt(total['sum_abs2'] / total['npts'])
        sigma_im  = sigma_vis / math.sqrt(total['npts'])
        print(" " + " | ".join([
            f"{'All':>14}",
            f"{total['nbl']:14d}",
            f"{total['npts']:14d}",
            f"{sigma_vis:14.4e}",
            f"{sigma_im:14.4e}",
        ]))


def _amp_stats_ms(ms_path):
    """
    Amplitude + weight stats on an MS.

    Returns dict with:
       mean_amp, std_amp, npts, meanwt
       (from |DATA|, FLAG, WEIGHT/WEIGHT_SPECTRUM)
    """
    tb.open(ms_path)
    try:
        data = tb.getcol("DATA")          # (pol, chan, row)
        colnames = set(tb.colnames())
        flag = tb.getcol("FLAG") if "FLAG" in colnames else None

        if "WEIGHT_SPECTRUM" in colnames:
            ws = tb.getcol("WEIGHT_SPECTRUM")  # (pol, chan, row)
            W = None
        else:
            ws = None
            W = tb.getcol("WEIGHT") if "WEIGHT" in colnames else None
    finally:
        tb.close()

    if flag is not None:
        good = ~flag
        vals = data[good]
    else:
        vals = data.ravel()

    if vals.size == 0:
        return dict(mean_amp=float("nan"),
                    std_amp=float("nan"),
                    npts=0,
                    meanwt=float("nan"))

    amp = np.abs(vals)
    mean_amp = float(np.mean(amp))
    std_amp = float(np.std(amp))
    npts = int(amp.size)

    if ws is not None:
        w_vals = ws[good] if flag is not None else ws.ravel()
    elif W is not None:
        # W shape: (pol, row) → broadcast over channels
        npol, nrow = W.shape
        nchan = data.shape[1]
        W_b = np.repeat(W, nchan, axis=1).reshape(npol, nchan, nrow)
        w_vals = W_b[good] if flag is not None else W_b.ravel()
    else:
        w_vals = np.ones_like(amp)

    meanwt = float(np.mean(w_vals)) if w_vals.size > 0 else float("nan")
    return dict(mean_amp=mean_amp, std_amp=std_amp, npts=npts, meanwt=meanwt)


def estimate_sefd_by_diameter(msname):
    """
    Estimate effective SEFD (Jy) per dish diameter directly from a noisy MS.

    Uses:
      * per-baseline-type RMS from checkvals()
      * radiometer eq: σ_ij = sqrt(SEFD_i * SEFD_j) / sqrt(2 Δν t)

    For homogeneous baselines D–D:
      SEFD(D) = σ_DD * sqrt(2 Δν t)

    Returns
    -------
    dict
        {diameter_m: sefd_Jy}
    """
    import math
    from collections import defaultdict

    baseline_stats = checkvals(msname, datacol='DATA')  # already prints summary

    # Group RMS by (D1, D2) (smallest first)
    group_rms = defaultdict(lambda: {'sum_sig2': 0.0, 'n': 0})

    for bl in baseline_stats:
        d1 = float(bl['D1_m'])
        d2 = float(bl['D2_m'])
        sig = float(bl['rms_abs_Jy'])
        n   = int(bl['npts'])
        if not (n > 0 and np.isfinite(sig)):
            continue
        if d2 < d1:
            d1, d2 = d2, d1
        key = (round(d1, 3), round(d2, 3))
        group_rms[key]['sum_sig2'] += (sig ** 2) * n
        group_rms[key]['n']        += n

    # Δν from SPW0; t_int from MAIN
    tb.open(msname + "/SPECTRAL_WINDOW")
    try:
        chan_width = float(tb.getcell("CHAN_WIDTH", 0))  # Hz
        dnu = abs(chan_width)
    finally:
        tb.close()

    tb.open(msname)
    try:
        t_int = float(tb.getcell("INTERVAL", 0))  # s
    finally:
        tb.close()

    if not (dnu > 0 and t_int > 0):
        print(f"[SEFD] Invalid dnu={dnu}, t_int={t_int}; cannot estimate SEFD.")
        return {}

    sefd_map = {}
    eta_s = 0.88

    for (d1, d2), acc in group_rms.items():
        if d1 != d2:
            continue
        n = acc['n']
        if n <= 0:
            continue
        sigma_dd = math.sqrt(acc['sum_sig2'] / n)  # Jy
        sefd = sigma_dd * eta_s * math.sqrt(2.0 * dnu * t_int)  # Jy
        sefd_map[d1] = sefd

    if not sefd_map:
        print("[SEFD] No homogeneous baseline groups found; cannot estimate SEFDs.")
        return {}

    print("\n[SEFD] Effective SEFD per dish diameter (from noise in MS):")
    print("  Dia (m) |  SEFD (Jy)")
    print("  --------------------")
    for d in sorted(sefd_map.keys()):
        print(f"  {d:7.3f} | {sefd_map[d]:9.1f}")

    # Optional check for cross baselines
    if len(sefd_map) >= 2:
        diams = sorted(sefd_map.keys())
        dA, dB = diams[0], diams[1]
        sefd_A = sefd_map[dA]
        sefd_B = sefd_map[dB]
        sigma_AB_theory = (1/eta_s) * math.sqrt(sefd_A * sefd_B) / math.sqrt(2.0 * dnu * t_int)

        key_AB = (round(min(dA, dB), 3), round(max(dA, dB), 3))
        if key_AB in group_rms:
            acc_AB = group_rms[key_AB]
            n_AB = acc_AB['n']
            if n_AB > 0:
                sigma_AB_meas = math.sqrt(acc_AB['sum_sig2'] / n_AB)
                print("\n[SEFD] Cross-baseline check:")
                print(f"  σ_AB(meas)   = {sigma_AB_meas:.4e} Jy")
                print(f"  σ_AB(theory) = {sigma_AB_theory:.4e} Jy (sqrt(SEFD_A SEFD_B))")

    return sefd_map


def debug_ms_summary(msname, label: str = ""):
    """
    Summarize an MS: fields, spw/channels/bandwidth, pol products,
    antenna diameters, integration time, and weights.
    """
    print(f"\n====[ MS SUMMARY {label} ]==== {msname}")

    # --- FIELD info ---
    tb.open(f"{msname}/FIELD")
    try:
        nfields = tb.nrows()
        cols = set(tb.colnames())
        if "NAME" in cols:
            names = tb.getcol("NAME").tolist()
        else:
            names = []
        extra = " …" if len(names) > 6 else ""
        print(f" fields: {nfields}  names: {names[:6]}{extra}")
    finally:
        tb.close()

    # --- SPW info: channels + bandwidth ---
    tb.open(f"{msname}/SPECTRAL_WINDOW")
    try:
        nspw = tb.nrows()
        cols = set(tb.colnames())

        num_chan = tb.getcol("NUM_CHAN") if "NUM_CHAN" in cols else []
        nchan0 = int(num_chan[0]) if len(num_chan) else "NA"

        if "TOTAL_BANDWIDTH" in cols:
            bw = tb.getcol("TOTAL_BANDWIDTH").astype(float)
        elif "CHAN_WIDTH" in cols:
            cw = tb.getcol("CHAN_WIDTH")
            # CHAN_WIDTH usually (nchan, nspw)
            cw = np.asarray(cw, dtype=float)
            if cw.ndim == 2:
                bw = np.abs(cw).sum(axis=0)
            else:
                bw = np.array([np.abs(cw).sum()], dtype=float)
        else:
            bw = np.array([], dtype=float)

        bw0 = float(bw[0]) if bw.size else float("nan")
        print(f" spws: {nspw}  nchan(spw0): {nchan0}  bw_Hz(spw0): {bw0:.6g}")
    finally:
        tb.close()

    # --- POLARIZATION info ---
    tb.open(f"{msname}/POLARIZATION")
    try:
        npol = int(tb.getcol("NUM_CORR")[0])
        corr_types = tb.getcell("CORR_TYPE", 0).tolist()
        corr_names = _corr_type_names(corr_types)
        print(f" correlations: npol={npol}  types={corr_names}")
    finally:
        tb.close()

    # --- ANTENNA diameters (via helper) ---
    diams, _names = _read_antenna_table(msname)
    h = Counter(np.round(diams.astype(float), 3))
    print(f" antennas: {len(diams)}  diameters(m) histogram: {dict(h)}")

    # --- MAIN table: nvis, integration time, weights ---
    tb.open(msname)
    try:
        cols = set(tb.colnames())

        if "TIME_CENTROID" in cols:
            times = tb.getcol("TIME_CENTROID")
            times = np.asarray(times, dtype=float)
            if times.size > 1:
                tau = float(np.median(np.diff(np.unique(times))))
            else:
                tau = float("nan")
        else:
            tau = float("nan")
        print(f" rows(nvis): {tb.nrows()}  integration τ≈{tau:.6g}s")

        if "WEIGHT" in cols:
            sW = float(tb.getcol("WEIGHT").sum())
            print(f" sum(WEIGHT): {sW:.6g}")
        else:
            print(" WEIGHT column: MISSING")

        if "WEIGHT_SPECTRUM" in cols:
            sWS = float(tb.getcol("WEIGHT_SPECTRUM").sum())
            print(f" sum(WEIGHT_SPECTRUM): {sWS:.6g}")
        else:
            print(" WEIGHT_SPECTRUM: (absent)")
    finally:
        tb.close()

    print("====[ end MS SUMMARY ]====\n")


# -----------------------------------------------------------------------------
# IMAGE STATS & MASKS
# -----------------------------------------------------------------------------

def _central_box_slices(nx, ny, frac_area):
    """
    Return (x_slice, y_slice) for a central box covering `frac_area` of the image.
    """
    lin_frac = float(frac_area) ** 0.5
    half_x = int(lin_frac * nx / 2.0)
    half_y = int(lin_frac * ny / 2.0)
    cx, cy = nx // 2, ny // 2
    x_start = max(0, cx - half_x)
    x_end   = min(nx, cx + half_x)
    y_start = max(0, cy - half_y)
    y_end   = min(ny, cy + half_y)
    return slice(x_start, x_end), slice(y_start, y_end)

def get_rms_casa(
    image_name,
    pb_image=None,
    pb_min=None,
    center_area_frac=0.000,
    keep_center_frac=None,
    nsigma_clip=5.0,
    max_points=2_000_000,
    debug=False,
    label=None,
):
    """
    Robust RMS estimator for CASA images (Jy/beam), with optional debug output.

    Strategy:
      1. Start from CASA's internal mask (no-data regions excluded).
      2. Optionally require PB > pb_min if pb_image is given.
      3. Optionally exclude a central box (center_area_frac) OR keep only a central
         box (keep_center_frac).
      4. Drop exact zeros.
      5. Sigma-clip around the median and compute RMS.

    If debug=True, also compute a plain imstat RMS and print a comparison.
    """

    if label is None:
        label = image_name

    # --- Load image + mask ---
    ia.open(image_name)
    shape = ia.shape()
    data = ia.getchunk()
    mask = ia.getchunk(getmask=True)
    ia.close()

    nx, ny = shape[0], shape[1]

    # --- Optional PB mask ---
    if pb_image is not None and pb_min is not None:
        ia.open(pb_image)
        pb = ia.getchunk()
        ia.close()
        mask &= (pb >= pb_min)

    # --- Central region handling ---
    # 2a) Exclude a central box
    if center_area_frac is not None and center_area_frac > 0.0 and not keep_center_frac:
        sx, sy = _central_box_slices(nx, ny, center_area_frac)
        slices = [sx, sy] + [slice(None)] * (len(shape) - 2)
        mask[tuple(slices)] = False

    # 2b) KEEP only a central box, but only if frac > 0
    if keep_center_frac is not None and keep_center_frac > 0.0:
        sx, sy = _central_box_slices(nx, ny, keep_center_frac)
        new_mask = np.zeros_like(mask, dtype=bool)
        slices = [sx, sy] + [slice(None)] * (len(shape) - 2)
        new_mask[tuple(slices)] = mask[tuple(slices)]
        mask = new_mask

    # --- Apply mask + drop exact zeros ---
    valid = mask & (data != 0.0)
    values = data[valid]

    # If excluding the centre killed everything, fall back to mask+nonzero only
    if values.size == 0:
        valid = mask & (data != 0.0)
        values = data[valid]

    if values.size == 0:
        # Truly no valid pixels
        rms_robust = float("nan")
    else:
        # Downsample if needed
        if values.size > max_points:
            idx = np.random.choice(values.size, max_points, replace=False)
            values = values[idx]

        values = values.astype(np.float64).ravel()

        # Sigma-clipped RMS around the median
        med = np.median(values)
        mad = np.median(np.abs(values - med))

        if mad > 0:
            sigma = 1.4826 * mad
        else:
            sigma = np.std(values)

        if sigma <= 0:
            rms_robust = float(np.sqrt(np.mean(values**2)))
        else:
            good = np.abs(values - med) < nsigma_clip * sigma
            clipped = values[good]
            if clipped.size < 0.1 * values.size:
                clipped = values
            rms_robust = float(np.sqrt(np.mean(clipped**2)))

    # --- Optional debug: compare with imstat ---
    if debug:
        try:
            s = imstat(imagename=image_name)
            rms_imstat = float(s["rms"][0])
        except Exception as e:
            print(f"[rms-debug] imstat failed for {label}: {e}")
            rms_imstat = float("nan")

        pb_txt = f"> {pb_min}" if (pb_image is not None and pb_min is not None) else "none"
        print(f"\n[rms-debug] {label}")
        print(f"  imstat rms (full mask)                  = {rms_imstat:.3e} Jy/beam")
        print(
            f"  get_rms_casa (PB {pb_txt}, "
            f"centre_frac={center_area_frac}, keep_centre={keep_center_frac}) "
            f"= {rms_robust:.3e} Jy/beam"
        )

    return rms_robust



def get_peak(im):
    """
    Return the maximum pixel value in a CASA image.
    """

    ia.open(im)
    stats = ia.statistics()
    ia.close()
    return stats['max'][0]



def measure_image_stats(
    img_base,
    args=None,
    use_pb=None,
    pb_min=None,
    center_area_frac=None,
    keep_center_frac=None,
    nsigma_clip=None,
    max_points=None,
):
    """
    Robust summary stats for an image:
      - peak from <img_base>.image
      - imstat RMS from <img_base>.residual
      - robust RMS via get_rms_casa (PB + masks if requested)

    Behaviour is controlled by either explicit kwargs or (if args is given)
    by the following attributes:

      rms_mode            : 'full', 'pb', or 'auto'
      rms_use_pb          : (used only when rms_mode == 'auto')
      rms_pb_min
      rms_center_frac
      rms_keep_center_frac
      rms_nsigma_clip
      rms_max_points
    """

    residual = f"{img_base}.residual"
    image    = f"{img_base}.image"
    pb_image = f"{img_base}.pb"

    # ------------------------------------------------------------------
    # Resolve defaults from args (if given)
    # ------------------------------------------------------------------
    if args is not None:
        rms_mode = getattr(args, "rms_mode", "auto")

        if pb_min is None:
            pb_min = getattr(args, "rms_pb_min", 0.05)
        if center_area_frac is None:
            center_area_frac = getattr(args, "rms_center_frac", 0.0)
        if keep_center_frac is None:
            keep_center_frac = getattr(args, "rms_keep_center_frac", None)
        if nsigma_clip is None:
            nsigma_clip = getattr(args, "rms_nsigma_clip", 5.0)
        if max_points is None:
            max_points = getattr(args, "rms_max_points", 2_000_000)

        # Decide whether to use PB:
        #  - if use_pb is explicitly passed to the function, respect that
        #  - otherwise follow rms_mode:
        #      'full' -> never use PB
        #      'pb'   -> always use PB
        #      'auto' -> fall back to args.rms_use_pb
        if use_pb is None:
            if rms_mode == "full":
                use_pb = False
            elif rms_mode == "pb":
                use_pb = True
            else:  # 'auto'
                use_pb = getattr(args, "rms_use_pb", False)

    else:
        # Hard defaults if args=None
        rms_mode = "auto"
        if use_pb is None:
            use_pb = False
        if pb_min is None:
            pb_min = 0.05
        if center_area_frac is None:
            center_area_frac = 0.0
        if keep_center_frac is None:
            keep_center_frac = None
        if nsigma_clip is None:
            nsigma_clip = 5.0
        if max_points is None:
            max_points = 2_000_000

    # ------------------------------------------------------------------
    # If PB requested but pb image doesn't exist → silently disable
    # ------------------------------------------------------------------
    if use_pb and not os.path.isdir(pb_image):
        use_pb = False

    eff_pb_image = pb_image if use_pb else None
    eff_pb_min   = pb_min if use_pb else None

    # ------------------------------------------------------------------
    # imstat RMS on residual
    # ------------------------------------------------------------------
    try:
        s_res = imstat(residual)
        rms_imstat = float(s_res["rms"][0])
    except Exception as e:
        print(f"[measure_image_stats] imstat failed on {residual}: {e}")
        rms_imstat = float("nan")

    # ------------------------------------------------------------------
    # robust RMS via get_rms_casa
    # ------------------------------------------------------------------
    try:
        rms_robust = get_rms_casa(
            image_name=residual,
            pb_image=eff_pb_image,
            pb_min=eff_pb_min,
            center_area_frac=center_area_frac,
            keep_center_frac=keep_center_frac,
            nsigma_clip=nsigma_clip,
            max_points=max_points,
        )
    except Exception as e:
        print(f"[measure_image_stats] get_rms_casa failed on {residual}: {e}")
        rms_robust = float("nan")

    # ------------------------------------------------------------------
    # peak from main image
    # ------------------------------------------------------------------
    try:
        peak = get_peak(image)
    except Exception as e:
        print(f"[measure_image_stats] get_peak failed on {image}: {e}")
        peak = float("nan")

    return {
        "peak": peak,
        "rms_imstat": rms_imstat,
        "rms_robust": rms_robust,
        "used_pb": bool(use_pb),
        "pb_min": eff_pb_min,
        "center_area_frac": center_area_frac,
        "keep_center_frac": keep_center_frac,
        "rms_mode": rms_mode,
    }




def read_restoring_beam(imname):
    ia = image()
    ia.open(imname)
    rb = ia.restoringbeam()
    ia.close()
    # Single-beam case
    if isinstance(rb, dict) and 'major' in rb:
        return (float(rb['major']['value']),
                float(rb['minor']['value']),
                float(rb['positionangle']['value']))
    # Cube/per-plane case
    if isinstance(rb, dict) and 'beams' in rb:
        b0 = rb['beams']['*0']['*0']  # first plane
        return (float(b0['major']['value']),
                float(b0['minor']['value']),
                float(b0['positionangle']['value']))
    # Fallback
    return (float('nan'), float('nan'), float('nan'))



def make_center_box_mask_crtf(imagename, frac_area=0.25, out_crtf=None):
    hd = imhead(imagename, mode='list')
    nx, ny = hd['shape'][0], hd['shape'][1]
    sx, sy = _central_box_slices(nx, ny, frac_area)
    x0, x1 = sx.start, sx.stop
    y0, y1 = sy.start, sy.stop

    region = f"box[[{x0:.3f}pix,{y0:.3f}pix],[{x1:.3f}pix,{y1:.3f}pix]]"
    out_crtf = out_crtf or (os.path.splitext(imagename)[0] + "_center25.crtf")
    with open(out_crtf, "w") as f:
        f.write("#CRTFv0\n")          # <-- REQUIRED HEADER
        f.write(region + "\n")        # the region line
    print(f"[mask] wrote {out_crtf}: {region}")
    return out_crtf

# === SKY MODEL & VP TABLE ===
def generate_sky_model(dec, sky_script, sky_config_path, casa_bin, overrides=None, output_suffix=None):
    cfg = Path(sky_config_path)
    if not cfg.is_absolute():
        for candidate in (SCRIPT_DIR / cfg, START_DIR / cfg):
            if candidate.exists(): cfg = candidate; break
    if not cfg.exists(): raise FileNotFoundError(f"Missing sky config: {cfg}")

    sky_py = Path(sky_script)
    if not sky_py.is_absolute():
        for candidate in (SCRIPT_DIR / sky_py, START_DIR / sky_py):
            if candidate.exists(): sky_py = candidate; break
    if not sky_py.exists(): raise FileNotFoundError(f"Missing sky script: {sky_py}")

    with open(cfg) as f:
        sky_config = json.load(f)
    sky_config["dec_center"] = dec
    if overrides: sky_config.update(overrides)

    output_base = sky_config.get("output_base", "skyModel")
    if output_suffix: output_base = f"{output_base}_{output_suffix}"
    sky_config["output_base"] = output_base

    tag = dec.replace('-', 'm').replace('+', 'p').replace('d', '').replace('.', '')
    fitsname = (START_DIR / f"{output_base}_dec{tag}.fits").resolve()
    temp_config = (START_DIR / f"skyconfig_{tag}_{output_suffix or 'nosuf'}.json").resolve()

    with open(temp_config, "w") as f:
        json.dump(sky_config, f, indent=2)
    if fitsname.exists(): fitsname.unlink()

    cmd = [casa_bin, "--nogui", "--nologger", "-c", str(sky_py), str(temp_config)]
    logging.info(f"Running sky generator: {cmd}")
    subprocess.run(cmd, check=True)

    if not fitsname.exists():
        raise RuntimeError(f"Failed to generate sky model: {fitsname}")
    return str(fitsname)

def _alma_band_from_freq_ghz(freq_ghz: float):
    """
    Map observing frequency (GHz) to nominal ALMA band.
    Ranges from ALMA documentation.:contentReference[oaicite:0]{index=0}
    Returns an int band number or None if out of range.
    """
    band_ranges = {
        1:  (35.0, 50.0),
        3:  (84.0, 116.0),
        4:  (125.0, 163.0),
        5:  (163.0, 211.0),
        6:  (211.0, 275.0),
        7:  (275.0, 373.0),
        8:  (385.0, 500.0),
        9:  (602.0, 720.0),
        10: (787.0, 950.0),
    }
    for band, (fmin, fmax) in band_ranges.items():
        if fmin <= freq_ghz <= fmax:
            return band
    return None


def trx_from_freq_ghz(freq_ghz: float, trx_override: float | None = None) -> float:
    """
    Return receiver temperature (K).

    If trx_override is not None, that value is used directly.
    Otherwise, we mimic CASA simobserve: T_rx is linearly interpolated
    between tabulated values as a function of frequency.

    ALMA receiver temperatures (K) and frequencies (GHz) from casadocs
    (simobserve thermalnoise='tsys-atm'):

        T_rx : 25, 30, 40, 42, 50, 50, 72, 135, 105, 230 K
        ν    : 35, 75, 110, 145, 185, 230, 345, 409, 675, 867 GHz
    """
    if trx_override is not None:
        return float(trx_override)

    # Frequencies (GHz) and corresponding T_rx (K) from CASA docs
    freq_tab = np.array([35, 75, 110, 145, 185, 230, 345, 409, 675, 867], dtype=float)
    trx_tab  = np.array([25, 30,  40,  42,  50,  50,  72, 135, 105, 230], dtype=float)

    # Clamp outside the tabulated range
    if freq_ghz <= freq_tab[0]:
        val = float(trx_tab[0])
    elif freq_ghz >= freq_tab[-1]:
        val = float(trx_tab[-1])
    else:
        # Linear interpolation in frequency
        val = float(np.interp(freq_ghz, freq_tab, trx_tab))

    band = _alma_band_from_freq_ghz(freq_ghz)
    if band is not None:
        logging.info(
            "[TRX] CASA-style interpolated T_rx=%.1f K at %.1f GHz (ALMA Band %d)",
            val, freq_ghz, band
        )
    else:
        logging.info(
            "[TRX] CASA-style interpolated T_rx=%.1f K at %.1f GHz (outside nominal bands)",
            val, freq_ghz
        )

    return val



