#=================================================================================================================================#
#                               Homogeneous + Heterogeneous Array Simulator (with arbitrary dish sizes)                           #
#                                                                                                                                 #
#                                                     Hampus Österbom (2025)                                                      #
#                                                                                                                                 #
#                                                                                                                                 #
#                                       This code is based on / took inspiration from  R. V. Urvashi:                             #
#               https://github.com/urvashirau/Simulation-in-CASA/tree/master/Heterogeneous_Array_Simulation_and_Imaging           #
#=================================================================================================================================#
# // for questions about the pipeline or bug reporting: hampusosterbom1@gmail.com
"""
High-level heterogeneous + homogeneous ALMA simulation pipeline.

Responsibilities:
  * Parse configuration (dish sizes, declination, integration time)
  * Generate sky models and MSs (hetero or homo)
  * Build per-antenna primary-beam VP tables
  * Run imaging (dirty + cleaned) and optional baseline-group imaging
  * Validate image-plane noise vs theoretical σ_vis/√N
  * Write structured logs and FITS outputs
ps: Only tested on CASA 6.7!!! also, if the program cant fint your custom cfgs, try placing them where all other ALMA cfgs live. 
This is the main orchestrator tying together all QoL, PB/VP, and validation tools.

To run the program:
(i) Make sure the following files (and any others the pipeline imports) are in the same directory.
(ii) Set the CASA binary path, e.g., casa_bin = "/path/to/your/casa/bin/casa"
(iii) Adjust parameters in EXAMPLE_SETTINGS.
(iv) cd into the folder and run the pipeline using CASA in non-GUI mode: 
    casa --nogui --nologger -c run_simulation_pipeline.py
"""

import os
import json
import numpy as np
from pathlib import Path
import random
import logging
import sys
import subprocess
from types import SimpleNamespace
from casatools import  image, table, vpmanager, simulator, coordsys, measures, quanta, ms
from casatasks import (
    simobserve, tclean, exportfits, imhead, 
    importfits, 
)
from casatasks.private import simutil

sm = simulator()
ia = image()
tb = table()
cs = coordsys()
me = measures()
qa = quanta()
mysu = simutil.simutil()
myms = ms()
vp = vpmanager()

# search for imports 
sys.path[:0] = [str(Path(__file__).resolve().parent)]
from qol_pipeline import (
    makeMSFrame,                                        # builds empty MS with your cfg/pointings/spw/times
    predictImager,                                      # (optional, we won't rely on it for prediction)
    addNoiseSim,                                        # CASA tsys-atm (with per D efficiency scaling)
    build_baseline_group_selectors_by_diameter,         # Helper to build baseline groups by diameter
    make_startmodel_match,                              # regrid / resize sky model to match MS imaging geometry    
    read_restoring_beam,                                # read beam major/minor/PA from .image/.psf
    find_config_path,                                   # resolve ALMA .cfg path from name or alias
    build_ha_scans,                                     # create hour-angle scan blocks for custom simulations
    make_center_box_mask_crtf,                          # build a CRTF mask string for CLEAN (central box)
    measure_image_stats,                                # peak + RMS + beam measurement with PB/centering options
    compute_imaging_geometry,                           # derive cell, imsize, beam FWHM from array + freq
    trx_from_freq_ghz,                                  # look up receiver temperature T_rx based on ALMA band
)

from utils import(
    setup_logging,                                      # configure global logging format + levels
    safe_rm_tree,                                       # safely remove a directory tree (w/ guardrails)
    wipe,                                               # delete all tclean-generated files for an imagename
    make_short_tag,                                     # create compact run identifier from parameters
)


from pb_vp import(
    build_per_antenna_pb_vptable,                       # make PB images per diameter & assemble VP table
    inspect_vptable,                                    # quick inspection of VP table contents + diameters
)

from validation import(
    run_noise_validation_single_field,                  # image-plane noise validation on noise-only MS
    hetero_checkVals,                                   # vis-domain hetero checks: A/A, B/B, A/B noise + stats
    estimate_sefd_by_diameter,                          # estimate SEFD(D) from σ_vis on same-diameter baselines
    debug_ms_summary,                                   # quick summary: nants, spws, vis ranges, diameters

)

# ==============================================================================
# User-editable settings for example_run()
# Edit these values and re-run the script.
# // Most of these values do not need changing, they are purely here to give the
# freedom to modify (almost) anything!
# ==============================================================================

EXAMPLE_SETTINGS = dict(

    # --- Paths / project structure ---
    project_base="ALMA.validate",                 # Folder where all outputs go
    casa_bin="/home/hampus/casa-6.7.0-31-py3.10.el8/bin/casa",  # CASA executable 
    extra_dirs=[],                                # Extra dirs to search for cfg files

    # --- Array configuration(s) ---
    config_map={"alma.demo3.cfg": "GG"},           # {cfg_file : short_tag}      
    declinations=["-23d00m00.00"],                # Target declination(s)
    integration_times=["1h"],                     # On-source time(s)

    # --- Frequency setup ---
    center_freq="343.5GHz",                       # Reference frequency
    width="7.5GHz",                               # Total bandwidth
    pwv=1.262,                                    # PWV for ATM noise

    # --- Imaging / sky setup ---
    phasecenter_ra="12h00m00.00s",                # Phase center RA
    fov_arcsec=1.28,                              # Desired field of view (arcsec)
    sampling_factor=5.0,                          # Beam/cell sampling (higher -> smaller pixels)

    # --- Sky model selection ---
    # sky_mode controls *where* the sky model comes from:
    #   "builtin" : use one of the internal test skies (sky_type below)
    #   "fits"    : use a user-supplied FITS image directly
    sky_mode="builtin",

    # If sky_mode == "builtin", choose between:
    #   "ring"      : 8+1 Gaussian ring (central Gaussian + 8 around, ring Gaussians are 100x fainter)
    #   "pointdisk" : bright central point source + faint small disk
    sky_type="ring",

    # If sky_mode == "fits", this must point to an existing FITS file
    # that will be imported and used as the sky brightness model.
    sky_fits=None,                                # e.g. "/path/to/my_sky_model.fits"

    # --- Built-in sky parameters (only used for sky_mode="builtin") ---
    # Ring model (sky_type="ring"):
    ring_flux=30e-8,                              # Jy of central Gaussian (set VERY low for an essentially noise-only image!)
    ring_n_sources=8,                             # Number of Gaussians in the ring
    ring_radius_as=0.15,                          # Ring radius in arcsec
    ring_major_as=0.035,                          # Major axis of each Gaussian (arcsec)
    ring_minor_as=0.035,                          # Minor axis of each Gaussian (arcsec)
    ring_pa_deg=0.0,                              # Position angle of Gaussians (deg)

    # Point+disk model (sky_type="pointdisk"):
    pnd_flux=1.0,                                 # Base total flux (Jy) (set VERY low for an essentially noise-only image!)
    pnd_point_flux=None,                          # If None -> uses pnd_flux
    pnd_disk_flux=None,                           # If None -> uses 0.1 * pnd_flux
    pnd_disk_diam_as=0.10,                        # Disk diameter (arcsec),

    # --- Optional manual geometry overrides ---
    # The pipeline normally computes "good" imaging geometry automatically
    # (cell size + image size) from the array beam and fov_arcsec.
    # These two knobs let you override that behaviour:
    #
    #   * If cell_arcsec is not None:
    #       - this fixed cell size (arcsec / pixel) is used for BOTH
    #         the sky model and all imaging.
    #   * If imsize_manual is not None:
    #       - this fixed image size (pixels) is used for BOTH the
    #         sky model and all imaging.
    #
    # If either is None, the auto-geometry from compute_imaging_geometry()
    # is used for that parameter instead.
    cell_arcsec=None,                             # Fixed cell size (arcsec), overrides auto if set
    imsize_manual=None,                           # Fixed image size (pixels), overrides auto if set

    # --- tclean / deconvolution ---
    weightings=["briggs"],                        # Weighting scheme(s)
    robust_values=[0.5],                          # Briggs robust parameter(s)
    uvtaper_values=["0.0arcsec"],                 # UV taper(s)
    niter=0,                                    # CLEAN iterations (0 = dirty)
    deconvolver="multiscale",                     # Deconvolver type
    threshold_factor=2,                           # CLEAN threshold = factor x dirty RMS

    # global pblimit used for all tclean calls that specify pblimit
    pblimit=0.2,
    # Cleaning mask settings
    clean_use_mask=True,                          # If True, use a central box mask for CLEAN (only if niter>1)
    clean_mask_center_frac=0.15,                   # Fraction of image size to keep as central mask

    # --- Noise / Primary Beam ---
    noise_mode="atm",                             # 'atm', 'manual', or 'none' (atm is HIGHLY recommended)

    # --- VP / PB usage for homogeneous configs ---
    use_vptable_in_homo=True,                     # If True: build/use a VP table even for homogeneous arrays (PB-aware imaging)

    # Antenna/aperture efficiencies for tsys-atm noise model // Eff. for 7-and 12m at various freqs can be found in 
    # the ALMA technical handbook page 147 (cycle 12 handbook)
    eta_A=0.63,                                   # Efficiency for large dishes (e.g. 12m)
    eta_B=0.63,                                   # Efficiency for small dishes (e.g. 7m)
    spillefficiency=0.96,                         # Spillover efficiency
    trx_K=None,                                   # Receiver temperature (K) ; None => auto from ALMA band & center_freq
    correfficiency=0.88,                          # Correlator efficiency
    waterheight=2000.0,                           # Water vapour scale height (m)

    pb_model="alma",                              # 'alma', 'airy', or 'gauss'  ('alma' is highly recommended, but you can try 'airy' as well)

    # --- Validation / diagnostic "buttons" ---
    validate_noise_theory=True,                   # Run noise-validation imaging?
    noiseval_field="0",                           # FIELD id used for noise validation (keep as 0, only 0 is currently supported)
    noiseval_do_groups=True,                      # For heterogeneous arrays: also do A/B/cross groups?
    print_vptable=False,                          # If True: prints the combined vptable with the PB mapping and which antennas use which PB. 

    image_baseline_groups=True,                   # Also image A/B/cross groups for the main run?

    # --- Logging / cleanup ---
    log_file="pipeline_test.log",                 # Log file (None = stdout only)
    clean_project_base=True,                      # Remove previous outputs before run

    # --- RMS measurement settings (used in measure_image_stats) ---
    # PS: note that when FoV << PB, aliasing will occur, for accurate RMS, one should mask the edges of the image! 
    # Otherwise, it is found to have weird effects, like underestimating the true RMS
    rms_mode="pb",                              # 'full' = whole image, 'pb' = PB-limited region, 'auto' = apply masks below (PB mask + geometric masks)                                                                                     
    rms_use_pb=True,                              # If True: apply PB mask (keep pixels where PB >= rms_pb_min)
    rms_pb_min=0.2,                              # Primary-beam cutoff for RMS regio
    # --- Geometric masking controls ---
    rms_center_frac=None,                          # Exclude a central box of this fractional size 
                                                  # (e.g., 0.10 -> remove inner 10% area; useful to mask bright source cores)
 
    rms_keep_center_frac=None,                    # Keep only a central box of this fractional size (crops away noisy edges).
                                                  # If both rms_keep_center_frac AND rms_center_frac are set:
                                                  # -> RMS is computed in an ANNULUS-LIKE region:
                                                  # central box defined by keep_center_frac
                                                  # MINUS the inner excluded box defined by center_frac.
    # Summary of combinations:
    #   Only rms_center_frac set:       mask *only* the inner core.
    #   Only rms_keep_center_frac set:  use *only* central part of image; discard edges.
    #   Both set:                       keep middle region but punch out the inner core 
    #                                   (annulus / donut mask).
    #   Neither set:                    no geometric masking (PB mask only, if enabled).
    rms_nsigma_clip=5.0,                          # Sigma clipping for robust RMS estimation
    rms_max_points=2_000_000,                     # Max number of pixels sampled for RMS (should not have to touch this)
)

# --------------------------------------------------------------------
# CONSTANTS & LOGGING
# --------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
START_DIR = Path.cwd()


# --------------------------------------------------------------------
# Helper for tclean
# --------------------------------------------------------------------

def apply_mosaic_settings(params, args, dec, vptab=None):
    """
    Add standard mosaic / AW-projection settings to a tclean params dict.
    """
    params.update({
        'specmode': 'mfs',
        'nchan': -1,
        'phasecenter': f"J2000 {args.phasecenter_ra} {dec}",
        'spw': '',
        'outframe': 'LSRK',
        'restfreq': args.center_freq,
        'gridder': 'mosaic',
        'usepointing': True,
        'normtype': 'flatnoise',
        'pblimit': getattr(args, "pblimit", 0.1),
        'stokes': 'I',
    })
    if vptab is not None:
        params['vptable'] = vptab



#X#X#X#X#X#X#X#X#X#X#X#X#X#X#X#X#X#X#X#X#X#X#X#X Skymodel creation  #X#X#X#X#X#X#X#X#X#X#X#X#X#X#X#X#X#X#X#X#X#X#X#X#X#X#X#X#X#X#X#X#X

def generate_sky_for_config(args, dec, dyn_cell, imsize, config_tag):
    """
    Return (fits_path, fits_name) for the sky model.

    sky_mode:
      - 'fits'    -> use user-supplied FITS (args.sky_fits)
      - 'builtin' -> call generate_builtin_sky_model.py with sky_type, geometry, fluxes

    Geometry rules:
      - cell size: use args.cell_arcsec if not None, else dyn_cell from compute_imaging_geometry
      - imsize   : use args.imsize_manual if not None, else imsize from compute_imaging_geometry
    """

    mode = getattr(args, "sky_mode", "builtin")

    # -------------------------------------------------------------
    # 1) User-provided FITS
    # -------------------------------------------------------------
    if mode == "fits":
        if not getattr(args, "sky_fits", None):
            raise ValueError("sky_mode='fits' but sky_fits is not set")

        fits_path = Path(args.sky_fits).expanduser().resolve()
        if not fits_path.exists():
            raise FileNotFoundError(f"sky_fits does not exist: {fits_path}")

        return fits_path, fits_path.stem

    # -------------------------------------------------------------
    # 2) Built-in models (ring or point+disk)
    # -------------------------------------------------------------
    if mode != "builtin":
        raise ValueError(f"Unknown sky_mode='{mode}'. Use 'builtin' or 'fits'.")

    # --- Geometry with overrides ---
    # dyn_cell is like "0.004arcsec"; args.cell_arcsec is a float in arcsec
    if getattr(args, "cell_arcsec", None) is not None:
        cell_as = float(args.cell_arcsec)          # arcsec / pixel (manual override)
    else:
        cell_as = float(str(dyn_cell).rstrip("arcsec"))  # arcsec / pixel (auto)

    if getattr(args, "imsize_manual", None) is not None:
        imsize_use = int(args.imsize_manual)       # manual override
    else:
        imsize_use = int(imsize)                   # auto from compute_imaging_geometry

    # --- Script location ---
    sky_script = Path.home() / "generate_builtin_sky_model.py"
    if not sky_script.exists():
        raise FileNotFoundError(f"Cannot find builtin sky script at {sky_script}")

    sky_type = getattr(args, "sky_type", "ring")   # 'ring' or 'pointdisk'

    def make_output_base(prefix):
        return f"{prefix}_{config_tag}"

    def dec_tag(d):
        return d.replace("-", "m").replace("+", "p").replace("d", "").replace(".", "")

    output_prefix = "Ring" if sky_type == "ring" else "PplusDisk"
    output_base = make_output_base(output_prefix)

    cmd = [
        str(args.casa_bin), "--nologger", "--nogui", "--logfile", f"sky_{sky_type}.log",
        "-c", str(sky_script),
        f"--sky_type={sky_type}",
        f"--ra_center={args.phasecenter_ra}",
        f"--dec_center={dec}",
        f"--freq={args.center_freq}",
        "--im_shape", str(imsize_use), str(imsize_use),
        f"--cell_size={cell_as}",
        f"--output_base={output_base}",
    ]

    # Ring-specific parameters
    if sky_type == "ring":
        flux       = getattr(args, "ring_flux", 27e-5)
        n_ring     = getattr(args, "ring_n_sources", 8)
        radius_as  = getattr(args, "ring_radius_as", 0.044)
        major_as   = getattr(args, "ring_major_as", cell_as * 5.0)
        minor_as   = getattr(args, "ring_minor_as", cell_as * 5.0)
        pa_deg     = getattr(args, "ring_pa_deg", 0.0)

        cmd.extend([
            f"--ring_flux={flux}",
            f"--ring_n={n_ring}",
            f"--ring_radius={radius_as}",
            f"--ring_major_beam={major_as}arcsec",
            f"--ring_minor_beam={minor_as}arcsec",
            f"--ring_pa_beam={pa_deg}deg",
        ])

    # Point+disk parameters
    if sky_type == "pointdisk":
        base_flux  = getattr(args, "pnd_flux", 1.0)
        point_flux = getattr(args, "pnd_point_flux", None)
        disk_flux  = getattr(args, "pnd_disk_flux", None)
        disk_d_as  = getattr(args, "pnd_disk_diam_as", 0.10)

        cmd.append(f"--pnd_flux={base_flux}")
        if point_flux is not None:
            cmd.append(f"--pnd_point_flux={point_flux}")
        if disk_flux is not None:
            cmd.append(f"--pnd_disk_flux={disk_flux}")
        if disk_d_as is not None:
            cmd.append(f"--pnd_disk_diameter={disk_d_as}")

    logging.info(f"[sky] Running builtin sky generator: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    fits_name = f"{output_base}_dec{dec_tag(dec)}.fits"
    fits_path = (Path.cwd() / fits_name).resolve()
    if not fits_path.exists():
        raise RuntimeError(f"Expected sky FITS not found: {fits_path}")

    return fits_path, fits_path.stem

##################################################### Simulation & Imaging #########################################################################################################

def simulate_hetero_ms(args, cfg_path, proj_dir, proj_name, fits_path, dec, itime, imsize, cell_arcsec):   #vptab
    """Full heterogeneous simulation path; returns MS path and is_hetero=True."""


    dc = me.direction('J2000', args.phasecenter_ra, dec)
    pointings = [{'name': 'p0', 'dir': dc}]
    print(f"[DEBUG] Mosaic has {len(pointings)} pointings: {[p['name'] for p in pointings]}")

    scans = build_ha_scans(pointings, itime, ncycles=1, debug=True)

    ms_out = (proj_dir / f"{proj_name}.ms").as_posix()
    _ = makeMSFrame(
        msn=(proj_dir / proj_name).as_posix(),  # function appends _ALMA.ms internally
        tel='ALMA',
        cfg_path=str(cfg_path),
        pointings=pointings,
        spwname='SIM',
        freq=str(args.center_freq),
        deltafreq=str(args.width),
        freqresolution=str(args.width),
        nchannels=1,
        stokes='XX YY',
        integrationtime='10s',
        usehourangle=True,
        referencetime='2025/01/01/00:00:00',
        scans=scans,
        overwrite=True,
    )

    ms_out = f"{(proj_dir / proj_name).as_posix()}_ALMA.ms"
    tb.open(ms_out + "/ANTENNA")
    print("Diameters (m):", np.unique(tb.getcol("DISH_DIAMETER")))
    diameters = tb.getcol("DISH_DIAMETER")
    ant_names = tb.getcol("NAME")
    ant_stations = tb.getcol("STATION") if 'STATION' in tb.colnames() else None

    logging.info(f"Diameters (m): {np.unique(diameters)}")
    logging.info(f"Antenna names: {ant_names}")
    logging.info(f"Antenna stations: {ant_stations}")


    tb.close()
    debug_ms_summary(ms_out, label="HETERO before-noise")

    vptab = os.path.join(proj_dir, f"{proj_name}_perant_vp.tbl")
    vptab = build_per_antenna_pb_vptable(args, ms_out, vptab,imsize,cell_arcsec)
    
    # Optional VP table inspection (controlled by EXAMPLE_SETTINGS["print_vptable"])
    if getattr(args, "print_vptable", False):
        inspect_vptable(vptab)

    if args.noise_mode == 'atm':
        # Parse center frequency like "343.5GHz" -> 343.5
        # and map correct reciever temperatures (table 9.2 in technical handbook)

        freq_ghz = float(str(args.center_freq).rstrip("GHz"))
        trx_val = trx_from_freq_ghz(freq_ghz, getattr(args, "trx_K", None))
        addNoiseSim(
            msname=str(ms_out),
            pwv_mm=args.pwv,
            t_ground_K=getattr(args, 't_ground', 270.0),
            altitude_m=getattr(args, 'altitude', 5000.0),
            relhum=getattr(args, 'relhum', 20.0),
            pground=getattr(args, 'pressure', 650.0),
            eta_A=getattr(args, "eta_A", 0.63),
            eta_B=getattr(args, "eta_B", 0.63),
            spillefficiency=getattr(args, "spillefficiency", 0.96),
            trx_K=trx_val,
            correfficiency=getattr(args, "correfficiency", 0.88),
            waterheight=getattr(args, "waterheight", 2000.0),
            tau1=getattr(args, "tau1", 0.224),
        )

    elif args.noise_mode in ('manual', 'custom'):
        print('Woopsie,,,, the pipeline currently only supports the ATM model')

    # measure effective SEFDs from the noisy visibilities
    try:
        sefd_map = estimate_sefd_by_diameter(ms_out)
        if sefd_map:
            logging.info(f"[SEFD] Estimated SEFDs (Jy) per diameter: {sefd_map}")
    except Exception as e:
        logging.warning(f"[SEFD] Failed to estimate SEFDs for {ms_out}: {e}")

    casa_model = (proj_dir / f"{fits_path.stem}.im").as_posix()
    os.system(f"rm -rf {casa_model}")
    importfits(fitsimage=str(fits_path), imagename=casa_model, overwrite=True)


    startmodel_rg = (proj_dir / f"{fits_path.stem}.startmodel.im").as_posix()
    make_startmodel_match(ms_out, casa_model, startmodel_rg)

    predictImager(
        msname=ms_out,
        imname_true=startmodel_rg,
        gridder='mosaic',
        tel='ALMA',
        vptable=vptab,
        pblimit=getattr(args, "pblimit", 0.1),
    )


    # --- Debug: interpret MODEL_DATA vs DATA amplitudes ---
    tb.open(ms_out)
    mmax = np.nanmax(np.abs(tb.getcol('MODEL_DATA'))) if tb.nrows() > 0 else 0.0
    tb.close()

    logging.debug(
        "[predict] Max |MODEL_DATA| = %.3g Jy (per visibility). "
        "This is the predicted sky signal only; for typical faint sources this "
        "value is much smaller than the thermal noise per visibility.",
        mmax
    )

    # Add MODEL_DATA to existing noisy DATA
    tb.open(ms_out, nomodify=False)
    data  = tb.getcol("DATA")
    model = tb.getcol("MODEL_DATA")
    data += model
    tb.putcol("DATA", data)
    tb.close()

    tb.open(ms_out)
    dmax = np.nanmax(np.abs(tb.getcol('DATA'))) if tb.nrows() > 0 else 0.0
    tb.close()

    logging.debug(
        "[predict] Max |DATA| after adding model = %.3g Jy (per visibility). "
        "This is dominated by thermal noise from tsys-atm. "
        "It is normal for |DATA| >> |MODEL_DATA| unless the sky model "
        "is extremely bright or the integration time is very long.",
        dmax
    )


    return ms_out, True, vptab


def simulate_homo_ms(args, cfg_path, proj_dir, proj_base_dir, proj_name, fits_path, dec, itime, pipeline_log):
    """Homogeneous simobserve path; returns MS path and is_hetero=False."""

    # 1) Make a single-pointing ptg file at the phase center
    ptg_path = proj_dir / f"{proj_name}.single.ptg.txt"

    with open(ptg_path, "w") as f:
        # CASA format: Epoch  RA  DEC  [TIME]
        # TIME is optional; let it default to 'integration'
        f.write(f"J2000 {args.phasecenter_ra} {dec}\n")

    simargs = dict(
        project=proj_name,
        skymodel=str(fits_path),
        incenter=args.center_freq,
        inwidth=args.width,
        integration='10s',
        totaltime=itime,
        hourangle='transit',
        antennalist=str(cfg_path),
        setpointings=False,
        ptgfile=str(ptg_path),
        indirection=f"J2000 {args.phasecenter_ra} {dec}",
        mapsize=[f"{args.fov_arcsec}arcsec"],
        graphics='both',
        verbose=True,
        overwrite=True
    )

    noise_mode = getattr(args, 'noise_mode', 'none')
    if noise_mode == 'manual' and hasattr(args, 't_sky') and hasattr(args, 'tau0'):
        simargs.update(dict(
            thermalnoise='tsys-manual',
            t_sky=float(args.t_sky),
            tau0=float(args.tau0)
        ))
    elif noise_mode == 'atm':
        simargs.update(dict(
            thermalnoise='tsys-atm',
            user_pwv=float(args.pwv),
            t_ground=270
        ))
    else:
        simargs.update(dict(thermalnoise=''))

    logging.info(f"Running single simobserve for homogeneous array: {simargs}")
    os.chdir(proj_base_dir)
    simobserve(**simargs)

    pipeline_log["simobserve"] = simargs
    noisy = sorted(proj_dir.glob(f"{proj_name}.*.noisy.ms"))
    plain = sorted(proj_dir.glob(f"{proj_name}.*.ms"))

    ms_noisy = noisy[0] if noisy else None
    ms_plain = plain[0] if plain else None

    if ms_noisy is None or not ms_noisy.exists():
        raise RuntimeError(f"Could not find noisy MS for project {proj_name}. Looked in {proj_dir}.")

    if ms_plain is None or not ms_plain.exists():
        logging.warning(
            "simulate_homo_ms: could not find noiseless MS for %s; "
            "homogeneous noise validation will fall back to DATA-only.",
            proj_name,
        )

    return str(ms_noisy), False, (str(ms_plain) if ms_plain is not None else None)


def image_baseline_groups(vis, img, tclean_params_dirty, tclean_params, vptab, args, rms_dirty):
    """
    A/B/cross imaging for heterogeneous arrays.
    Returns meas_peaks, meas_rms dicts including the 'all' entry.

    Uses per-group dirty images for RMS, and (optionally) per-group CLEANed
    images when args.niter > 1, with mask settings controlled by:
      * args.clean_use_mask
      * args.clean_mask_center_frac
    """
    meas_peaks = {}
    meas_rms = {}

    # --- ALL (already imaged by caller) ---
    try:
        all_stats = measure_image_stats(img_base=img, args=args)
        meas_peaks['all'] = all_stats["peak"]
        meas_rms['all']   = all_stats["rms_robust"]
        print(
            "[meas] all: peak=%.3e Jy/bm, rms=%.3e Jy/bm (used_pb=%s)"
            % (all_stats["peak"], all_stats["rms_robust"], all_stats["used_pb"])
        )
    except Exception as e:
        print(f"[meas] WARNING: could not measure ALL image peak/rms: {e}")

    # Decide if we actually CLEAN per group: only if niter > 1
    niter_val = int(getattr(args, "niter", 0) or 0)
    do_clean = niter_val > 1

    antsels = build_baseline_group_selectors_by_diameter(vis)
    print("A:", antsels.get("A"))
    print("B:", antsels.get("B"))
    print("cross:", antsels.get("cross"))

    for key in ["A", "B", "cross"]:
        antsel = antsels.get(key)
        if not antsel:
            continue

        imtag = key
        dirty_im_g = f"{img}_dirty_{imtag}"
        print(f"[hetero imaging] Dirty image for group {key} (antenna='{antsel[:60]}...')")

        # --- DIRTY IMAGE FOR THIS GROUP ---
        tparams_dirty_g = dict(tclean_params_dirty)
        tparams_dirty_g.update(
            imagename=dirty_im_g,
            antenna=antsel,
            gridder="mosaic",
            vptable=vptab,
            specmode="mfs",
            nchan=-1,
            fastnoise=True,
            normtype="flatnoise",
            pblimit=getattr(args, "pblimit", 0.1),
            usepointing=True,
            pbcor=False,
            savemodel="modelcolumn",
            niter=0,
        )
        tclean(**tparams_dirty_g)

        # RMS on group dirty image
        try:
            dirty_stats_g = measure_image_stats(
                img_base=dirty_im_g,
                args=args,
            )
            rms_dirty_g = dirty_stats_g["rms_robust"]
            print(
                f"[meas] dirty {key}: RMS_imstat={dirty_stats_g['rms_imstat']:.3e}, "
                f"RMS_robust={rms_dirty_g:.3e} (used_pb={dirty_stats_g['used_pb']})"
            )
        except Exception as e:
            print(f"[meas] WARNING: could not get dirty RMS for {key}: {e}")
            rms_dirty_g = rms_dirty   # fallback to ALL-group dirty rms

        threshold_g = float(args.threshold_factor) * rms_dirty_g

        # ------------------------------------------------------------------
        # CLEANED IMAGE FOR THIS GROUP (optional)
        # ------------------------------------------------------------------
        im_g = f"{img}_{imtag}"
        img_base_for_stats = dirty_im_g  # default: use dirty image

        if do_clean:
            print(f"[hetero imaging] Clean image for group {key} → {im_g}")

            tparams_clean_g = dict(tclean_params)
            tparams_clean_g.update(
                imagename=im_g,
                antenna=antsel,
                gridder="mosaic",
                vptable=vptab,
                specmode="mfs",
                nchan=-1,
                fastnoise=True,
                normtype="flatnoise",
                pblimit=getattr(args, "pblimit", 0.1),
                usepointing=True,
                pbcor=False,
                threshold=threshold_g,
            )

            # Mask behaviour controlled by global cleaning settings
            if getattr(args, "clean_use_mask", True):
                mask_frac = getattr(args, "clean_mask_center_frac", 0.1)
                mask27 = make_center_box_mask_crtf(f"{dirty_im_g}.image", mask_frac)
                tparams_clean_g.update(
                    usemask="user",
                    mask=mask27,
                )
            else:
                tparams_clean_g.update(
                    usemask="auto-multithresh",
                )

            tclean(**tparams_clean_g)
            img_base_for_stats = im_g

        # ------------------------------------------------------------------
        # Measure stats for this group (cleaned if available, else dirty)
        # ------------------------------------------------------------------
        try:
            stats_g = measure_image_stats(img_base_for_stats, args=args)
            meas_peaks[key] = stats_g["peak"]
            meas_rms[key]   = stats_g["rms_robust"]
            print(
                "[meas] %s: peak=%.3e Jy/bm, rms=%.3e Jy/bm (used_pb=%s, cleaned=%s)"
                % (key, stats_g["peak"], stats_g["rms_robust"], stats_g["used_pb"], do_clean)
            )

        except Exception as e:
            print(f"[meas] WARNING: could not measure peak/rms for group {key}: {e}")

    print("\n[Validate] Expected vs measured (baseline groups):")
    hetero_checkVals(vis, meas_peaks=meas_peaks, meas_rms=meas_rms)

    return meas_peaks, meas_rms



def image_and_log_for_vis_single(
    args,
    vis,
    is_hetero,
    fits_name,
    config,
    config_tag,
    dec,
    itime,
    dyn_cell,
    imsize,
    proj_base_dir,
    pipeline_log,
    vptab=None,
):
    """Image a single MS with one (weighting, robust, taper) combo."""

    os.chdir(proj_base_dir)

    wt  = args.weightings[0]
    rb  = args.robust_values[0] if wt == "briggs" and args.robust_values else None
    tap = args.uvtaper_values[0]

    tag   = make_short_tag(fits_name, config_tag, itime, dec, wt, rb, tap)
    img   = f"{tag}_d{imsize}"      # name for CLEANed image (if any)
    dirty = f"{img}_dirty"          # name for dirty image

    base_tclean = dict(
        vis=vis,
        cell=dyn_cell,
        imsize=imsize,
        weighting=wt,
        uvtaper=[tap],
        phasecenter=f"J2000 {args.phasecenter_ra} {dec}",
        nchan=-1,
        datacolumn="data",
        fastnoise=True,
        interactive=False,
    )
    if wt == "briggs" and rb is not None:
        base_tclean["robust"] = rb

    # ------------------------------------------------------------------
    # 1) DIRTY IMAGE
    # ------------------------------------------------------------------
    tclean_dirty = dict(
        base_tclean,
        imagename=dirty,
        niter=0,
        calcpsf=True,
        calcres=True,
        pbcor=False,
        usepointing=False,
    )
    if is_hetero or (vptab is not None):
        apply_mosaic_settings(tclean_dirty, args, dec, vptab=vptab)

    wipe(dirty)
    tclean(**tclean_dirty)

    # PSF → multiscale scales
    ia_local = image()
    ia_local.open(f"{dirty}.psf")
    psf_summary = ia_local.summary()
    ia_local.close()

    beam_major = psf_summary["restoringbeam"]["major"]["value"]
    cell_arcsec = float(dyn_cell.rstrip("arcsec"))
    beam_pix = beam_major / cell_arcsec
    scalefactors = [1, 2, 5]
    scales = sorted({0} | {max(1, int(round(f * beam_pix))) for f in scalefactors})

    # RMS on dirty
    dirty_stats = measure_image_stats(dirty, args=args)
    rms_dirty = dirty_stats["rms_robust"]
    logging.info(
        "[dirty] %s: peak=%.3e Jy/bm, rms_robust=%.3e Jy/bm (used_pb=%s)",
        dirty,
        dirty_stats["peak"],
        rms_dirty,
        dirty_stats["used_pb"],
    )
    threshold = float(args.threshold_factor) * rms_dirty
    # Decide if we actually CLEAN
    do_clean = (getattr(args, "niter", 0) or 0) > 0

    # ------------------------------------------------------------------
    # 2) CLEAN IMAGE (optional)
    # ------------------------------------------------------------------
    # Base CLEAN params
    tclean_clean = dict(
        base_tclean,
        imagename=img,
        deconvolver=args.deconvolver,
        scales=scales,
        niter=args.niter,
        threshold=threshold,
        savemodel="modelcolumn",
        pbcor=False,
        usepointing=False,
    )
    if is_hetero or (vptab is not None):
        apply_mosaic_settings(tclean_clean, args, dec, vptab=vptab)

    if do_clean:
        # Mask behaviour controlled by args.clean_use_mask / args.clean_mask_center_frac
        if getattr(args, "clean_use_mask", True):
            mask_frac = getattr(args, "clean_mask_center_frac", 0.1)
            mask = make_center_box_mask_crtf(f"{dirty}.image", mask_frac)
            tclean_clean.update(
                usemask="user",
                mask=mask,
            )
        else:
            # No explicit user mask: let tclean choose
            tclean_clean.update(
                usemask="auto-multithresh",
            )

        wipe(img)
        tclean(**tclean_clean)
        img_base_for_stats = img
    else:
        # No CLEAN: we will treat the dirty image as the final product
        img_base_for_stats = dirty

    # ------------------------------------------------------------------
    # 3) Export + stats from the chosen image (clean or dirty)
    # ------------------------------------------------------------------
    imhead(
        imagename=f"{img_base_for_stats}.image",
        mode="put",
        hdkey="OBJECT",
        hdvalue=img_base_for_stats,
    )
    exportfits(
        imagename=f"{img_base_for_stats}.image",
        fitsimage=f"{img_base_for_stats}.image.fits",
        dropdeg=True,
        dropstokes=True,
        history=False,
        velocity=False,
        optical=False,
        overwrite=True,
    )

    clean_stats = measure_image_stats(img_base_for_stats, args=args)
    pk = clean_stats["peak"]

    logging.info(
        "[final image] %s: peak=%.3e Jy/beam, rms=%.3e Jy/beam (PB masked=%s)",
        img_base_for_stats,
        clean_stats["peak"],
        clean_stats["rms_robust"],
        clean_stats["used_pb"],
    )
        # Baseline-group imaging
    # image_baseline_groups itself decides whether to CLEAN groups
    # based on args.niter.
    if is_hetero and getattr(args, "image_baseline_groups", True):
        image_baseline_groups(
            vis=vis,
            img=img_base_for_stats,
            tclean_params_dirty=tclean_dirty,
            tclean_params=tclean_clean,
            vptab=vptab,
            args=args,
            rms_dirty=rms_dirty,
        )

    bm_maj, bm_min, bm_pa = read_restoring_beam(f"{img_base_for_stats}.image")
    log_entry = {
        "fits": fits_name,
        "config": config,
        "int_time": itime,
        "dec": dec,
        "weighting": wt,
        "uvtaper": tap,
        "rms": clean_stats["rms_robust"],
        "peak_flux": pk,
        "beam_major": bm_maj,
        "beam_minor": bm_min,
        "beam_pa": bm_pa,
        "cleaned": do_clean,
    }
    if wt == "briggs":
        log_entry["robust"] = rb
    pipeline_log["results"].append(log_entry)



def run_single_integration(
    args,
    dec,
    config,
    config_tag,
    cfg_path,
    fits_path,
    fits_name,
    dyn_cell,
    imsize,
    proj_base_dir,
    itime,
    is_hetero,
    pipeline_log):
    """
    Run exactly one integration time / MS / imaging.
    """
    proj_name = (
        f"{args.project_base}_{fits_name}_{config_tag}_MIXED_"
        f"{itime}_{dec.replace('d','_').replace('m','').replace('.','')}"
    )
    proj_dir = proj_base_dir / proj_name
    proj_dir.mkdir(exist_ok=True)
    
    vis_plain = None  # only used for homogeneous simobserve
    if is_hetero:
        vis, is_hetero, vptab = simulate_hetero_ms(
            args=args,
            cfg_path=cfg_path,
            proj_dir=proj_dir,
            proj_name=proj_name,
            fits_path=fits_path,
            dec=dec,
            itime=itime,
            imsize=imsize,
            cell_arcsec=float(dyn_cell.rstrip("arcsec")),
        )
        pipeline_log["vptable"] = {
            "used": bool(vptab),
            "path": vptab,
            "pb_model": args.pb_model,
        }
    else:
        vis, is_hetero, vis_plain = simulate_homo_ms(
            args=args,
            cfg_path=cfg_path,
            proj_dir=proj_dir,
            proj_base_dir=proj_base_dir,
            proj_name=proj_name,
            fits_path=fits_path,
            dec=dec,
            itime=itime,
            pipeline_log=pipeline_log,
        )

        vptab = None
        # Optionally use a vptable even for homogeneous configs
        if getattr(args, "use_vptable_in_homo", False):
            # Build a per-antenna vptable (single diameter) from the homogeneous MS
            vptab_path = proj_dir / f"{proj_name}_perant_vp.tbl"
            vptab = build_per_antenna_pb_vptable(
                args=args,
                msname=vis,
                vptab_path=str(vptab_path),
                imsize=imsize,
                cell_arcsec=float(dyn_cell.rstrip("arcsec")),
            )
            logging.info(f"[VP] Built homogeneous per-antenna vptable: {vptab}")
            pipeline_log["vptable"] = {
                "used": bool(vptab),
                "path": vptab,
                "pb_model": args.pb_model,
            }

    # Optional noise validation – BOTH homo and hetero arrays
    if getattr(args, "validate_noise_theory", True):
        field = getattr(args, "noiseval_field", "0")
        tag = f"noiseval_{config_tag}_{itime}_{dec.replace('d','_')}"

        # Hetero: also do A/B/cross; Homo: just ALL
        if is_hetero:
            do_groups = getattr(args, "noiseval_do_groups", True)
        else:
            do_groups = False

        nv = run_noise_validation_single_field(
            vis=vis,
            cell=dyn_cell,
            vptable=vptab,
            imsize=imsize,
            field=field,
            tag=tag,
            do_groups=do_groups,
            args=args,
            is_hetero=is_hetero,   # pass in what compute_imaging_geometry told us
            vis_plain=vis_plain,
        )
        pipeline_log["noise_validation"] = nv
    image_and_log_for_vis_single(
        args=args,
        vis=vis,
        is_hetero=is_hetero,
        fits_name=fits_name,
        config=config,
        config_tag=config_tag,
        dec=dec,
        itime=itime,
        dyn_cell=dyn_cell,
        imsize=imsize,
        proj_base_dir=proj_base_dir,
        pipeline_log=pipeline_log,
        vptab=vptab,
    )

# === MAIN PIPELINE ===
def main(args):
    setup_logging(args.log_file)
    try:
        root_dir = Path(__file__).parent.resolve()
        proj_base_dir = root_dir / args.project_base

        # optional clean of existing project_base
        if getattr(args, "clean_project_base", False) and proj_base_dir.exists():
            safe_rm_tree(proj_base_dir)

        proj_base_dir.mkdir(exist_ok=True)

        # --- pick a single config / dec / integration time ---
        config_map = args.config_map
        if not isinstance(config_map, dict) or not config_map:
            raise ValueError("config_map must be a non-empty JSON dict")

        # Just take the first (config_file, tag) pair
        (config, config_tag), = list(config_map.items())[:1]

        # First declination and integration time only
        dec = args.declinations[0]
        itime = args.integration_times[0]

        seed = random.randint(1, 2**31 - 1)
        logging.info(f"Random seed for this run: {seed}")
        logging.info(f"Using single combo: config={config} ({config_tag}), "
                     f"dec={dec}, itime={itime}")
        pipeline_log = {
            "random_seed": seed,
            "sky_model": {},
            "geometry": {},
            "simobserve": {},
            "vptable": {},
            "noise_validation": {},
            "results": []
        }
        pipeline_log["tclean_settings"] = {
            "pblimit": getattr(args, "pblimit", 0.1),
        }

        cfg_path = find_config_path(config, args.casa_bin, args.extra_dirs)

        beam, dyn_cell, pixsize, imsize, is_hetero = compute_imaging_geometry(
            args, cfg_path, config_tag
        )

        pipeline_log["geometry"] = {
            "beam_est": beam,
            "dyn_cell": dyn_cell,
            "imsize": imsize,
            "pixsize_arcsec": pixsize,
        }

        fits_path, fits_name = generate_sky_for_config(
            args=args,
            dec=dec,
            dyn_cell=dyn_cell,
            imsize=imsize,
            config_tag=config_tag,
        )

        pipeline_log["sky_model"] = {
            "mode": args.sky_mode,
            "sky_type": getattr(args, "sky_type", None),
            "fits_path": str(fits_path),
            "fov_arcsec": args.fov_arcsec,
        }

        # Run a *single* integration / imaging combo
        run_single_integration(
            args=args,
            dec=dec,
            config=config,
            config_tag=config_tag,
            cfg_path=cfg_path,
            fits_path=fits_path,
            fits_name=fits_name,
            dyn_cell=dyn_cell,
            imsize=imsize,
            proj_base_dir=proj_base_dir,
            itime=itime,
            is_hetero=is_hetero,
            pipeline_log=pipeline_log,
        )

        with open(proj_base_dir / 'pipeline_log.json', 'w') as f:
            json.dump(pipeline_log, f, indent=4)

        logging.info(f"Pipeline complete. Log at {proj_base_dir / 'pipeline_log.json'}")
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise

    # Log full configuration in a compact way
    logging.info("=== Pipeline configuration ===")
    for k, v in sorted(vars(args).items()):
        logging.info(f"  {k} = {v}")
    logging.info("================================")



def example_run():
    """
    Run a single end-to-end example using EXAMPLE_SETTINGS defined above.
    Edit EXAMPLE_SETTINGS, then call example_run() from CASA or Python.
    """
    args = SimpleNamespace(**EXAMPLE_SETTINGS)
    main(args)


# === ARG PARSER ===
if __name__ == "__main__":
    # Simple “script mode”: just use EXAMPLE_SETTINGS above.
    example_run()


