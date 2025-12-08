"""
H. Österbom

validation.py

Noise and heterogeneity validation utilities.

Provides:
  * Visibility -> image noise conversion helpers
  * Theoretical heterogeneous σ_im estimators (A/B/cross)
  * Baseline-group RMS and amplitude checks
  * End-to-end natural-weight noise validation imaging

Used to confirm correct noise scaling in heterogeneous simulations.
"""

import os
import json
import math
import shutil
import logging
import numpy as np
from collections import Counter, defaultdict
from casatools import table
from casatasks import tclean, mstransform
from utils import safe_rm_tree
from utils import wipe
from qol_pipeline import (
    build_baseline_group_selectors_by_diameter,
    measure_image_stats,
    _read_antenna_table,
)
from pb_vp import calc_ang

tb = table()

#======================================Validation Functions=====================================================#
def vis_to_image_noise(msname):
    sigma_vis, N = load_sigma_vis_and_N(msname)
    sigma_im = sigma_vis / math.sqrt(N) if N > 0 else float("nan")
    return sigma_vis, N, sigma_im


def load_sigma_vis_and_N(msname):
    """
    Reads the DATA column and returns:
        σ_vis   – combined RMS of real+imag
        N_good  – number of unflagged points
    """
    tb.open(msname)
    try:
        data = tb.getcol("DATA")       # shape: (pol, chan, row)
        if "FLAG" in tb.colnames():
            flag = tb.getcol("FLAG")
            mask = ~flag
        else:
            mask = np.ones_like(data.real, dtype=bool)

        re = data.real[mask]
        im = data.imag[mask]

        sigma_vis = 0.5 * (np.std(re) + np.std(im))
        N = re.size
    finally:
        tb.close()

    return sigma_vis, N



def _amp_stats_ms(ms_path):
    """
    Compute basic amplitude and weight statistics for an MS.

    Uses unflagged DATA values to compute mean and standard deviation of
    |DATA|, along with the number of samples. Also computes the mean
    weight from WEIGHT_SPECTRUM if present, otherwise from WEIGHT (broadcast
    over channels), or defaults to 1 if neither is available.

    Returns a dict with keys: mean_amp, std_amp, npts, meanwt.
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
        # W shape: (pol, row) -> broadcast over channels
        npol, nrow = W.shape
        nchan = data.shape[1]
        W_b = np.repeat(W, nchan, axis=1).reshape(npol, nchan, nrow)
        w_vals = W_b[good] if flag is not None else W_b.ravel()
    else:
        w_vals = np.ones_like(amp)

    meanwt = float(np.mean(w_vals)) if w_vals.size > 0 else float("nan")
    return dict(mean_amp=mean_amp, std_amp=std_amp, npts=npts, meanwt=meanwt)


def build_group_ms(vis, antsels, suffix):
    """
    Split a full MS into separate A–A, B–B, and A–B (cross) baseline
    sub-MSs using mstransform. Each antsels[key] is an antenna-selection
    string that selects the baselines belonging to that heterogeneity
    group. This enables per-group noise analysis, SEFD checks, and
    imaging diagnostics for heterogeneous arrays.

    Returns a dict {group: ms_path}.
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


def is_ms_heterogeneous(vis: str) -> bool:
    """
    Decide if an MS is heterogeneous based on the DISH_DIAMETER column.
    Uses _read_antenna_table from qol_simulation_pipeline.
    """
    diams_m, _ = _read_antenna_table(vis)
    # Filter out NaNs, round a bit to avoid tiny numerical differences
    uniq = {round(float(d), 3) for d in diams_m if np.isfinite(d)}
    return len(uniq) > 1



def compute_theoretical_rms(ms_all, ms_A, ms_B, ms_cross):
    """
    Computes σ_im,A, σ_im,B, σ_im,cross and heterogeneous σ_im,all
    using naive σ_im ≈ σ_vis / sqrt(N) per group and the
    natural-weight combination:

        σ_im(all) = 1 / sqrt( Σ_k N_k / σ_vis,k^2 )
    """

    results = {}

    # --- per-group theory from DATA ---
    for label, ms in zip(["A", "B", "cross"], [ms_A, ms_B, ms_cross]):
        sigma_vis, N, sigma_im = vis_to_image_noise(ms)

        results[label] = {
            "sigma_vis": sigma_vis,
            "N": N,
            "sigma_im": sigma_im,
        }
        
    # --- heterogeneous ALL ---
    W_tot = 0.0
    for k in ["A", "B", "cross"]:
        sigma_vis_k = results[k]["sigma_vis"]
        N_k = results[k]["N"]
        if np.isfinite(sigma_vis_k) and sigma_vis_k > 0 and N_k > 0:
            W_tot += N_k / (sigma_vis_k**2)

    if W_tot > 0.0:
        sigma_im_all = 1.0 / math.sqrt(W_tot)
    else:
        sigma_im_all = float("nan")

    results["ALL"] = {
        "sigma_im": sigma_im_all,
        "W_tot": W_tot,
    }

    return results



def hetero_checkVals(
    vis,
    meas_peaks=None,
    meas_rms=None,
    keep_group_ms=False,
):
    """
    Heterogeneous baseline sanity check.

    - Splits baselines into A–A, B–B, A–B using dish diameters.
    - Builds A/B/cross sub-MSs via mstransform (build_group_ms).
    - On each sub-MS, computes:
        * VisMean, VisStd(sim), Weight(sim) from |DATA| and WEIGHT(SPECTRUM)
        * σ_vis, N, σ_im,calc via vis_to_image_noise()
    - For 'all', combines A/B/cross via:
          1 / σ_im,all^2  = Σ_g N_g / σ_vis,g^2
      and also computes σ_im,all,naive directly from the full MS.
    """

    if meas_peaks is None:
        meas_peaks = {}
    if meas_rms is None:
        meas_rms = {}

    logging.info("[hetero_checkVals] Starting for vis=%s", vis)

    # ------------------------------------------------------------------
    # 1) Read dish diameters and define A/B dish classes
    #    A = largest diameter, B = second-largest (if present).
    # ------------------------------------------------------------------
    diams, names = _read_antenna_table(vis)

    uniq_d = sorted({round(float(d), 3) for d in diams if d == d}, reverse=True)
    if not uniq_d:
        print("[hetero_checkVals] No valid dish diameters found; aborting.")
        return {}

    D_A = uniq_d[0]                      # largest dish
    D_B = uniq_d[1] if len(uniq_d) > 1 else None
    label_A = f"{D_A:.0f}m"
    label_B = f"{D_B:.0f}m" if D_B is not None else ""


    # --- 2) Reference frequency, for PB-size printout ---
    tb.open(vis + "/SPECTRAL_WINDOW")
    try:
        nu_hz = float(tb.getcell("REF_FREQUENCY", 0))
    finally:
        tb.close()
    freq_ghz = nu_hz / 1e9


    # ------------------------------------------------------------------
    # 3) Split MS into A, B, and cross baseline groups (sub-MSs).
    #    If this fails, we still continue with only the full MS stats.
    # ------------------------------------------------------------------
    try:
        antsels = build_baseline_group_selectors_by_diameter(vis)
        group_ms = build_group_ms(vis, antsels, suffix="hetero")
    except Exception as e:
        logging.warning("[hetero_checkVals] Could not build baseline group MSs: %s", e)
        group_ms = {}

    # --- 4) For each group MS (A, B, cross), measure 
    # amplitude stats + noise stats ---
    amp_stats = {}
    noise_stats = {}

    for key, ms_path in group_ms.items():
        amp_stats[key] = _amp_stats_ms(ms_path)
        sigma_vis, N, sigma_im = vis_to_image_noise(ms_path)
        noise_stats[key] = dict(sigma_vis=sigma_vis, N=int(N), sigma_im_calc=sigma_im)

    # 5) For the full MS ("all"), measure the same visibility noise and
    #    compute a naive σ_im directly from the combined MS.
    amp_stats["all"] = _amp_stats_ms(vis)
    sigma_vis_all, N_all, sigma_im_all_naive = vis_to_image_noise(vis)
    noise_stats["all_naive"] = dict(
        sigma_vis=sigma_vis_all,
        N=N_all,
        sigma_im_calc=sigma_im_all_naive,
    )

    # --- 6) Heterogeneous ALL theory from A/B/cross ---
    denom = 0.0
    for key in ("A", "B", "cross"):
        s = noise_stats.get(key)
        if not s:
            continue
        if s["N"] > 0 and s["sigma_vis"] > 0:
            denom += s["N"] / (s["sigma_vis"] ** 2)

    if denom > 0.0:
        sigma_im_all_hetero = 1.0 / math.sqrt(denom)
    else:
        sigma_im_all_hetero = sigma_im_all_naive

    noise_stats["all"] = dict(
        sigma_vis=sigma_vis_all,
        N=N_all,
        sigma_im_calc=sigma_im_all_hetero,
        sigma_im_naive=sigma_im_all_naive,
    )

    # --- 7) Theoretical ratios for VisStd and weight between A/B/cross ---
    D_A2 = D_A**2
    D_B2 = D_B**2 if D_B is not None else None

    def _visstd_calc(tag):
        # Noise scaling relative to A–A.
        if tag == "A":
            return 1.0
        if tag == "B" and D_B2 is not None:
            return D_A2 / D_B2           # (σ_B / σ_A) expectation
        if tag == "cross" and D_B2 is not None:
            return D_A2 / (D_A * D_B)    # geometric mean
        return float("nan")

    def _w_calc(tag):
        # Expected relative weights 
        if tag == "A":
            return 1.0
        if tag == "B" and D_B2 is not None:
            return (D_B2 / D_A2)**2      
        if tag == "cross" and D_B2 is not None:
            return ((D_A * D_B) / D_A2)**2       # geometric mean
        return float("nan")

    # --- 8) Theoretical peaks from VisMean (still computed but no longer printed) ---
    peak_calc = {}
    if amp_stats.get("A", {}).get("npts", 0) > 0:
        peak_calc["A"] = amp_stats["A"]["mean_amp"]
    if amp_stats.get("B", {}).get("npts", 0) > 0:
        peak_calc["B"] = amp_stats["B"]["mean_amp"]
    if "A" in peak_calc and "B" in peak_calc:
        peak_calc["cross"] = math.sqrt(peak_calc["A"] * peak_calc["B"])
    peak_calc["all"] = float("nan")  # keep as NaN for completeness

    # --- 9) Pretty-printed summary table  ---
    hdr = (
        "Baseline", "VisMean", "VisStd(sim)", "VisStd(calc)",
        "#DataPts", "Weight(calc)", "Weight(sim)",
        "RMS calc (mJy)", "RMS meas (mJy)",
    )
    print("\n  " + " | ".join(f"{h:>14}" for h in hdr))
    print("  " + "-" * 93)


    baseline_labels = {
        "A": f"{label_A}-{label_A}",
        "B": f"{label_B}-{label_B}" if D_B is not None else None,
        "cross": f"{label_A}-{label_B}" if D_B is not None else None,
        "all": "All",
    }

    def _fmt_row(baseline_label, tag):
        """
        Format one summary-table row for a given baseline group (A, B, cross, all).
        Returns a tuple of values to be printed, or None if stats are missing.
        """

        # Special case: "all" row only compares image RMS (calc vs. meas).
        if tag == "all":
            ns = noise_stats["all"]
            rms_calc_mJy = ns["sigma_im_calc"] * 1e3
            rms_meas_mJy = meas_rms.get("all", float("nan")) * 1e3
            return (
                baseline_label,
                float("nan"), float("nan"), float("nan"),
                float("nan"), float("nan"), float("nan"),
                rms_calc_mJy, rms_meas_mJy,
            )
        
        # --- For real groups (A, B, cross): fetch amplitude and noise stats.
        a = amp_stats.get(tag, {})
        ns = noise_stats.get(tag, {})
        if not a or not ns:
            return None

        # Visibility-domain statistics
        vismean     = a["mean_amp"]     # mean |DATA|
        visstd_sim  = a["std_amp"]      # simulated visibility RMS
        npts        = float(a["npts"])  # number of vis samples
        w_sim       = a["meanwt"]       # mean weight

        # Theoretical expectations (based on dish sizes)
        visstd_th   = _visstd_calc(tag)     # predicted vis RMS ratio
        w_calc      = _w_calc(tag)          # predicted weight ratio

        # Image-domain noise: calculated vs measured
        sigma_im    = ns["sigma_im_calc"]
        rms_calc_mJy = sigma_im * 1e3
        rms_meas_mJy = meas_rms.get(tag, float("nan")) * 1e3

        # Return the row contents (order matches the printed table columns)
        return (
            baseline_label,
            vismean, visstd_sim, visstd_th,
            npts, w_calc, w_sim,
            rms_calc_mJy, rms_meas_mJy,
        )

    # Print one table row per baseline group.
    # Skip missing groups; format numbers nicely aligned; show NaN placeholders.
    for tag, label in baseline_labels.items():
            if not label:
                continue
            row = _fmt_row(label, tag)
            if not row:
                continue
            print("  " + " | ".join(
                f"{x:>14.4f}" if isinstance(x, (int, float)) and x == x else f"{x:>14}"
                for x in row
            ))
    # ------------------------------------------------------------------
    # 10) Print primary beam FWHM for each dish type (for context).
    # ------------------------------------------------------------------
    print(
        "\nCalculated PB size for type A (dia={0:2.2f} m): {1:3.5f} arcmin"
        .format(D_A, calc_ang(freq_ghz, D_A))
    )
    if D_B is not None:
        print(
            "Calculated PB size for type B (dia={0:2.2f} m): {1:3.5f} arcmin"
            .format(D_B, calc_ang(freq_ghz, D_B))
        )

    # ------------------------------------------------------------------
    # 11) Clean up temporary A/B/cross sub-MSs unless asked to keep them.
    # ------------------------------------------------------------------
    if not keep_group_ms:
        for key, ms_path in group_ms.items():
            try:
                safe_rm_tree(ms_path)
            except Exception as e:
                logging.warning("[hetero_checkVals] Failed to remove %s: %s", ms_path, e)

    return {
        "amp_stats": amp_stats,
        "noise_stats": noise_stats,
        "freq_ghz": freq_ghz,
        "D_A": D_A,
        "D_B": D_B,
    }



def run_noise_validation_single_field(
    vis,
    cell,
    vptable,
    imsize,
    field="0",
    tag="noiseval",
    do_groups=True,
    args=None,
    is_hetero=None,
    vis_plain=None, 
):
    """
    Validate that vis-domain noise maps to image-domain noise for a
    single-field, natural-weight, niter=0 dirty image (noise-only MS).
    """

    logging.info("[noiseval] Starting noise-validation for vis=%s field=%s", vis, field)

    # pick up pblimit from args or use default
    pblimit_val = getattr(args, "pblimit", 0.1)
    # --- Normalize vptable for tclean: '' means "no VP table" ---
    vpt_for_tclean = "" if vptable in (None, "", "None") else vptable

    if args is None:
        raise ValueError("run_noise_validation_single_field requires args for measure_image_stats")
    
    # 1) Copy MS -> noise-only sandbox
    base_noise = vis.rstrip("/") + "_noiseval.ms"
    if os.path.exists(base_noise):
        logging.info("[noiseval] Removing existing %s", base_noise)
        safe_rm_tree(base_noise)
    logging.info("[noiseval] Copying %s → %s", vis, base_noise)
    shutil.copytree(vis, base_noise)

    # Decide hetero vs homo if not explicitly given
    if is_hetero is None:
        is_hetero = is_ms_heterogeneous(base_noise)
        logging.info("[noiseval] MS %s heterogeneous? %s", base_noise, is_hetero)

    # 2) Subtract sky from DATA in the copy -> leave only noise
    #    - heterogeneous: DATA - MODEL_DATA
    #    - homogeneous (simobserve): DATA(noisy) - DATA(plain)
    if (not is_hetero) and vis_plain is not None:
        # Homogeneous simobserve: use plain MS as the model
        tb.open(base_noise, nomodify=False)
        try:
            data_noisy = tb.getcol("DATA")
        finally:
            tb.close()

        tb.open(vis_plain)
        try:
            data_plain = tb.getcol("DATA")
        finally:
            tb.close()

        if data_noisy.shape != data_plain.shape:
            logging.error(
                "[noiseval] Homo: noisy and plain DATA shapes differ: %s vs %s",
                data_noisy.shape, data_plain.shape,
            )
        else:
            dmax = float(np.nanmax(np.abs(data_noisy)))
            mmax = float(np.nanmax(np.abs(data_plain)))
            logging.info(
                "[noiseval] (homo) before subtract: max|DATA_noisy|=%.3e, max|DATA_plain|=%.3e",
                dmax, mmax,
            )

            noise = data_noisy - data_plain

            tb.open(base_noise, nomodify=False)
            try:
                tb.putcol("DATA", noise)
            finally:
                tb.close()

            d2max = float(np.nanmax(np.abs(noise)))
            logging.info(
                "[noiseval] (homo) after subtract: max|DATA_noise|=%.3e", d2max
            )

    else:
        # Heterogeneous (or no plain MS): fallback to MODEL_DATA subtraction
        tb.open(base_noise, nomodify=False)
        try:
            colnames = set(tb.colnames())
            if "DATA" not in colnames:
                raise RuntimeError("MS has no DATA column")

            data = tb.getcol("DATA")

            if "MODEL_DATA" in colnames:
                model = tb.getcol("MODEL_DATA")

                dmax = float(np.nanmax(np.abs(data)))
                mmax = float(np.nanmax(np.abs(model)))
                logging.info(
                    f"[noiseval] before subtract: max|DATA|={dmax:.3e}, max|MODEL_DATA|={mmax:.3e}"
                )

                tb.putcol("DATA", data - model)

                data2 = tb.getcol("DATA")
                d2max = float(np.nanmax(np.abs(data2)))
                logging.info(f"[noiseval] after subtract: max|DATA|={d2max:.3e}")

            else:
                logging.info(
                    "[noiseval] No MODEL_DATA; assuming DATA already noise-only."
                )
        finally:
            tb.close()

    # 3) Optional: build A/B/cross baseline-group MSs from noise-only MS
    group_ms_paths = {}
    if do_groups and is_hetero:
        try:
            antsels = build_baseline_group_selectors_by_diameter(base_noise)
            group_ms_paths = build_group_ms(base_noise, antsels, suffix="noise")
        except Exception as e:
            logging.warning("[noiseval] Could not build baseline group MSs: %s", e)
            group_ms_paths = {}

    # 4) Theoretical σ_im for A/B/cross + heterogeneous ALL
    theory = None
    if is_hetero and do_groups and all(k in group_ms_paths for k in ("A", "B", "cross")):
        theory = compute_theoretical_rms(
            ms_all=base_noise,
            ms_A=group_ms_paths["A"],
            ms_B=group_ms_paths["B"],
            ms_cross=group_ms_paths["cross"],
        )

    # 5) Helper: image + compare for one (sub)MS
    def _noise_check_for_ms(ms_path, imagename, label, theory_entry=None):
        logging.info("[noiseval] Validating noise for %s (label=%s)", ms_path, label)

        # Always get σ_vis and N from the MS itself
        sigma_vis, N = load_sigma_vis_and_N(ms_path)
        sigma_im_naive = sigma_vis / math.sqrt(N) if N > 0 else float("nan")

        # If theory gives us a better σ_im (heterogeneous ALL), use that as "eff"
        if theory_entry is not None and "sigma_im" in theory_entry:
            sigma_im_calc = float(theory_entry["sigma_im"])
        else:
            sigma_im_calc = sigma_im_naive

        # Natural, mosaic dirty image
        logging.info(
            "[noiseval] tclean: %s (label=%s), weighting='natural', niter=0",
            ms_path, label,
        )
        wipe(imagename)
        
        tclean(
            vis=ms_path,
            imagename=imagename,
            niter=0,
            weighting="natural",
            gridder="mosaic",
            vptable=vpt_for_tclean,
            pblimit=pblimit_val,
            cell=cell,
            imsize=imsize,
            specmode="mfs",
            nchan=-1,
            datacolumn="data",
            phasecenter="",   # use FIELD table phasecenter
            calcpsf=True,
            calcres=True,
            interactive=False,
            fastnoise=True,
        )

        # Measured image RMS (and peak) via measure_image_stats
        try:
            stats = measure_image_stats(
                img_base=imagename,
                args=args,
                use_pb=True,   # force PB use for noise validation
            )
            sigma_im_meas = stats["rms_robust"]
            logging.info(
                "[noiseval] %s: peak=%.3e Jy/beam, σ_im,meas=%.3e Jy/beam (used_pb=%s)",
                label,
                stats["peak"],
                sigma_im_meas,
                stats["used_pb"],
            )
        except Exception as e: 
            logging.error("[noiseval] Failed RMS measurement for %s: %s", label, e)
            sigma_im_meas = float("nan")

        ratio = sigma_im_meas / sigma_im_calc if sigma_im_calc > 0 else float("nan")

        print("\n[noiseval] --- Natural-noise validation ---")
        print(f"    label       : {label}")
        print(f"    vis(subset) : {ms_path}")
        print(f"    σ_vis(data) : {sigma_vis:.3e} Jy")
        print(f"    σ_im,calc   : {sigma_im_calc:.3e} Jy/beam")
        print(f"    σ_im,meas   : {sigma_im_meas:.3e} Jy/beam")
        print(f"    meas/calc   : {ratio:.3f}")
        print("[noiseval] ---------------------------------\n")

        return {
            "label": label,
            "ms_path": ms_path,
            "sigma_vis_data": sigma_vis,
            "N_good_vis": N,
            "sigma_im_calc": sigma_im_calc,
            "sigma_im_meas": sigma_im_meas,
            "ratio_meas_over_naive": ratio,
        }

    # 6) Run checks for ALL + groups
    results = {}
    noise_dir = os.path.dirname(base_noise)

    # ALL
    base_label = f"field{field}_all"
    imagename_all = os.path.join(noise_dir, f"{tag}_{base_label}_natdirty")
    theory_all = theory["ALL"] if theory and "ALL" in theory else None
    results["all"] = _noise_check_for_ms(base_noise, imagename_all, base_label, theory_all)

    # A/B/cross
    if do_groups:
        for key in ("A", "B", "cross"):
            ms_path = group_ms_paths.get(key)
            if not ms_path:
                continue
            label = f"field{field}_{key}"
            imagename_g = os.path.join(noise_dir, f"{tag}_{label}_natdirty")
            th = theory.get(key) if theory else None
            results[key] = _noise_check_for_ms(ms_path, imagename_g, label, th)

    # 7) Summary + JSON dump
    print(f"\n[noiseval] === Baseline-group noise summary (field {field}) ===")
    print(f"  {'group':5s}  {'σ_im,calc':14s}  {'σ_im,meas':14s}  {'meas/calc':10s}")
    print("  ---------------------------------------------------------------")

    meas_rms_for_hetero = {}
    keys_for_summary = ["all", "A", "B", "cross"] if is_hetero and do_groups else ["all"]
    for key in keys_for_summary:
        res = results.get(key)
        if not res:
            continue
        calc = res["sigma_im_calc"]
        meas = res["sigma_im_meas"]
        r    = res["ratio_meas_over_naive"]
        meas_rms_for_hetero[key] = meas
        print(f"  {key:5s}  {calc:14.3e}   {meas:14.3e}   {r:10.3f}")

    print("[noiseval] ================================================\n")

    # Heterogeneous baseline validation in vis-domain on noise-only MS
    hetero_info = None
    if is_hetero:
        try:
            hetero_info = hetero_checkVals(
                base_noise,
                meas_peaks={},      
                meas_rms=meas_rms_for_hetero,
            )
        except Exception as e:
            print(f"[noiseval] WARNING: hetero_checkVals failed on {base_noise}: {e}")
            hetero_info = None

    out = {
        "vis": vis,
        "base_noise_ms": base_noise,
        "field": field,
        "tag": tag,
        "results": results,
        "hetero_check": hetero_info,
    }

    def _json_default(o):
        if isinstance(o, np.generic):
            return o.item()
        return str(o)

    json_path = vis.rstrip("/") + f"_noiseval_{tag}.json"
    try:
        with open(json_path, "w") as f:
            json.dump(out, f, indent=2, default=_json_default)
        logging.info("[noiseval] Saved results JSON to %s", json_path)
    except Exception as e:
        logging.error("[noiseval] Failed to write JSON summary: %s", e)

    return out



def _corr_type_names(corr_type_ids):
    # https://casacore.github.io/casacore/Stokes_8h_source.html
    m = {
        1: "I",   2: "Q",   3: "U",   4: "V",
        5: "RR",  6: "RL",  7: "LR",  8: "LL",
        9: "XX", 10: "XY", 11: "YX", 12: "YY",
    }
    return [m.get(int(x), str(int(x))) for x in corr_type_ids]


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

    baseline_stats = checkvals(msname, datacol='DATA')  # already prints summary

    # Group RMS by (D1, D2) (smallest first)
    group_rms = defaultdict(lambda: {'sum_sig2': 0.0, 'n': 0})

    for bl in baseline_stats:
        d1 = float(bl['D1_m'])
        d2 = float(bl['D2_m'])
        n  = int(bl['npts'])
        if n <= 0:
            continue

        sig_re = float(bl['rms_re_Jy'])
        sig_im = float(bl['rms_im_Jy'])
        if not (np.isfinite(sig_re) and np.isfinite(sig_im)):
            continue

        # sigma_vis from Re/Im
        sigma_ri = math.sqrt(0.5 * (sig_re**2 + sig_im**2))

        if d2 < d1:
            d1, d2 = d2, d1
        key = (round(d1, 3), round(d2, 3))

        group_rms[key]['sum_sig2'] += (sigma_ri ** 2) * n
        group_rms[key]['n']        += n

    # Delta ν from SPW0; t_int from MAIN
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




def checkvals(vis, datacol='DATA', combine_conj=True, max_baselines=None):
    """
    Measure per-baseline visibility RMS from an MS.

    For each (ANTENNA1, ANTENNA2) pair, accumulate the unflagged samples
    from the given data column and compute RMS of the real part, imaginary
    part, and complex amplitude. Conjugate baselines (i,j) and (j,i) can
    optionally be combined.

    The function prints a baseline-type noise summary grouped by dish-size
    pairs (e.g. '7m-7m', '7m-12m', '12m-12m'), and returns a list of per-
    baseline statistics dictionaries.
    // This is more of a debugger tool, we mainly use "hetero_checkVals()" = for structured heterogeneity check (pipeline)
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


