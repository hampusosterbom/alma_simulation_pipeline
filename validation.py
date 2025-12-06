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
from casatools import table
from casatasks import tclean
from utils import safe_rm_tree
from utils import wipe
from qol_pipeline import (
    build_baseline_group_selectors_by_diameter,
    build_group_ms,
    _amp_stats_ms,
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

    # --- Dish diameters → define A/B labels ---
    tb.open(vis + "/ANTENNA")
    try:
        diams = tb.getcol("DISH_DIAMETER").astype(float)
    finally:
        tb.close()

    uniq_d = sorted({round(float(d), 3) for d in diams if d == d}, reverse=True)
    if not uniq_d:
        print("[hetero_checkVals] No valid dish diameters found; aborting.")
        return {}

    D_A = uniq_d[0]                      # largest dish
    D_B = uniq_d[1] if len(uniq_d) > 1 else None
    label_A = f"{D_A:.0f}m"
    label_B = f"{D_B:.0f}m" if D_B is not None else ""


    # --- Reference frequency, for PB-size printout ---
    tb.open(vis + "/SPECTRAL_WINDOW")
    try:
        nu_hz = float(tb.getcell("REF_FREQUENCY", 0))
    finally:
        tb.close()
    freq_ghz = nu_hz / 1e9


    # --- Build A/B/cross antenna selectors + sub-MSs ---
    try:
        antsels = build_baseline_group_selectors_by_diameter(vis)
        group_ms = build_group_ms(vis, antsels, suffix="hetero")
    except Exception as e:
        logging.warning("[hetero_checkVals] Could not build baseline group MSs: %s", e)
        group_ms = {}

    # --- Per-group amplitude + noise stats ---
    amp_stats = {}
    noise_stats = {}

    for key, ms_path in group_ms.items():
        amp_stats[key] = _amp_stats_ms(ms_path)
        sigma_vis, N, sigma_im = vis_to_image_noise(ms_path)
        noise_stats[key] = dict(sigma_vis=sigma_vis, N=int(N), sigma_im_calc=sigma_im)

    # "all" stats directly from full MS
    amp_stats["all"] = _amp_stats_ms(vis)
    sigma_vis_all, N_all, sigma_im_all_naive = vis_to_image_noise(vis)
    noise_stats["all_naive"] = dict(
        sigma_vis=sigma_vis_all,
        N=N_all,
        sigma_im_calc=sigma_im_all_naive,
    )

    # --- Heterogeneous ALL theory from A/B/cross ---
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

    # --- Theoretical VisStd(calc) and Weight(calc) ratios ---
    D_A2 = D_A**2
    D_B2 = D_B**2 if D_B is not None else None

    def _visstd_calc(tag):
        if tag == "A":
            return 1.0
        if tag == "B" and D_B2 is not None:
            return D_A2 / D_B2           # (σ_B / σ_A) expectation
        if tag == "cross" and D_B2 is not None:
            return D_A2 / (D_A * D_B)    # geometric mean
        return float("nan")

    def _w_calc(tag):
        if tag == "A":
            return 1.0
        if tag == "B" and D_B2 is not None:
            return (D_B2 / D_A2)**2      # weight ∝ 1/σ^2
        if tag == "cross" and D_B2 is not None:
            return ((D_A * D_B) / D_A2)**2
        return float("nan")

    # --- Theoretical peaks from VisMean (still computed but no longer printed) ---
    peak_calc = {}
    if amp_stats.get("A", {}).get("npts", 0) > 0:
        peak_calc["A"] = amp_stats["A"]["mean_amp"]
    if amp_stats.get("B", {}).get("npts", 0) > 0:
        peak_calc["B"] = amp_stats["B"]["mean_amp"]
    if "A" in peak_calc and "B" in peak_calc:
        peak_calc["cross"] = math.sqrt(peak_calc["A"] * peak_calc["B"])
    peak_calc["all"] = float("nan")  # keep as NaN for completeness

    # --- Pretty-printed table (simplified) ---
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

        a = amp_stats.get(tag, {})
        ns = noise_stats.get(tag, {})
        if not a or not ns:
            return None

        vismean     = a["mean_amp"]
        visstd_sim  = a["std_amp"]
        npts        = float(a["npts"])
        w_sim       = a["meanwt"]

        visstd_th   = _visstd_calc(tag)
        w_calc      = _w_calc(tag)

        sigma_im    = ns["sigma_im_calc"]
        rms_calc_mJy = sigma_im * 1e3
        rms_meas_mJy = meas_rms.get(tag, float("nan")) * 1e3

        return (
            baseline_label,
            vismean, visstd_sim, visstd_th,
            npts, w_calc, w_sim,
            rms_calc_mJy, rms_meas_mJy,
        )

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

    print(
        "\nCalculated PB size for type A (dia={0:2.2f} m): {1:3.5f} arcmin"
        .format(D_A, calc_ang(freq_ghz, D_A))
    )
    if D_B is not None:
        print(
            "Calculated PB size for type B (dia={0:2.2f} m): {1:3.5f} arcmin"
            .format(D_B, calc_ang(freq_ghz, D_B))
        )

    # --- Cleanup temporary group MSs ---
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

