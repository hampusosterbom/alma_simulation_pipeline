"""
H. Österbom

utils.py

Shared small utilities for the pipeline.

Includes:
  * Logging configuration
  * Safe file/directory removal
  * Cleanup of CASA imaging products
  * Smooth-number image-size helper
  * Short descriptive naming helper for images and logs
"""

import logging
import shutil
from pathlib import Path


def setup_logging(log_file=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    if log_file:
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        ))
        logging.getLogger().addHandler(handler)

def safe_rm_tree(path):
    path = Path(path).expanduser().resolve()

    forbidden = {Path("/"), Path.home()}
    if path in forbidden:
        raise RuntimeError(f"Refusing to delete dangerous path: {path}")
    if not path.exists():
        return

    logging.info("[cleanup] Removing existing path: %s", path)
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def wipe(base: str | Path) -> None:
    """
    Remove common tclean output products for a given imagename base.
    """
    suffixes = (
        ".image", ".model", ".residual", ".psf", ".pb", ".sumwt",
        ".image.tt0", ".psf.tt0", ".pb.tt0", ".sumwt.tt0",
    )
    base = str(base)
    for suf in suffixes:
        safe_rm_tree(base + suf)


def next_smooth(target):
    """
    Find the smallest number >= target whose prime factors are only 2, 3, 5.

    Useful for FFT-friendly imsize values.
    """
    candidates = []
    max_exp = 20
    for a in range(1, max_exp):
        for b in range(max_exp):
            for c in range(max_exp):
                s = (2 ** a) * (3 ** b) * (5 ** c)
                if target <= s < 2 * target:
                    candidates.append(s)
    if not candidates:
        raise ValueError(f"No smooth number found near {target}; increase max_exp")
    return min(candidates)

def make_short_tag(sky, cfg_short, itime, dec, wt, rb, tap):
    """
    Short tag used in imagenames / filenames:
      sky_cfg_t[it]_[dec]_[weighting+robust]_t[taper]
    """
    sky_t = sky.replace('ninePlusCenter', '9C')
    it_t = 't' + itime.rstrip('h')
    sign = 'm' if dec.startswith('-') else 'p'
    val = dec.lstrip('+-').split('d')[0]
    dec_t = f"{sign}{val}"
    wm = {'natural': 'N', 'uniform': 'U', 'briggs': 'B'}[wt]
    rb_t = f"r{str(rb).replace('.', 'p')}" if wt == 'briggs' and rb is not None else ''
    tap_t = 't' + tap.replace('arcsec', '').replace('.', 'p')
    return '_'.join([sky_t, cfg_short, it_t, dec_t, wm + rb_t, tap_t])

