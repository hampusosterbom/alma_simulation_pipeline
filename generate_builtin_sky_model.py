"""
H. Österbom

generate_builtin_sky_model.py

Unified sky generator for the ALMA simulation pipeline.

Sky types:
  - 'ring'      : 8+1 Gaussian ring (central Gaussian 100x brighter than ring Gaussians)
  - 'pointdisk' : bright point source + faint small disk
"""

import sys
import argparse
import logging
import re
from math import sin, cos, pi

from casatools import quanta, componentlist, image
from casatasks import exportfits


# ---------- Shared helpers ----------

def setup_logging(log_file=None):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    if log_file:
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        ))
        logging.getLogger().addHandler(handler)


def extract_number(s):
    """
    Extract the numeric part from strings like '0.01arcsec', '0.01 arcsec',
    '1e-3Jy', '5.2 deg', etc. Returns float.
    """
    s = str(s)
    m = re.findall(r'[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', s)
    if not m:
        raise ValueError(f"Cannot extract a number from '{s}'")
    return float(m[0])


def validate_ra_dec(value, type_str):
    logging.debug(f"Validating {type_str}: {value}")
    try:
        qa = quanta()
        qa.convert(value, "rad")
        return value
    except Exception as e:
        logging.error(f"Failed to validate {type_str} {value}: {e}")
        raise ValueError(
            f"Invalid {type_str} format: {value}. "
            "Expected format like '12h00m00.00s' for RA or '-70d00m00.00' for Dec."
        )


def make_imagename(output_base, dec_center):
    tag = dec_center.replace("-", "m").replace("+", "p").replace("d", "").replace(".", "")
    return f"{output_base}_dec{tag}"


# ---------- Ring model (8+1 Gaussians) ----------

def make_ring_model(args):
    """
    8+1 Gaussian ring model (based on generate_sky_model.py).
    """
    qa_tool = quanta()
    comp_list = componentlist()
    img_tool = image()
    comp_list.done()

    flux = args.ring_flux
    sflux = flux / 100.0  # ring sources are 1/100 of central

    major_beam_arcsec = extract_number(args.ring_major_beam)
    minor_beam_arcsec = extract_number(args.ring_minor_beam)

    # Central Gaussian
    comp_list.addcomponent(
        dir=f"J2000 {args.ra_center} {args.dec_center}",
        flux=flux,
        fluxunit="Jy",
        freq=args.freq,
        shape="Gaussian",
        majoraxis=f"{major_beam_arcsec}arcsec",
        minoraxis=f"{minor_beam_arcsec}arcsec",
        positionangle=args.ring_pa_beam,
    )

    # Ring Gaussians
    dec_rad = qa_tool.convert(args.dec_center, "rad")
    cos_dec = cos(dec_rad["value"])
    for i in range(args.ring_n):
        ang = 2 * pi * i / args.ring_n
        dx_arcsec = args.ring_radius * cos(ang) / cos_dec
        dy_arcsec = args.ring_radius * sin(ang)

        ra_rad = qa_tool.convert(args.ra_center, "rad")
        dec_rad = qa_tool.convert(args.dec_center, "rad")
        dx_rad = qa_tool.convert(f"{dx_arcsec}arcsec", "rad")
        dy_rad = qa_tool.convert(f"{dy_arcsec}arcsec", "rad")
        new_ra_rec = qa_tool.add(ra_rad, dx_rad)
        new_dec_rec = qa_tool.add(dec_rad, dy_rad)
        offs_ra = qa_tool.tos(new_ra_rec)
        offs_dec = qa_tool.tos(new_dec_rec)

        comp_list.addcomponent(
            dir=f"J2000 {offs_ra} {offs_dec}",
            flux=sflux,
            fluxunit="Jy",
            freq=args.freq,
            shape="Gaussian",
            majoraxis=args.ring_major_beam,
            minoraxis=args.ring_minor_beam,
            positionangle=args.ring_pa_beam,
        )

    imagename = make_imagename(args.output_base, args.dec_center)
    im_shape = list(args.im_shape) + [1, 1]

    img_tool.fromshape(f"{imagename}.im", im_shape, overwrite=True)
    cs = img_tool.coordsys()
    cs.setunits(["rad", "rad", "", "Hz"])
    cs.setreferencevalue(
        [
            qa_tool.convert(args.ra_center, "rad")["value"],
            qa_tool.convert(args.dec_center, "rad")["value"],
        ],
        type="direction",
    )
    cell = qa_tool.convert(f"{args.cell_size}arcsec", "rad")["value"]
    cs.setincrement([-cell, cell], "direction")
    cs.setreferencevalue(args.freq, "spectral")
    cs.setreferencepixel([args.im_shape[0] // 2, args.im_shape[1] // 2, 0, 0])
    cs.setincrement("7.5GHz", "spectral")
    img_tool.setcoordsys(cs.torecord())
    img_tool.setbrightnessunit("Jy/pixel")

    img_tool.modify(comp_list.torecord(), subtract=False)
    img_tool.done()

    fitsname = f"{imagename}.fits"
    exportfits(imagename=f"{imagename}.im", fitsimage=fitsname, overwrite=True)
    logging.info(f"Saved ring sky model to {fitsname}")
    return fitsname

# ---------- Point + disk model ----------

def add_disk(cl, center_dir, diameter_arcsec, total_flux_jy, freq):
    d = f"{diameter_arcsec}arcsec"
    cl.addcomponent(
        dir=center_dir,
        flux=total_flux_jy,
        fluxunit="Jy",
        freq=freq,
        shape="Disk",
        majoraxis=d,
        minoraxis=d,
        positionangle="0deg",
    )


def add_point(cl, center_dir, flux_jy, freq):
    cl.addcomponent(
        dir=center_dir,
        flux=flux_jy,
        fluxunit="Jy",
        freq=freq,
        shape="Point",
    )


def make_pointdisk_model(args):
    """
    Point + disk model (based on generate_point_plus_disk_sky_model.py).
    """
    qa = quanta()
    cl = componentlist()
    ia_tool = image()
    cl.done()

    ra_rad = qa.convert(args.ra_center, "rad")
    dec_rad = qa.convert(args.dec_center, "rad")
    center_dir = f"J2000 {qa.tos(ra_rad)} {qa.tos(dec_rad)}"

    base_flux = args.pnd_flux
    point_flux = args.pnd_point_flux if args.pnd_point_flux is not None else base_flux
    disk_flux = args.pnd_disk_flux if args.pnd_disk_flux is not None else 0.1 * base_flux

    add_point(cl, center_dir, point_flux, args.freq)
    add_disk(cl, center_dir, args.pnd_disk_diameter, disk_flux, args.freq)

    logging.info(f"Point source: flux={point_flux:.3e} Jy at center.")
    logging.info(
        f"Disk: diameter={args.pnd_disk_diameter:.4f}\" flux={disk_flux:.3e} Jy."
    )

    imagename = make_imagename(args.output_base, args.dec_center)
    shape = list(args.im_shape) + [1, 1]
    ia_tool.fromshape(f"{imagename}.im", shape, overwrite=True)

    cs = ia_tool.coordsys()
    cs.setunits(["rad", "rad", "", "Hz"])
    cs.setreferencevalue([ra_rad["value"], dec_rad["value"]], type="direction")
    cell_rad = qa.convert(f"{args.cell_size}arcsec", "rad")["value"]
    cs.setincrement([-cell_rad, cell_rad], "direction")
    cs.setreferencevalue(args.freq, "spectral")
    cs.setreferencepixel([args.im_shape[0] // 2, args.im_shape[1] // 2, 0, 0])
    cs.setincrement("7.5GHz", "spectral")
    ia_tool.setcoordsys(cs.torecord())
    ia_tool.setbrightnessunit("Jy/pixel")

    ia_tool.modify(cl.torecord(), subtract=False)
    ia_tool.done()

    fitsname = f"{imagename}.fits"
    exportfits(imagename=f"{imagename}.im", fitsimage=fitsname, overwrite=True)
    logging.info(f"Saved point+disk sky model to {fitsname}")
    return fitsname


# ---------- CLI / JSON glue ----------

def parse_args_from_cli(args_list):
    p = argparse.ArgumentParser(
        description="Generate builtin ALMA sky model (ring or point+disk)",
        allow_abbrev=False,
    )

    # selector
    p.add_argument("--sky_type", choices=["ring", "pointdisk"], default="ring")

    # common params
    p.add_argument("--ra_center", default="12h00m00.00s", type=str)
    p.add_argument("--dec_center", default="-23d00m00.00", type=str)
    p.add_argument("--freq", default="343.5GHz", type=str)
    p.add_argument("--im_shape", type=int, nargs=2, default=[160, 160])
    p.add_argument("--cell_size", type=float, default=0.001, help="cell size in arcsec")
    p.add_argument("--output_base", default="builtinSky", type=str)
    p.add_argument("--log_file", default=None, type=str)

    # ring-specific params
    p.add_argument("--ring_flux", type=float, default=27e-5)
    p.add_argument("--ring_n", type=int, default=8)
    p.add_argument("--ring_radius", type=float, default=0.044, help="arcsec")
    p.add_argument("--ring_major_beam", default="0.022arcsec")
    p.add_argument("--ring_minor_beam", default="0.022arcsec")
    p.add_argument("--ring_pa_beam", default="0deg")

    # point+disk-specific params
    p.add_argument("--pnd_flux", type=float, default=1.0)
    p.add_argument("--pnd_point_flux", type=float, default=None)
    p.add_argument("--pnd_disk_flux", type=float, default=None)
    p.add_argument("--pnd_disk_diameter", type=float, default=0.10, help="arcsec")

    return p.parse_args(args_list)


def main(args):
    setup_logging(args.log_file)

    # basic validation
    args.ra_center = validate_ra_dec(args.ra_center, "RA")
    args.dec_center = validate_ra_dec(args.dec_center, "Dec")

    logging.info(
        f"Generating builtin sky model: type={getattr(args, 'sky_type', 'ring')} "
        f"at dec {args.dec_center}"
    )

    if args.sky_type == "ring":
        fitsname = make_ring_model(args)
    elif args.sky_type == "pointdisk":
        fitsname = make_pointdisk_model(args)
    else:
        raise ValueError(f"Unknown sky_type={args.sky_type}")

    print(fitsname)


if __name__ == "__main__":
    # JSON mode: casa -c generate_builtin_sky_model.py sky_config.json
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        import json
        from argparse import Namespace

        with open(sys.argv[1]) as f:
            config = json.load(f)
        args = Namespace(**config)
        if not hasattr(args, "sky_type"):
            args.sky_type = "ring"
        main(args)
        sys.exit(0)

    # CLI mode
    args_list = sys.argv[1:]
    args = parse_args_from_cli(args_list)
    main(args)
