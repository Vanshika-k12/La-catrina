"""Colour naming and HSV colour-harmony rules (no ML here, just simple rules)."""
import colorsys
import json
import os

import numpy as np

# Readable colour names the user can pick from. RGB is a typical swatch for each.
PALETTE = {
    "black": (20, 20, 20),
    "charcoal": (55, 55, 60),
    "grey": (128, 128, 128),
    "light grey": (200, 200, 200),
    "white": (245, 245, 245),
    "cream": (245, 235, 210),
    "beige": (220, 200, 165),
    "khaki": (190, 175, 120),
    "tan": (190, 145, 95),
    "brown": (115, 75, 45),
    "navy": (25, 35, 80),
    "denim": (75, 105, 150),
    "blue": (40, 90, 200),
    "light blue": (150, 190, 230),
    "teal": (0, 128, 128),
    "green": (40, 140, 60),
    "olive": (95, 95, 50),
    "yellow": (240, 210, 40),
    "mustard": (200, 160, 30),
    "orange": (240, 130, 30),
    "red": (200, 30, 40),
    "maroon": (110, 20, 35),
    "pink": (240, 150, 180),
    "purple": (110, 50, 140),
    "lavender": (180, 160, 215),
}
COLOUR_NAMES = list(PALETTE.keys())

# Colours treated as "go with anything" neutrals.
NEUTRALS = {"black", "charcoal", "grey", "light grey", "white", "cream",
            "beige", "khaki", "tan", "brown", "navy", "denim"}


def _srgb_to_lab(rgb):
    """Convert an (N,3) array of 0-255 RGB to CIELAB (D65) so nearest-name lookup is perceptual."""
    c = np.asarray(rgb, dtype=float) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


_PALETTE_LAB = _srgb_to_lab(np.array(list(PALETTE.values())))

# Colour NAMING uses reference points in CIELAB. By default these are the ideal swatches above.
# colour_calibration.json (made by `python evaluate_public.py --calibrate`) replaces them with the
# average colour actually extracted from labelled photos, e.g. shaded black fabric. The swatches
# shown in the app and the harmony rules always use PALETTE.
CALIBRATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "colour_calibration.json")


def _load_calibration():
    if not os.path.exists(CALIBRATION_FILE):
        return {}
    with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
        return {k: v for k, v in json.load(f).get("lab", {}).items() if k in PALETTE}


def set_naming_reference(calibrated_lab=None):
    """Use ideal swatches (None) or {name: [L, a, b]} overrides for colour naming."""
    global _NAMING_LAB
    _NAMING_LAB = _PALETTE_LAB.copy()
    for name, lab in (calibrated_lab or {}).items():
        _NAMING_LAB[COLOUR_NAMES.index(name)] = lab


set_naming_reference(_load_calibration())


def rgb_to_lab(rgb):
    return _srgb_to_lab(np.array([rgb], dtype=float))[0]


def rgb_to_name(rgb):
    """Nearest palette colour name for one RGB triple (perceptual distance in CIELAB)."""
    d = np.linalg.norm(_NAMING_LAB - rgb_to_lab(rgb), axis=1)
    return COLOUR_NAMES[int(np.argmin(d))]


def rgb_to_hex(rgb):
    return "#%02x%02x%02x" % tuple(int(v) for v in rgb)


def name_to_rgb(name):
    return PALETTE[name]


def _hsv(rgb):
    r, g, b = (v / 255.0 for v in rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return h * 360.0, s, v


def is_neutral(name, rgb):
    _, s, v = _hsv(rgb)
    return name in NEUTRALS or s < 0.2 or v < 0.2


def pair_harmony(name_a, rgb_a, name_b, rgb_b):
    """Score one colour pair in [-1, 1] and return (score, kind)."""
    if is_neutral(name_a, rgb_a) or is_neutral(name_b, rgb_b):
        return 1.0, "neutral"
    ha, sa, _ = _hsv(rgb_a)
    hb, sb, _ = _hsv(rgb_b)
    d = abs(ha - hb)
    d = min(d, 360 - d)  # hue distance on the colour wheel, 0..180
    if d <= 30:
        return 0.8, "analogous"
    if d >= 150:
        return 0.7, "complementary"
    if sa > 0.5 and sb > 0.5:
        return -1.0, "clash"
    return 0.0, "mixed"
