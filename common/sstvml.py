# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Hany El Imam
"""
Shared code of the scripts: folders, the two C tools, picture preparation,
the SSTV channel simulation, picture comparison.

The scripts of the numbered folders import it with

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
    import sstvml

Every folder this package creates is at the top of the repository and has a
name in CAPITALS (DATA, TEST_SET, TRAINED, RESULTS): they can be deleted and
made again at any time.
"""
import os
import re
import subprocess
import sys
import zlib

import numpy as np
from PIL import Image, ImageFilter, ImageOps

# ---- folders ----------------------------------------------------------------------

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
BIN = os.path.join(ROOT, "bin")
DATA = os.path.join(ROOT, "DATA")              # training data (step 2)
TEST_SET = os.path.join(ROOT, "TEST_SET")      # test sets (step 1)
TRAINED = os.path.join(ROOT, "TRAINED")        # trained models (step 3)
RESULTS = os.path.join(ROOT, "RESULTS")        # scores and HTML pages (steps 4, 5, 6)
REFERENCE_MODEL = os.path.join(ROOT, "models", "sstv_dnn_20x6.bin")

# ---- SSTV ---------------------------------------------------------------------------

WIDTH, HEIGHT = 320, 256                       # size of an SSTV picture (Martin, Scottie)
MODES = {                                      # simulator name -> mode
    "m1": "Martin M1", "m2": "Martin M2",
    "s1": "Scottie S1", "s2": "Scottie S2",
    "r36": "Robot 36", "r72": "Robot 72",
}
NO_NOISE_SNR = 60                              # dB: the "clean" reception of a pair
MISALIGNED_DB = 20.0                           # blurred PSNR under this = the decoder lost the sync
PICTURE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp")


# ---- the C tools ------------------------------------------------------------------------

def tool(name):
    """Path of bin/<name> (bin/<name>.exe on Windows); stops with a hint if missing"""
    for p in (os.path.join(BIN, name), os.path.join(BIN, name + ".exe")):
        if os.path.isfile(p):
            return p
    sys.exit("bin/%s is missing. Build the tools with  make  (macOS, Linux, Windows MSYS2)\n"
             "or download the zip for your system from the GitHub Releases page and unzip it\n"
             "at the top of the repository (it contains the bin/ folder)." % name)


def run_tool(args):
    """Runs a C tool; returns (ok, {key: value}) from its "key=value" output line"""
    r = subprocess.run([str(a) for a in args], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    fields = {}
    for word in r.stdout.split():
        if "=" in word:
            k, v = word.split("=", 1)
            fields[k] = v
    return r.returncode == 0, fields


# ---- pictures -------------------------------------------------------------------------------

def list_pictures(folder):
    """Picture files of a folder (not of its subfolders), sorted by name.
    Files whose name starts with "." or "_" are skipped: hidden files, and
    the check sheets the tools write next to their pictures (_check.jpg,
    _examples.jpg)."""
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if f.lower().endswith(PICTURE_EXT) and not f.startswith((".", "_")))


def clean_name(path):
    """A file name usable everywhere: letters, digits, - and single _ only.
    A double underscore separates the picture from the noise case in the
    noisy file names, so it must never appear in a picture name."""
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = re.sub(r"[^A-Za-z0-9-]+", "_", stem).strip("_")
    return stem or "picture"


def fit_picture(path):
    """Opens any picture and brings it to a full 320x256 picture:
    - turned as the camera says (EXIF orientation);
    - transparency over black, then RGB;
    - cropped at the centre to the 5:4 shape of 320x256 (the long side is cut,
      nothing is squeezed and no black bands are added);
    - scaled down or up to exactly 320x256 (Lanczos).
    Returns (picture, (original width, original height), scale factor)."""
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (0, 0, 0, 255))
        im = Image.alpha_composite(bg, im)
    im = im.convert("RGB")
    w, h = im.size
    scale = max(WIDTH / w, HEIGHT / h)
    out = ImageOps.fit(im, (WIDTH, HEIGHT), method=Image.LANCZOS, centering=(0.5, 0.5))
    return out, (w, h), scale


def prepare_pictures(src_files, out_folder, log=print):
    """fit_picture() on every file, saved as out_folder/<clean name>.png.
    Returns the list of the PNG written.  Warns about pictures enlarged more
    than 2 times (they come out soft) and about names used twice."""
    os.makedirs(out_folder, exist_ok=True)
    done, seen = [], set()
    for f in src_files:
        try:
            im, (w, h), scale = fit_picture(f)
        except Exception as e:                  # not a picture, or damaged
            log("  skipped %s: %s" % (os.path.basename(f), e))
            continue
        name = clean_name(f)
        base, k = name, 2
        while name in seen:                     # photo.jpg and photo.png
            name = "%s-%d" % (base, k)
            k += 1
        seen.add(name)
        out = os.path.join(out_folder, name + ".png")
        im.save(out)
        note = ""
        if scale > 2:
            note = "  (enlarged %.1f times: it will be soft, a bigger original is better)" % scale
        log("  %-40s %5dx%-5d -> 320x256%s" % (os.path.basename(f), w, h, note))
        done.append(out)
    return done


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def psnr(a, b):
    """Peak signal to noise ratio in dB between two pictures of the same size
    (higher = more alike; 99 = identical)"""
    mse = float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))
    return 99.0 if mse < 1e-10 else 10 * np.log10(255.0 ** 2 / mse)


# ---- the "detail" part of a picture: text, lines, edges ----------------------------------

DETAIL_EDGE = 40        # brightness step between two neighbour pixels that makes an edge (0..255)


def detail_mask(rgb):
    """True where the picture has detail: pixels next to a strong brightness
    step (letters, lines, contours), widened by one pixel all around.

    It is computed on the CLEAN picture, so it marks where the detail really
    is, not where the noise is.  Smooth areas (sky, skin, flat backgrounds)
    are left out: the PSNR on this mask tells how well a denoiser keeps text
    and fine lines, which the PSNR of the whole picture hides, because most
    pixels of a picture are smooth."""
    a = np.asarray(rgb, np.float32)
    y = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    edge = np.zeros(y.shape, bool)
    gx = np.abs(y[:, 1:] - y[:, :-1]) > DETAIL_EDGE
    gy = np.abs(y[1:, :] - y[:-1, :]) > DETAIL_EDGE
    edge[:, 1:] |= gx
    edge[:, :-1] |= gx
    edge[1:, :] |= gy
    edge[:-1, :] |= gy
    wide = edge.copy()                  # widen by one pixel (3x3)
    wide[1:, :] |= edge[:-1, :]
    wide[:-1, :] |= edge[1:, :]
    wide2 = wide.copy()
    wide2[:, 1:] |= wide[:, :-1]
    wide2[:, :-1] |= wide[:, 1:]
    return wide2


def psnr_masked(a, b, mask):
    """PSNR (dB) only on the pixels where mask is True; None if the mask is empty"""
    if not mask.any():
        return None
    d = np.asarray(a, np.float64)[mask] - np.asarray(b, np.float64)[mask]
    mse = float(np.mean(d ** 2))
    return 99.0 if mse < 1e-10 else 10 * np.log10(255.0 ** 2 / mse)


def blurred(path):
    """Grey and strongly blurred: compares the layout of two pictures, not their noise"""
    return np.asarray(Image.open(path).convert("L").filter(ImageFilter.GaussianBlur(3)), dtype=np.float64)


def aligned(noisy_path, clean_blurred):
    """False if the decoder lost the sync (picture shifted, cut or of another size)"""
    b = blurred(noisy_path)
    return b.shape == clean_blurred.shape and psnr(b, clean_blurred) > MISALIGNED_DB


# ---- near duplicates ------------------------------------------------------------------------

def fingerprint(path):
    """64-bit difference hash: the same picture, even resized, recompressed or
    slightly retouched, gives (almost) the same bits"""
    g = np.asarray(Image.open(path).convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def near_duplicate(fp, fps, max_bits=6):
    """The first fingerprint of fps that differs from fp by at most max_bits, or None"""
    for other, name in fps:
        if bin(fp ^ other).count("1") <= max_bits:
            return name
    return None


# ---- the SSTV channel ------------------------------------------------------------------------

def seed_of(text):
    """A fixed seed from a name: the same noise for the same file, everywhere"""
    return zlib.crc32(text.encode()) & 0x7FFFFFFF


def simulate(picture, mode, snr, seed, out, impulses=0, fading=0, bursts=0):
    """One SSTV reception through bin/sstv_channel_sim; True if a picture came out.
    impulses and bursts are per minute, fading is the depth in dB."""
    r = subprocess.run([tool("sstv_channel_sim"), picture, mode, str(snr), str(seed), out,
                        str(impulses), str(fading), str(bursts)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return r.returncode == 0 and os.path.exists(out)


def make_pairs(picture, mode, name, cases, out_folder):
    """The clean reception of picture in mode, and the noisy ones of cases.

    cases: list of (case name, snr, impulses/min, fading dB, bursts/min, seed)
    Writes  out_folder/clean/<name>.png  and  out_folder/noisy/<name>__<case>.png
    A noisy reception whose decoder lost the sync is thrown away: a pair that
    is not aligned teaches the network to blur.
    Returns (kept, dropped)."""
    clean = os.path.join(out_folder, "clean", name + ".png")
    if not simulate(picture, mode, NO_NOISE_SNR, 1, clean):
        return 0, len(cases)
    cb = blurred(clean)
    kept = dropped = 0
    for case, snr, imp, fade, burst, seed in cases:
        path = os.path.join(out_folder, "noisy", "%s__%s.png" % (name, case))
        if simulate(picture, mode, snr, seed, path, imp, fade, burst) and aligned(path, cb):
            kept += 1
        else:
            dropped += 1
            if os.path.exists(path):
                os.remove(path)
    return kept, dropped


def find_pairs(folder):
    """(noisy path, clean path, case) for every noisy picture that has its clean one.
    The case is what follows "__" in the noisy name (e.g. "snr6", "burst_snr12")."""
    pairs = []
    noisy_dir = os.path.join(folder, "noisy")
    if not os.path.isdir(noisy_dir):
        return pairs
    for f in sorted(os.listdir(noisy_dir)):
        if not f.endswith(".png") or "__" not in f:
            continue
        base, case = f[:-4].split("__", 1)
        c = os.path.join(folder, "clean", base + ".png")
        if os.path.exists(c):
            pairs.append((os.path.join(noisy_dir, f), c, case))
    return pairs


def test_sets():
    """The test sets made by step 1: {"reference": path, "mine": path}, only those that exist"""
    sets = {}
    for name in ("reference", "mine"):
        p = os.path.join(TEST_SET, name)
        if find_pairs(p):
            sets[name] = p
    return sets


def model_labels(paths):
    """Short names for model files: sstv_dnn_20x6.bin -> 20x6; when two are the
    same, the name of the folder is added"""
    labels = [os.path.splitext(os.path.basename(p))[0].replace("sstv_dnn_", "") for p in paths]
    for i, l in enumerate(labels):
        if labels.count(l) > 1:
            labels[i] = "%s (%s)" % (l, os.path.basename(os.path.dirname(os.path.abspath(paths[i]))))
    if len(set(labels)) < len(labels):
        sys.exit("two models have the same name and folder: %s" % ", ".join(paths))
    return labels
