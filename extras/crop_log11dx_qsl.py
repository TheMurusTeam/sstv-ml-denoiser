#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Hany El Imam
"""
Crops eQSL cards from LOG11DX (log11dx.com) to their picture.

A LOG11DX card is the sender's picture on the left and, on the right, a light
"Radio Postcard" panel with the QSO data (From/To, date, frequency, RST...).
This script keeps only the picture on the left, at its original size: it is
NOT resized (the SSTV denoiser scripts bring pictures to 320x256 by themselves).

    python3 crop_log11dx_qsl.py INPUT_FOLDER OUTPUT_FOLDER

    INPUT_FOLDER    the LOG11DX cards (PNG, JPG, WEBP), subfolders included
    OUTPUT_FOLDER   made if missing: one PNG per card, same name as the card,
                    plus _check.jpg, a sheet of the first cards with the kept
                    part outlined in magenta, to check the cuts at a glance

How the cut is found
  1. The light panel is searched FROM THE RIGHT: walking left from the right
     edge, every column that is mostly light (cream/white) belongs to the
     panel; the first column that is not is the right edge of the picture.
     Searching from the right matters: a picture can have large light areas
     of its own (the white of a flag, a sky), which a search from the left
     would take for the panel.
  2. The thin border lines left around the picture (flat dark or light
     lines, the edge between picture and panel) are trimmed: at most 3% of
     each side, so a smooth area of the picture itself is never cut.
  3. A card without a light panel on the right (not a LOG11DX card, or a
     different layout) is reported and skipped, not copied.

Needs: Python 3 with numpy and Pillow.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageOps

EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
LIGHT = 200             # a pixel is "light" when R, G and B are all above this
PANEL_COLUMN = 0.5      # a column belongs to the panel when this share of it is light
SMOOTH = 15             # columns averaged together (the panel has thin lines and text)
MIN_PANEL = 0.15        # the panel is at least this share of the card width
FLAT_STD = 12.0         # a border line is "flat" when its brightness varies less than this
MAX_TRIM = 0.03         # at most this share of each side is trimmed as border
CHECK_CARDS = 8         # cards shown in _check.jpg


def light_columns(a):
    """Share of light pixels in every column, smoothed over SMOOTH columns"""
    col = (a.min(axis=2) > LIGHT).mean(axis=0)
    k = np.ones(SMOOTH) / SMOOTH
    return np.convolve(col, k, mode="same")


def panel_start(a):
    """Column where the light panel starts, searched from the right edge;
    None if there is no panel"""
    col = light_columns(a)
    w = a.shape[1]
    x = w - 1 - SMOOTH                 # skip the very edge (rounded corners, shadow)
    while x > 0 and col[x] > PANEL_COLUMN:
        x -= 1
    if w - x < MIN_PANEL * w:
        return None                    # no light panel on the right
    # the smoothing moves the edge a little: find the real first light column
    raw = (a.min(axis=2) > LIGHT).mean(axis=0)
    while x < w - 1 and raw[x] <= PANEL_COLUMN:
        x += 1
    return x


def trim_flat_border(a):
    """(top, bottom, left, right) of the picture without the thin border lines
    around it.  Only lines that are flat AND very dark or very light (a frame,
    the edge of the panel) are removed, and at most MAX_TRIM of each side:
    a smooth area of the picture itself (a red flag, a sky) is never cut."""
    g = a.mean(axis=2)
    h, w = g.shape

    def border(v):
        return v.std() < FLAT_STD and (v.mean() < 60 or v.mean() > 180)

    top, bot, left, right = 0, h, 0, w
    while top < MAX_TRIM * h and border(g[top, left:right]):
        top += 1
    while h - bot < MAX_TRIM * h and border(g[bot - 1, left:right]):
        bot -= 1
    while left < MAX_TRIM * w and border(g[top:bot, left]):
        left += 1
    while w - right < MAX_TRIM * w and border(g[top:bot, right - 1]):
        right -= 1
    return top, bot, left, right


def find_cards(folder):
    files = []
    for root, _dirs, names in os.walk(folder):
        for n in names:
            if n.lower().endswith(EXTENSIONS) and not n.startswith("."):
                files.append(os.path.join(root, n))
    return sorted(files)


def main():
    ap = argparse.ArgumentParser(description="Crops LOG11DX (log11dx.com) eQSL cards to their picture")
    ap.add_argument("input", help="folder with the LOG11DX cards")
    ap.add_argument("output", help="folder for the cropped pictures (made if missing)")
    a = ap.parse_args()

    cards = find_cards(a.input)
    if not cards:
        sys.exit("no pictures in %s" % a.input)
    os.makedirs(a.output, exist_ok=True)
    done, skipped, checks, used = 0, 0, [], set()
    for f in cards:
        try:
            img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
        except Exception as e:
            print("skipped %s: %s" % (f, e))
            skipped += 1
            continue
        arr = np.asarray(img).astype(np.int16)
        x = panel_start(arr)
        if x is None:
            print("skipped %s: no light panel on the right (not a LOG11DX card?)" % f)
            skipped += 1
            continue
        t, b, l, r = trim_flat_border(arr[:, :x])
        if r - l < 100 or b - t < 80:
            print("skipped %s: the picture part is too small" % f)
            skipped += 1
            continue

        name = os.path.splitext(os.path.basename(f))[0]
        base, k = name, 2
        while name in used:                     # same name in two subfolders
            name = "%s-%d" % (base, k)
            k += 1
        used.add(name)
        img.crop((l, t, r, b)).save(os.path.join(a.output, name + ".png"))
        done += 1
        print("%-40s %5dx%-5d -> picture %dx%d" % (os.path.basename(f), img.width, img.height, r - l, b - t))

        if len(checks) < CHECK_CARDS:
            c = img.copy()
            ImageDraw.Draw(c).rectangle((l, t, r - 1, b - 1), outline=(255, 0, 255),
                                        width=max(3, img.width // 250))
            c.thumbnail((900, 450))
            checks.append(c)

    if checks:
        sheet = Image.new("RGB", (900, 460 * len(checks)), "white")
        for i, c in enumerate(checks):
            sheet.paste(c, (0, 460 * i))
        sheet.save(os.path.join(a.output, "_check.jpg"), quality=85)
    print("\n%d cards cropped, %d skipped. Pictures in %s" % (done, skipped, a.output))
    if checks:
        print("Check the cuts in %s" % os.path.join(a.output, "_check.jpg"))


if __name__ == "__main__":
    main()
