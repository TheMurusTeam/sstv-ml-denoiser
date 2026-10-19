#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Hany El Imam
"""
Makes pictures full of text, the hardest case for an SSTV denoiser, to add
to the training data.

    python3 2_training_data/make_text_pictures.py
    python3 2_training_data/make_text_pictures.py --count 800 --backgrounds ~/Pictures/for_training

Then give the folder to step 2 together with your other pictures:

    python3 2_training_data/make_training_data.py --pictures ~/Pictures/for_training PICTURES_TEXT

Why
    Photos have little text, so a network trained on photos sees few letters
    and learns to soften them.  SSTV pictures - QSL cards above all - are
    full of callsigns, reports and greetings.  These pictures look like that:
    text of every size, from big outlined callsigns to small lines of 7-8
    pixels, in many fonts and colours, with outlines, shadows and boxes,
    over photos, gradients or flat colours.

What a picture holds
    - a background: a piece of one of your photos (--backgrounds; photos
      that are test pictures of step 1 are left out), or a colour gradient,
      a flat colour, soft colour blobs, or black;
    - 3 to 7 pieces of text: a callsign (amateur like IZ1ABC, or CB like
      1AT123), words and short sentences of a QSO (CQ, QSL, 73, RST 59,
      frequencies, dates, locators, names...);
    - each piece with its own font, size, colour and style: plain, outlined,
      with a shadow, on a box, or a little rotated;
    - sometimes thin lines and frames, as on QSL cards.

The pictures are 640x512: step 2 brings them to 320x256 (exactly half), so
the letters are smooth as in a real picture.  The same --seed gives the same
pictures.

Fonts: the free fonts in 2_training_data/fonts (SIL Open Font License, see
FONTS.md), plus any .ttf/.otf you put there; --system-fonts also uses the
fonts installed on your computer, for more variety.

Output (default PICTURES_TEXT at the top of the package, made from scratch):
    text_00001.png ...      the pictures
    _examples.jpg           the first 12, to have a look (skipped by step 2)

Needs: Python 3 with numpy and Pillow.
"""
import argparse
import glob
import os
import random
import shutil
import string
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import sstvml  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 640, 512                 # twice 320x256
DEFAULT_OUT = os.path.join(sstvml.ROOT, "PICTURES_TEXT")
SYSTEM_FONT_FOLDERS = [
    "/System/Library/Fonts", "/Library/Fonts", os.path.expanduser("~/Library/Fonts"),     # macOS
    os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"),                       # Windows
    "/usr/share/fonts", os.path.expanduser("~/.local/share/fonts"),                       # Linux
]

# ---- the words ------------------------------------------------------------------------

WORDS = ["CQ", "CQ CQ", "QSL", "QSO", "SSTV", "DX", "73", "73!", "TNX", "TU", "PSE QSL", "DE", "UR", "OP",
         "QTH", "LOC", "RIG", "ANT", "PWR", "RST", "Thanks for the QSO", "Best 73", "vy 73", "Good DX",
         "Hello from", "Radio Club", "Worldwide DX Group", "Division", "Contest", "Award", "Not only radio",
         "Martin M1", "Scottie S1", "Robot 36", "PD120", "eQSL", "Confirmed", "Mobile", "Portable", "/P", "/M"]
NAMES = ["Doug", "Joel", "Mario", "Anna", "Luca", "Peter", "Hans", "Maria", "Ivan", "Kenji", "Paolo", "Sara",
         "Tom", "Olga", "Carlos", "Yuki", "Ahmed", "Marco", "Elena", "Jan"]
PLACES = ["Rome", "Milano", "Berlin", "Tokyo", "Toronto", "Madrid", "Lisbon", "Oslo", "Sydney", "Napoli",
          "Bavaria", "Tuscany", "Sicily", "Canada", "Italy", "Brazil", "Japan", "Texas", "Alps", "Sardinia"]


def callsign(rng):
    """An amateur (IZ1ABC, K1AB, 9A2XY) or CB style (1AT123, 14JG036) callsign"""
    up, dig = string.ascii_uppercase, string.digits
    kind = rng.random()
    if kind < 0.55:
        prefix = rng.choice([rng.choice(up), rng.choice(up) + rng.choice(up), rng.choice(dig) + rng.choice(up),
                             rng.choice(up) + rng.choice(dig)])
        return prefix + rng.choice(dig) + "".join(rng.choice(up) for _ in range(rng.randint(1, 3)))
    return (str(rng.randint(1, 199)) + "".join(rng.choice(up) for _ in range(2))
            + str(rng.randint(1, 999)).zfill(rng.choice([2, 3])))


def phrase(rng):
    """A short piece of QSO text"""
    r = rng.random()
    if r < 0.25:
        return rng.choice(WORDS)
    if r < 0.35:
        return "RST %d%d%s" % (rng.randint(3, 5), rng.randint(5, 9), rng.choice(["", "9"]))
    if r < 0.45:
        return "%.3f MHz" % rng.choice([3.735, 7.165, 14.230, 21.340, 28.680, 27.555, 27.267, 145.500])
    if r < 0.55:
        return "%04d-%02d-%02d  %02d%02d UTC" % (rng.randint(2015, 2030), rng.randint(1, 12), rng.randint(1, 28),
                                                 rng.randint(0, 23), rng.randint(0, 59))
    if r < 0.62:
        up = string.ascii_uppercase
        return "%s%s%d%d%s%s" % (rng.choice(up[:18]), rng.choice(up[:18]), rng.randint(0, 9), rng.randint(0, 9),
                                 rng.choice(up[:24]).lower(), rng.choice(up[:24]).lower())
    if r < 0.75:
        return "%s %s %s" % (rng.choice(["de", "op", "from", "73 de"]), rng.choice(NAMES), callsign(rng))
    if r < 0.85:
        return "%s, %s" % (rng.choice(PLACES), rng.choice(PLACES))
    words = [rng.choice(WORDS + NAMES + PLACES) for _ in range(rng.randint(2, 5))]
    return " ".join(words)


# ---- fonts ------------------------------------------------------------------------------------

def find_fonts(system):
    files = glob.glob(os.path.join(HERE, "fonts", "*.ttf")) + glob.glob(os.path.join(HERE, "fonts", "*.otf"))
    if system:
        for d in SYSTEM_FONT_FOLDERS:
            if os.path.isdir(d):
                for root, _dirs, names in os.walk(d):
                    files += [os.path.join(root, n) for n in names if n.lower().endswith((".ttf", ".otf"))]
    good = []
    for f in sorted(set(files)):
        try:                                    # a font that cannot draw Latin letters is left out
            font = ImageFont.truetype(f, 40)
            box = font.getbbox("Ag7")
            if box[2] - box[0] > 20:
                good.append(f)
        except Exception:
            pass
    return good


# ---- backgrounds -----------------------------------------------------------------------------

def random_colour(rng, dark=None):
    if dark is True:
        return tuple(rng.randint(0, 90) for _ in range(3))
    if dark is False:
        return tuple(rng.randint(150, 255) for _ in range(3))
    return tuple(rng.randint(0, 255) for _ in range(3))


def background(rng, photos):
    r = rng.random()
    if photos and r < 0.45:
        try:
            im = ImageOps.exif_transpose(Image.open(rng.choice(photos))).convert("RGB")
            scale = max(W / im.width, H / im.height) * rng.uniform(1.0, 1.6)
            im = im.resize((max(W, int(im.width * scale)), max(H, int(im.height * scale))), Image.LANCZOS)
            x, y = rng.randint(0, im.width - W), rng.randint(0, im.height - H)
            return im.crop((x, y, x + W, y + H))
        except Exception:
            pass
    if r < 0.65:                                # gradient at a random angle
        a, b = np.array(random_colour(rng)), np.array(random_colour(rng))
        ang = rng.uniform(0, np.pi)
        yy, xx = np.mgrid[0:H, 0:W]
        t = (np.cos(ang) * xx / W + np.sin(ang) * yy / H)
        t = (t - t.min()) / (t.max() - t.min() + 1e-9)
        return Image.fromarray((a * (1 - t[..., None]) + b * t[..., None]).astype(np.uint8))
    if r < 0.8:                                 # soft colour blobs
        small = np.random.default_rng(rng.randint(0, 2 ** 31)).integers(0, 256, (rng.randint(3, 8), rng.randint(3, 8), 3))
        return Image.fromarray(small.astype(np.uint8)).resize((W, H), Image.BICUBIC).filter(ImageFilter.GaussianBlur(20))
    if r < 0.93:
        return Image.new("RGB", (W, H), random_colour(rng))
    return Image.new("RGB", (W, H), random_colour(rng, dark=True))


# ---- text -------------------------------------------------------------------------------------

def contrast_colour(rng, fill):
    """A colour that stands out from fill (for outlines and shadows)"""
    lum = 0.299 * fill[0] + 0.587 * fill[1] + 0.114 * fill[2]
    return random_colour(rng, dark=lum > 128)


def draw_text(canvas, rng, fonts, text, size):
    """Draws one piece of text, with a random style, at a random free-ish place"""
    font = ImageFont.truetype(rng.choice(fonts), size)
    fill = random_colour(rng)
    style = rng.random()
    stroke = rng.randint(1, max(1, size // 14)) if style < 0.45 else 0
    stroke_col = contrast_colour(rng, fill)
    pad = stroke + 4
    box = font.getbbox(text, stroke_width=stroke)
    tw, th = box[2] - box[0] + 2 * pad, box[3] - box[1] + 2 * pad
    if tw > W - 8:                              # too long: smaller
        return draw_text(canvas, rng, fonts, text, max(12, int(size * (W - 8) / tw)))
    layer = Image.new("RGBA", (tw + 12, th + 12), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    org = (pad - box[0], pad - box[1])
    if 0.45 <= style < 0.6:                     # box behind the text
        alpha = rng.choice([255, 255, 200, 140])
        d.rounded_rectangle((0, 0, tw - 1, th - 1), radius=rng.randint(0, 12),
                            fill=contrast_colour(rng, fill) + (alpha,))
    if 0.6 <= style < 0.8:                      # drop shadow
        off = rng.randint(2, max(2, size // 12))
        shadow = Image.new("RGBA", layer.size, (0, 0, 0, 0))
        ImageDraw.Draw(shadow).text((org[0] + off, org[1] + off), text, font=font, fill=(0, 0, 0, 200))
        if rng.random() < 0.5:
            shadow = shadow.filter(ImageFilter.GaussianBlur(off / 2))
        layer = Image.alpha_composite(layer, shadow)
        d = ImageDraw.Draw(layer)
    d.text(org, text, font=font, fill=fill + (255,), stroke_width=stroke, stroke_fill=stroke_col + (255,))
    if rng.random() < 0.15:                     # a little rotated
        layer = layer.rotate(rng.uniform(-10, 10), resample=Image.BICUBIC, expand=True)
    x = rng.randint(0, max(0, W - layer.width))
    y = rng.randint(0, max(0, H - layer.height))
    canvas.alpha_composite(layer, (x, y))


def draw_lines(canvas, rng):
    """Thin lines and frames, as on QSL cards"""
    d = ImageDraw.Draw(canvas)
    col = random_colour(rng) + (255,)
    if rng.random() < 0.5:
        m, w = rng.randint(4, 30), rng.randint(1, 4)
        d.rectangle((m, m, W - 1 - m, H - 1 - m), outline=col, width=w)
    for _ in range(rng.randint(0, 3)):
        y = rng.randint(10, H - 10)
        x0, x1 = sorted((rng.randint(0, W // 2), rng.randint(W // 2, W)))
        d.line((x0, y, x1, y), fill=col, width=rng.randint(1, 3))


def without_test_pictures(photos):
    """Leaves out the background photos that are test pictures (step 1): a
    test picture hidden behind some text would still be a test picture in
    the training data, and step 2 could not recognise it any more."""
    test = []
    for set_name in ("reference", "mine"):
        for f in sstvml.list_pictures(os.path.join(sstvml.TEST_SET, set_name, "pictures_320")):
            test.append((sstvml.fingerprint(f), "%s/%s" % (set_name, os.path.basename(f))))
    if not test:
        if photos:
            print("warning: no test sets yet (step 1), the backgrounds cannot be checked against them")
        return photos
    kept = []
    for p in photos:
        try:
            im, _size, _scale = sstvml.fit_picture(p)
        except Exception:
            continue
        g = np.asarray(im.convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
        fp = int("".join("1" if b else "0" for b in (g[:, 1:] > g[:, :-1]).flatten()), 2)
        twin = sstvml.near_duplicate(fp, test)
        if twin:
            print("  background left out: %s is the test picture %s" % (os.path.basename(p), twin))
        else:
            kept.append(p)
    return kept


def make_picture(rng, fonts, photos):
    canvas = background(rng, photos).convert("RGBA")
    if rng.random() < 0.3:
        draw_lines(canvas, rng)
    draw_text(canvas, rng, fonts, callsign(rng), rng.randint(60, 150))     # the big callsign
    for _ in range(rng.randint(2, 6)):
        r = rng.random()
        size = rng.randint(14, 24) if r < 0.4 else (rng.randint(24, 50) if r < 0.8 else rng.randint(50, 90))
        draw_text(canvas, rng, fonts, callsign(rng) if rng.random() < 0.2 else phrase(rng), size)
    return canvas.convert("RGB")


def main():
    ap = argparse.ArgumentParser(description="Makes pictures full of text for the training data")
    ap.add_argument("--count", type=int, default=500, help="how many pictures (default 500)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output folder (default PICTURES_TEXT)")
    ap.add_argument("--backgrounds", help="folder of photos to use as backgrounds (optional)")
    ap.add_argument("--system-fonts", action="store_true", help="also use the fonts installed on the computer")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()

    fonts = find_fonts(a.system_fonts)
    if not fonts:
        sys.exit("no usable fonts in %s" % os.path.join(HERE, "fonts"))
    photos = sstvml.list_pictures(a.backgrounds) if a.backgrounds else []
    if a.backgrounds and not photos:
        sys.exit("no pictures in %s" % a.backgrounds)
    photos = without_test_pictures(photos)
    print("%d fonts, %d background photos, %d pictures..." % (len(fonts), len(photos), a.count))

    if os.path.isdir(a.out):
        shutil.rmtree(a.out)
    os.makedirs(a.out)
    rng = random.Random(a.seed)
    examples = []
    for i in range(1, a.count + 1):
        im = make_picture(rng, fonts, photos)
        im.save(os.path.join(a.out, "text_%05d.png" % i))
        if len(examples) < 12:
            examples.append(im.resize((320, 256), Image.LANCZOS))
        if i % 100 == 0 or i == a.count:
            print("  %d/%d" % (i, a.count), flush=True)
    sheet = Image.new("RGB", (320 * 4, 256 * 3), "white")
    for k, im in enumerate(examples):
        sheet.paste(im, (320 * (k % 4), 256 * (k // 4)))
    sheet.save(os.path.join(a.out, "_examples.jpg"), quality=88)
    print("pictures in %s (a look at the first ones: _examples.jpg)" % a.out)
    print("next: python3 2_training_data/make_training_data.py --pictures YOUR_PICTURES %s" % a.out)


if __name__ == "__main__":
    main()
