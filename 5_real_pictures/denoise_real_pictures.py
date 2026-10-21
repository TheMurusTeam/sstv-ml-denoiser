#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Hany El Imam
"""
Step 5 - a model on real received SSTV pictures, as the radio would use it,
with an HTML page to look at the result.

    python3 5_real_pictures/denoise_real_pictures.py --pictures FOLDER [--model MODEL.bin]

FOLDER: SSTV pictures you received (PNG or JPG, as saved by the radio or by
any SSTV program).  They are used as they are, not resized: they are already
SSTV pictures.  Files whose name ends in _dn (copies already cleaned by the
radio) are skipped.

Each picture is treated exactly as the radio treats it at the end of a
reception with Denoise ML (bin/sstv_denoise_tool radio):
  1. the noise is measured (the measure of the classic denoiser);
  2. a clean picture (noise under the gate) is LEFT AS IT IS: the network
     does not even run;
  3. otherwise the network cleans the received picture; black rows at the
     bottom (a picture not complete) are left out.

Output (default RESULTS/real_pictures/<model>/):
    index.html        received | cleaned, one row per picture; press and hold
                      a cleaned picture to see the received one in its place
    pictures/         the pictures of the page
    results.csv       measured noise, whether the network ran, multiply-adds

Options
    --model FILE   the model (default models/sstv_dnn_20x6.bin, the reference)
    --classic      adds a column with the classic (non-ML) denoiser
    --always       runs the network on every picture, clean ones too
                   (to see what it does to a clean picture)
    --out FOLDER   another output folder
    --jobs N       pictures cleaned at the same time (default: all the cores)

Needs: Python 3 with numpy and Pillow, and the tools in bin/.
"""
import argparse
import csv
import os
import pathlib
import shutil
import sys
import webbrowser
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import html_page  # noqa: E402
import sstvml  # noqa: E402


def received_pictures(folder):
    """The pictures of the folder, without the copies cleaned by the radio (*_dn)"""
    return [f for f in sstvml.list_pictures(folder)
            if not os.path.splitext(os.path.basename(f))[0].lower().endswith("_dn")]


def main():
    ap = argparse.ArgumentParser(description="Step 5: a model on real received SSTV pictures, with an HTML page")
    ap.add_argument("--pictures", required=True, help="folder of received SSTV pictures")
    ap.add_argument("--model", default=sstvml.REFERENCE_MODEL)
    ap.add_argument("--classic", action="store_true", help="add the classic denoiser")
    ap.add_argument("--always", action="store_true", help="run the network on clean pictures too")
    ap.add_argument("--out", help="output folder (default RESULTS/real_pictures/<model>)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--no-browser", action="store_true", help="do not open the page at the end")
    a = ap.parse_args()

    if not os.path.isfile(a.model):
        sys.exit("model not found: %s" % a.model)
    label = sstvml.model_labels([a.model])[0]
    files = received_pictures(a.pictures)
    if not files:
        sys.exit("no pictures in %s" % a.pictures)
    out = a.out or os.path.join(sstvml.RESULTS, "real_pictures", label.replace(" ", "_"))
    if os.path.isdir(out):
        shutil.rmtree(out)
    pics = os.path.join(out, "pictures")
    os.makedirs(pics)
    tool = sstvml.tool("sstv_denoise_tool")

    def one(i_f):
        i, f = i_f
        name = "%03d_%s" % (i, sstvml.clean_name(f))
        rx = os.path.join(pics, name + "_received.png")
        Image.open(f).convert("RGB").save(rx)
        ml = os.path.join(pics, name + "_ml.png")
        ok, info = sstvml.run_tool([tool, "ml" if a.always else "radio", a.model, rx, ml, 1])
        if not ok:
            return None
        if a.always:                        # the measure, for the page
            info.update(sstvml.run_tool([tool, "measure", rx])[1])
            info["ml"] = "1"
        cl = None
        if a.classic:
            cl = os.path.join(pics, name + "_classic.png")
            sstvml.run_tool([tool, "classic", rx, cl])
        return f, name, info, cl is not None

    print("cleaning %d pictures with %s..." % (len(files), label))
    with ThreadPoolExecutor(a.jobs) as ex:
        results = [r for r in ex.map(one, enumerate(files)) if r]

    rows, table = [], []
    ran = 0
    for f, name, info, has_classic in results:
        ml_ran = info.get("ml") == "1"
        ran += ml_ran
        noise = "noise Y %s, Cb %s, Cr %s" % (info.get("noise_y", "?"), info.get("noise_cb", "?"),
                                             info.get("noise_cr", "?"))
        what = ("network ran (%s G multiply-adds)" % info.get("gmacs", "?")) if ml_ran else \
            "clean picture: left as it is, as the radio does"
        size = info.get("size", "?")
        note = "" if size == "320x256" or size.startswith("320x") else " (not an SSTV size)"
        rx = "pictures/%s_received.png" % name
        figs = [html_page.figure("received", rx)]
        if has_classic:
            figs.append(html_page.figure("classic", "pictures/%s_classic.png" % name, rx))
        figs.append(html_page.figure("ML %s" % label, "pictures/%s_ml.png" % name, rx))
        rows.append(html_page.row(os.path.basename(f), "%s%s · %s · %s" % (size, note, noise, what), figs))
        table.append([os.path.basename(f), size, info.get("noise_y"), info.get("noise_cb"), info.get("noise_cr"),
                      int(ml_ran), info.get("gmacs")])

    intro = ("Model <b>%s</b> on %d received pictures, treated as the radio does: the network runs only on "
             "noisy pictures (%d of them); clean pictures are left as they are."
             % (html_page.esc(label), len(results), ran))
    if a.always:
        intro = ("Model <b>%s</b> on %d received pictures, run on every picture (--always), clean ones too."
                 % (html_page.esc(label), len(results)))
    html_page.write_page(os.path.join(out, "index.html"), "Real pictures: %s" % label, intro, "".join(rows),
                         columns=3 if a.classic else 2)
    with open(os.path.join(out, "results.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["picture", "size", "noise_y", "noise_cb", "noise_cr", "network_ran", "gmacs"])
        w.writerows(table)

    if a.always:
        print("%d pictures, the network ran on all of them (--always)." % len(results))
    else:
        print("%d pictures: the network ran on %d noisy ones, %d clean ones were left as they are."
              % (len(results), ran, len(results) - ran))
    page = os.path.join(out, "index.html")
    print("page: %s" % page)
    if not a.no_browser:
        webbrowser.open(pathlib.Path(page).resolve().as_uri())


if __name__ == "__main__":
    main()
