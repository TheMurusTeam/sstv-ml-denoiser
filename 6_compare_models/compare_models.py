#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Hany El Imam
"""
Step 6 - compares several models side by side, in one HTML page.

    python3 6_compare_models/compare_models.py MODEL_A.bin MODEL_B.bin [...] [--pictures FOLDER]

    python3 6_compare_models/compare_models.py models/sstv_dnn_20x6.bin TRAINED/16x7/sstv_dnn_16x7.bin \\
            --pictures ~/my_received_sstv

The page (default RESULTS/compare_models/index.html) has:

  1. the scores of every model on each test set (TEST_SET/reference and, if
     made, TEST_SET/mine), as step 4 prints them; the best of each row is
     in green;

  2. one example per test picture, the same reception for every model:
         received | clean (the target) | classic | model A | model B ...
     The examples are the "snr3" reception in Martin M1: noisy enough to show
     the differences;

  3. with --pictures, your real received pictures, each treated as the radio
     does (a clean picture is left as it is):
         received | model A | model B ...

Press and hold a cleaned picture to see the received one in its place.

Options
    --pictures FOLDER   real received SSTV pictures (PNG/JPG), see step 5
    --example CASE      the test reception shown in part 2 (default snr3)
    --out FOLDER        another output folder
    --jobs N            pictures cleaned at the same time (default: all the cores)

Needs: Python 3 with numpy and Pillow, and the tools in bin/.
"""
import argparse
import os
import pathlib
import shutil
import sys
import tempfile
import webbrowser
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import html_page  # noqa: E402
import scoring  # noqa: E402
import sstvml  # noqa: E402

EXAMPLE_MODE = "m1"


def copy_png(src, dst):
    shutil.copyfile(src, dst)
    return dst


def test_examples(set_name, set_path, work, names, pics, case):
    """Part 2: one row per test picture, the `case` reception in EXAMPLE_MODE"""
    rows = []
    pairs = sstvml.find_pairs(set_path)
    by_clean = {}
    for noisy, clean, c in pairs:
        by_clean.setdefault(clean, []).append((noisy, c))
    for clean, items in sorted(by_clean.items()):
        base = os.path.splitext(os.path.basename(clean))[0]
        if not base.endswith("_" + EXAMPLE_MODE):
            continue
        pick = [n for n, c in items if c == case] or [n for n, _ in items]
        noisy = pick[0]
        stem = "%s_%s" % (set_name, os.path.splitext(os.path.basename(noisy))[0])
        rx = "pictures/%s_received.png" % stem
        copy_png(noisy, os.path.join(pics, os.path.basename(rx)))
        copy_png(clean, os.path.join(pics, "%s_clean.png" % stem))
        figs = [html_page.figure("received", rx),
                html_page.figure("clean (target)", "pictures/%s_clean.png" % stem)]
        received = sstvml.load_rgb(noisy)
        target = sstvml.load_rgb(clean)
        info = ["received %.1f dB" % sstvml.psnr(received, target)]
        for n in names:
            src = os.path.join(work, n, "noisy", os.path.basename(noisy))
            dst = "pictures/%s_%s.png" % (stem, sstvml.clean_name(n))
            copy_png(src, os.path.join(pics, os.path.basename(dst)))
            figs.append(html_page.figure(n, dst, rx))
            info.append("%s %+.1f dB" % (n, sstvml.psnr(sstvml.load_rgb(src), target) - sstvml.psnr(received, target)))
        rows.append(html_page.row(os.path.basename(noisy), " · ".join(info), figs))
    return rows


def real_pictures(folder, models, labels, pics, jobs):
    """Part 3: every real picture through every model, as the radio does"""
    tool = sstvml.tool("sstv_denoise_tool")
    files = [f for f in sstvml.list_pictures(folder)
             if not os.path.splitext(os.path.basename(f))[0].lower().endswith("_dn")]
    todo, names = [], []
    for i, f in enumerate(files):
        stem = "real_%03d_%s" % (i, sstvml.clean_name(f))
        rx = os.path.join(pics, stem + "_received.png")
        Image.open(f).convert("RGB").save(rx)
        names.append((f, stem))
        for m, l in zip(models, labels):
            todo.append([tool, "radio", m, rx, os.path.join(pics, "%s_%s.png" % (stem, sstvml.clean_name(l))), 1])
    with ThreadPoolExecutor(jobs) as ex:
        res = list(ex.map(sstvml.run_tool, todo))
    rows = []
    k = 0
    for f, stem in names:
        rx = "pictures/%s_received.png" % stem
        figs = [html_page.figure("received", rx)]
        info = None
        for l in labels:
            ok, fields = res[k]
            k += 1
            info = info or fields
            figs.append(html_page.figure(l, "pictures/%s_%s.png" % (stem, sstvml.clean_name(l)), rx))
        what = "noisy: the models ran" if info and info.get("ml") == "1" else \
            "clean picture: left as it is, as the radio does"
        rows.append(html_page.row(os.path.basename(f), "%s · noise Y %s · %s"
                                  % (info.get("size", "?") if info else "?", info.get("noise_y", "?") if info else "?",
                                     what), figs))
    return rows, len(files)


def main():
    ap = argparse.ArgumentParser(description="Step 6: compares models side by side in an HTML page")
    ap.add_argument("models", nargs="+", help="model files (.bin)")
    ap.add_argument("--pictures", help="folder of real received SSTV pictures")
    ap.add_argument("--example", default="snr3", help="test reception shown for each picture (default snr3)")
    ap.add_argument("--out", default=os.path.join(sstvml.RESULTS, "compare_models"))
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--no-browser", action="store_true", help="do not open the page at the end")
    a = ap.parse_args()

    for m in a.models:
        if not os.path.isfile(m):
            sys.exit("model not found: %s" % m)
    labels = sstvml.model_labels(a.models)
    sets = sstvml.test_sets()
    if not sets:
        sys.exit("no test set: run step 1 first (python3 1_test_set/make_test_set.py)")
    if os.path.isdir(a.out):
        shutil.rmtree(a.out)
    pics = os.path.join(a.out, "pictures")
    os.makedirs(pics)

    body = []
    tables, examples = [], []
    for set_name, path in sets.items():
        print("scoring on the test set %s..." % set_name)
        work = tempfile.mkdtemp(prefix="sstv_compare_")
        r = scoring.score_set(a.models, labels, path, work, a.jobs)
        print(scoring.text_table(r))
        tables.append(html_page.scores_table(r, "Scores on the test set %s (dB added to the received picture; "
                                                "ALL = the mean)" % set_name))
        examples.append("<h2>Test set %s: one reception per picture (%s, %s)</h2>"
                        % (set_name, a.example, sstvml.MODES[EXAMPLE_MODE]))
        examples += test_examples(set_name, path, work, r["names"], pics, a.example)
        shutil.rmtree(work)
    body += tables + examples

    if a.pictures:
        print("real pictures...")
        rows, n = real_pictures(a.pictures, a.models, labels, pics, a.jobs)
        body.append("<h2>Real received pictures (%d), treated as the radio does</h2>" % n)
        body += rows

    intro = "Models compared: %s. Every picture is cleaned with the C code of the radio." % ", ".join(
        "<b>%s</b> (%s)" % (html_page.esc(l), html_page.esc(m))
        for l, m in zip(labels, a.models))
    page = os.path.join(a.out, "index.html")
    html_page.write_page(page, "Model comparison", intro, "".join(body), columns=3 + len(labels))
    print("page: %s" % page)
    if not a.no_browser:
        webbrowser.open(pathlib.Path(page).resolve().as_uri())


if __name__ == "__main__":
    main()
