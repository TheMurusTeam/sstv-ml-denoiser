#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Hany El Imam
"""
Step 4b - scores one or more models on the test sets, with the C code of the
radio, next to the classic (non-ML) denoiser.

    python3 4_check_and_score/score.py TRAINED/20x6/sstv_dnn_20x6.bin models/sstv_dnn_20x6.bin

Any number of models.  Each test set of step 1 (TEST_SET/reference and, if
made, TEST_SET/mine) is scored on its own: their numbers must not be mixed,
the reference one is the one to compare with other people.

What it prints, for each test set
    one row per kind of noise (snr0 ... burst_snr12 ...), and ALL = the mean
    over every noisy picture, THE number to compare; ALL detail = the same
    mean measured only on the detail of the pictures (text, thin lines,
    edges: sstvml.detail_mask), where softening shows most:
      received   PSNR of the received picture against the clean one, in dB
                 (higher = closer to the clean picture)
      classic    dB the classic denoiser adds to it
      <model>    dB each model adds to it
    then how much each model changes the CLEAN pictures (higher = less
    changed; above about 40 dB the change cannot be seen).

Everything is also written to RESULTS/scores/scores.csv.

Options
    --jobs N     pictures cleaned at the same time (default: all the cores)
    --keep       keeps the cleaned pictures in RESULTS/scores/<set>/<model>/

Needs: Python 3 with numpy and Pillow, and the tools in bin/.
"""
import argparse
import csv
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import scoring  # noqa: E402
import sstvml  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Step 4b: scores models on the test sets")
    ap.add_argument("models", nargs="+", help="model files (.bin)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--keep", action="store_true", help="keep the cleaned pictures in RESULTS/scores")
    a = ap.parse_args()
    for m in a.models:
        if not os.path.isfile(m):
            sys.exit("model not found: %s" % m)
    labels = sstvml.model_labels(a.models)
    sets = sstvml.test_sets()
    if not sets:
        sys.exit("no test set: run step 1 first (python3 1_test_set/make_test_set.py)")

    out = os.path.join(sstvml.RESULTS, "scores")
    os.makedirs(out, exist_ok=True)
    rows = []
    for set_name, path in sets.items():
        work = os.path.join(out, set_name) if a.keep else tempfile.mkdtemp(prefix="sstv_score_")
        print("\n=== test set %s (%s)" % (set_name, os.path.relpath(path, sstvml.ROOT)))
        r = scoring.score_set(a.models, labels, path, work, a.jobs)
        print(scoring.text_table(r))
        rows += scoring.csv_rows(set_name, r)
        if not a.keep:
            shutil.rmtree(work)

    with open(os.path.join(out, "scores.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["test_set", "noise", "pictures", "received_db", "classic_gain_db"]
                   + ["%s_gain_db" % l for l in labels])
        w.writerows(rows)
    print("\nwritten: %s" % os.path.join(out, "scores.csv"))


if __name__ == "__main__":
    main()
