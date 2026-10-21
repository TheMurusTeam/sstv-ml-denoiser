#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Hany El Imam
"""
Step 1 - makes the test sets: the pictures a model is measured on.
A test set is NEVER used for training.

    python3 1_test_set/make_test_set.py

Two test sets, kept apart so their scores are never mixed:

    1_test_set/reference_pictures/  ->  TEST_SET/reference/
        the pictures that come with this package: the same for everybody,
        so scores can be compared with the ones published in models/README.md
        and with anybody else's.

    1_test_set/my_pictures/         ->  TEST_SET/mine/
        your own pictures (any size, any shape, PNG/JPG/...), the kind of
        pictures you want the denoiser to be good at.  Optional.

For every picture:

  1. it is brought to a full 320x256 (cropped at the centre to the 5:4 shape,
     then scaled down or up; nothing is squeezed, no black bands):
         TEST_SET/<set>/pictures_320/<name>.png
     The SSTV encoder sends exactly 320x256; a picture of another shape would
     be stretched by the encoder, so it is prepared first, as for training.

  2. it goes through the real SSTV encoder and decoder of the radio
     (bin/sstv_channel_sim) in 3 modes - Martin M1, Scottie S1, Robot 36:
       - once without noise: the "clean" picture, the target
             TEST_SET/<set>/clean/<name>_<mode>.png
       - then with the 8 fixed noise cases of NOISE_CASES below
             TEST_SET/<set>/noisy/<name>_<mode>__<case>.png

The noise is fixed: the seed of each reception comes from its file name, so
the same picture gives the same noisy receptions on every computer, and
adding a picture does not change the others.  A noisy reception where the
decoder lost the sync (picture shifted or cut) is dropped and reported.

Run it again after adding pictures: what is already there is made again,
pictures removed from the source folder are removed from the test set.

Needs: Python 3 with numpy and Pillow, and the tools in bin/ (see README.md).
"""
import argparse
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import sstvml  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = {
    "reference": os.path.join(HERE, "reference_pictures"),
    "mine": os.path.join(HERE, "my_pictures"),
}
TEST_MODES = ["m1", "s1", "r36"]        # Martin M1, Scottie S1, Robot 36

# (case name, SNR dB, impulses per minute, fading depth dB, bursts per minute)
# SNR is measured in 2500 Hz, as for SSB.  0 dB is a picture barely readable,
# 10 dB a good one.  Bursts are 0.3..1.2 s of strong interference (the
# coloured bands), impulses are clicks of 2..32 ms, fading is a slow
# (0.15..0.45 Hz) drop of the signal.
NOISE_CASES = [
    ("snr0", 0, 0, 0, 0),
    ("snr3", 3, 0, 0, 0),
    ("snr6", 6, 0, 0, 0),
    ("snr10", 10, 0, 0, 0),
    ("burst_snr6", 6, 0, 0, 3),
    ("burst_snr12", 12, 0, 0, 3),
    ("impulses_snr10", 10, 10, 0, 0),
    ("fading_snr10", 10, 0, 10, 0),
]


def build(set_name, src_folder, threads):
    out = os.path.join(sstvml.TEST_SET, set_name)
    files = sstvml.list_pictures(src_folder)
    if not files:
        if os.path.isdir(out):
            shutil.rmtree(out)              # the source folder is now empty
        return None
    print("\n== %s: %d pictures from %s" % (set_name, len(files), os.path.relpath(src_folder, sstvml.ROOT)))
    if os.path.isdir(out):
        shutil.rmtree(out)                  # always made from scratch: no leftovers
    for sub in ("pictures_320", "clean", "noisy"):
        os.makedirs(os.path.join(out, sub))

    print("1. bringing the pictures to 320x256")
    pictures = sstvml.prepare_pictures(files, os.path.join(out, "pictures_320"))

    print("2. SSTV receptions (encoder -> channel -> decoder of the radio)")
    jobs = []
    for pic in pictures:
        base = os.path.splitext(os.path.basename(pic))[0]
        for mode in TEST_MODES:
            name = "%s_%s" % (base, mode)
            cases = [(c, snr, imp, fade, burst, sstvml.seed_of(name + "__" + c))
                     for c, snr, imp, fade, burst in NOISE_CASES]
            jobs.append((pic, mode, name, cases))
    with ThreadPoolExecutor(threads) as ex:
        res = list(ex.map(lambda j: sstvml.make_pairs(j[0], j[1], j[2], j[3], out), jobs))
    kept, dropped = sum(r[0] for r in res), sum(r[1] for r in res)
    print("   %d pictures x %d modes: %d noisy receptions kept, %d dropped (decoder lost the sync)"
          % (len(pictures), len(TEST_MODES), kept, dropped))
    return kept


def main():
    ap = argparse.ArgumentParser(description="Step 1: makes TEST_SET/reference and TEST_SET/mine")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4,
                    help="simulations running at the same time (default: all the cores)")
    a = ap.parse_args()
    sstvml.tool("sstv_channel_sim")          # stops here with a hint if the tools are missing

    made = {}
    for name, src in SOURCES.items():
        n = build(name, src, a.threads)
        if n:
            made[name] = n
    if "reference" not in made:
        sys.exit("no pictures in %s" % SOURCES["reference"])
    print("\nTest sets ready in %s:" % sstvml.TEST_SET)
    for name, n in made.items():
        print("  %-10s %d noisy receptions" % (name, n))
    if "mine" not in made:
        print("  (put your own pictures in 1_test_set/my_pictures/ for a second test set)")


if __name__ == "__main__":
    main()
