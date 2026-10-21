#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Hany El Imam
"""
Step 2 - makes the training data from your pictures.

    python3 2_training_data/make_training_data.py --pictures FOLDER [FOLDER ...]

FOLDER: any pictures (PNG, JPG, ...), any size and shape (subfolders are not
read).  Several folders can be given, for example your photos, your QSL
cards and the text pictures of make_text_pictures.py:

    python3 2_training_data/make_training_data.py --pictures ~/photos ~/qsl PICTURES_TEXT

A few hundred varied pictures is a good start; more and more varied is
better.

What it does:

  1. every picture is brought to a full 320x256 (cropped at the centre to the
     5:4 shape, then scaled down or up; nothing squeezed, no black bands):
         DATA/pictures_320/<name>.png

  2. pictures that are also in the test sets (TEST_SET/reference and
     TEST_SET/mine, made by step 1) are left out, even if resized or
     recompressed: a model trained on its test pictures gets scores that are
     too good.  So step 1 must be done first.

  3. every picture goes through the real SSTV encoder and decoder of the radio
     (bin/sstv_channel_sim) in --modes-per-picture modes picked at random
     among Martin M1/M2, Scottie S1/S2, Robot 36/72:
       - once without noise: the "clean" picture.  It has the small defects
         of the SSTV codec but no noise: the network must learn to remove the
         noise, not to "improve" the codec;
             DATA/clean/<name>_<mode>.png
       - then --variants times with noise: SNR picked among 0..15 dB, and
         about 1 in 4 with bursts, 1 in 5 with impulses, 1 in 7 with fading:
             DATA/noisy/<name>_<mode>__v<k>_snr<dB>.png
     A noisy reception whose decoder lost the sync (picture shifted or cut)
     is dropped: a pair that is not aligned teaches the network to blur.

The name before "__" links each noisy picture to its clean one: it is the
layout that 3_train/train.py reads.

The random choices use a fixed seed (--seed): the same pictures give the
same data.  300 pictures take a few minutes and make about 2800 pairs
(about 750 MB).  DATA is made from scratch every time.

Needs: Python 3 with numpy and Pillow, and the tools in bin/ (see README.md).
"""
import argparse
import os
import random
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import sstvml  # noqa: E402

SNRS = [0, 2, 4, 6, 8, 10, 12, 15]      # dB in 2500 Hz


def noise_plan(rng, variants, name):
    """The noisy receptions of one picture in one mode:
    (case name, snr, impulses/min, fading dB, bursts/min, seed)"""
    cases = []
    for v in range(variants):
        snr = rng.choice(SNRS)
        r = rng.random()
        imp = fade = burst = 0
        if r < 0.25:
            burst = rng.choice([0.5, 1, 2])         # bursts per minute
        elif r < 0.45:
            imp = rng.choice([2, 5, 10])            # impulses per minute
        elif r < 0.6:
            fade = rng.choice([3, 6, 10])           # fading depth, dB
        case = "v%d_snr%d" % (v, snr)
        cases.append((case, snr, imp, fade, burst, sstvml.seed_of(name + "__" + case)))
    return cases


def test_fingerprints():
    """Fingerprints of all the test pictures (step 1)"""
    fps = []
    for set_name in ("reference", "mine"):
        for f in sstvml.list_pictures(os.path.join(sstvml.TEST_SET, set_name, "pictures_320")):
            fps.append((sstvml.fingerprint(f), "%s/%s" % (set_name, os.path.basename(f))))
    return fps


def main():
    ap = argparse.ArgumentParser(description="Step 2: makes DATA/clean and DATA/noisy for the training")
    ap.add_argument("--pictures", required=True, nargs="+",
                    help="one or more folders with your training pictures (any size)")
    ap.add_argument("--out", default=sstvml.DATA, help="output folder (default DATA at the top of the package)")
    ap.add_argument("--modes-per-picture", type=int, default=2, help="SSTV modes per picture (default 2)")
    ap.add_argument("--variants", type=int, default=5, help="noisy receptions per picture and mode (default 5)")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4,
                    help="simulations running at the same time (default: all the cores)")
    ap.add_argument("--seed", type=int, default=7, help="seed of the random choices (default 7)")
    ap.add_argument("--no-test-check", action="store_true",
                    help="do not compare with the test sets (not recommended)")
    a = ap.parse_args()
    sstvml.tool("sstv_channel_sim")

    files = []
    for folder in a.pictures:
        found = sstvml.list_pictures(folder)
        if not found:
            sys.exit("no pictures in %s" % folder)
        print("%5d pictures in %s" % (len(found), folder))
        files += found
    test_fps = []
    if not a.no_test_check:
        test_fps = test_fingerprints()
        if not test_fps:
            sys.exit("the test sets are missing: run step 1 first (python3 1_test_set/make_test_set.py),\n"
                     "so that test pictures can be kept out of the training data.")

    if os.path.isdir(a.out):
        shutil.rmtree(a.out)
    for sub in ("pictures_320", "clean", "noisy"):
        os.makedirs(os.path.join(a.out, sub))

    print("1. bringing %d pictures to 320x256" % len(files))
    pictures = sstvml.prepare_pictures(files, os.path.join(a.out, "pictures_320"))

    if test_fps:
        print("2. leaving out the pictures that are also in the test sets")
        kept = []
        for p in pictures:
            twin = sstvml.near_duplicate(sstvml.fingerprint(p), test_fps)
            if twin:
                print("  left out %-36s it is the test picture %s" % (os.path.basename(p), twin))
                os.remove(p)
            else:
                kept.append(p)
        if len(kept) == len(pictures):
            print("  none: no training picture is in the test sets")
        pictures = kept
    if not pictures:
        sys.exit("no pictures left")

    print("3. SSTV receptions (encoder -> channel -> decoder of the radio)")
    rng = random.Random(a.seed)
    jobs = []
    for pic in pictures:
        base = os.path.splitext(os.path.basename(pic))[0]
        for mode in rng.sample(sorted(sstvml.MODES), a.modes_per_picture):
            name = "%s_%s" % (base, mode)
            jobs.append((pic, mode, name, noise_plan(rng, a.variants, name)))
    done = [0]

    def job(j):
        r = sstvml.make_pairs(j[0], j[1], j[2], j[3], a.out)
        done[0] += 1
        if done[0] % 20 == 0 or done[0] == len(jobs):
            print("   %d/%d" % (done[0], len(jobs)), flush=True)
        return r

    with ThreadPoolExecutor(a.threads) as ex:
        res = list(ex.map(job, jobs))
    kept, dropped = sum(r[0] for r in res), sum(r[1] for r in res)
    print("\n%d pictures x %d modes: %d training pairs, %d noisy receptions dropped (decoder lost the sync)"
          % (len(pictures), a.modes_per_picture, kept, dropped))
    print("Training data ready in %s" % a.out)


if __name__ == "__main__":
    main()
