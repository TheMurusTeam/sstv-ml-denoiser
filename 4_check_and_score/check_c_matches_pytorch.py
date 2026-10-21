#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Hany El Imam
"""
Step 4a - checks that the C code gives the same picture as PyTorch.

    python3 4_check_and_score/check_c_matches_pytorch.py TRAINED/20x6

The folder is the output of 3_train/train.py.  It holds the model
(sstv_dnn_*.bin), a picture (check_in.png) and what PyTorch made of it
(check_torch.png).  Here the same picture is cleaned by the C code
(bin/sstv_denoise_tool, the code of the radio) and the two results are
compared: they may differ by 1 or 2 levels (out of 255) because of rounding,
not more.  It then runs the C code's own check: the fast vector loops must
match the plain C loops.

    OK          the model works in C exactly as it was trained
    DIFFERENT   the model file and the C code do not agree: do not use it

Needs: Python 3 with numpy and Pillow, and the tools in bin/.
"""
import glob
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import sstvml  # noqa: E402

MAX_LEVELS = 2          # allowed difference, levels out of 255


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: check_c_matches_pytorch.py TRAINED/<CxL>   (the output folder of train.py)")
    folder = sys.argv[1]
    models = glob.glob(os.path.join(folder, "sstv_dnn_*.bin"))
    pic_in, pic_torch = os.path.join(folder, "check_in.png"), os.path.join(folder, "check_torch.png")
    if len(models) != 1 or not os.path.exists(pic_in) or not os.path.exists(pic_torch):
        sys.exit("%s must hold one sstv_dnn_*.bin, check_in.png and check_torch.png (made by train.py)" % folder)
    tool = sstvml.tool("sstv_denoise_tool")

    out = os.path.join(tempfile.mkdtemp(prefix="sstv_check_"), "c.png")
    ok, f = sstvml.run_tool([tool, "ml", models[0], pic_in, out, 1])
    if not ok:
        sys.exit("the C code could not run the model %s" % models[0])
    c, t = sstvml.load_rgb(out).astype(int), sstvml.load_rgb(pic_torch).astype(int)
    if c.shape != t.shape:
        sys.exit("DIFFERENT: the pictures have different sizes")
    d = np.abs(c - t)
    print("model %s: %s G multiply-adds per picture" % (os.path.basename(models[0]), f.get("gmacs", "?")))
    print("C code against PyTorch: largest difference %d levels, %.3f%% of the values differ"
          % (d.max(), 100.0 * np.count_nonzero(d) / d.size))
    same = d.max() <= MAX_LEVELS

    ok2, f2 = sstvml.run_tool([tool, "selfcheck", models[0], pic_in])
    print("fast loops against plain C loops: largest difference %s levels" % f2.get("max_diff", "?"))
    same = same and ok2
    print("OK" if same else "DIFFERENT: do not use this model")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
