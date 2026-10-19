# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Hany El Imam
"""
Scores denoisers on a test set (used by 4_check_and_score/score.py and
6_compare_models/compare_models.py).

For every noisy picture of the test set, every denoiser cleans it with the C
code of the radio (bin/sstv_denoise_tool), and the result is compared with
the clean picture (PSNR, dB).  The score of a denoiser is how many dB it adds
to the received picture:

    gain = PSNR(cleaned, clean) - PSNR(received, clean)

The networks run here on every picture (no noise gate), to measure the
network itself.  They also clean the CLEAN pictures, to see how much they
change a picture that has no noise: a good denoiser must leave it alone.
"""
import collections
import os
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import sstvml

CLASSIC = "classic"
DETAIL = "ALL detail"       # the mean gain on the detail pixels only (text, lines, edges)


def case_order(case):
    """ALL and ALL detail first, then plain noise, bursts, impulses, fading; each by SNR"""
    if case == "ALL":
        return (0, 0, "")
    if case == DETAIL:
        return (0, 1, "")
    kinds = ["", "burst", "impulses", "fading"]
    kind = case.rsplit("snr", 1)[0].rstrip("_")
    m = re.search(r"snr(-?\d+)", case)
    return (1 + (kinds.index(kind) if kind in kinds else len(kinds)), int(m.group(1)) if m else 0, case)


def clean_with_all(models, labels, set_path, work, jobs, with_classic=True):
    """Cleans every picture of set_path with every denoiser.
    work/<label>/noisy/<file> and work/<label>/clean/<file> are the results."""
    pairs = sstvml.find_pairs(set_path)
    cleans = sorted(set(c for _, c, _ in pairs))
    tool = sstvml.tool("sstv_denoise_tool")
    todo = []
    names = ([CLASSIC] if with_classic else []) + labels
    paths = ([None] if with_classic else []) + list(models)
    for name, model in zip(names, paths):
        for sub, files in (("noisy", [n for n, _, _ in pairs]), ("clean", cleans)):
            if model is None and sub == "clean":
                continue                    # the classic is measured on the noisy pictures only
            os.makedirs(os.path.join(work, name, sub), exist_ok=True)
            for f in files:
                out = os.path.join(work, name, sub, os.path.basename(f))
                if model is None:
                    todo.append([tool, "classic", f, out])
                else:
                    todo.append([tool, "ml", model, f, out, 1])
    with ThreadPoolExecutor(jobs) as ex:
        results = list(ex.map(sstvml.run_tool, todo))
    failed = [t for t, (ok, _) in zip(todo, results) if not ok]
    if failed:
        raise SystemExit("the denoiser failed on %s" % failed[0][-2 if failed[0][1] == "ml" else -1])
    return pairs, cleans, names


def score_set(models, labels, set_path, work, jobs):
    """Returns {"cases": {case: {"received": dB, "classic": gain, label: gain}},
                "count": {case: n}, "clean": {label: (mean dB, worst dB)}}

    Besides one row per noise case and "ALL", the row "ALL detail" is the
    same mean measured only on the detail pixels of each picture
    (sstvml.detail_mask of the clean picture: text, thin lines, edges).  A
    denoiser that softens letters loses more here than in "ALL"."""
    pairs, cleans, names = clean_with_all(models, labels, set_path, work, jobs)
    gains = collections.defaultdict(lambda: collections.defaultdict(list))
    targets, masks = {}, {}
    for noisy, clean, case in pairs:
        if clean not in targets:
            targets[clean] = sstvml.load_rgb(clean)
            masks[clean] = sstvml.detail_mask(targets[clean])
        t, m = targets[clean], masks[clean]
        rx = sstvml.load_rgb(noisy)
        base = sstvml.psnr(rx, t)
        base_d = sstvml.psnr_masked(rx, t, m)
        outs = {n: sstvml.load_rgb(os.path.join(work, n, "noisy", os.path.basename(noisy))) for n in names}
        for key in ("ALL", case):
            gains[key]["received"].append(base)
            for name in names:
                gains[key][name].append(sstvml.psnr(outs[name], t) - base)
        if base_d is not None:              # a picture without any detail is left out of this row
            gains[DETAIL]["received"].append(base_d)
            for name in names:
                gains[DETAIL][name].append(sstvml.psnr_masked(outs[name], t, m) - base_d)
    cases = {k: {n: float(np.mean(v)) for n, v in g.items()} for k, g in gains.items()}
    count = {k: len(g["received"]) for k, g in gains.items()}
    clean_scores = {}
    for name in labels:
        v = [sstvml.psnr(sstvml.load_rgb(os.path.join(work, name, "clean", os.path.basename(c))),
                         targets.get(c) if c in targets else sstvml.load_rgb(c)) for c in cleans]
        clean_scores[name] = (float(np.mean(v)), float(np.min(v)))
    return {"cases": cases, "count": count, "clean": clean_scores, "names": names}


def text_table(result):
    """The scores as a text table"""
    names = result["names"]
    lines = ["%-16s%6s%10s" % ("noise", "n", "received") + "".join("%12s" % n[:11] for n in names)]
    for case in sorted(result["cases"], key=case_order):
        r = result["cases"][case]
        lines.append("%-16s%6d%9.2f " % (case, result["count"][case], r["received"])
                     + "".join("%+11.2f " % r[n] for n in names))
    lines.append("")
    lines.append("clean pictures left alone (PSNR dB of the cleaned clean picture; higher = less changed,")
    lines.append("above about 40 dB the change cannot be seen):")
    for name, (mean, worst) in result["clean"].items():
        lines.append("  %-24s mean %.1f  worst %.1f" % (name, mean, worst))
    return "\n".join(lines)


def csv_rows(set_name, result):
    """The scores as CSV rows: set, case, n, received, then one column per denoiser"""
    rows = []
    for case in sorted(result["cases"], key=case_order):
        r = result["cases"][case]
        rows.append([set_name, case, result["count"][case], "%.3f" % r["received"]]
                    + ["%.3f" % r[n] for n in result["names"]])
    for name, (mean, worst) in result["clean"].items():
        rows.append([set_name, "clean_pictures_" + name, "", "", "mean %.2f worst %.2f" % (mean, worst)])
    return rows
