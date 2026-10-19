#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Hany El Imam
"""
Step 3 - trains the SSTV denoiser network with PyTorch.  Runs on the Apple
GPU of a Mac (MPS), on an Nvidia card (CUDA) or, slowly, on the CPU.

    python3 3_train/train.py                          20x6, the reference recipe
    python3 3_train/train.py --channels 16 --layers 7 --steps 120000

The network (see 3_train/README.md for how to choose its size)
    A small DnCNN-like network: --layers layers of 3x3 convolutions, with
    --channels channels between them and ReLU after every layer but the last;
    no batch normalisation.  It estimates the NOISE of the picture and the
    output is  input - noise  ("residual"): a network that knows nothing
    leaves the picture as it is.  The same network runs in C in
    src/sstv/sstv_dnn.c (and in the 1KO125 firmware): the best weights are
    written in its model file format, SSTVDNN1 (common/model_file.py).

The data
    --train  DATA (step 2):            clean/<name>.png, noisy/<name>__<case>.png
    --val    TEST_SET/reference (step 1), same layout, NEVER used for
             training: it tells how good the network is on pictures it has
             never seen, and which save is the best.

How it learns
    - random 96x96 pieces ("patches") of the pictures, 32 per step;
    - L1 loss: the mean absolute difference from the clean picture (with
      SSTV it gives sharper pictures than the squared difference);
    - Adam, learning rate 1e-3 going down to 1e-5 along a cosine curve;
    - patches are mirrored only up/down: the SSTV noise runs along the
      lines, left to right, so a left/right mirror or a rotation would show
      the network a noise that never happens;
    - --identity (0.15): that share of the patches is a clean picture as
      input AND as target, so the network learns to leave clean pictures
      alone;
    - every --val-every steps the network is measured on --val, on the whole
      pictures and on their detail only (text, lines, edges: see
      sstvml.detail_mask); when it is the best so far (--select), it is saved
      and exported at once, so an interrupted training always leaves a
      usable model.

For text and fine detail (all off by default: the reference recipe)
    --detail-share S  that share of the patches is taken where the clean
                      picture has the most detail (the best of 8 random
                      places), instead of anywhere: the network spends more
                      of its training on letters and lines.  0.5 is a good try.
    --edge-weight E   adds to the loss E x the difference between the edges
                      of the output and of the clean picture (the differences
                      between neighbour pixels, across and along the lines):
                      softening a letter costs more.  0.5 is a good try.
    --select detail   keeps the save that is best on the detail of the
                      test pictures, instead of on the whole pictures.
    None of them changes the network: the model has the same size and runs
    at the same speed.

Output in --out (default TRAINED/<channels>x<layers>)
    sstv_dnn_<C>x<L>.bin        the model (the best one on --val)
    log.csv                     step, loss, validation PSNR (whole pictures, detail)
    best.pt, last.pt            PyTorch saves: the best model, and the state
                                to go on from (--resume)
    check_in.png, check_torch.png   a picture and the PyTorch result, for
                                4_check_and_score/check_c_matches_pytorch.py

Other options
    --resume     goes on from last.pt: an interrupted training is not lost
    --selftest   a few hundred steps on synthetic pictures, in a minute:
                 checks that PyTorch, the GPU and the export work

Needs: Python 3, numpy, Pillow and PyTorch (see 3_train/README.md).
"""
import argparse
import copy
import math
import os
import random
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import model_file  # noqa: E402
import sstvml  # noqa: E402

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    sys.exit("PyTorch is missing: see 3_train/README.md for the install")


# ---- data ---------------------------------------------------------------------------

DETAIL_TRIES = 8        # random places tried for a detail patch: the one with most detail wins


def detail_table(clean):
    """Summed-area table of the detail mask: the detail pixels of any
    rectangle are counted with 4 lookups"""
    m = sstvml.detail_mask(clean).astype(np.int32)
    t = np.zeros((m.shape[0] + 1, m.shape[1] + 1), np.int32)
    t[1:, 1:] = m.cumsum(0).cumsum(1)
    return t


class PairSet:
    """All the pairs in memory (uint8); random patches on request.

    items:  (noisy, clean, detail table or None)
    cleans: (clean, detail table or None), for the identity patches
    masks:  the detail mask of each item (validation sets only)"""

    def __init__(self, pairs, identity=0.15, detail_share=0.0, with_masks=False):
        self.items, self.masks = [], []
        cache = {}
        for noisy, clean, _case in pairs:
            if clean not in cache:
                y = sstvml.load_rgb(clean)
                cache[clean] = (y, detail_table(y) if detail_share > 0 else None)
            x, (y, table) = sstvml.load_rgb(noisy), cache[clean]
            h, w = min(x.shape[0], y.shape[0]), min(x.shape[1], y.shape[1])
            self.items.append((x[:h, :w], y[:h, :w], table))
            if with_masks:
                self.masks.append(sstvml.detail_mask(y[:h, :w]))
        self.cleans = list(cache.values())
        self.identity = identity
        self.detail_share = detail_share

    def place(self, h, w, patch, table, rng):
        """Top-left corner of a patch: anywhere, or (detail_share of the
        times) the place with the most detail among DETAIL_TRIES random ones.
        With detail_share 0 the random numbers drawn are exactly those of
        the reference recipe."""
        if self.detail_share > 0 and table is not None and rng.random() < self.detail_share:
            best, best_n = None, -1
            for _ in range(DETAIL_TRIES):
                t, l = rng.randrange(h - patch + 1), rng.randrange(w - patch + 1)
                n = table[t + patch, l + patch] - table[t, l + patch] - table[t + patch, l] + table[t, l]
                if n > best_n:
                    best, best_n = (t, l), n
            return best
        return rng.randrange(h - patch + 1), rng.randrange(w - patch + 1)

    def batch(self, size, patch, rng):
        xs = np.empty((size, patch, patch, 3), np.uint8)
        ys = np.empty_like(xs)
        for i in range(size):
            if self.cleans and rng.random() < self.identity:
                y, table = self.cleans[rng.randrange(len(self.cleans))]
                x = y                       # a clean picture must come out unchanged
            else:
                x, y, table = self.items[rng.randrange(len(self.items))]
            h, w = x.shape[:2]
            if h < patch or w < patch:
                raise ValueError("pictures smaller than the patch size")
            top, left = self.place(h, w, patch, table, rng)
            px, py = x[top:top + patch, left:left + patch], y[top:top + patch, left:left + patch]
            if rng.random() < 0.5:          # up/down mirror only (see the top of the file)
                px, py = px[::-1], py[::-1]
            xs[i], ys[i] = px, py
        return xs, ys


def to_tensor(a, dev):
    """uint8 [n][h][w][3] -> float [n][3][h][w] in 0..1"""
    return torch.from_numpy(np.array(a, copy=True)).to(dev).permute(0, 3, 1, 2).float() / 255.0


def to_picture(t):
    """float [1][3][h][w] -> uint8 [h][w][3], rounded as the C code does"""
    o = t.clamp(0, 1)[0].permute(1, 2, 0).detach().cpu().numpy()
    return (o * 255.0 + 0.5).astype(np.uint8)


# ---- network --------------------------------------------------------------------------

class SSTVNet(nn.Module):
    """layers convolutions 3x3: 3 -> channels -> ... -> channels -> 3"""

    def __init__(self, channels, layers):
        super().__init__()
        convs = [nn.Conv2d(3, channels, 3, padding=1)]
        convs += [nn.Conv2d(channels, channels, 3, padding=1) for _ in range(layers - 2)]
        convs += [nn.Conv2d(channels, 3, 3, padding=1)]
        self.convs = nn.ModuleList(convs)
        for c in self.convs[:-1]:
            nn.init.kaiming_normal_(c.weight, nonlinearity="relu")
            nn.init.zeros_(c.bias)
        nn.init.zeros_(self.convs[-1].weight)       # starts as "change nothing"
        nn.init.zeros_(self.convs[-1].bias)

    def noise(self, x):
        h = x
        for c in self.convs[:-1]:
            h = F.relu(c(h))
        return self.convs[-1](h)

    def forward(self, x):
        return x - self.noise(x)


def edge_loss(out, target):
    """Mean absolute difference between the edges of out and of target: the
    differences between neighbour pixels, along the lines (x) and across
    them (y).  It grows when the output softens a letter or a line."""
    dx = F.l1_loss(out[..., :, 1:] - out[..., :, :-1], target[..., :, 1:] - target[..., :, :-1])
    dy = F.l1_loss(out[..., 1:, :] - out[..., :-1, :], target[..., 1:, :] - target[..., :-1, :])
    return dx + dy


def export(model, path):
    """Writes the model file read by src/sstv/sstv_dnn.c"""
    layers = []
    n = len(model.convs)
    for i, c in enumerate(model.convs):
        layers.append((c.weight.detach().cpu().numpy().astype(np.float32),
                       c.bias.detach().cpu().numpy().astype(np.float32), i < n - 1))
    model_file.write_model(path, layers, residual=True)


def save_best(model, out, name, check_picture):
    """The best model so far: PyTorch save, model file, and the check pictures"""
    torch.save(model.state_dict(), os.path.join(out, "best.pt"))
    cpu = copy.deepcopy(model).cpu().eval()
    export(cpu, os.path.join(out, name))
    with torch.no_grad():
        o = to_picture(cpu(to_tensor(check_picture[None], "cpu")))
    Image.fromarray(check_picture).save(os.path.join(out, "check_in.png"))
    Image.fromarray(o).save(os.path.join(out, "check_torch.png"))


# ---- validation -----------------------------------------------------------------------

@torch.no_grad()
def validate(model, val, dev):
    """Mean PSNR, in dB, of the received pictures and of the network output:
    {"all": (received, network), "detail": (received, network)}, "detail"
    measured only on the detail pixels (val.masks)"""
    model.eval()
    sums = {"all": [0.0, 0.0, 0], "detail": [0.0, 0.0, 0]}
    for k, (x, y, _table) in enumerate(val.items):
        o = to_picture(model(to_tensor(x[None], dev)))
        s = sums["all"]
        s[0] += sstvml.psnr(x, y)
        s[1] += sstvml.psnr(o, y)
        s[2] += 1
        m = val.masks[k] if k < len(val.masks) else None
        if m is not None and m.any():
            s = sums["detail"]
            s[0] += sstvml.psnr_masked(x, y, m)
            s[1] += sstvml.psnr_masked(o, y, m)
            s[2] += 1
    model.train()
    return {k: (v[0] / max(1, v[2]), v[1] / max(1, v[2])) for k, v in sums.items()}


# ---- helpers ---------------------------------------------------------------------------------

def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def keep_awake():
    """Windows: no sleep while this program runs (macOS: train_mac_linux.sh uses caffeinate)"""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # CONTINUOUS | SYSTEM
        except Exception:
            pass


def selftest_sets():
    """Smooth pictures (gradients and soft shapes) with noise on top: a network
    learns to clean these in a few hundred steps.  Pure random pictures could
    not be cleaned at all (the picture itself would look like noise)."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:64, 0:80].astype(np.float64)
    pairs = []
    for _ in range(16):
        y = np.empty((64, 80, 3))
        for c in range(3):
            a, b, f, ph = rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0.03, 0.12), rng.uniform(0, 6.3)
            y[..., c] = 128 + 50 * a * xx / 80 + 50 * b * yy / 64 + 50 * np.sin(f * (xx + yy) + ph)
        y = np.clip(y, 0, 255)
        noise = rng.normal(0, 18, y.shape) + rng.normal(0, 8, (64, 1, 1))   # pixels + whole lines
        x = np.clip(y + noise, 0, 255)
        pairs.append((x.astype(np.uint8), y.astype(np.uint8)))
    train, val = PairSet([]), PairSet([], identity=0)
    train.items = [(x, y, None) for x, y in pairs[:12]]
    train.cleans = [(y, None) for _x, y in pairs[:12]]
    val.items = [(x, y, None) for x, y in pairs[12:]]
    val.masks = [sstvml.detail_mask(y) for _x, y in pairs[12:]]
    return train, val


# ---- main -------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Step 3: training of the SSTV denoiser")
    ap.add_argument("--train", default=sstvml.DATA, help="training data (default DATA)")
    ap.add_argument("--val", default=os.path.join(sstvml.TEST_SET, "reference"),
                    help="validation set, never trained on (default TEST_SET/reference)")
    ap.add_argument("--channels", type=int, default=20, help="channels between the layers (default 20)")
    ap.add_argument("--layers", type=int, default=6, help="convolution layers (default 6)")
    ap.add_argument("--steps", type=int, default=200000, help="training steps (default 200000)")
    ap.add_argument("--batch", type=int, default=32, help="patches per step (default 32)")
    ap.add_argument("--patch", type=int, default=96, help="patch size in pixels (default 96)")
    ap.add_argument("--lr", type=float, default=1e-3, help="starting learning rate (default 0.001)")
    ap.add_argument("--identity", type=float, default=0.15,
                    help="share of patches that are clean pictures, to be left unchanged (default 0.15)")
    ap.add_argument("--val-every", type=int, default=2000, help="steps between validations (default 2000)")
    ap.add_argument("--detail-share", type=float, default=0.0,
                    help="share of patches taken where the picture has most detail (default 0; try 0.5)")
    ap.add_argument("--edge-weight", type=float, default=0.0,
                    help="weight of the edge difference in the loss (default 0; try 0.5)")
    ap.add_argument("--select", choices=["all", "detail"], default="all",
                    help="keep the save best on the whole pictures (all, default) or on their detail")
    ap.add_argument("--out", help="output folder (default TRAINED/<channels>x<layers>)")
    ap.add_argument("--device", default="auto", help="auto, mps, cuda or cpu")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resume", action="store_true", help="go on from the last save in --out")
    ap.add_argument("--selftest", action="store_true", help="one-minute test of PyTorch and the export")
    a = ap.parse_args()

    if a.out is None:
        a.out = os.path.join(sstvml.TRAINED, ("selftest" if a.selftest else "%dx%d" % (a.channels, a.layers)))
    dev = pick_device(a.device)
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    keep_awake()

    if a.selftest:
        train, val = selftest_sets()
        a.steps, a.val_every, a.patch, a.batch = 400, 100, 48, 16
    else:
        tp, vp = sstvml.find_pairs(a.train), sstvml.find_pairs(a.val)
        if not tp:
            sys.exit("no training pairs in %s: run step 2 first" % a.train)
        if not vp:
            sys.exit("no validation pairs in %s: run step 1 first" % a.val)
        print("loading %d training pairs, %d validation pairs..." % (len(tp), len(vp)))
        train = PairSet(tp, identity=a.identity, detail_share=a.detail_share)
        val = PairSet(vp, identity=0, with_masks=True)

    model = SSTVNet(a.channels, a.layers).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    macs = sum(c.weight.numel() for c in model.convs) * sstvml.WIDTH * sstvml.HEIGHT
    print("network %d channels x %d layers: %d weights, %.2f G multiply-adds per 320x256 picture; device %s"
          % (a.channels, a.layers, nparam, macs / 1e9, dev))
    print("detail patches %.2f, edge weight %.2f, best save chosen on: %s"
          % (a.detail_share, a.edge_weight, "whole pictures" if a.select == "all" else "detail"))

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps, eta_min=a.lr * 0.01)
    name = "sstv_dnn_%dx%d.bin" % (a.channels, a.layers)
    last_path = os.path.join(a.out, "last.pt")
    check_picture = val.items[0][0]

    first = 1
    if a.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=dev, weights_only=False)   # our own file
        if ck["net"] != [a.channels, a.layers]:
            sys.exit("the save in %s is a %dx%d network" % (a.out, *ck["net"]))
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        if ck.get("select", "all") != a.select:
            sys.exit("the save in %s chooses the best on '%s': resume with --select %s"
                     % (a.out, ck.get("select", "all"), ck.get("select", "all")))
        first, received, best = ck["step"] + 1, ck["received"] if "received" in ck else \
            {"all": ck["b0"], "detail": ck["b0"]}, ck["best"]
        rng.setstate(ck["rng"])
        print("going on from step %d (best so far %.2f dB)" % (first - 1, best))
        log = open(os.path.join(a.out, "log.csv"), "a")
    else:
        if a.resume:
            print("nothing to resume in %s, starting from the beginning" % a.out)
        v = validate(model, val, dev)
        received = {k: v[k][0] for k in v}
        best = v[a.select][1]
        print("validation: received pictures %.2f dB (detail %.2f dB)" % (received["all"], received["detail"]))
        save_best(model, a.out, name, check_picture)
        log = open(os.path.join(a.out, "log.csv"), "w")
        log.write("step,loss,val_psnr,val_detail_psnr\n")

    t0 = time.time()
    run = 0.0
    for step in range(first, a.steps + 1):
        xs, ys = train.batch(a.batch, a.patch, rng)
        x, y = to_tensor(xs, dev), to_tensor(ys, dev)
        out = model(x)
        loss = F.l1_loss(out, y)
        if a.edge_weight > 0:
            loss = loss + a.edge_weight * edge_loss(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        run = loss.item() if step == first else 0.98 * run + 0.02 * loss.item()
        if step % 100 == 0:
            el = time.time() - t0
            done = step - first + 1
            print("step %6d/%d  loss %.5f  %.1f min, %.0f min left" %
                  (step, a.steps, run, el / 60, el / done * (a.steps - step) / 60), flush=True)
        if step % a.val_every == 0 or step == a.steps:
            v = validate(model, val, dev)
            log.write("%d,%.6f,%.3f,%.3f\n" % (step, run, v["all"][1], v["detail"][1]))
            log.flush()
            mark = ""
            if v[a.select][1] > best:
                best = v[a.select][1]
                save_best(model, a.out, name, check_picture)
                mark = "  (best: saved)"
            print("  validation: %.2f dB, %+.2f dB over the received pictures; detail %+.2f dB%s"
                  % (v["all"][1], v["all"][1] - received["all"], v["detail"][1] - received["detail"], mark),
                  flush=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step, "b0": received["all"], "received": received, "best": best,
                        "select": a.select, "rng": rng.getstate(),
                        "net": [a.channels, a.layers]}, last_path + ".tmp")
            os.replace(last_path + ".tmp", last_path)

    r = received[a.select]
    print("\nbest validation (%s) %.2f dB (received %.2f dB, %+.2f dB)" % (a.select, best, r, best - r))
    print("model: %s" % os.path.join(a.out, name))
    print("next:  python3 4_check_and_score/check_c_matches_pytorch.py %s" % a.out)


if __name__ == "__main__":
    main()
