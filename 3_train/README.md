# Step 3 – Training

`train.py` trains the network with PyTorch on `DATA` (step 2), measures it
on `TEST_SET/reference` (step 1) and writes the model file.

## How to run it

| Computer | Command | Notes |
|---|---|---|
| Mac with Apple Silicon (M1…M4) | `3_train/train_mac_linux.sh` | uses the Apple GPU (MPS) |
| Windows with an Nvidia card | double click `3_train\train_windows.bat` | installs Python 3.12, the private environment and PyTorch for CUDA by itself |
| Linux with an Nvidia card | `3_train/train_mac_linux.sh` | uses CUDA |
| any computer, CPU only | `python3 3_train/train.py --device cpu` | very slow, only for tests |

All three start the reference recipe (20 channels × 6 layers, 200 000
steps). With other values:

    3_train/train_mac_linux.sh 16 7 120000          # CHANNELS LAYERS STEPS
    python3 3_train/train.py --channels 16 --layers 7 --steps 120000

On Windows, change `CHANNELS`, `LAYERS` and `STEPS` at the top of
`train_windows.bat`.

**Interrupted?** Run the same command again: `--resume` (always on in the
two launchers) goes on from the last save, made at every validation.

**Check first (one minute).** Before a long training, check that PyTorch,
the GPU and the model export work:

    python3 3_train/train.py --selftest
    python3 4_check_and_score/check_c_matches_pytorch.py TRAINED/selftest

The second command must print `OK`.

### Installing PyTorch by hand

The launchers install it. To do it yourself:

- **Mac:** `python3 -m pip install torch` (PyTorch uses the Apple GPU by itself; see `INSTALL_MAC.md`).
- **Windows + Nvidia:** 64-bit Python 3.12, then
  `python -m pip install torch --index-url https://download.pytorch.org/whl/cu130`
  (RTX 20xx and newer, GTX 16xx; for a GTX 9xx/10xx use `cu126`).
  A 32-bit or too new Python gives "no matching distribution".
  `INSTALL_WIN.md` goes through it step by step.
- **Linux:** `python3 -m pip install torch` (CUDA included).

## What it prints

    network 20 channels x 6 layers: 15583 weights, 1.27 G multiply-adds per 320x256 picture; device mps
    step    100/200000  loss 0.04012  0.3 min, 610 min left
    ...
      validation: 28.72 dB, +9.15 dB over the received pictures; detail +4.52 dB  (best: saved)

- **loss:** how far the network is from the clean pictures in training. It
  goes down quickly, then slowly.
- **validation:** the mean PSNR of the network's output on the test set, and
  the gain over the received pictures: the number that matters. Then the
  gain on the detail of the pictures only (text, lines, edges). `(best:
  saved)` means it is the best so far and the model file has been written.

## What it writes (`TRAINED/<C>x<L>/`)

| File | What |
|---|---|
| `sstv_dnn_<C>x<L>.bin` | **the model**, the best one on the test set; written at every new best |
| `log.csv` | step, loss, validation PSNR of the whole pictures and of their detail: open it in a spreadsheet to see when it stops improving |
| `best.pt`, `last.pt` | PyTorch saves: the best model, and the full state for `--resume` |
| `check_in.png`, `check_torch.png` | a picture and the PyTorch result, for `4_check_and_score/check_c_matches_pytorch.py` |

## The network: channels and layers

The network is a small **DnCNN**: a stack of 3×3 convolutions.

    picture (R,G,B) → [3×3 conv, 3 → C] → ReLU
                    → [3×3 conv, C → C] → ReLU      (L − 2 times)
                    → [3×3 conv, C → 3]             = the noise it sees
    output = picture − noise

- **Layers (L)**: how many convolutions are stacked. Each one looks at a
  3×3 neighbourhood of the one before, so the network sees an area of
  **(2L + 1) × (2L + 1) pixels** around each pixel: 13×13 with 6 layers,
  15×15 with 7. More layers → it sees more context (a long stripe of noise,
  the edge of a letter) and costs more.
- **Channels (C)**: how many values each layer computes for every pixel: how
  many different "features" the network can follow at the same time
  (edges, dashes along the lines, colour noise…). More channels → more
  capacity, and the cost grows with the square of C.
- **Residual**: the network estimates the noise, not the picture. A network
  that has learned nothing outputs zero noise and leaves the picture as it
  is; the last layer starts at zero, so the training starts from "change
  nothing" and only learns to remove.

### What a size costs

For a 320×256 picture:

    weights        = 3·C·9 + (L−2)·C·C·9 + C·3·9   (+ biases)
    multiply-adds  = weights × 320 × 256

| Network | Weights | Multiply-adds | Gain on the reference test set |
|---|---|---|---|
| 16 × 7 | 12 483 | 1.01 G | see `models/README.md` |
| **20 × 6 (reference)** | 15 583 | 1.27 G | see `models/README.md` |
| 20 × 7 | 19 203 | 1.56 G | |
| 24 × 7 | 27 363 | 2.23 G | |
| 32 × 8 | 57 251 | 4.67 G | |

The multiply-adds tell the time to clean one picture. On the Xiegu X6100 (4
Cortex-A7 cores) about 1 G multiply-adds take one second, which is why the
radio uses the 20 × 6; a computer is many times faster.

Use **multiples of 4 for the channels** (16, 20, 24, 32): the C code computes
4 channels at a time, so 18 channels cost almost as much as 20.

### Which size to choose

Bigger networks are a little better, with diminishing returns: on the
reference test set going from 20 × 6 to 32 × 8 costs almost 4 times the
work for well under 1 dB. Very small networks (C < 12 or L < 5) lose a lot.
More, and more varied, **pictures** usually help more than a bigger network.

## How it learns

| Setting | Default | What it does |
|---|---|---|
| `--steps` | 200000 | training steps. The gain grows fast at first, then slowly: after about 120 000 steps it adds only a few hundredths of a dB |
| `--batch` | 32 | patches per step |
| `--patch` | 96 | patches are random 96×96 pieces of the training pictures |
| `--lr` | 0.001 | learning rate (Adam), going down to 1/100 along a cosine curve |
| `--identity` | 0.15 | share of patches that are clean pictures as input AND target: teaches the network to leave clean pictures alone |
| `--val-every` | 2000 | steps between two measures on the test set |
| `--device` | auto | `mps` (Mac), `cuda` (Nvidia), `cpu` |
| `--train`, `--val`, `--out` | `DATA`, `TEST_SET/reference`, `TRAINED/<C>x<L>` | folders |

Other choices, built in:

- **L1 loss** (mean absolute difference from the clean picture): with SSTV it
  gives sharper pictures than the squared difference.
- **Only up/down mirroring.** The SSTV noise runs along the lines, left to
  right. Mirroring left/right or turning the patches would show the network
  a noise that never happens.
- **The test set is only measured, never trained on.** It also picks which
  save is kept. Never point `--train` at a test set: the scores would be
  meaningless.

## Text and fine detail

Three options, all off by default (the reference recipe), to make the
network better at keeping text and thin lines. None of them changes the
network: the model has the same size and runs at the same speed.

| Option | Try | What it does |
|---|---|---|
| `--detail-share` | 0.5 | that share of the patches is taken where the clean picture has the most detail (the best of 8 random places) instead of anywhere. Most of a picture is smooth (sky, skin, backgrounds), so random patches show the network few letters |
| `--edge-weight` | 0.5 | adds to the loss the difference between the **edges** of the output and of the clean picture (the differences between neighbour pixels, along and across the lines). Softening a letter costs more than with the plain L1 loss |
| `--select detail` | | keeps the save that is best on the **detail** of the test pictures (`ALL detail`, see step 4), instead of on the whole pictures |

"Detail" is the same everywhere in the package: the pixels next to a strong
brightness step in the clean picture (letters, lines, contours), see
`detail_mask` in `common/sstvml.py`.

A good experiment, with text pictures in the data (step 2,
`make_text_pictures.py`):

    python3 3_train/train.py --detail-share 0.5 --edge-weight 0.5 --out TRAINED/20x6_text

then compare it with the reference in step 6. Look at both rows, `ALL`
and `ALL detail`: the aim is a clearly better `ALL detail` without losing
much on `ALL`, and pictures where the text is sharper but **not changed**.
A network pushed too hard on edges can start to "invent" strokes; a
callsign that looks sharp but wrong is worse than one that is a little
soft.

## What to look at in `log.csv`

The `val_psnr` column (whole pictures) and `val_detail_psnr` (detail only)
should rise quickly and then flatten. If it is still
rising at the end, more steps may help; if it went down at the end, the
model file still holds the best one. A model trained on few pictures reaches
its best early and then gets worse ("over-fitting"): add pictures.
