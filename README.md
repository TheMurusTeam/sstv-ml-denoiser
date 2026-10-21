# SSTV ML denoiser – train and test it on your computer

This package rebuilds on a computer the **SSTV picture denoiser** of the
**1KO125 firmware for the Xiegu X6100**, and lets you **train your own
model** from your own pictures, test it, and compare it with others.

What makes it different from a generic denoiser: every picture goes through
the **real SSTV encoder and decoder of the radio** (the same C files as in
the firmware), so the network learns the noise SSTV really produces – the
dashes along the lines, the coloured bands of a burst of interference, the
colour speckle of Martin and Scottie – and the C code that cleans the
pictures here is the same code that runs in the radio.

    your pictures ─▶ 320×256 ─▶ SSTV encoder ─▶ noise, bursts, fading ─▶ SSTV decoder ─▶ received picture
                                     (radio)                                 (radio)
                                                                     the network learns: received → clean

## The steps

| Folder | Step | What you get |
|---|---|---|
| `1_test_set/` | **1. Test sets**: the reference pictures (given) and yours, brought to 320×256 and received through SSTV | `TEST_SET/` |
| `2_training_data/` | **2. Training data**: your pictures, brought to 320×256, received with and without noise; optionally pictures full of text | `DATA/clean`, `DATA/noisy` |
| `3_train/` | **3. Training** with PyTorch (Mac GPU, Nvidia card) – channels, layers, steps explained | `TRAINED/<C>x<L>/sstv_dnn_<C>x<L>.bin` |
| `4_check_and_score/` | **4. Check** that the C code matches PyTorch, and **score** models on the test sets | tables, `RESULTS/scores/` |
| `5_real_pictures/` | **5. Real pictures**: run model on SSTV pictures you received, as the radio uses it | `RESULTS/real_pictures/…/index.html` |
| `6_compare_models/` | **6. Compare** several models: scores and pictures side by side | `RESULTS/compare_models/index.html` |
| `models/` | the reference model, `sstv_dnn_20x6.bin` (the one of the firmware), with its scores | |
| `extras/` | tools for special pictures, e.g. cropping LOG11DX eQSL cards to their picture | |
| `src/` | the C code: the radio's SSTV files and the two tools | `bin/` |
| `common/` | Python code shared by the steps | |

Each folder has its own README with all the details. Folders in **CAPITALS**
(`DATA`, `TEST_SET`, `TRAINED`, `RESULTS`, `PICTURES_TEXT`) are made by the scripts at the
top of the package: they can be deleted and made again at any time.

## What you need

- **Python 3.9 or newer (3.12 recommended)** with **numpy** and
  **Pillow** for every step; **PyTorch** for step 3 only
  (`3_train/README.md`). On an older Python, pip installs older versions of
  the packages, which do the same job here. **`INSTALL_MAC.md`** and
  **`INSTALL_WIN.md`** go through the whole installation step by step.
- The two **C tools** in `bin/`: `sstv_channel_sim` (the SSTV channel) and
  `sstv_denoise_tool` (the denoisers of the radio). Either:
  - **download** them: on the GitHub *Releases* page, the zip for your
    system (`tools-windows-x64.zip`, `tools-macos-universal.zip`,
    `tools-linux-x64.zip`); unzip it at the top of the package, it contains
    the `bin/` folder; or
  - **build** them: `make` at the top of the package. It needs only a C
    compiler: on a Mac `xcode-select --install`, on Linux `gcc` and `make`,
    on Windows MSYS2 (`pacman -S make mingw-w64-ucrt-x86_64-gcc`, then
    `make` in the UCRT64 shell).

**macOS:** tools downloaded from the internet are blocked by Gatekeeper the
first time. After unzipping, run once: `xattr -dr com.apple.quarantine bin`

**Windows:** install Python 3.12 (`winget install Python.Python.3.12`, or
from python.org) and, in the private environment below, use `python` where
this README says `python3` (see `INSTALL_WIN.md`).

### Python packages

    python3 -m pip install -r requirements.txt

If your system refuses ("externally managed environment", common on recent
macOS and Linux), use a private environment in the package folder:

    python3 -m venv .venv
    source .venv/bin/activate          # Windows: .venv\Scripts\activate
    python3 -m pip install -r requirements.txt

## Quick start

From the top of the package (macOS / Linux; on Windows `python` instead of `python3`):

    # 1. the test sets (add your own pictures to 1_test_set/my_pictures/ first, if you want)
    python3 1_test_set/make_test_set.py

    # 2. the training data, from a folder of your pictures (a few hundred, any size)
    python3 2_training_data/make_training_data.py --pictures ~/Pictures/for_training

    # 3. the training: a one-minute check, then the real thing (15-20 minutes on nvidia RTX 5070, 2 hours on Apple M4 Pro)
    python3 3_train/train.py --selftest
    python3 4_check_and_score/check_c_matches_pytorch.py TRAINED/selftest
    3_train/train_mac_linux.sh                # Windows + Nvidia: double click 3_train\train_windows.bat

    # 4. check and score against the reference model
    python3 4_check_and_score/check_c_matches_pytorch.py TRAINED/20x6
    python3 4_check_and_score/score.py TRAINED/20x6/sstv_dnn_20x6.bin models/sstv_dnn_20x6.bin

    # 5. your real received SSTV pictures, as the radio would clean them
    python3 5_real_pictures/denoise_real_pictures.py --pictures ~/received_sstv --model TRAINED/20x6/sstv_dnn_20x6.bin

    # 6. models side by side
    python3 6_compare_models/compare_models.py models/sstv_dnn_20x6.bin TRAINED/20x6/sstv_dnn_20x6.bin --pictures ~/received_sstv

Steps 4, 5 and 6 work with the reference model alone, without training:
try them first to see what the denoiser does.

## Why 320×256, and how pictures are prepared

SSTV sends a fixed number of pixels: 320×256 in Martin and Scottie (the
modes this package trains on, with Robot 36 and 72 at 320×240). The
encoder would stretch any other picture, so every picture – the reference
ones too, which are given in their original sizes on purpose – is first
made a **full 320×256**: cropped at the centre to the 5:4 shape (the long
side is cut, nothing is squeezed, no black bands), then scaled down or up.
The prepared pictures are kept (`TEST_SET/<set>/pictures_320`,
`DATA/pictures_320`), so you can see exactly what is sent.

## The rules that keep the scores honest

- **Never train on a test picture.** Step 2 checks it and leaves out of the
  training data any picture that is also in a test set, even resized or
  recompressed.
- **Compare models on the reference test set.** It is the same for
  everybody; your own test set (`TEST_SET/mine`) is scored apart.
- **The target is the picture received without noise**, not the original:
  the network learns to remove the noise, not to "improve" SSTV.
- **Pairs that are not aligned are dropped**: when the decoder loses the
  sync, the picture is shifted and cannot be compared pixel by pixel.

## Credits and license

Written by **Hany El Imam** for the 1KO125 firmware of the Xiegu X6100.
GNU LGPL version 2.1 or later, the license of the 1KO125 firmware: see
`NOTICE.md` and `LICENSE`. The reference pictures
have their own credits, in `1_test_set/reference_pictures/CREDITS.md`.

This program comes with ABSOLUTELY NO WARRANTY.
