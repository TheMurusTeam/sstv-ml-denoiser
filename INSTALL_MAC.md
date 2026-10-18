# Installing everything on a Mac

Step by step, from a Mac with nothing installed to the first test. Every
command goes in the **Terminal** (Applications → Utilities → Terminal),
one line at a time.

## What you will install

| Component | Version | Why this version |
|---|---|---|
| macOS | 12.3 or newer | needed by PyTorch to use the Apple GPU (training only) |
| Xcode Command Line Tools | the one Apple offers | the C compiler, to build the two tools with `make` |
| **Python** | **3.12** (3.11 to 3.13 work too) | the current Pillow and PyTorch need at least 3.11; the very newest Python (3.14, 3.15) can wait a few months for ready-made packages, so 3.12 is the safe choice |
| numpy | 2.x (anything from 1.24 works) | picture maths |
| Pillow | the latest (10 or newer works) | reading and writing pictures |
| PyTorch | the latest (only for step 3, the training) | the training, on the Apple GPU (MPS) |

The Python packages go in a **private environment** inside the package
folder (`.venv`): nothing is changed in macOS, and deleting the folder
removes everything.

## 1. Which Mac you have

Apple menu → **About This Mac**:

- **Chip: Apple M1, M2, M3, M4…** – everything works, the training too
  (on the GPU).
- **Processor: Intel** – steps 1, 2, 4, 5 and 6 work. The training is slow:
  PyTorch stopped making Intel Mac packages after version 2.2, and Intel Macs
  have no GPU PyTorch can use. See the note in step 7.

Check also the macOS version on the same window: 12.3 (Monterey) or newer.

## 2. The C compiler

    xcode-select --install

A window opens: click **Install** (a few minutes). If it says the tools are
already installed, nothing to do. Check:

    cc --version

It prints `Apple clang version …`.

## 3. Python 3.12

1. Go to https://www.python.org/downloads/macos/ and download the latest
   **Python 3.12.x** "macOS 64-bit universal2 installer" (it works on both
   Apple and Intel Macs).
2. Open the `.pkg` and install it.
3. At the end a Finder window opens on the folder *Python 3.12*: double
   click **Install Certificates.command** (without it, downloads from Python
   can fail with "certificate verify failed").

Check in a **new** Terminal window:

    python3.12 --version

It prints `Python 3.12.x`. Use `python3.12` (not just `python3`) until the
private environment is made: `python3` may be the older Python of macOS.

> With Homebrew instead: `brew install python@3.12`. Both are fine; do not
> install packages into Homebrew's Python directly (it refuses with
> "externally-managed-environment"): the private environment of step 4
> avoids that.

## 4. The package and its Python packages

Unzip the package (or `git clone` it), then go into its folder, for example:

    cd ~/Downloads/sstv-ml-denoiser

Make the private environment and install numpy and Pillow in it:

    python3.12 -m venv .venv
    source .venv/bin/activate
    python3 -m pip install --upgrade pip
    python3 -m pip install -r requirements.txt

After `source .venv/bin/activate` the prompt starts with `(.venv)`: from
now on `python3` is the Python of the environment. Check:

    python3 -c "import sys, numpy, PIL; print(sys.version.split()[0], numpy.__version__, PIL.__version__)"

It prints three versions, for example `3.12.10 2.3.5 12.0.0`.

**Every new Terminal window:** go into the folder and activate again:

    cd ~/Downloads/sstv-ml-denoiser
    source .venv/bin/activate

## 5. The two C tools

Either **build** them (needs step 2):

    make

or **download** `tools-macos-universal.zip` from the *Releases* page of the
GitHub repository, unzip it in the package folder (it contains `bin/`),
and unblock it once, because macOS blocks programs downloaded from the
internet:

    xattr -dr com.apple.quarantine bin

Check:

    bin/sstv_denoise_tool

It prints its usage (five lines starting with `sstv_denoise_tool`).

## 6. First test (no training needed)

    python3 1_test_set/make_test_set.py
    python3 4_check_and_score/score.py models/sstv_dnn_20x6.bin

The first makes the reference test set (about a minute), the second scores
the reference model on it (a few minutes): the `ALL` row must show about
`+5.2` for classic and `+9.1` for `20x6`.

With some SSTV pictures you received:

    python3 5_real_pictures/denoise_real_pictures.py --pictures ~/Desktop/my_sstv_pictures

The page opens in the browser at the end.

## 7. PyTorch, for the training (step 3)

### Apple chip (M1…M4)

In the activated environment:

    python3 -m pip install torch

Check that it sees the GPU:

    python3 -c "import torch; print(torch.__version__, 'GPU (MPS):', torch.backends.mps.is_available())"

It must print `GPU (MPS): True`. Then the one-minute check:

    python3 3_train/train.py --selftest
    python3 4_check_and_score/check_c_matches_pytorch.py TRAINED/selftest

The last line must be `OK`. The real training: `3_train/train_mac_linux.sh`
(it keeps the Mac awake while it runs; see `3_train/README.md`).

### Intel Mac

The last PyTorch for Intel Macs is 2.2.2, and it needs numpy 1.x and a
Python up to 3.12:

    python3 -m pip install "torch==2.2.2" "numpy<2"

It runs on the CPU only: the selftest works, a real training takes days.
Better to train on an Apple-chip Mac or on a PC with an Nvidia card
(`3_train/train_windows.bat`), then bring the `.bin` back here for steps 4
to 6.

## If something goes wrong

| Message | What to do |
|---|---|
| `xcrun: error: invalid active developer path` | the C compiler is missing: step 2 |
| `command not found: python3.12` | Python 3.12 is not installed, or the Terminal was open before: open a new window |
| `externally-managed-environment` | you are not in the private environment: `source .venv/bin/activate` |
| `No module named numpy` (or `PIL`, `torch`) | same: activate the environment, then install again |
| `certificate verify failed` | run *Install Certificates.command* in Applications → Python 3.12 |
| `No matching distribution found for torch` | Python too new or too old for PyTorch: use 3.12 (step 3) and make the environment again (`rm -rf .venv`, then step 4) |
| `"sstv_denoise_tool" cannot be opened because the developer cannot be verified` | `xattr -dr com.apple.quarantine bin` (step 5) |
| `bin/sstv_channel_sim is missing` | step 5 |
| `GPU (MPS): False` on an Apple chip | macOS older than 12.3, or an Intel Python: reinstall Python from the universal2 installer |

## Removing everything

Delete the package folder (with it go `.venv`, `bin` and all the results).
Python 3.12 stays: to remove it, delete *Applications → Python 3.12* and
`/Library/Frameworks/Python.framework/Versions/3.12`.
