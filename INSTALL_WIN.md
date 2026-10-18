# Installing everything on Windows

Step by step, from a PC with nothing installed to the first test. Every
command goes in the **Command Prompt**: Start menu → type `cmd` → *Command
Prompt*. One line at a time. (PowerShell works too: see the note in step 4.)

## What you will install

| Component | Version | Why this version |
|---|---|---|
| Windows | 10 or 11, **64-bit** | PyTorch exists only for 64-bit Windows |
| **Python** | **3.12, 64-bit** (3.11 to 3.13 work too) | the current Pillow and PyTorch need at least 3.11; the very newest Python (3.14, 3.15) can wait a few months for ready-made packages, so 3.12 is the safe choice |
| numpy | 2.x (anything from 1.24 works) | picture maths |
| Pillow | the latest (10 or newer works) | reading and writing pictures |
| The two C tools | the Windows zip of the GitHub Releases | the SSTV channel and the denoisers of the radio; no compiler needed |
| Nvidia driver | recent (580 or newer) | only for the training on an Nvidia card |
| PyTorch | the latest, CUDA build (only for step 3, the training) | the training, on the Nvidia card |

The Python packages go in a **private environment** inside the package
folder (`.venv`): nothing else on the PC is changed, and deleting the folder
removes everything.

## 1. Which PC you have

Settings → System → **About**:

- **System type** must say **64-bit operating system**.

Task Manager (Ctrl+Shift+Esc) → **Performance** → **GPU**: the name of the
graphics card, top right.

- **NVIDIA GeForce RTX 20xx, 30xx, 40xx, 50xx or GTX 16xx** – everything
  works, the training too (on the card).
- **NVIDIA GeForce GTX 9xx or 10xx** – the training works with an older
  build of PyTorch: see step 7.
- **AMD, Intel, or no separate card** – steps 1, 2, 4, 5 and 6 work; the
  training would run on the processor only, which takes days. Better to
  train on a Mac with an Apple chip or a PC with an Nvidia card, then bring
  the `.bin` model back here.

## 2. Where to put the package

Unzip the package (right click → *Extract All…*) into a **short folder
that is not synced by OneDrive**, for example:

    C:\sstv-ml-denoiser

Not on the Desktop or in Documents when OneDrive is on: OneDrive would try
to upload the thousands of pictures the steps make (hundreds of MB), and
can lock files while the scripts write them.

## 3. Python 3.12

Either with **winget** (already in Windows 10 and 11), in the Command Prompt:

    winget install -e --id Python.Python.3.12

or from **python.org**: https://www.python.org/downloads/windows/ →
the latest **Python 3.12.x**, "Windows installer (64-bit)". In the first
window of the installer tick **Add python.exe to PATH**, then *Install Now*.

Close the Command Prompt and open a **new** one, then check:

    py -3.12 --version

It prints `Python 3.12.x`. `py` is the Python launcher of Windows: `py -3.12`
always finds Python 3.12, even if other versions are installed.

> Do not use the Python of the Microsoft Store, and ignore it if typing
> `python` opens the Store: from step 4 on, `python` is the one of the
> private environment.

## 4. The private environment, numpy and Pillow

Go into the package folder:

    cd /d C:\sstv-ml-denoiser

Make the private environment and install numpy and Pillow in it:

    py -3.12 -m venv .venv
    .venv\Scripts\activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

After `.venv\Scripts\activate` the prompt starts with `(.venv)`: from now
on **`python`** is the Python of the environment. (The READMEs write
`python3`, the name on macOS and Linux: on Windows type `python`.) Check:

    python -c "import sys, numpy, PIL; print(sys.version.split()[0], numpy.__version__, PIL.__version__)"

It prints three versions, for example `3.12.10 2.3.5 12.0.0`.

**Every new Command Prompt:** go into the folder and activate again:

    cd /d C:\sstv-ml-denoiser
    .venv\Scripts\activate

> **PowerShell** (the default of Windows Terminal) refuses to run
> `.venv\Scripts\Activate.ps1` at first ("running scripts is disabled on
> this system"). Allow it once for your user:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, answer `Y`. Or
> simply use the Command Prompt.

## 5. The two C tools

Download **`tools-windows-x64.zip`** from the *Releases* page of the GitHub
repository. Before unzipping it: right click the zip → *Properties* → tick
**Unblock** (bottom of the window) → OK. Then unzip it **into the package
folder**: it contains a `bin` folder, so you get

    C:\sstv-ml-denoiser\bin\sstv_channel_sim.exe
    C:\sstv-ml-denoiser\bin\sstv_denoise_tool.exe

Check:

    bin\sstv_denoise_tool

It prints its usage (five lines starting with `sstv_denoise_tool`).

> Windows Defender SmartScreen may say "Windows protected your PC" if you
> double click the programs: they are not meant to be double clicked, the
> scripts run them. If an antivirus deletes them, restore them and add the
> `bin` folder to its exceptions.

**To build them yourself instead** (only if you want to): install MSYS2
from https://www.msys2.org, open *MSYS2 UCRT64* from the Start menu, then:

    pacman -S --needed make mingw-w64-ucrt-x86_64-gcc
    cd /c/sstv-ml-denoiser
    make LDFLAGS=-static

`LDFLAGS=-static` puts everything inside the `.exe`: without it, the tools
work only inside the MSYS2 window (they would miss its DLLs elsewhere).

## 6. First test (no training needed)

In the activated environment:

    python 1_test_set\make_test_set.py
    python 4_check_and_score\score.py models\sstv_dnn_20x6.bin

The first makes the reference test set (about a minute), the second scores
the reference model on it (a few minutes): the `ALL` row must show about
`+5.2` for classic and `+9.1` for `20x6`.

With some SSTV pictures you received:

    python 5_real_pictures\denoise_real_pictures.py --pictures C:\Users\%USERNAME%\Pictures\my_sstv

The page opens in the browser at the end.

## 7. PyTorch, for the training (step 3)

Only with an Nvidia card (step 1).

### The easy way

Double click **`3_train\train_windows.bat`**. It uses the environment of
step 4 (or makes it), installs PyTorch for Nvidia in it (a few GB the first
time), checks that the card is seen, then trains. Do steps 1 and 2 of the
package first. The network size is set at the top of the file.

### By hand

1. **Nvidia driver:** update it from https://www.nvidia.com/drivers (or
   the NVIDIA App). Check in the Command Prompt:

       nvidia-smi

   It shows the card and, top right, "CUDA Version": it must be **13.0 or
   higher** for the default build below. A lower number means the driver is
   too old.

2. **PyTorch**, in the activated environment:

   - RTX 20xx and newer, GTX 16xx:

         python -m pip install torch --index-url https://download.pytorch.org/whl/cu130

   - GTX 9xx or 10xx (not supported by the CUDA 13 builds):

         python -m pip install torch --index-url https://download.pytorch.org/whl/cu126

   The `--index-url` matters: plain `pip install torch` installs a build
   **without** Nvidia support on Windows.

3. Check that it sees the card:

       python -c "import torch; print(torch.__version__, 'Nvidia:', torch.cuda.is_available())"

   It must print `Nvidia: True` (and a version ending in `+cu130` or
   `+cu126`).

4. The one-minute check:

       python 3_train\train.py --selftest
       python 4_check_and_score\check_c_matches_pytorch.py TRAINED\selftest

   The last line must be `OK`. Then the real training: `python
   3_train\train.py` (see `3_train\README.md`). The PC is kept awake while
   it runs; leave it on mains power.

## If something goes wrong

| Message | What to do |
|---|---|
| `'py' is not recognized` | Python is not installed, or the Command Prompt was open before: open a new one (step 3) |
| typing `python` opens the Microsoft Store | the environment is not active: `.venv\Scripts\activate` (step 4) |
| `No module named numpy` (or `PIL`, `torch`) | same: activate the environment, then install again |
| `running scripts is disabled on this system` | PowerShell: see the note in step 4, or use the Command Prompt |
| `No matching distribution found for torch` | Python 32-bit, or too new: install Python 3.12 64-bit, delete `.venv` and redo step 4 |
| `Nvidia: False` | the build without CUDA was installed (`python -m pip uninstall -y torch`, then step 7 with `--index-url`), or the driver is too old (`nvidia-smi`), or a GTX 9xx/10xx with the `cu130` build: use `cu126` |
| `bin/sstv_channel_sim is missing` | step 5: the `bin` folder must be inside the package folder |
| `libwinpthread-1.dll was not found` | tools built with MSYS2 without `LDFLAGS=-static`: build again as in step 5 |
| `The system cannot find the path specified` | the path has a typo, or contains spaces: put it in quotes, `"C:\My pictures\sstv"` |
| very slow, files locked | the package is in a OneDrive folder: move it (step 2) |

## Removing everything

Delete the package folder (with it go `.venv`, `bin` and all the results).
Python 3.12 stays: to remove it, Settings → Apps → *Python 3.12* →
Uninstall (and *Python Launcher* if no other Python is left).
