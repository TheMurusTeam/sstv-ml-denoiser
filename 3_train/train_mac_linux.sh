#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Hany El Imam
#
# Step 3 on macOS or Linux: installs what is missing and trains.
#
#   3_train/train_mac_linux.sh [CHANNELS] [LAYERS] [STEPS]
#
#   3_train/train_mac_linux.sh              20 x 6, 200000 steps (the reference recipe)
#   3_train/train_mac_linux.sh 16 7 120000
#
# Reads DATA (step 2) and TEST_SET/reference (step 1); the model ends up in
# TRAINED/<C>x<L>/sstv_dnn_<C>x<L>.bin.  If it stops (power cut, Ctrl-C),
# run the same command again: it goes on from the last save.
#
# Python: if numpy, Pillow and PyTorch are already installed, they are used.
# Otherwise, the first time, a private Python environment is made in .venv at
# the top of the package and they are installed there (nothing changes in
# the system).  On a Mac PyTorch uses the Apple GPU by itself; on Linux the
# default PyTorch includes the Nvidia (CUDA) support.
set -e
cd "$(dirname "$0")/.."

CH="${1:-20}"
LY="${2:-6}"
STEPS="${3:-200000}"

PY=python3
if [ -x .venv/bin/python ]; then
    PY=.venv/bin/python
fi
if ! "$PY" -c "import torch, numpy, PIL" 2>/dev/null; then
    echo "Installing numpy, Pillow and PyTorch in .venv (the first time it downloads a few hundred MB)..."
    [ -x .venv/bin/python ] || python3 -m venv .venv
    PY=.venv/bin/python
    "$PY" -m pip install --upgrade pip
    "$PY" -m pip install torch numpy pillow
fi

if command -v caffeinate >/dev/null 2>&1; then
    # macOS: keep the Mac awake while training (caffeinate ends with it)
    exec caffeinate -i "$PY" 3_train/train.py --channels "$CH" --layers "$LY" --steps "$STEPS" --resume
fi
exec "$PY" 3_train/train.py --channels "$CH" --layers "$LY" --steps "$STEPS" --resume
