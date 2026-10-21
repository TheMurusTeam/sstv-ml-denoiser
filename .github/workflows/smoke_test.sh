#!/bin/sh
# SPDX-License-Identifier: LGPL-2.1-or-later
#
# Short test of the freshly built tools (run by build-tools.yml):
# one picture through the channel simulator, then the ML denoiser of the
# radio on it, and the check that the fast loops match the plain C ones.
set -e
PIC=1_test_set/reference_pictures/testcard.png
mkdir -p smoke
bin/sstv_channel_sim "$PIC" r36 6 1 smoke/noisy.png 2>/dev/null
bin/sstv_denoise_tool ml models/sstv_dnn_20x6.bin smoke/noisy.png smoke/ml.png
bin/sstv_denoise_tool classic smoke/noisy.png smoke/classic.png
bin/sstv_denoise_tool selfcheck models/sstv_dnn_20x6.bin smoke/noisy.png
rm -rf smoke
echo "smoke test OK"
