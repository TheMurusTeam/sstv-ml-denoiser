# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Hany El Imam
#
# Builds the two C tools into bin/:
#
#   bin/sstv_channel_sim     picture -> SSTV audio -> noise -> SSTV decoder -> picture
#   bin/sstv_denoise_tool    the classic and the ML denoiser of the radio
#
#   make          build (macOS, Linux, Windows with MSYS2)
#   make clean    remove bin/
#
# Nothing has to be installed besides a C compiler: PNG and JPG are read and
# written with stb_image (src/third_party), and src/stubs stands in for the
# libpng/libsndfile calls of the firmware files.
#
# -ffast-math lets the compiler use vector instructions (SSE/AVX/NEON) for
# the float loops of the network; results stay within 1 level of the plain C
# loops (bin/sstv_denoise_tool selfcheck tells).

CC      ?= cc
CFLAGS  ?= -O3 -ffast-math
LDLIBS   = -lm -lpthread
INC      = -Isrc/sstv -Isrc/third_party -Isrc/stubs

SIM_SRC  = src/sstv_channel_sim.c src/sstv/encoder.c src/sstv/decoder.c src/sstv/vis.c src/sstv/linesync.c
DN_SRC   = src/sstv_denoise_tool.c src/sstv/sstv_dnn.c src/sstv/sstv_denoise.c
HEADERS  = $(wildcard src/sstv/*.h src/stubs/*.h src/third_party/*.h)

all: bin/sstv_channel_sim bin/sstv_denoise_tool

bin/sstv_channel_sim: $(SIM_SRC) $(HEADERS)
	@mkdir -p bin
	$(CC) $(CFLAGS) -w $(INC) -o $@ $(SIM_SRC) $(LDFLAGS) $(LDLIBS)

bin/sstv_denoise_tool: $(DN_SRC) $(HEADERS)
	@mkdir -p bin
	$(CC) $(CFLAGS) $(INC) -o $@ $(DN_SRC) $(LDFLAGS) $(LDLIBS)

clean:
	rm -rf bin

.PHONY: all clean
