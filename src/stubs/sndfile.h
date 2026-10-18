/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Hany El Imam
 *
 * Minimal stand-in for libsndfile: sstv/decoder.c can also read WAV files,
 * which the tools of this package never do, so these calls only have to
 * exist for the build.  Nothing has to be installed.
 */
#pragma once
#include <stdint.h>
typedef int64_t sf_count_t;
typedef struct { sf_count_t frames; int samplerate, channels, format, sections, seekable; } SF_INFO;
typedef struct SNDFILE_tag SNDFILE;
#define SFM_READ 0x10
static inline SNDFILE *sf_open(const char *p, int m, SF_INFO *i) { (void)p; (void)m; (void)i; return 0; }
static inline const char *sf_strerror(SNDFILE *f) { (void)f; return "stub"; }
static inline int sf_close(SNDFILE *f) { (void)f; return 0; }
static inline sf_count_t sf_readf_float(SNDFILE *f, float *b, sf_count_t n) { (void)f; (void)b; (void)n; return 0; }
