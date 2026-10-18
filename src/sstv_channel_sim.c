/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Hany El Imam - 1KO125 firmware for the Xiegu X6100
 *
 * SSTV channel simulator.
 *
 * Does to a picture what a real SSTV contact does to it:
 *
 *   clean PNG --> SSTV audio (the 1KO125 encoder, sstv/encoder.c)
 *             --> radio channel: white noise, fading, impulses, bursts
 *             --> the 1KO125 decoder (sstv/decoder.c) --> received PNG
 *
 * The encoder and the decoder (src/sstv/) are the same files as in the
 * 1KO125 firmware of the Xiegu X6100, so the received picture has exactly
 * the defects the radio produces (codec softness, line timing, colour
 * bleeding) plus the noise added here.
 *
 * Usage:
 *
 *   sstv_channel_sim CLEAN.png MODE SNR_DB SEED OUT.png [IMPULSES_MIN] [FADE_DB] [BURSTS_MIN]
 *
 *   MODE          m1 m2 s1 s2 r36 r72  (Martin M1/M2, Scottie S1/S2, Robot 36/72)
 *   SNR_DB        signal to noise ratio, noise measured in 2500 Hz (the usual
 *                 SSB reference).  60 = practically no noise.
 *   SEED          any number: the same seed gives the same noise
 *   IMPULSES_MIN  short impulses (2..32 ms) per minute, 0 = none
 *   FADE_DB       depth of a slow fading (0.15..0.45 Hz), 0 = none
 *   BURSTS_MIN    long bursts of interference (0.3..1.2 s) per minute, 0 = none
 *
 * The noise comes from a small random generator of our own (xorshift64), not
 * from rand(): the same SEED gives the same noise on Windows, macOS and Linux.
 *
 * The "clean" picture of a training pair is the same picture through the
 * same mode with SNR 60 and nothing else: it carries the codec defects but
 * no noise, so the network learns to remove only the noise.
 *
 * Exit code: 0 ok, 1 bad arguments, 2 encoder error, 3 decoder found nothing.
 */
#define _USE_MATH_DEFINES      /* M_PI on Windows */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "encoder.h"
#include "decoder.h"

/* small fast random generator (xorshift64), seeded from the command line */
static unsigned long long rs = 88172645463325252ull;

static double urand(void)              /* uniform 0..1 */
{
    rs ^= rs << 13;
    rs ^= rs >> 7;
    rs ^= rs << 17;
    return (rs >> 11) * (1.0 / 9007199254740992.0);
}

static double grand(void)              /* gaussian, mean 0, sigma 1 */
{
    double u = urand() + 1e-12, v = urand();
    return sqrt(-2 * log(u)) * cos(2 * M_PI * v);
}

int main(int argc, char **argv)
{
    if (argc < 6) {
        fprintf(stderr, "sstv_channel_sim CLEAN.png MODE(m1|m2|s1|s2|r36|r72) SNR_DB SEED OUT.png "
                        "[IMPULSES/min] [FADE_DB] [BURSTS/min]\n");
        return 1;
    }
    const char *mode = argv[2];
    double snr = atof(argv[3]);
    rs += strtoull(argv[4], 0, 10) * 7919;
    double imp_per_min   = argc > 6 ? atof(argv[6]) : 0;
    double fade_db       = argc > 7 ? atof(argv[7]) : 0;
    double burst_per_min = argc > 8 ? atof(argv[8]) : 0;

    /* 1. picture -> SSTV audio, 16-bit, 12000 samples per second */
    int16_t *pcm;
    size_t n;
    sstv_mode_t m;
    int r;
    if (!strcmp(mode, "m1"))       { r = sstv_encode_martin_m1_png_pcm(argv[1], SSTV_SAMPLE_RATE, &pcm, &n);  m = SSTV_MODE_MARTIN_M1; }
    else if (!strcmp(mode, "m2"))  { r = sstv_encode_martin_m2_png_pcm(argv[1], SSTV_SAMPLE_RATE, &pcm, &n);  m = SSTV_MODE_MARTIN_M2; }
    else if (!strcmp(mode, "s1"))  { r = sstv_encode_scottie_s1_png_pcm(argv[1], SSTV_SAMPLE_RATE, &pcm, &n); m = SSTV_MODE_SCOTTIE_S1; }
    else if (!strcmp(mode, "s2"))  { r = sstv_encode_scottie_s2_png_pcm(argv[1], SSTV_SAMPLE_RATE, &pcm, &n); m = SSTV_MODE_SCOTTIE_S2; }
    else if (!strcmp(mode, "r72")) { r = sstv_encode_robot72_png_pcm(argv[1], SSTV_SAMPLE_RATE, &pcm, &n);    m = SSTV_MODE_ROBOT72; }
    else if (!strcmp(mode, "r36")) { r = sstv_encode_robot36_png_pcm(argv[1], SSTV_SAMPLE_RATE, &pcm, &n);    m = SSTV_MODE_ROBOT36; }
    else {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 1;
    }
    if (r) {
        fprintf(stderr, "encode failed %d\n", r);
        return 2;
    }

    /* 2. the channel.  One second of noise before and after the picture, as
     * a real reception has.  ps = power of the signal. */
    long pre = SSTV_SAMPLE_RATE, total = pre + (long)n + SSTV_SAMPLE_RATE;
    float *x = calloc(total, sizeof(float));
    double ps = 0;
    for (size_t i = 0; i < n; i++) {
        double v = pcm[i] / 32768.0;
        ps += v * v;
    }
    ps /= n;
    /* white noise: SNR referred to 2500 Hz; the audio band is SR/2 = 6000 Hz */
    double nsig = sqrt(ps / pow(10, snr / 10) * (SSTV_SAMPLE_RATE / 2.0) / 2500.0);
    double fph = urand() * 6.28, fr = 0.15 + urand() * 0.3;    /* fading phase and rate (Hz) */
    long imp_left = 0;                                          /* samples left in the current impulse/burst */
    double imp_amp = 0;
    for (long i = 0; i < total; i++) {
        double s = 0;
        if (i >= pre && i < pre + (long)n) {
            double g = 1;                                       /* fading: gain between 1 and -FADE_DB */
            if (fade_db > 0) g = pow(10, -fade_db / 20 * (0.5 + 0.5 * sin(fph + 2 * M_PI * fr * i / SSTV_SAMPLE_RATE)));
            s = pcm[i - pre] / 32768.0 * g;
        }
        s += nsig * grand();
        if (imp_left > 0) {                                     /* inside an impulse or a burst */
            s += imp_amp * grand();
            imp_left--;
        } else if (imp_per_min > 0 && urand() < imp_per_min / 60.0 / SSTV_SAMPLE_RATE) {
            imp_left = (long)(SSTV_SAMPLE_RATE * (0.002 + urand() * 0.03));    /* 2..32 ms */
            imp_amp = sqrt(ps) * (1 + urand() * 4);
        } else if (burst_per_min > 0 && urand() < burst_per_min / 60.0 / SSTV_SAMPLE_RATE) {
            imp_left = (long)(SSTV_SAMPLE_RATE * (0.3 + urand() * 0.9));       /* 0.3..1.2 s */
            imp_amp = sqrt(ps) * (0.7 + urand() * 1.5);
        }
        x[i] = (float)s;
    }

    /* 3. the firmware decoder, in the given mode (as after a VIS) */
    static uint8_t rgb[SSTV_MAX_W * SSTV_MAX_H * 3];
    sstv_info_t info;
    memset(rgb, 0, sizeof(rgb));
    int rows = sstv_decode_samples(x, total, m, rgb, &info);
    if (rows <= 0) return 3;
    return stbi_write_png(argv[5], info.width, info.height, 3, rgb, info.width * 3) ? 0 : 3;
}
