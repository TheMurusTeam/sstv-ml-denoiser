/*
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2026 Hany El Imam
 *
 * The two SSTV picture denoisers of the 1KO125 firmware (src/sstv/), on a
 * computer.  The C code is the same as in the radio, so what this tool gives
 * is what the radio gives.
 *
 *   sstv_denoise_tool ml MODEL.bin IN.png OUT.png [THREADS]
 *       The neural network, ALWAYS (no noise gate), on the whole picture.
 *       Used to score a model on the test set.
 *
 *   sstv_denoise_tool radio MODEL.bin IN.png OUT.png [THREADS]
 *       What the radio does at the end of a reception with Denoise ML:
 *         1. it measures the noise with the classic denoiser's measure;
 *         2. a clean picture (noise under the gate) is left as it is;
 *         3. otherwise the network cleans the picture; black rows at the
 *            bottom (a picture not complete) are left out.
 *       Used on real received pictures.
 *
 *   sstv_denoise_tool classic IN.png OUT.png
 *       The classic (non-ML) denoiser, as the radio does with Denoise CLASSIC.
 *
 *   sstv_denoise_tool measure IN.png
 *       Only the noise measure.
 *
 *   sstv_denoise_tool selfcheck MODEL.bin IN.png
 *       Runs the network twice, with the fast vector loops and with the plain
 *       C loops, and compares: they must give the same picture (within 1).
 *
 * Every command prints one line of "key=value" fields, read by the Python
 * scripts, for example:
 *
 *   size=320x256 noise_y=6.1 noise_cb=3.2 noise_cr=3.0 rows=256 noisy=1 ml=1 gmacs=1.268 ms=410
 *
 * Exit code: 0 ok, 1 error (message on stderr), 2 bad arguments.
 */
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "sstv_denoise.h"
#include "sstv_dnn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DEFAULT_THREADS 4   /* the radio uses 4 threads */

/* milliseconds from a fixed point (C11 timespec_get: Windows, macOS, Linux) */
static double now_ms(void)
{
    struct timespec t;
    timespec_get(&t, TIME_UTC);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static int usage(void)
{
    fprintf(stderr, "sstv_denoise_tool ml        MODEL.bin IN.png OUT.png [THREADS]\n"
                    "sstv_denoise_tool radio     MODEL.bin IN.png OUT.png [THREADS]\n"
                    "sstv_denoise_tool classic   IN.png OUT.png\n"
                    "sstv_denoise_tool measure   IN.png\n"
                    "sstv_denoise_tool selfcheck MODEL.bin IN.png\n");
    return 2;
}

/* reads a PNG or JPG as 8-bit RGB */
static uint8_t *load(const char *path, int *w, int *h)
{
    int c;
    uint8_t *px = stbi_load(path, w, h, &c, 3);
    if (!px) fprintf(stderr, "cannot read %s\n", path);
    return px;
}

static int save(const char *path, const uint8_t *px, int w, int h)
{
    if (stbi_write_png(path, w, h, 3, px, w * 3)) return 1;
    fprintf(stderr, "cannot write %s\n", path);
    return 0;
}

static void print_noise(const sstv_denoise_stats_t *st, int noisy)
{
    printf(" noise_y=%.2f noise_cb=%.2f noise_cr=%.2f rows=%d noisy=%d", st->sigma_y, st->sigma_cb, st->sigma_cr,
           st->rows, noisy);
}

static int threads_arg(int argc, char **argv, int i)
{
    int t = argc > i ? atoi(argv[i]) : DEFAULT_THREADS;
    return t < 1 ? 1 : (t > 8 ? 8 : t);
}

/* ml and radio */
static int run_network(int argc, char **argv, int as_radio)
{
    if (argc < 5) return usage();
    sstv_dnn_t *m = sstv_dnn_load(argv[2]);
    if (!m) return 1;
    int w, h, threads = threads_arg(argc, argv, 5);
    uint8_t *px = load(argv[3], &w, &h);
    if (!px) return 1;
    printf("size=%dx%d", w, h);

    int rows = h, run = 1;
    if (as_radio) {
        /* the gate of the radio: the network runs only on a noisy picture */
        sstv_denoise_stats_t st;
        int noisy = sstv_denoise_measure(px, w, h, &st);
        if (noisy < 0) {
            fprintf(stderr, "out of memory\n");
            return 1;
        }
        print_noise(&st, noisy);
        run = noisy > 0 && st.rows >= 5;
        rows = st.rows;
    }
    double t0 = now_ms();
    if (run && sstv_dnn_run(m, px, w, rows, threads)) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    printf(" ml=%d gmacs=%.3f ms=%.0f threads=%d\n", run, run ? sstv_dnn_macs(m, w, rows) / 1e9 : 0.0,
           now_ms() - t0, threads);
    int ok = save(argv[4], px, w, h);
    sstv_dnn_free(m);
    stbi_image_free(px);
    return ok ? 0 : 1;
}

static int run_classic(const char *in, const char *out)
{
    int w, h;
    uint8_t *px = load(in, &w, &h);
    if (!px) return 1;
    sstv_denoise_stats_t st;
    double t0 = now_ms();
    if (sstv_denoise(px, w, h, &st) < 0) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    printf("size=%dx%d", w, h);
    print_noise(&st, 1);
    printf(" ms=%.0f\n", now_ms() - t0);
    int ok = save(out, px, w, h);
    stbi_image_free(px);
    return ok ? 0 : 1;
}

static int run_measure(const char *in)
{
    int w, h;
    uint8_t *px = load(in, &w, &h);
    if (!px) return 1;
    sstv_denoise_stats_t st;
    int noisy = sstv_denoise_measure(px, w, h, &st);
    printf("size=%dx%d", w, h);
    print_noise(&st, noisy > 0);
    printf("\n");
    stbi_image_free(px);
    return noisy < 0 ? 1 : 0;
}

static int run_selfcheck(const char *model, const char *in)
{
    sstv_dnn_t *m = sstv_dnn_load(model);
    if (!m) return 1;
    int w, h;
    uint8_t *a = load(in, &w, &h);
    if (!a) return 1;
    size_t sz = (size_t)w * h * 3;
    uint8_t *b = malloc(sz);
    memcpy(b, a, sz);

    double t0 = now_ms();
    sstv_dnn_run(m, a, w, h, DEFAULT_THREADS);      /* fast loops */
    double t1 = now_ms();
    sstv_dnn_generic = 1;                           /* plain C loops */
    sstv_dnn_run(m, b, w, h, DEFAULT_THREADS);
    double t2 = now_ms();

    int maxd = 0;
    long nd = 0;
    for (size_t i = 0; i < sz; i++) {
        int d = abs((int)a[i] - (int)b[i]);
        if (d > maxd) maxd = d;
        if (d) nd++;
    }
    printf("size=%dx%d neon=%d fast_ms=%.0f plain_ms=%.0f max_diff=%d values_diff=%ld result=%s\n", w, h,
           sstv_dnn_has_neon(), t1 - t0, t2 - t1, maxd, nd, maxd <= 1 ? "OK" : "WRONG");
    sstv_dnn_free(m);
    stbi_image_free(a);
    free(b);
    return maxd <= 1 ? 0 : 1;
}

int main(int argc, char **argv)
{
    if (argc < 3) return usage();
    const char *cmd = argv[1];
    if (!strcmp(cmd, "ml")) return run_network(argc, argv, 0);
    if (!strcmp(cmd, "radio")) return run_network(argc, argv, 1);
    if (!strcmp(cmd, "classic") && argc >= 4) return run_classic(argv[2], argv[3]);
    if (!strcmp(cmd, "measure")) return run_measure(argv[2]);
    if (!strcmp(cmd, "selfcheck") && argc >= 4) return run_selfcheck(argv[2], argv[3]);
    return usage();
}
