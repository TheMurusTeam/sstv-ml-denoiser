/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Hany El Imam
 */
/*
 * Minimal stand-in for libpng: the few read calls of sstv/encoder.c, done
 * with stb_image (third_party/, public domain), so nothing has to be
 * installed on any system.  Every picture is read as 8-bit RGB.
 */
#ifndef SIM_PNG_SHIM_H
#define SIM_PNG_SHIM_H

#include <setjmp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "stb_image.h"

#define PNG_LIBPNG_VER_STRING      "shim"
#define PNG_COLOR_TYPE_GRAY        0
#define PNG_COLOR_TYPE_PALETTE     3
#define PNG_COLOR_TYPE_RGB         2
#define PNG_COLOR_TYPE_GRAY_ALPHA  4
#define PNG_INFO_tRNS              0x10

typedef uint32_t png_uint_32;
typedef size_t   png_size_t;
typedef uint8_t *png_bytep;

typedef struct {
    jmp_buf  jb;
    FILE    *fp;
    uint8_t *px;
    int      w, h;
} png_struct;
typedef png_struct *png_structp;
typedef int        *png_infop;

static int png_shim_info;

static inline png_structp png_create_read_struct(const char *v, void *a, void *b, void *c)
{
    (void)v; (void)a; (void)b; (void)c;
    return (png_structp)calloc(1, sizeof(png_struct));
}
static inline png_infop png_create_info_struct(png_structp p) { (void)p; return &png_shim_info; }
static inline void png_destroy_read_struct(png_structp *p, png_infop *i, void *x)
{
    (void)i; (void)x;
    if (p && *p) {
        if ((*p)->px) stbi_image_free((*p)->px);
        free(*p);
        *p = NULL;
    }
}
#define png_jmpbuf(p) ((p)->jb)
static inline void png_init_io(png_structp p, FILE *fp) { p->fp = fp; }
static inline void png_read_info(png_structp p, png_infop i)
{
    (void)i;
    int c;
    p->px = stbi_load_from_file(p->fp, &p->w, &p->h, &c, 3);
    if (!p->px) longjmp(p->jb, 1);
}
static inline png_uint_32 png_get_image_width(png_structp p, png_infop i) { (void)i; return (png_uint_32)p->w; }
static inline png_uint_32 png_get_image_height(png_structp p, png_infop i) { (void)i; return (png_uint_32)p->h; }
static inline int png_get_bit_depth(png_structp p, png_infop i) { (void)p; (void)i; return 8; }
static inline int png_get_color_type(png_structp p, png_infop i) { (void)p; (void)i; return PNG_COLOR_TYPE_RGB; }
static inline int png_get_valid(png_structp p, png_infop i, int f) { (void)p; (void)i; (void)f; return 0; }
static inline void png_set_strip_16(png_structp p) { (void)p; }
static inline void png_set_palette_to_rgb(png_structp p) { (void)p; }
static inline void png_set_expand_gray_1_2_4_to_8(png_structp p) { (void)p; }
static inline void png_set_tRNS_to_alpha(png_structp p) { (void)p; }
static inline void png_set_gray_to_rgb(png_structp p) { (void)p; }
static inline void png_read_update_info(png_structp p, png_infop i) { (void)p; (void)i; }
static inline int png_get_channels(png_structp p, png_infop i) { (void)p; (void)i; return 3; }
static inline png_size_t png_get_rowbytes(png_structp p, png_infop i) { (void)i; return (png_size_t)p->w * 3; }
static inline void png_read_image(png_structp p, png_bytep *rows)
{
    for (int y = 0; y < p->h; y++) memcpy(rows[y], p->px + (size_t)y * p->w * 3, (size_t)p->w * 3);
}
static inline void png_read_end(png_structp p, png_infop i) { (void)p; (void)i; }

#endif
