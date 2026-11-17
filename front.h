// lviv_gl's front end for mali_bench (front.c): a glTF scene's frame as gl_test's GPU gets it, its
// triangle records and textures. No GL names here: main.c has libGLESv2's.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "gpu_triangle.h"

#define FRONT_TEXTURES 128 // gpu_map.h's GPU_TEXTURES

typedef struct
{
    const gpu_triangle_t *triangles; // as gpu_frame gets them
    uint32_t count;
    uint32_t clear_color; // ARGB
    // By tex_id: ARGB words in Morton order (gpu_texel_index), 1 << log2_size square; NULL none
    const uint32_t *texels[FRONT_TEXTURES];
    uint32_t log2_size[FRONT_TEXTURES];
    int file_triangles, textures, lights, own_camera;
} front_frame_t;

// The frame of the glTF file at path, width x height, the model turned by angle degrees as gl_test
// turns it; frame_bytes of frame memory (a multiple of 64). Returns 0, or -1 with a message in error.
int front_build(const char *path, int width, int height, float angle, size_t frame_bytes, front_frame_t *out,
                char *error, size_t error_size);
