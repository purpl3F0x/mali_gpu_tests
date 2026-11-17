// A glTF 2.0 scene ready to draw: its triangles' vertices already transformed to clip space and lit
// per vertex on the CPU (gltf.c), so a pass-through vertex shader draws them.
#pragma once

#include <stddef.h>
#include <stdint.h>

typedef struct
{
    float position[4]; // clip space
    uint8_t color[4];  // lit, RGBA
    float uv[2];
} scene_vertex_t;

// A primitive: its triangles' indices, from first_index
typedef struct
{
    uint32_t first_index, index_count;
    int texture; // in textures; -1 none
    int blended, double_sided;
    int clockwise; // front faces clockwise (a mirroring transform)
} scene_draw_t;

// RGBA, row 0 at t = 0; sizes powers of two
typedef struct
{
    int width, height;
    uint8_t *rgba;
    int repeat_s, repeat_t;
} scene_texture_t;

typedef struct
{
    scene_vertex_t *vertices;
    uint32_t vertex_count;
    uint32_t *indices;
    uint32_t index_count;
    scene_draw_t *draws; // opaque, then blended
    int draw_count;
    scene_texture_t *textures;
    int texture_count;
    int own_camera;
} scene_t;

// The file's scene (a .glb, or a .gltf with its buffers and images embedded or beside it) for a
// width x height frame, the model turned by angle degrees about the vertical axis through its
// center; its first camera, else one 1.5 of its spans back along +z. Returns 0, or -1 with a message.
int scene_load(const char *path, int width, int height, float angle, scene_t *out, char *error,
               size_t error_size);
