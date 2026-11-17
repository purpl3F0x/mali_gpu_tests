// front.h on lviv_gl, gl_test's way: gltf_scene_setup, gltf_scene_draw, lviv_gl_frame. The GPU driver
// is replaced by gpu_frame and gpu_set_texture keeping what they get. Built with lviv_rename.h, so
// lviv_gl's gl* names are lviv_gl* and do not take libGLESv2's.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "front.h"
#include "gltf_scene.h"
#include "lviv_gl.h"

static gpu_frame_t last_frame;
static int framed;
static const uint32_t *texels[FRONT_TEXTURES];
static uint32_t log2_sizes[FRONT_TEXTURES];

gpu_status_t gpu_frame(gpu_t *gpu, const gpu_frame_t *frame, uint32_t *failed_tile)
{
    (void)gpu;
    if (failed_tile)
        *failed_tile = 0;
    last_frame = *frame;
    framed = 1;
    return GPU_OK;
}

int gpu_set_texture(const gpu_t *gpu, uint32_t id, const uint32_t *t, uint32_t log2_size)
{
    (void)gpu;
    if (id >= FRONT_TEXTURES)
        return -1;
    texels[id] = t;
    log2_sizes[id] = log2_size;
    return 0;
}

uint32_t gpu_texel_index(uint32_t x, uint32_t y)
{
    uint32_t index = 0;
    for (int bit = 0; bit < 11; bit++)
        index |= (x >> bit & 1u) << (2 * bit) | (y >> bit & 1u) << (2 * bit + 1);
    return index;
}

// The file, whole, with the padding gltf_scene_load reads after it
static unsigned char *read_file(const char *path, size_t *size)
{
    FILE *in = fopen(path, "rb");
    if (!in)
        return NULL;
    unsigned char *data = NULL;
    if (fseek(in, 0, SEEK_END) == 0)
    {
        const long n = ftell(in);
        data = n >= 0 ? calloc((size_t)n + GLTF_SCENE_PADDING, 1) : NULL;
        *size = (size_t)n;
        if (data && (fseek(in, 0, SEEK_SET) != 0 || fread(data, 1, *size, in) != *size))
        {
            free(data);
            data = NULL;
        }
    }
    fclose(in);
    return data;
}

int front_build(const char *path, int width, int height, float angle, size_t frame_bytes, front_frame_t *out,
                char *error, size_t error_size)
{
    static gpu_t gpu;
    static uint32_t target; // gpu_frame's, not drawn into
    size_t size;
    unsigned char *data = read_file(path, &size);
    if (!data)
    {
        snprintf(error, error_size, "%s: cannot read it", path);
        return -1;
    }
    gltf_scene_t *scene = gltf_scene_load(data, size, error, error_size);
    free(data);
    if (!scene)
        return -1;
    void *memory = aligned_alloc(64, frame_bytes);
    if (!memory || lviv_gl_init(&gpu, width, height, memory, frame_bytes) != 0)
    {
        snprintf(error, error_size, "lviv_gl_init failed (%d x %d: at most 64 tiles of 128)", width, height);
        return -1;
    }
    glClearColor(0x63 / 255.0f, 0x63 / 255.0f, 0x63 / 255.0f, 1.0f); // gl_test's
    gltf_scene_setup(scene, width, height);
    glGetError(); // setup's own (textures halved to fit)
    gltf_scene_draw(scene, angle);
    const GLenum e = glGetError();
    if (e != GL_NO_ERROR)
    {
        snprintf(error, error_size, "lviv_gl error 0x%04x%s", e,
                 e == GL_OUT_OF_MEMORY ? ": more triangles than the frame memory holds (-m)" : "");
        return -1;
    }
    framed = 0;
    lviv_gl_frame(&target, (uint32_t)width * 4, NULL);
    if (!framed)
    {
        snprintf(error, error_size, "lviv_gl sent no frame");
        return -1;
    }
    // The scene stays: its textures' texels are lviv_gl's, which gpu_set_texture's pointers are into
    out->triangles = last_frame.triangles;
    out->count = last_frame.triangle_count;
    out->clear_color = last_frame.clear_color;
    memcpy(out->texels, texels, sizeof texels);
    memcpy(out->log2_size, log2_sizes, sizeof log2_sizes);
    out->file_triangles = gltf_scene_triangles(scene);
    out->textures = gltf_scene_textures(scene);
    out->lights = gltf_scene_lights(scene);
    out->own_camera = gltf_scene_has_camera(scene);
    return 0;
}
