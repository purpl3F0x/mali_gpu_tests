// gltf.h's scenes: a small JSON parser, the file's buffers, accessors, nodes, materials (base color,
// its texture, emission, alpha mode, double sided) and first camera. Lit by a white light from
// (1, 1, 1) in eye space, ambient 0.2. Images decoded by stb_image (PNG, JPEG), resized to powers of
// two (at most 2048) by nearest texels. Not read: skins, morph targets, sparse accessors, texture
// transforms, other material maps, lights.

#define _GNU_SOURCE
#include "gltf.h"

#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "stb_image.h"

#define MAX_TEXTURE 2048

// ---------------------------------------------------------------------------------------------
// JSON

typedef enum
{
    J_NULL,
    J_FALSE,
    J_TRUE,
    J_NUMBER,
    J_STRING,
    J_ARRAY,
    J_OBJECT
} json_type_t;

typedef struct json
{
    json_type_t type;
    double number;
    char *string;
    struct json *items;
    char **keys; // an object's
    int count;
} json_t;

static const char *skip_space(const char *p)
{
    while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')
        p++;
    return p;
}

static const char *parse_string(const char *p, char **out)
{
    size_t cap = 16, n = 0;
    char *s = malloc(cap);
    for (p++; *p && *p != '"';)
    {
        if (n + 4 > cap)
            s = realloc(s, cap *= 2);
        if (*p != '\\')
        {
            s[n++] = *p++;
            continue;
        }
        p++;
        unsigned c;
        switch (*p++)
        {
        case 'n':
            c = '\n';
            break;
        case 't':
            c = '\t';
            break;
        case 'r':
            c = '\r';
            break;
        case 'b':
            c = '\b';
            break;
        case 'f':
            c = '\f';
            break;
        case 'u':
            c = 0;
            for (int i = 0; i < 4; i++, p++)
            {
                if (!isxdigit((unsigned char)*p))
                    return NULL;
                c = c * 16 + (unsigned)(isdigit((unsigned char)*p) ? *p - '0' : (*p | 32) - 'a' + 10);
            }
            break;
        case '\0':
            return NULL;
        default:
            c = (unsigned char)p[-1];
        }
        if (c < 0x80)
            s[n++] = (char)c;
        else if (c < 0x800)
        {
            s[n++] = (char)(0xC0 | c >> 6);
            s[n++] = (char)(0x80 | (c & 63));
        }
        else
        {
            s[n++] = (char)(0xE0 | c >> 12);
            s[n++] = (char)(0x80 | (c >> 6 & 63));
            s[n++] = (char)(0x80 | (c & 63));
        }
    }
    if (*p != '"')
        return NULL;
    s[n] = '\0';
    *out = s;
    return p + 1;
}

// On an error the value is left as it is: the caller gives up on the file
static const char *parse_value(const char *p, json_t *v, int depth)
{
    memset(v, 0, sizeof *v);
    p = skip_space(p);
    if (depth > 64)
        return NULL;
    if (*p == '{' || *p == '[')
    {
        const int object = *p == '{';
        const char end = object ? '}' : ']';
        v->type = object ? J_OBJECT : J_ARRAY;
        p = skip_space(p + 1);
        if (*p == end)
            return p + 1;
        for (int cap = 0;;)
        {
            if (v->count == cap)
            {
                cap = cap ? 2 * cap : 4;
                v->items = realloc(v->items, (size_t)cap * sizeof *v->items);
                if (object)
                    v->keys = realloc(v->keys, (size_t)cap * sizeof *v->keys);
            }
            if (object)
            {
                p = skip_space(p);
                if (*p != '"' || !(p = parse_string(p, &v->keys[v->count])))
                    return NULL;
                p = skip_space(p);
                if (*p++ != ':')
                    return NULL;
            }
            if (!(p = parse_value(p, &v->items[v->count], depth + 1)))
                return NULL;
            v->count++;
            p = skip_space(p);
            if (*p == ',')
                p++;
            else if (*p == end)
                return p + 1;
            else
                return NULL;
        }
    }
    if (*p == '"')
    {
        v->type = J_STRING;
        return parse_string(p, &v->string);
    }
    if (strncmp(p, "true", 4) == 0)
    {
        v->type = J_TRUE;
        return p + 4;
    }
    if (strncmp(p, "false", 5) == 0)
    {
        v->type = J_FALSE;
        return p + 5;
    }
    if (strncmp(p, "null", 4) == 0)
        return p + 4;
    char *after;
    v->number = strtod(p, &after);
    v->type = J_NUMBER;
    return after == p ? NULL : after;
}

static void json_free(json_t *v)
{
    for (int i = 0; i < v->count; i++)
    {
        json_free(&v->items[i]);
        if (v->keys)
            free(v->keys[i]);
    }
    free(v->items);
    free(v->keys);
    free(v->string);
}

static const json_t *get(const json_t *o, const char *key)
{
    if (o && o->type == J_OBJECT)
        for (int i = 0; i < o->count; i++)
            if (strcmp(o->keys[i], key) == 0)
                return &o->items[i];
    return NULL;
}

static const json_t *at(const json_t *a, int i)
{
    return a && a->type == J_ARRAY && i >= 0 && i < a->count ? &a->items[i] : NULL;
}

static int length(const json_t *a)
{
    return a && a->type == J_ARRAY ? a->count : 0;
}

static double num(const json_t *v, double otherwise)
{
    return v && v->type == J_NUMBER ? v->number : otherwise;
}

static const char *str(const json_t *v)
{
    return v && v->type == J_STRING ? v->string : NULL;
}

// ---------------------------------------------------------------------------------------------
// Matrices: column-major, as GL's

typedef float mat4[16];

static void mat_identity(mat4 m)
{
    memset(m, 0, sizeof(mat4));
    m[0] = m[5] = m[10] = m[15] = 1.0f;
}

// out = a * b (out may be either)
static void mat_mul(mat4 out, const mat4 a, const mat4 b)
{
    mat4 r;
    for (int c = 0; c < 4; c++)
        for (int row = 0; row < 4; row++)
        {
            float s = 0;
            for (int k = 0; k < 4; k++)
                s += a[k * 4 + row] * b[c * 4 + k];
            r[c * 4 + row] = s;
        }
    memcpy(out, r, sizeof r);
}

static int mat_inverse(mat4 out, const mat4 m)
{
    float inv[16];
    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] +
             m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] -
             m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] +
             m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] -
              m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] -
             m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] +
             m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] -
             m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] +
              m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] +
             m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] -
             m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] +
              m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] -
              m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] -
             m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] +
             m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] -
              m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] +
              m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
    const float det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    if (det == 0.0f)
        return -1;
    for (int i = 0; i < 16; i++)
        out[i] = inv[i] / det;
    return 0;
}

static void mat_translate(mat4 m, float x, float y, float z)
{
    mat_identity(m);
    m[12] = x;
    m[13] = y;
    m[14] = z;
}

// T * R * S, the rotation a unit quaternion (x, y, z, w)
static void mat_trs(mat4 m, const float t[3], const float q[4], const float s[3])
{
    const float x = q[0], y = q[1], z = q[2], w = q[3];
    const float r[9] = {1 - 2 * (y * y + z * z), 2 * (x * y + z * w),     2 * (x * z - y * w),
                        2 * (x * y - z * w),     1 - 2 * (x * x + z * z), 2 * (y * z + x * w),
                        2 * (x * z + y * w),     2 * (y * z - x * w),     1 - 2 * (x * x + y * y)};
    mat_identity(m);
    for (int c = 0; c < 3; c++)
        for (int row = 0; row < 3; row++)
            m[c * 4 + row] = r[c * 3 + row] * s[c];
    m[12] = t[0];
    m[13] = t[1];
    m[14] = t[2];
}

static void transform(const mat4 m, const float v[4], float out[4])
{
    for (int row = 0; row < 4; row++)
        out[row] = m[row] * v[0] + m[4 + row] * v[1] + m[8 + row] * v[2] + m[12 + row] * v[3];
}

// ---------------------------------------------------------------------------------------------
// The file

typedef struct
{
    const uint8_t *data;
    size_t size;
    uint8_t *owned; // freed after loading, if not NULL
} blob_t;

// A primitive as the file has it, in its node's space
typedef struct
{
    mat4 world;
    float *positions, *normals, *uvs; // 3, 3 and 2 a vertex; normals and uvs may be NULL
    uint32_t vertex_count;
    uint32_t *indices; // triangles
    uint32_t index_count;
    float base[4], emissive[3];
    int texture, blended, double_sided;
} primitive_t;

typedef struct
{
    json_t root;
    blob_t *buffers;
    int buffer_count;
    char dir[1024];
    primitive_t *primitives;
    int primitive_count, primitive_cap;
    int *texture_slots; // by the file's texture: in the scene's textures; -1 not yet, -2 none
    scene_t *scene;
    int camera;
    mat4 camera_world;
    int orthographic;
    float yfov, aspect, xmag, ymag, znear, zfar;
    char *error;
    size_t error_size;
} loader_t;

static int fail(loader_t *l, const char *format, ...)
{
    va_list args;
    va_start(args, format);
    vsnprintf(l->error, l->error_size, format, args);
    va_end(args);
    return -1;
}

static uint8_t *read_file(const char *path, size_t *size)
{
    FILE *in = fopen(path, "rb");
    if (!in)
        return NULL;
    uint8_t *data = NULL;
    long n;
    if (fseek(in, 0, SEEK_END) == 0 && (n = ftell(in)) >= 0 && fseek(in, 0, SEEK_SET) == 0 &&
        (data = malloc((size_t)n + 1)) && fread(data, 1, (size_t)n, in) == (size_t)n)
    {
        data[n] = '\0';
        *size = (size_t)n;
    }
    else
    {
        free(data);
        data = NULL;
    }
    fclose(in);
    return data;
}

static uint8_t *base64_decode(const char *s, size_t *size)
{
    uint8_t *out = malloc(strlen(s) / 4 * 3 + 3);
    uint32_t bits = 0;
    int have = 0;
    size_t n = 0;
    for (; *s && *s != '='; s++)
    {
        const char *digits = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        const char *d = strchr(digits, *s);
        if (!d || !*s)
            continue;
        bits = bits << 6 | (uint32_t)(d - digits);
        if ((have += 6) >= 8)
            out[n++] = (uint8_t)(bits >> (have -= 8));
    }
    *size = n;
    return out;
}

// A buffer's or image's uri: a data: URI in base64, or a file beside the glTF file
static int load_uri(loader_t *l, const char *uri, blob_t *out)
{
    size_t size;
    uint8_t *data;
    if (strncmp(uri, "data:", 5) == 0)
    {
        const char *b64 = strstr(uri, ";base64,");
        if (!b64)
            return fail(l, "a data URI not in base64");
        data = base64_decode(b64 + 8, &size);
    }
    else
    {
        char path[2048];
        size_t n = (size_t)snprintf(path, sizeof path, "%s", l->dir);
        for (const char *p = uri; *p && n + 1 < sizeof path; p++) // %XX escapes
        {
            unsigned c;
            if (p[0] == '%' && isxdigit((unsigned char)p[1]) && isxdigit((unsigned char)p[2]) &&
                sscanf(p + 1, "%2x", &c) == 1)
            {
                path[n++] = (char)c;
                p += 2;
            }
            else
                path[n++] = *p;
        }
        path[n] = '\0';
        if (!(data = read_file(path, &size)))
            return fail(l, "%s: cannot read it", path);
    }
    *out = (blob_t){data, size, data};
    return 0;
}

// A buffer view's bytes; its stride (0 if not given)
static int view_bytes(loader_t *l, int index, const uint8_t **data, size_t *size, size_t *stride)
{
    const json_t *v = at(get(&l->root, "bufferViews"), index);
    const int buffer = (int)num(get(v, "buffer"), -1);
    const size_t offset = (size_t)num(get(v, "byteOffset"), 0), n = (size_t)num(get(v, "byteLength"), 0);
    if (!v || buffer < 0 || buffer >= l->buffer_count || offset + n > l->buffers[buffer].size)
        return fail(l, "buffer view %d past its buffer", index);
    *data = l->buffers[buffer].data + offset;
    *size = n;
    if (stride)
        *stride = (size_t)num(get(v, "byteStride"), 0);
    return 0;
}

static int component_bytes(int type)
{
    switch (type)
    {
    case 5120:
    case 5121:
        return 1;
    case 5122:
    case 5123:
        return 2;
    case 5125:
    case 5126:
        return 4;
    }
    return 0;
}

static int type_components(const char *type)
{
    static const char *names[] = {"SCALAR", "VEC2", "VEC3", "VEC4", "MAT2", "MAT3", "MAT4"};
    static const int counts[] = {1, 2, 3, 4, 4, 9, 16};
    for (int i = 0; type && i < 7; i++)
        if (strcmp(type, names[i]) == 0)
            return counts[i];
    return 0;
}

static float component(const uint8_t *p, int type, int normalized)
{
    switch (type)
    {
    case 5126: {
        float f;
        memcpy(&f, p, 4);
        return f;
    }
    case 5121:
        return normalized ? p[0] / 255.0f : p[0];
    case 5120: {
        const float v = (int8_t)p[0];
        return normalized ? fmaxf(v / 127.0f, -1.0f) : v;
    }
    case 5123: {
        uint16_t v;
        memcpy(&v, p, 2);
        return normalized ? v / 65535.0f : v;
    }
    case 5122: {
        int16_t v;
        memcpy(&v, p, 2);
        return normalized ? fmaxf(v / 32767.0f, -1.0f) : v;
    }
    case 5125: {
        uint32_t v;
        memcpy(&v, p, 4);
        return (float)v;
    }
    }
    return 0.0f;
}

// An accessor's layout: its first element (NULL: all 0, no buffer view), stride, component type and
// count, components, count
typedef struct
{
    const uint8_t *data;
    size_t stride;
    int type, components, normalized;
    size_t count;
} layout_t;

static int accessor_layout(loader_t *l, int index, layout_t *out)
{
    const json_t *a = at(get(&l->root, "accessors"), index);
    out->type = (int)num(get(a, "componentType"), 0);
    out->components = type_components(str(get(a, "type")));
    out->normalized = get(a, "normalized") && get(a, "normalized")->type == J_TRUE;
    out->count = (size_t)num(get(a, "count"), 0);
    out->data = NULL;
    const size_t element = (size_t)out->components * component_bytes(out->type);
    if (!a || !element)
        return fail(l, "accessor %d: no such accessor or type", index);
    const json_t *view = get(a, "bufferView");
    if (!view || out->count == 0)
        return 0;
    size_t size;
    if (view_bytes(l, (int)num(view, -1), &out->data, &size, &out->stride) != 0)
        return -1;
    const size_t offset = (size_t)num(get(a, "byteOffset"), 0);
    if (!out->stride)
        out->stride = element;
    if (offset + out->stride * (out->count - 1) + element > size)
        return fail(l, "accessor %d past its buffer view", index);
    out->data += offset;
    return 0;
}

// An accessor's elements as floats, want of them each (past its own components 0); count elements
static float *read_floats(loader_t *l, int index, int want, uint32_t *count)
{
    layout_t a;
    if (accessor_layout(l, index, &a) != 0)
        return NULL;
    float *out = calloc(a.count * want + 1, sizeof *out);
    *count = (uint32_t)a.count;
    const int k_max = a.components < want ? a.components : want, bytes = component_bytes(a.type);
    for (size_t i = 0; a.data && i < a.count; i++)
        for (int k = 0; k < k_max; k++)
            out[i * want + k] = component(a.data + i * a.stride + (size_t)k * bytes, a.type, a.normalized);
    return out;
}

// An index accessor's elements, exactly
static uint32_t *read_indices(loader_t *l, int index, uint32_t *count)
{
    layout_t a;
    if (accessor_layout(l, index, &a) != 0)
        return NULL;
    uint32_t *out = calloc(a.count + 1, sizeof *out);
    *count = (uint32_t)a.count;
    for (size_t i = 0; a.data && i < a.count; i++)
    {
        const uint8_t *p = a.data + i * a.stride;
        uint16_t u16;
        switch (a.type)
        {
        case 5121:
            out[i] = p[0];
            break;
        case 5123:
            memcpy(&u16, p, 2);
            out[i] = u16;
            break;
        case 5125:
            memcpy(&out[i], p, 4);
            break;
        default:
            free(out);
            fail(l, "accessor %d: indices of component type %d", index, a.type);
            return NULL;
        }
    }
    return out;
}

// The file's texture in the scene's textures, decoded the first time; -1 if it has none
static int load_texture(loader_t *l, int index)
{
    const json_t *textures = get(&l->root, "textures");
    if (index < 0 || index >= length(textures))
        return -1;
    if (l->texture_slots[index] != -1)
        return l->texture_slots[index] >= 0 ? l->texture_slots[index] : -1;
    l->texture_slots[index] = -2;
    const json_t *t = at(textures, index);
    const json_t *image = at(get(&l->root, "images"), (int)num(get(t, "source"), -1));
    blob_t file = {0};
    if (get(image, "bufferView"))
    {
        if (view_bytes(l, (int)num(get(image, "bufferView"), -1), &file.data, &file.size, NULL) != 0)
            return -1;
    }
    else if (!str(get(image, "uri")) || load_uri(l, str(get(image, "uri")), &file) != 0)
        return -1;
    int w, h, channels;
    uint8_t *pixels = stbi_load_from_memory(file.data, (int)file.size, &w, &h, &channels, 4);
    free(file.owned);
    if (!pixels)
        return -1;
    scene_texture_t out;
    out.width = out.height = 1;
    while (out.width * 2 <= w && out.width < MAX_TEXTURE)
        out.width *= 2;
    while (out.height * 2 <= h && out.height < MAX_TEXTURE)
        out.height *= 2;
    out.rgba = malloc((size_t)out.width * out.height * 4);
    for (int y = 0; y < out.height; y++)
        for (int x = 0; x < out.width; x++)
            memcpy(&out.rgba[4 * ((size_t)y * out.width + x)],
                   &pixels[4 * ((size_t)(y * h / out.height) * w + (size_t)(x * w / out.width))], 4);
    stbi_image_free(pixels);
    const json_t *sampler = at(get(&l->root, "samplers"), (int)num(get(t, "sampler"), -1));
    out.repeat_s = num(get(sampler, "wrapS"), 10497) != 33071; // mirrored: repeated
    out.repeat_t = num(get(sampler, "wrapT"), 10497) != 33071;
    scene_t *s = l->scene;
    s->textures = realloc(s->textures, (size_t)(s->texture_count + 1) * sizeof *s->textures);
    s->textures[s->texture_count] = out;
    return l->texture_slots[index] = s->texture_count++;
}

static int add_primitive(loader_t *l, const json_t *p, const mat4 world)
{
    const int mode = (int)num(get(p, "mode"), 4);
    const json_t *attributes = get(p, "attributes");
    if ((mode != 4 && mode != 5 && mode != 6) || !get(attributes, "POSITION"))
        return 0; // points and lines: not drawn
    primitive_t q;
    memset(&q, 0, sizeof q);
    memcpy(q.world, world, sizeof(mat4));
    if (!(q.positions = read_floats(l, (int)num(get(attributes, "POSITION"), -1), 3, &q.vertex_count)))
        return -1;
    uint32_t n;
    if (get(attributes, "NORMAL") && (q.normals = read_floats(l, (int)num(get(attributes, "NORMAL"), -1), 3, &n)) &&
        n != q.vertex_count)
    {
        free(q.normals);
        q.normals = NULL;
    }

    // Material: glTF's default, white, if none
    const json_t *m = at(get(&l->root, "materials"), (int)num(get(p, "material"), -1));
    const json_t *pbr = get(m, "pbrMetallicRoughness");
    for (int k = 0; k < 4; k++)
        q.base[k] = (float)num(at(get(pbr, "baseColorFactor"), k), 1.0);
    for (int k = 0; k < 3; k++)
        q.emissive[k] = (float)num(at(get(m, "emissiveFactor"), k), 0.0);
    q.blended = str(get(m, "alphaMode")) && strcmp(str(get(m, "alphaMode")), "BLEND") == 0;
    q.double_sided = get(m, "doubleSided") && get(m, "doubleSided")->type == J_TRUE;
    q.texture = -1;
    const json_t *info = get(pbr, "baseColorTexture");
    if (info)
    {
        char name[32];
        snprintf(name, sizeof name, "TEXCOORD_%d", (int)num(get(info, "texCoord"), 0));
        if (get(attributes, name))
        {
            q.texture = load_texture(l, (int)num(get(info, "index"), -1));
            if (q.texture >= 0 && (!(q.uvs = read_floats(l, (int)num(get(attributes, name), -1), 2, &n)) ||
                                   n != q.vertex_count))
                q.texture = -1;
        }
    }

    // Indices, as triangles
    uint32_t *indices, count;
    if (get(p, "indices"))
    {
        if (!(indices = read_indices(l, (int)num(get(p, "indices"), -1), &count)))
            return -1;
    }
    else
    {
        count = q.vertex_count;
        indices = malloc(((size_t)count + 1) * sizeof *indices);
        for (uint32_t i = 0; i < count; i++)
            indices[i] = i;
    }
    for (uint32_t i = 0; i < count; i++)
        if (indices[i] >= q.vertex_count)
            return fail(l, "an index past its primitive's vertices");
    if (mode == 4)
    {
        q.indices = indices;
        q.index_count = count / 3 * 3;
    }
    else
    {
        const uint32_t triangles = count >= 3 ? count - 2 : 0;
        q.indices = malloc(((size_t)triangles * 3 + 1) * sizeof *q.indices);
        for (uint32_t t = 0; t < triangles; t++)
        {
            uint32_t *out = &q.indices[3 * t];
            if (mode == 6) // fan
            {
                out[0] = indices[0];
                out[1] = indices[t + 1];
                out[2] = indices[t + 2];
            }
            else // strip, every other triangle turned back
            {
                out[0] = indices[t + (t & 1)];
                out[1] = indices[t + 1 - (t & 1)];
                out[2] = indices[t + 2];
            }
        }
        q.index_count = triangles * 3;
        free(indices);
    }
    if (l->primitive_count == l->primitive_cap)
    {
        l->primitive_cap = l->primitive_cap ? 2 * l->primitive_cap : 16;
        l->primitives = realloc(l->primitives, (size_t)l->primitive_cap * sizeof *l->primitives);
    }
    l->primitives[l->primitive_count++] = q;
    return 0;
}

static int visit(loader_t *l, int index, const mat4 parent, int depth)
{
    const json_t *node = at(get(&l->root, "nodes"), index);
    if (!node || depth > 64)
        return fail(l, "node %d: no such node, or nodes in a cycle", index);
    mat4 local, world;
    const json_t *matrix = get(node, "matrix");
    if (length(matrix) == 16)
        for (int i = 0; i < 16; i++)
            local[i] = (float)num(at(matrix, i), 0);
    else
    {
        float t[3], r[4], s[3];
        for (int k = 0; k < 3; k++)
        {
            t[k] = (float)num(at(get(node, "translation"), k), 0);
            s[k] = (float)num(at(get(node, "scale"), k), 1);
        }
        for (int k = 0; k < 4; k++)
            r[k] = (float)num(at(get(node, "rotation"), k), k == 3);
        mat_trs(local, t, r, s);
    }
    mat_mul(world, parent, local);

    const json_t *mesh = at(get(&l->root, "meshes"), (int)num(get(node, "mesh"), -1));
    for (int i = 0; i < length(get(mesh, "primitives")); i++)
        if (add_primitive(l, at(get(mesh, "primitives"), i), world) != 0)
            return -1;
    const json_t *camera = at(get(&l->root, "cameras"), (int)num(get(node, "camera"), -1));
    if (camera && !l->camera)
    {
        l->camera = 1;
        memcpy(l->camera_world, world, sizeof(mat4));
        const json_t *p = get(camera, "perspective"), *o = get(camera, "orthographic");
        if (o)
        {
            l->orthographic = 1;
            l->xmag = (float)num(get(o, "xmag"), 1);
            l->ymag = (float)num(get(o, "ymag"), 1);
            l->znear = (float)num(get(o, "znear"), 0.01);
            l->zfar = (float)num(get(o, "zfar"), 100);
        }
        else
        {
            l->yfov = (float)num(get(p, "yfov"), 0.8);
            l->aspect = (float)num(get(p, "aspectRatio"), 0);
            l->znear = (float)num(get(p, "znear"), 0.01);
            l->zfar = (float)num(get(p, "zfar"), l->znear * 1e4); // none: infinite, far enough here
        }
    }
    const json_t *children = get(node, "children");
    for (int i = 0; i < length(children); i++)
        if (visit(l, (int)num(at(children, i), -1), world, depth + 1) != 0)
            return -1;
    return 0;
}

static uint8_t to_byte(float f)
{
    return (uint8_t)(f <= 0.0f ? 0 : f >= 1.0f ? 255 : f * 255.0f + 0.5f);
}

// The primitives, transformed and lit, into the scene: opaque ones first, then blended
static void build_scene(loader_t *l, int width, int height, float angle)
{
    float lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
    uint32_t vertices = 0, indices = 0;
    for (int i = 0; i < l->primitive_count; i++)
    {
        const primitive_t *p = &l->primitives[i];
        vertices += p->vertex_count;
        indices += p->index_count;
        for (uint32_t v = 0; v < p->vertex_count; v++)
        {
            const float in[4] = {p->positions[3 * v], p->positions[3 * v + 1], p->positions[3 * v + 2], 1};
            float w[4];
            transform(p->world, in, w);
            for (int k = 0; k < 3; k++)
            {
                lo[k] = fminf(lo[k], w[k]);
                hi[k] = fmaxf(hi[k], w[k]);
            }
        }
    }
    float center[3], span = 1e-6f;
    for (int k = 0; k < 3; k++)
    {
        center[k] = 0.5f * (lo[k] + hi[k]);
        span = fmaxf(span, hi[k] - lo[k]);
    }

    mat4 view, projection, model, m;
    if (l->camera)
        mat_inverse(view, l->camera_world);
    else
    {
        mat_translate(view, -center[0], -center[1], -(center[2] + 1.5f * span));
        l->yfov = (float)(45.0 * M_PI / 180.0);
        l->znear = span * 0.01f;
        l->zfar = span * 100.0f;
    }
    memset(projection, 0, sizeof projection);
    const float n = l->znear, f = l->zfar;
    if (l->orthographic)
    {
        projection[0] = 1.0f / l->xmag;
        projection[5] = 1.0f / l->ymag;
        projection[10] = -2.0f / (f - n);
        projection[14] = -(f + n) / (f - n);
        projection[15] = 1.0f;
    }
    else
    {
        const float c = 1.0f / tanf(0.5f * l->yfov), aspect = l->aspect > 0 ? l->aspect : (float)width / height;
        projection[0] = c / aspect;
        projection[5] = c;
        projection[10] = (f + n) / (n - f);
        projection[11] = -1.0f;
        projection[14] = 2.0f * f * n / (n - f);
    }
    // Turned about the vertical axis through its center
    const float a = angle * (float)(M_PI / 180.0);
    mat_translate(model, center[0], center[1], center[2]);
    mat4 turn;
    mat_identity(turn);
    turn[0] = turn[10] = cosf(a);
    turn[2] = -sinf(a);
    turn[8] = sinf(a);
    mat_mul(model, model, turn);
    mat_translate(m, -center[0], -center[1], -center[2]);
    mat_mul(model, model, m);

    scene_t *s = l->scene;
    s->vertices = malloc(((size_t)vertices + 1) * sizeof *s->vertices);
    s->indices = malloc(((size_t)indices + 1) * sizeof *s->indices);
    s->draws = malloc(((size_t)l->primitive_count + 1) * sizeof *s->draws);
    const float light[3] = {0.57735f, 0.57735f, 0.57735f}; // eye space
    for (int pass = 0; pass < 2; pass++)
        for (int i = 0; i < l->primitive_count; i++)
        {
            const primitive_t *p = &l->primitives[i];
            if (p->blended != pass || p->index_count == 0)
                continue;
            mat4 mv, mvp, normal;
            mat_mul(mv, view, model);
            mat_mul(mv, mv, p->world);
            mat_mul(mvp, projection, mv);
            if (mat_inverse(normal, mv) != 0)
                mat_identity(normal);
            const float det = mv[0] * (mv[5] * mv[10] - mv[9] * mv[6]) - mv[4] * (mv[1] * mv[10] - mv[9] * mv[2]) +
                              mv[8] * (mv[1] * mv[6] - mv[5] * mv[2]);
            const uint32_t base = s->vertex_count;
            for (uint32_t v = 0; v < p->vertex_count; v++)
            {
                scene_vertex_t *out = &s->vertices[base + v];
                const float in[4] = {p->positions[3 * v], p->positions[3 * v + 1], p->positions[3 * v + 2], 1};
                transform(mvp, in, out->position);
                float shade = 1.0f;
                if (p->normals)
                {
                    // The inverse's transpose: rows of the inverse as columns
                    const float *nv = &p->normals[3 * v];
                    float e[3];
                    for (int k = 0; k < 3; k++)
                        e[k] = normal[k * 4] * nv[0] + normal[k * 4 + 1] * nv[1] + normal[k * 4 + 2] * nv[2];
                    const float len = sqrtf(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
                    float d = len > 0 ? (e[0] * light[0] + e[1] * light[1] + e[2] * light[2]) / len : 1.0f;
                    if (p->double_sided)
                        d = fabsf(d);
                    shade = 0.2f + 0.8f * fmaxf(d, 0.0f);
                }
                for (int k = 0; k < 3; k++)
                    out->color[k] = to_byte(p->base[k] * shade + p->emissive[k]);
                out->color[3] = to_byte(p->blended ? p->base[3] : 1.0f);
                out->uv[0] = p->uvs ? p->uvs[2 * v] : 0.0f;
                out->uv[1] = p->uvs ? p->uvs[2 * v + 1] : 0.0f;
            }
            scene_draw_t *d = &s->draws[s->draw_count++];
            *d = (scene_draw_t){s->index_count, p->index_count, p->texture, p->blended, p->double_sided, det < 0};
            for (uint32_t k = 0; k < p->index_count; k++)
                s->indices[s->index_count++] = base + p->indices[k];
            s->vertex_count += p->vertex_count;
        }
}

int scene_load(const char *path, int width, int height, float angle, scene_t *out, char *error, size_t error_size)
{
    memset(out, 0, sizeof *out);
    loader_t l;
    memset(&l, 0, sizeof l);
    l.scene = out;
    l.error = error;
    l.error_size = error_size;
    const char *slash = strrchr(path, '/');
    snprintf(l.dir, sizeof l.dir, "%.*s", slash ? (int)(slash - path + 1) : 0, path);

    size_t size;
    uint8_t *file = read_file(path, &size);
    if (!file)
        return fail(&l, "%s: cannot read it", path);
    // A .glb: its JSON chunk, then its binary one; else the file is JSON
    char *json = (char *)file;
    blob_t bin = {0};
    uint32_t header[5];
    if (size >= sizeof header && memcmp(file, "glTF", 4) == 0)
    {
        memcpy(header, file, sizeof header);
        if (header[4] != 0x4E4F534Au || header[3] > size - sizeof header) // "JSON"
            return fail(&l, "%s: not a glTF binary", path);
        json = malloc(header[3] + 1);
        memcpy(json, file + sizeof header, header[3]);
        json[header[3]] = '\0';
        const size_t next = sizeof header + header[3];
        uint32_t chunk[2];
        if (next + 8 <= size)
        {
            memcpy(chunk, file + next, 8);
            if (chunk[1] == 0x004E4942u && chunk[0] <= size - next - 8) // "BIN"
                bin = (blob_t){file + next + 8, chunk[0], NULL};
        }
    }
    if (!parse_value(json, &l.root, 0) || l.root.type != J_OBJECT)
        return fail(&l, "%s: not glTF's JSON", path);

    const json_t *buffers = get(&l.root, "buffers");
    l.buffer_count = length(buffers);
    l.buffers = calloc((size_t)l.buffer_count + 1, sizeof *l.buffers);
    for (int i = 0; i < l.buffer_count; i++)
    {
        const char *uri = str(get(at(buffers, i), "uri"));
        if (uri ? load_uri(&l, uri, &l.buffers[i]) != 0 : !bin.data)
            return uri ? -1 : fail(&l, "buffer %d has no data", i);
        if (!uri)
            l.buffers[i] = bin;
        if (num(get(at(buffers, i), "byteLength"), 0) > l.buffers[i].size)
            return fail(&l, "buffer %d shorter than its byteLength", i);
    }
    const int textures = length(get(&l.root, "textures"));
    l.texture_slots = malloc(((size_t)textures + 1) * sizeof *l.texture_slots);
    for (int i = 0; i < textures; i++)
        l.texture_slots[i] = -1;

    mat4 identity;
    mat_identity(identity);
    const json_t *scenes = get(&l.root, "scenes");
    const json_t *scene = at(scenes, (int)num(get(&l.root, "scene"), 0));
    if (scene)
    {
        for (int i = 0; i < length(get(scene, "nodes")); i++)
            if (visit(&l, (int)num(at(get(scene, "nodes"), i), -1), identity, 0) != 0)
                return -1;
    }
    else // no scene: every node no other has as a child
    {
        const json_t *nodes = get(&l.root, "nodes");
        char *child = calloc((size_t)length(nodes) + 1, 1);
        for (int i = 0; i < length(nodes); i++)
            for (int k = 0; k < length(get(at(nodes, i), "children")); k++)
            {
                const int c = (int)num(at(get(at(nodes, i), "children"), k), -1);
                if (c >= 0 && c < length(nodes))
                    child[c] = 1;
            }
        for (int i = 0; i < length(nodes); i++)
            if (!child[i] && visit(&l, i, identity, 0) != 0)
                return -1;
        free(child);
    }
    if (l.primitive_count == 0)
        return fail(&l, "%s: no triangles", path);
    build_scene(&l, width, height, angle);
    out->own_camera = l.camera;

    for (int i = 0; i < l.primitive_count; i++)
    {
        free(l.primitives[i].positions);
        free(l.primitives[i].normals);
        free(l.primitives[i].uvs);
        free(l.primitives[i].indices);
    }
    free(l.primitives);
    for (int i = 0; i < l.buffer_count; i++)
        free(l.buffers[i].owned);
    free(l.buffers);
    free(l.texture_slots);
    json_free(&l.root);
    if (json != (char *)file)
        free(json);
    free(file);
    return 0;
}
