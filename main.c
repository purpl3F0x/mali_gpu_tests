// mali_bench: the ZCU104's Mali-400 MP2 (Ubuntu, OpenGL ES 2.0 on EGL: libmali or Lima) drawing a
// glTF scene, its raster and write-back timed. The vertices are transformed and lit once on the CPU
// (gltf.c); the Mali's vertex shader only passes them on. Into a width x height FBO: the base color
// times its texture (nearest texels), the depth test, back faces culled but where double sided,
// BLEND materials blended after the rest; depth discarded rather than written back.
//
// Frames timed (glFinish), each kind its own run:
//   full        the frame
//   scissored   its draws scissored to 1 x 1: the vertices and binning, no raster
//   clear       glClear alone: every tile cleared and written back
//   clear 16    glClear of a 16 x 16 FBO: one tile, the fixed cost of a frame
// Their differences estimate the raster and the write-back. As root, the kernel's GPU job events
// (tracefs) time the GP (vertices, binning) and PP (raster, write-back) jobs themselves: the Mali
// driver's mali_timeline_event (built with CONFIG_MALI400_PROFILING), or Lima's drm scheduler events.
//
// Usage: mali_bench [-n FRAMES] [-a ANGLE] [-s WxH] [-o PNG] [-t TXT] [-T] MODEL
// Writes MODEL_mali_times.txt (every frame's times) and MODEL_mali.png (the frame) by default.

#define _GNU_SOURCE
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <glob.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "gltf.h"
#include "stb_image_write.h"

#ifndef EGL_PLATFORM_SURFACELESS_MESA
#define EGL_PLATFORM_SURFACELESS_MESA 0x31DD
#endif

#define WARMUP 10
#define TINY 16 // a Mali-400 tile

enum
{
    FULL,
    SCISSORED,
    CLEARED,
    CLEARED_TINY,
    MODES
};
static const char *mode_names[MODES] = {"full frame", "draws scissored to 1x1", "clear only",
                                        "clear only, 16x16"};

static int width = 1280, height = 720;
static scene_t scene;
static GLuint programs[2]; // untextured, textured
static GLuint fbo, tiny_fbo;
static GLuint *textures;
// A draw's triangles within one window of at most 65536 vertices, 16-bit indices into it
typedef struct
{
    int draw;
    uint32_t first_index, index_count;
    size_t vertex_offset; // the window's first vertex, in bytes
} batch_t;
static batch_t *batches;
static int batch_count;
static PFNGLDISCARDFRAMEBUFFEREXTPROC discard;

static void die(const char *format, ...);

static void check_gl(const char *what)
{
    const GLenum e = glGetError();
    if (e != GL_NO_ERROR)
        die("GL error 0x%04x %s%s", e, what, e == GL_OUT_OF_MEMORY ? " (out of memory)" : "");
}

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static void die(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    fprintf(stderr, "mali_bench: ");
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    va_end(args);
    exit(1);
}

static int has_extension(const char *name)
{
    const char *all = (const char *)glGetString(GL_EXTENSIONS);
    const size_t n = strlen(name);
    for (const char *p = all; p && (p = strstr(p, name)); p += n)
        if ((p == all || p[-1] == ' ') && (p[n] == ' ' || p[n] == '\0'))
            return 1;
    return 0;
}

// ---------------------------------------------------------------------------------------------
// EGL and GL

// The display, initialized; else EGL_NO_DISPLAY, why told
static EGLDisplay try_display(const char *what, EGLDisplay display)
{
    EGLint major, minor;
    if (display == EGL_NO_DISPLAY)
        fprintf(stderr, "  %s: no display (0x%x)\n", what, eglGetError());
    else if (!eglInitialize(display, &major, &minor))
        fprintf(stderr, "  %s: eglInitialize failed (0x%x)\n", what, eglGetError());
    else
        return display;
    return EGL_NO_DISPLAY;
}

// A display without a window system, none opened: Mesa's surfaceless one or an EGL device (Lima),
// else the default one with DISPLAY and WAYLAND_DISPLAY unset (libmali's headless or fbdev backend)
static EGLDisplay open_display(void)
{
    EGLDisplay display = EGL_NO_DISPLAY;
    const char *client = eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS);
    PFNEGLGETPLATFORMDISPLAYEXTPROC get =
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    if (get && client && strstr(client, "EGL_MESA_platform_surfaceless"))
        display = try_display("surfaceless", get(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL));
#ifdef EGL_EXT_device_enumeration
    PFNEGLQUERYDEVICESEXTPROC query = (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
    EGLDeviceEXT devices[8];
    EGLint count = 0;
    if (display == EGL_NO_DISPLAY && get && query && client && strstr(client, "EGL_EXT_platform_device") &&
        query(8, devices, &count))
        for (EGLint i = 0; i < count && display == EGL_NO_DISPLAY; i++)
        {
            char what[32];
            snprintf(what, sizeof what, "EGL device %d", (int)i);
            display = try_display(what, get(EGL_PLATFORM_DEVICE_EXT, devices[i], NULL));
        }
#endif
    if (display == EGL_NO_DISPLAY)
    {
        unsetenv("DISPLAY");
        unsetenv("WAYLAND_DISPLAY");
        display = try_display("default display, no window system", eglGetDisplay(EGL_DEFAULT_DISPLAY));
    }
    if (display == EGL_NO_DISPLAY)
    {
        Dl_info library;
        fprintf(stderr, "  EGL library: %s\n",
                dladdr((void *)eglGetDisplay, &library) && library.dli_fname ? library.dli_fname : "unknown");
        fprintf(stderr, "  EGL client extensions: %s\n", client ? client : "(none)");
        die("no EGL display without a window system (libmali's X11 build has none: its headless or fbdev "
            "build, or Lima, does)");
    }
    return display;
}

static void egl_init(void)
{
    EGLDisplay display = open_display();
    eglBindAPI(EGL_OPENGL_ES_API);
    EGLint attributes[] = {EGL_SURFACE_TYPE,
                           EGL_PBUFFER_BIT,
                           EGL_RENDERABLE_TYPE,
                           EGL_OPENGL_ES2_BIT,
                           EGL_RED_SIZE,
                           8,
                           EGL_GREEN_SIZE,
                           8,
                           EGL_BLUE_SIZE,
                           8,
                           EGL_NONE};
    EGLConfig config;
    EGLint configs = 0;
    if (!eglChooseConfig(display, attributes, &config, 1, &configs) || configs == 0)
    {
        attributes[1] = 0; // any surface type: surfaceless
        if (!eglChooseConfig(display, attributes, &config, 1, &configs) || configs == 0)
            die("no OpenGL ES 2.0 EGL config");
    }
    const EGLint context_attributes[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};
    EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, context_attributes);
    if (context == EGL_NO_CONTEXT)
        die("eglCreateContext failed (0x%x)", eglGetError());
    const EGLint pbuffer_attributes[] = {EGL_WIDTH, 16, EGL_HEIGHT, 16, EGL_NONE};
    EGLSurface surface = eglCreatePbufferSurface(display, config, pbuffer_attributes);
    if (!eglMakeCurrent(display, surface, surface, context))
        die("eglMakeCurrent failed (0x%x)", eglGetError());
}

static GLuint compile(GLenum type, const char *source)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    GLint ok;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char log[1024];
        glGetShaderInfoLog(shader, sizeof log, NULL, log);
        die("shader: %s\n%s", log, source);
    }
    return shader;
}

// Pass-through vertex shaders: the positions are clip space already
static void make_programs(void)
{
    for (int textured = 0; textured < 2; textured++)
    {
        char vs[512], fs[512];
        snprintf(vs, sizeof vs,
                 "attribute vec4 position;\n"
                 "attribute vec4 color;\n"
                 "attribute vec2 uv;\n"
                 "varying lowp vec4 v_color;\n"
                 "%s"
                 "void main() { gl_Position = position; v_color = color;%s }\n",
                 textured ? "varying mediump vec2 v_uv;\n" : "", textured ? " v_uv = uv;" : "");
        snprintf(fs, sizeof fs,
                 "precision mediump float;\n"
                 "varying lowp vec4 v_color;\n"
                 "%s"
                 "void main() { gl_FragColor = v_color%s; }\n",
                 textured ? "varying mediump vec2 v_uv;\nuniform sampler2D tex;\n" : "",
                 textured ? " * texture2D(tex, v_uv)" : "");
        GLuint p = glCreateProgram();
        glAttachShader(p, compile(GL_VERTEX_SHADER, vs));
        glAttachShader(p, compile(GL_FRAGMENT_SHADER, fs));
        glBindAttribLocation(p, 0, "position");
        glBindAttribLocation(p, 1, "color");
        glBindAttribLocation(p, 2, "uv");
        glLinkProgram(p);
        GLint ok;
        glGetProgramiv(p, GL_LINK_STATUS, &ok);
        if (!ok)
            die("program %d did not link", textured);
        programs[textured] = p;
    }
}

static GLuint make_fbo(int w, int h, GLenum depth_format)
{
    GLuint color, depth, f;
    glGenTextures(1, &color);
    glBindTexture(GL_TEXTURE_2D, color);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenRenderbuffers(1, &depth);
    glBindRenderbuffer(GL_RENDERBUFFER, depth);
    glRenderbufferStorage(GL_RENDERBUFFER, depth_format, w, h);
    glGenFramebuffers(1, &f);
    glBindFramebuffer(GL_FRAMEBUFFER, f);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);
    const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE)
        die("FBO %d x %d incomplete (0x%x)", w, h, status);
    return f;
}

static void upload_textures(void)
{
    textures = calloc((size_t)scene.texture_count + 1, sizeof *textures);
    for (int i = 0; i < scene.texture_count; i++)
    {
        const scene_texture_t *t = &scene.textures[i];
        glGenTextures(1, &textures[i]);
        glBindTexture(GL_TEXTURE_2D, textures[i]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, t->repeat_s ? GL_REPEAT : GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, t->repeat_t ? GL_REPEAT : GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, t->width, t->height, 0, GL_RGBA, GL_UNSIGNED_BYTE, t->rgba);
        char what[64];
        snprintf(what, sizeof what, "uploading texture %d (%d x %d)", i, t->width, t->height);
        check_gl(what);
    }
}

static void point_attributes(size_t offset)
{
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(scene_vertex_t),
                          (const void *)(offset + offsetof(scene_vertex_t, position)));
    glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(scene_vertex_t),
                          (const void *)(offset + offsetof(scene_vertex_t, color)));
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(scene_vertex_t),
                          (const void *)(offset + offsetof(scene_vertex_t, uv)));
}

// The geometry in windows of at most 65536 vertices with 16-bit indices: Mali-400 has no
// 32-bit indices, and unindexed costs it 3x the vertices. ES 2 has no base vertex, so each
// batch points the attributes at its window. Returns the bytes uploaded.
static size_t upload_geometry(void)
{
    const uint32_t n = scene.index_count;
    uint32_t *owner = calloc((size_t)scene.vertex_count + 1, 4), *local = malloc((size_t)scene.vertex_count * 4 + 4);
    uint16_t *indices = malloc((size_t)n * 2 + 2);
    size_t capacity = (size_t)scene.vertex_count + 1, used = 0, window = 0, batch_capacity = 16;
    scene_vertex_t *vertices = malloc(capacity * sizeof *vertices);
    batches = malloc(batch_capacity * sizeof *batches);
    if (!owner || !local || !indices || !vertices || !batches)
        die("out of memory");
    uint32_t chunk = 1;
    batch_count = 0;
    for (int d = 0; d < scene.draw_count; d++)
    {
        const scene_draw_t *draw = &scene.draws[d];
        batch_t *b = NULL;
        for (uint32_t t = 0; t + 3 <= draw->index_count; t += 3)
        {
            const uint32_t at = draw->first_index + t, *tri = &scene.indices[at];
            const size_t fresh = (owner[tri[0]] != chunk) + (owner[tri[1]] != chunk) + (owner[tri[2]] != chunk);
            if (used - window + fresh > 65536)
            {
                chunk++;
                window = used;
                b = NULL;
            }
            if (!b)
            {
                if (batch_count == (int)batch_capacity &&
                    !(batches = realloc(batches, (batch_capacity *= 2) * sizeof *batches)))
                    die("out of memory");
                b = &batches[batch_count++];
                *b = (batch_t){d, at, 0, window * sizeof(scene_vertex_t)};
            }
            for (int k = 0; k < 3; k++)
            {
                const uint32_t v = tri[k];
                if (owner[v] != chunk)
                {
                    if (used == capacity && !(vertices = realloc(vertices, (capacity *= 2) * sizeof *vertices)))
                        die("out of memory");
                    owner[v] = chunk;
                    local[v] = (uint32_t)(used - window);
                    vertices[used++] = scene.vertices[v];
                }
                indices[at + k] = (uint16_t)local[v];
            }
            b->index_count += 3;
        }
    }
    GLuint buffers[2];
    glGenBuffers(2, buffers);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers[1]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)n * 2, indices, GL_STATIC_DRAW);
    check_gl("uploading the indices");
    glBindBuffer(GL_ARRAY_BUFFER, buffers[0]);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(used * sizeof *vertices), vertices, GL_STATIC_DRAW);
    check_gl("uploading the vertices");
    for (GLuint a = 0; a < 3; a++)
        glEnableVertexAttribArray(a);
    free(owner);
    free(local);
    free(indices);
    free(vertices);
    return (size_t)n * 2 + used * sizeof(scene_vertex_t);
}

static void render(int mode)
{
    const int tiny = mode == CLEARED_TINY;
    glBindFramebuffer(GL_FRAMEBUFFER, tiny ? tiny_fbo : fbo);
    glViewport(0, 0, tiny ? TINY : width, tiny ? TINY : height);
    glDisable(GL_SCISSOR_TEST);
    glDepthMask(GL_TRUE);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    if (mode == FULL || mode == SCISSORED)
    {
        if (mode == SCISSORED)
        {
            glEnable(GL_SCISSOR_TEST);
            glScissor(0, 0, 1, 1);
        }
        int program = -1, blended = -1, culled = -1, clockwise = -1;
        GLuint texture = 0;
        size_t offset = SIZE_MAX;
        for (int i = 0; i < batch_count; i++)
        {
            const batch_t *b = &batches[i];
            if (b->vertex_offset != offset)
                point_attributes(offset = b->vertex_offset);
            const scene_draw_t *d = &scene.draws[b->draw];
            const int textured = d->texture >= 0;
            if (textured != program)
                glUseProgram(programs[program = textured]);
            if (textured && textures[d->texture] != texture)
                glBindTexture(GL_TEXTURE_2D, texture = textures[d->texture]);
            if (d->blended != blended)
            {
                if ((blended = d->blended))
                {
                    glEnable(GL_BLEND);
                    glDepthMask(GL_FALSE);
                }
                else
                {
                    glDisable(GL_BLEND);
                    glDepthMask(GL_TRUE);
                }
            }
            if ((!d->double_sided) != culled)
            {
                if ((culled = !d->double_sided))
                    glEnable(GL_CULL_FACE);
                else
                    glDisable(GL_CULL_FACE);
            }
            if (d->clockwise != clockwise)
                glFrontFace((clockwise = d->clockwise) ? GL_CW : GL_CCW);
            glDrawElements(GL_TRIANGLES, (GLsizei)b->index_count, GL_UNSIGNED_SHORT,
                           (const void *)((size_t)b->first_index * 2));
        }
    }
    if (discard)
    {
        const GLenum depth = GL_DEPTH_ATTACHMENT;
        discard(GL_FRAMEBUFFER, 1, &depth);
    }
}

// ---------------------------------------------------------------------------------------------
// The kernel's GPU job events, by tracefs

enum
{
    TRACE_NONE,
    TRACE_MALI, // mali_timeline_event: the Mali driver's, with profiling
    TRACE_DRM,  // gpu_scheduler's drm_run_job and drm_sched_process_job: Lima's
};
static int trace_kind;
static const char *trace_dir;
static int marker_fd = -1;

static int trace_write(const char *name, const char *value)
{
    char path[256];
    snprintf(path, sizeof path, "%s/%s", trace_dir, name);
    const int fd = open(path, O_WRONLY | O_TRUNC);
    if (fd < 0)
        return -1;
    const ssize_t n = write(fd, value, strlen(value));
    close(fd);
    return n == (ssize_t)strlen(value) ? 0 : -1;
}

static int trace_has(const char *name)
{
    char path[256];
    snprintf(path, sizeof path, "%s/%s", trace_dir, name);
    return access(path, F_OK) == 0;
}

static const char *const mali_events[] = {"events/mali/mali_timeline_event/enable", NULL};
static const char *const drm_events[] = {"events/gpu_scheduler/drm_run_job/enable",
                                         "events/gpu_scheduler/drm_sched_process_job/enable", NULL};

static const char *const *trace_events(void)
{
    return trace_kind == TRACE_MALI ? mali_events : drm_events;
}

static void trace_open(void)
{
    static const char *const dirs[] = {"/sys/kernel/tracing", "/sys/kernel/debug/tracing"};
    for (size_t i = 0; i < sizeof dirs / sizeof dirs[0] && !trace_dir; i++)
    {
        char path[256];
        snprintf(path, sizeof path, "%s/trace_marker", dirs[i]);
        if (access(path, W_OK) == 0)
            trace_dir = dirs[i];
    }
    if (!trace_dir)
    {
        printf("GPU job events: no tracefs to write (run as root for them)\n");
        return;
    }
    if (trace_has("events/mali/mali_timeline_event"))
        trace_kind = TRACE_MALI;
    else if (trace_has("events/gpu_scheduler/drm_run_job") && trace_has("events/gpu_scheduler/drm_sched_process_job"))
        trace_kind = TRACE_DRM;
    else
    {
        printf("GPU job events: none in this kernel (the Mali driver's need CONFIG_MALI400_PROFILING)\n");
        return;
    }
    trace_write("tracing_on", "0");
    trace_write("buffer_size_kb", "8192");
    for (const char *const *e = trace_events(); *e; e++)
        if (trace_write(*e, "1") != 0)
        {
            printf("GPU job events: cannot enable %s\n", *e);
            trace_kind = TRACE_NONE;
            return;
        }
    char path[256];
    snprintf(path, sizeof path, "%s/trace_marker", trace_dir);
    marker_fd = open(path, O_WRONLY);
    if (marker_fd < 0)
        trace_kind = TRACE_NONE;
    printf("GPU job events: %s\n", trace_kind == TRACE_MALI ? "mali_timeline_event (Mali driver)"
                                   : trace_kind == TRACE_DRM ? "drm scheduler (Lima)"
                                                             : "none");
}

static void trace_close(void)
{
    if (trace_kind == TRACE_NONE)
        return;
    for (const char *const *e = trace_events(); *e; e++)
        trace_write(*e, "0");
    trace_write("tracing_on", "1");
    close(marker_fd);
}

static void trace_start(void)
{
    trace_write("tracing_on", "0");
    trace_write("trace", "");
    trace_write("tracing_on", "1");
}

static void trace_mark(const char *what, int frame)
{
    dprintf(marker_fd, "mali_bench %s %d\n", what, frame);
}

// A trace line's timestamp, event and payload: "  task-pid [cpu] flags  123.456789: event: payload"
static int trace_split(char *line, double *t, char **event, char **payload)
{
    for (char *c = strstr(line, ": "); c; c = strstr(c + 1, ": "))
    {
        char *s = c;
        while (s > line && s[-1] != ' ')
            s--;
        char *end;
        const double v = strtod(s, &end);
        if (end != c || end == s || !memchr(s, '.', (size_t)(c - s)))
            continue;
        char *e = c + 2, *colon = strchr(e, ':');
        if (!colon)
            return 0;
        *colon = '\0';
        *t = v;
        *event = e;
        *payload = colon + 1;
        (*payload)[strcspn(*payload, "\n")] = '\0';
        return 1;
    }
    return 0;
}

// Each frame's GP and PP job times (s) from the trace, between its markers: the first job start to
// the last end (PP: of all cores); NAN where none
static void trace_read(int frames, double *gp, double *pp)
{
    for (int i = 0; i < frames; i++)
        gp[i] = pp[i] = NAN;
    trace_write("tracing_on", "0");
    char path[256];
    snprintf(path, sizeof path, "%s/trace", trace_dir);
    FILE *in = fopen(path, "r");
    if (!in)
        return;
    char line[1024];
    int frame = -1, pending = 0; // DRM: the job run last, 1 GP, 2 PP
    double gp0 = NAN, gp1 = NAN, pp0 = NAN, pp1 = NAN;
    while (fgets(line, sizeof line, in))
    {
        double t;
        char *event, *payload;
        if (line[0] == '#' || !trace_split(line, &t, &event, &payload))
            continue;
        int n;
        if (strcmp(event, "tracing_mark_write") == 0)
        {
            if (sscanf(payload, " mali_bench begin %d", &n) == 1)
            {
                frame = n;
                gp0 = gp1 = pp0 = pp1 = NAN;
                pending = 0;
            }
            else if (sscanf(payload, " mali_bench end %d", &n) == 1 && n == frame && n >= 0 && n < frames)
            {
                gp[n] = gp1 - gp0;
                pp[n] = pp1 - pp0;
                frame = -1;
            }
        }
        else if (frame < 0)
            continue;
        else if (strcmp(event, "mali_timeline_event") == 0)
        {
            // mali_utgard_profiling_events.h: type << 24 (1 start, 2 stop), channel << 16 (1 GP0, 5 to
            // 12 PP0 to PP7)
            const char *e = strstr(payload, "event=");
            if (!e)
                continue;
            const unsigned long id = strtoul(e + 6, NULL, 10);
            const unsigned type = id >> 24 & 0xF, channel = id >> 16 & 0xFF;
            const int is_gp = channel == 1, is_pp = channel >= 5 && channel <= 12;
            if (type == 1 && is_gp && isnan(gp0))
                gp0 = t;
            if (type == 1 && is_pp && isnan(pp0))
                pp0 = t;
            if (type == 2 && is_gp)
                gp1 = t;
            if (type == 2 && is_pp)
                pp1 = t;
        }
        else if (strcmp(event, "drm_run_job") == 0)
        {
            // Lima's PP job waits for its GP job: they run one after the other
            const char *r = strstr(payload, "ring=");
            if (r && strncmp(r + 5, "gp", 2) == 0)
            {
                if (isnan(gp0))
                    gp0 = t;
                pending = 1;
            }
            else if (r && strncmp(r + 5, "pp", 2) == 0)
            {
                if (isnan(pp0))
                    pp0 = t;
                pending = 2;
            }
        }
        else if (strcmp(event, "drm_sched_process_job") == 0)
        {
            if (pending == 1)
                gp1 = t;
            else if (pending == 2)
                pp1 = t;
            pending = 0;
        }
    }
    fclose(in);
    trace_write("tracing_on", "1");
}

// ---------------------------------------------------------------------------------------------
// Runs and their statistics

typedef struct
{
    double median, min, mean;
    int n;
} stats_t;

static int compare(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

// Of the values not NAN, in ms
static stats_t stats(const double *values, int count)
{
    double *v = malloc((size_t)count * sizeof *v);
    stats_t s = {NAN, NAN, NAN, 0};
    double sum = 0;
    for (int i = 0; i < count; i++)
        if (!isnan(values[i]))
        {
            v[s.n++] = values[i] * 1e3;
            sum += values[i] * 1e3;
        }
    if (s.n)
    {
        qsort(v, (size_t)s.n, sizeof *v, compare);
        s.median = s.n % 2 ? v[s.n / 2] : 0.5 * (v[s.n / 2 - 1] + v[s.n / 2]);
        s.min = v[0];
        s.mean = sum / s.n;
    }
    free(v);
    return s;
}

typedef struct
{
    stats_t total, submit, gp, pp;
    double *total_s, *submit_s, *gp_s, *pp_s; // each frame's, in s (NAN none)
} result_t;

static result_t run(int mode, int frames)
{
    double *total = malloc((size_t)frames * sizeof(double)), *submit = malloc((size_t)frames * sizeof(double));
    double *gp = malloc((size_t)frames * sizeof(double)), *pp = malloc((size_t)frames * sizeof(double));
    if (!total || !submit || !gp || !pp)
        die("out of memory");
    const int tracing = trace_kind != TRACE_NONE;
    if (tracing)
        trace_start();
    for (int i = -WARMUP; i < frames; i++)
    {
        if (tracing && i >= 0)
            trace_mark("begin", i);
        const double t0 = now();
        render(mode);
        const double t1 = now();
        glFinish();
        const double t2 = now();
        if (tracing && i >= 0)
            trace_mark("end", i);
        if (i >= 0)
        {
            total[i] = t2 - t0;
            submit[i] = t1 - t0;
        }
        else if (i == -WARMUP)
        {
            char what[64];
            snprintf(what, sizeof what, "drawing the %s", mode_names[mode]);
            check_gl(what);
        }
    }
    if (tracing)
        trace_read(frames, gp, pp);
    else
        for (int i = 0; i < frames; i++)
            gp[i] = pp[i] = NAN;
    const result_t r = {stats(total, frames), stats(submit, frames), stats(gp, frames), stats(pp, frames),
                        total, submit, gp, pp};
    return r;
}

static void write_png(const char *path)
{
    render(FULL);
    glFinish();
    uint8_t *rgba = malloc((size_t)width * height * 4), *rows = malloc((size_t)width * height * 4);
    if (!rgba || !rows)
        die("out of memory");
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, rgba);
    for (int y = 0; y < height; y++) // GL's bottom row first
        memcpy(&rows[(size_t)y * width * 4], &rgba[(size_t)(height - 1 - y) * width * 4], (size_t)width * 4);
    if (!stbi_write_png(path, width, height, 4, rows, width * 4))
        die("%s: cannot write it", path);
    free(rgba);
    free(rows);
    printf("frame written to %s\n", path);
}

// The GPU's clock, where the kernel tells
static void print_clock(void)
{
    FILE *in = fopen("/sys/kernel/debug/clk/gpu_ref/clk_rate", "r");
    unsigned long long hz;
    if (in && fscanf(in, "%llu", &hz) == 1)
        printf("GPU clock: gpu_ref %.1f MHz\n", hz / 1e6);
    else
    {
        glob_t g;
        if (glob("/sys/class/devfreq/*gpu*/cur_freq", 0, NULL, &g) == 0 && g.gl_pathc)
        {
            FILE *f = fopen(g.gl_pathv[0], "r");
            if (f && fscanf(f, "%llu", &hz) == 1)
                printf("GPU clock: %.1f MHz (%s)\n", hz / 1e6, g.gl_pathv[0]);
            if (f)
                fclose(f);
        }
        else
            printf("GPU clock: unknown (root reads /sys/kernel/debug/clk/gpu_ref/clk_rate)\n");
        globfree(&g);
    }
    if (in)
        fclose(in);
}

static void print_ms(FILE *out, double ms)
{
    if (isnan(ms))
        fprintf(out, "%9s", "-");
    else
        fprintf(out, "%9.3f", ms);
}

// The table and the estimates, each line after prefix
static void report(FILE *out, const char *prefix, const result_t *results, int frames)
{
    const int traced = trace_kind != TRACE_NONE;
    fprintf(out, "%s%d frames of each (ms)      median      min     mean   submit%s\n", prefix, frames,
            traced ? "   GP job   PP job" : "");
    for (int m = 0; m < MODES; m++)
    {
        fprintf(out, "%s  %-24s", prefix, mode_names[m]);
        print_ms(out, results[m].total.median);
        print_ms(out, results[m].total.min);
        print_ms(out, results[m].total.mean);
        print_ms(out, results[m].submit.median);
        if (traced)
        {
            print_ms(out, results[m].gp.median);
            print_ms(out, results[m].pp.median);
        }
        fprintf(out, "\n");
    }
    const double full = results[FULL].total.median, scissored = results[SCISSORED].total.median;
    const double cleared = results[CLEARED].total.median, tiny = results[CLEARED_TINY].total.median;
    fprintf(out, "%s\n%sestimates from the medians (ms)\n", prefix, prefix);
    fprintf(out, "%s  raster                  full - scissored          %9.3f\n", prefix, full - scissored);
    fprintf(out, "%s  write-back              clear - clear 16x16       %9.3f\n", prefix, cleared - tiny);
    fprintf(out, "%s  raster + write-back                               %9.3f\n", prefix,
            full - scissored + cleared - tiny);
    fprintf(out, "%s  vertices + binning      scissored - clear         %9.3f\n", prefix, scissored - cleared);
    fprintf(out, "%s  fixed cost of a frame   clear 16x16               %9.3f\n", prefix, tiny);
    if (traced && results[FULL].pp.n)
        fprintf(out, "%s\n%sPP job of the full frame (raster + write-back): %.3f ms median, %.3f min (%d frames)\n",
                prefix, prefix, results[FULL].pp.median, results[FULL].pp.min, results[FULL].pp.n);
    else if (traced)
        fprintf(out, "%s\n%sno PP job events seen\n", prefix, prefix);
}

// Every frame's times, one line each, the report after them as comments
static void write_times(const char *path, const char *model, float angle, const result_t *results, int frames)
{
    FILE *out = fopen(path, "w");
    if (!out)
        die("%s: cannot write it", path);
    fprintf(out, "# mali_bench %s, angle %g: %u triangles, %d x %d\n", model, angle, scene.index_count / 3, width,
            height);
    fprintf(out, "# GL: %s, %s, %s\n", (const char *)glGetString(GL_VENDOR), (const char *)glGetString(GL_RENDERER),
            (const char *)glGetString(GL_VERSION));
    fprintf(out, "# kind frame total_ms submit_ms gp_job_ms pp_job_ms (nan: no job events)\n");
    static const char *kinds[MODES] = {"full", "scissored", "clear", "clear16"};
    for (int m = 0; m < MODES; m++)
        for (int i = 0; i < frames; i++)
            fprintf(out, "%s %d %.4f %.4f %.4f %.4f\n", kinds[m], i, results[m].total_s[i] * 1e3,
                    results[m].submit_s[i] * 1e3, results[m].gp_s[i] * 1e3, results[m].pp_s[i] * 1e3);
    fprintf(out, "#\n");
    report(out, "# ", results, frames);
    if (fclose(out) != 0)
        die("%s: cannot write it", path);
    printf("times written to %s\n", path);
}

int main(int argc, char **argv)
{
    int frames = 100, use_trace = 1;
    float angle = 0;
    const char *png = NULL, *times = NULL;
    int opt;
    while ((opt = getopt(argc, argv, "n:a:s:o:t:T")) != -1)
        switch (opt)
        {
        case 'n':
            frames = atoi(optarg);
            break;
        case 'a':
            angle = (float)atof(optarg);
            break;
        case 's':
            if (sscanf(optarg, "%dx%d", &width, &height) != 2)
                width = 0;
            break;
        case 'o':
            png = optarg;
            break;
        case 't':
            times = optarg;
            break;
        case 'T':
            use_trace = 0;
            break;
        default:
            frames = 0;
        }
    if (optind != argc - 1 || frames < 1 || width < 1 || height < 1)
    {
        fprintf(stderr,
                "usage: %s [-n FRAMES] [-a ANGLE] [-s WxH] [-o PNG] [-t TXT] [-T] MODEL\n"
                "  MODEL  a .glb, or a .gltf with its buffers and images embedded or beside it\n"
                "  -n     frames timed of each kind (100), after %d unmeasured\n"
                "  -a     the model turned by ANGLE degrees about its vertical axis (0)\n"
                "  -s     frame size (1280x720)\n"
                "  -o     the frame as a PNG (MODEL_mali.png here; - none)\n"
                "  -t     every frame's times (MODEL_mali_times.txt here; - none)\n"
                "  -T     no kernel GPU job events (else used when root)\n",
                argv[0], WARMUP);
        return 1;
    }
    const char *model = argv[optind];
    // The defaults: the model's name, without its directory and extension, in this directory
    char stem[256], default_png[300], default_times[300];
    const char *base = strrchr(model, '/') ? strrchr(model, '/') + 1 : model;
    snprintf(stem, sizeof stem, "%s", base);
    char *dot = strrchr(stem, '.');
    if (dot && dot != stem)
        *dot = '\0';
    snprintf(default_png, sizeof default_png, "%s_mali.png", stem);
    snprintf(default_times, sizeof default_times, "%s_mali_times.txt", stem);
    if (!png)
        png = default_png;
    if (!times)
        times = default_times;

    char error[256];
    if (scene_load(model, width, height, angle, &scene, error, sizeof error) != 0)
        die("%s", error);

    egl_init();
    printf("GL: %s, %s, %s\n", (const char *)glGetString(GL_VENDOR), (const char *)glGetString(GL_RENDERER),
           (const char *)glGetString(GL_VERSION));
    if (!strstr((const char *)glGetString(GL_RENDERER), "Mali"))
        printf("WARNING: the renderer is not the Mali; these are not its times\n");
    print_clock();

    make_programs();
    const int depth24 = has_extension("GL_OES_depth24");
    fbo = make_fbo(width, height, depth24 ? GL_DEPTH_COMPONENT24_OES : GL_DEPTH_COMPONENT16);
    tiny_fbo = make_fbo(TINY, TINY, depth24 ? GL_DEPTH_COMPONENT24_OES : GL_DEPTH_COMPONENT16);
    if (has_extension("GL_EXT_discard_framebuffer"))
        discard = (PFNGLDISCARDFRAMEBUFFEREXTPROC)eglGetProcAddress("glDiscardFramebufferEXT");
    glClearColor(0x63 / 255.0f, 0x63 / 255.0f, 0x63 / 255.0f, 1.0f);
    glClearDepthf(1.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glCullFace(GL_BACK);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_DITHER);
    check_gl("setting up");
    upload_textures();
    const size_t bytes = upload_geometry();

    printf("%s, angle %g: %u triangles, %u vertices, %d texture%s, %d draw calls, %s camera\n", model, angle,
           scene.index_count / 3, scene.vertex_count, scene.texture_count, scene.texture_count == 1 ? "" : "s",
           scene.draw_count, scene.own_camera ? "its" : "the default");
    printf("%d x %d, depth %d-bit%s; %.1f MiB of geometry, 16-bit indices, %d batches\n", width, height,
           depth24 ? 24 : 16, discard ? ", discarded" : ", written back (no EXT_discard_framebuffer)",
           bytes / 1048576.0, batch_count);

    if (use_trace)
        trace_open();
    result_t results[MODES];
    for (int m = 0; m < MODES; m++)
        results[m] = run(m, frames);
    trace_close();

    printf("\n");
    report(stdout, "", results, frames);
    if (strcmp(times, "-") != 0)
        write_times(times, model, angle, results, frames);
    if (strcmp(png, "-") != 0)
        write_png(png);
    return 0;
}
