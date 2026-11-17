# mali_bench: the ZCU104's Mali-400 drawing a glTF scene, its raster and write-back timed (main.c).
# Self-contained: copy this directory to the board's Ubuntu and
#
#   sudo apt install build-essential libegl-dev libgles-dev
#   make
#   sudo ./mali_bench MODEL.glb          root for the kernel's GPU job times; -h for the options

CC = gcc
CFLAGS = -O2 -std=c11 -Wall -Wextra
OBJS = main.o gltf.o stb.o

all: mali_bench

mali_bench: $(OBJS)
	$(CC) $(OBJS) -lEGL -lGLESv2 -lm -o $@

main.o: main.c gltf.h stb_image_write.h
	@printf '#include <EGL/egl.h>\n#include <GLES2/gl2.h>\n' | $(CC) -E -x c - >/dev/null 2>&1 || \
		{ echo "no EGL or GLES2 headers: sudo apt install libegl-dev libgles-dev"; exit 1; }
	$(CC) $(CFLAGS) -c $< -o $@

gltf.o: gltf.c gltf.h stb_image.h
	$(CC) $(CFLAGS) -c $< -o $@

stb.o: stb.c stb_image.h stb_image_write.h
	$(CC) -O2 -w -c $< -o $@

clean:
	rm -f mali_bench $(OBJS)

.PHONY: all clean
