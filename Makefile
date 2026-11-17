# mali_bench: the ZCU104's Mali-400 drawing a glTF frame's triangles as gl_test's GPU gets them,
# timed (main.c). Built on the board's Ubuntu, in this repository with its submodules:
#
#   sudo apt install build-essential libegl-dev libgles-dev
#   make
#   sudo ./mali_bench MODEL.glb          root for the kernel's GPU job times; -h for the options
#
# lviv_gl's front end is built with its gl* names as lviv_gl* (lviv_rename.h, from lviv_gl.h), so
# they do not take libGLESv2's.

CC = gcc
CXX = g++
SDK = ../../sdk
GLM = $(SDK)/third_party/glm
FASTGLTF = $(SDK)/third_party/fastgltf
SIMDJSON = $(SDK)/third_party/simdjson/singleheader
STB = $(SDK)/third_party/stb
INCLUDES = -I$(SDK)/lviv_gl -I$(SDK)/gltf -I$(SDK)/src -I$(SDK)/include -I$(GLM) -I$(FASTGLTF)/include \
           -I$(SIMDJSON) -I$(STB)
CFLAGS = -O2 -std=c11 -Wall -Wextra $(INCLUDES)
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra -fno-exceptions -DGLM_FORCE_INTRINSICS -DGLM_FORCE_DEFAULT_ALIGNED_GENTYPES \
           $(INCLUDES)
FRONT_OBJS = front.o gl_api.o gltf_scene.o
OBJS = main.o $(FRONT_OBJS) lviv_geom.o stb_image.o fastgltf.o base64.o io.o simdjson.o
HDRS = front.h $(SDK)/lviv_gl/lviv_geom.h $(SDK)/lviv_gl/lviv_gl.h $(SDK)/gltf/gltf_scene.h $(SDK)/src/gpu.h
SUBMODULES = $(GLM)/glm/glm.hpp $(FASTGLTF)/include/fastgltf/core.hpp $(SIMDJSON)/simdjson.h $(STB)/stb_image.h
CHECK_SUBMODULES = $(foreach f,$(SUBMODULES),$(if $(wildcard $(f)),,$(error $(f) is missing: git submodule update --init)))
vpath %.c $(SDK)/lviv_gl $(SDK)/gltf
vpath %.cpp $(SDK)/lviv_gl $(SDK)/gltf $(FASTGLTF)/src $(SIMDJSON)

all: mali_bench

mali_bench: $(OBJS)
	@if nm -g --defined-only $(FRONT_OBJS) | grep -E ' [A-Z] glu?[A-Z]'; then \
		echo "lviv_gl's names above would take libGLESv2's: lviv_rename.h misses them"; exit 1; fi
	$(CXX) $(OBJS) -lEGL -lGLESv2 -lm -o $@

lviv_rename.h: $(SDK)/lviv_gl/lviv_gl.h
	sed -n 's/^[A-Za-z][A-Za-z ]*[ *]\(glu\{0,1\}[A-Z][A-Za-z0-9]*\)(.*/#define \1 lviv_\1/p' $< > $@

$(FRONT_OBJS): lviv_rename.h
front.o gl_api.o: CFLAGS += -include lviv_rename.h
gltf_scene.o: CXXFLAGS += -include lviv_rename.h

%.o: %.c $(HDRS)
	$(CHECK_SUBMODULES)
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp $(HDRS)
	$(CHECK_SUBMODULES)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f mali_bench lviv_rename.h *.o

.PHONY: all clean
