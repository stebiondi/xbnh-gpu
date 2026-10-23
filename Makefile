# xbnh-gpu build. ARCH=native compiles for the GPU in this machine; set ARCH=sm_80 (A100), sm_90 (H100),
# sm_86 (RTX 30xx), sm_89 (RTX 40xx / Ada), sm_75 (RTX 20xx, T4), sm_70 (V100), sm_61 (Pascal) to cross-compile.
# netCDF output (outputformat = netcdf) is compiled in when nc-config is found (environment.yml installs
# libnetcdf); NETCDF=0 builds without it.
NVCC    ?= nvcc
ARCH    ?= native
NVFLAGS ?= -O3 -std=c++20
NETCDF  ?= $(shell command -v nc-config > /dev/null 2>&1 && echo 1 || echo 0)
ifeq ($(NETCDF),1)
NCFLAGS = -DXB_NETCDF $(shell nc-config --cflags) $(shell nc-config --libs) -Xlinker -rpath=$(shell nc-config --prefix)/lib
endif
SRC = src/xbnh_gpu.cu $(wildcard src/*.h src/*.cuh)

all: bin/xbnh_gpu bin/xbnh_gpu_sp

bin/xbnh_gpu: $(SRC) | bin
	$(NVCC) $(NVFLAGS) -arch=$(ARCH) -o $@ src/xbnh_gpu.cu $(NCFLAGS)

bin/xbnh_gpu_sp: $(SRC) | bin
	$(NVCC) $(NVFLAGS) -arch=$(ARCH) -DXB_SINGLE -o $@ src/xbnh_gpu.cu $(NCFLAGS)

bin:
	mkdir -p bin

clean:
	rm -rf bin

.PHONY: all clean
