# xbnh-gpu build. ARCH=native compiles for the GPU in this machine; set ARCH=sm_80 (A100), sm_90 (H100),
# sm_86 (RTX 30xx), sm_89 (RTX 40xx / Ada), sm_75 (RTX 20xx, T4), sm_70 (V100) to cross-compile.
NVCC    ?= nvcc
ARCH    ?= native
NVFLAGS ?= -O3 -std=c++20

all: bin/xbnh_gpu bin/xbnh_gpu_sp

bin/xbnh_gpu: src/xbnh_gpu.cu | bin
	$(NVCC) $(NVFLAGS) -arch=$(ARCH) -o $@ $<

bin/xbnh_gpu_sp: src/xbnh_gpu.cu | bin
	$(NVCC) $(NVFLAGS) -arch=$(ARCH) -DXB_SINGLE -o $@ $<

bin:
	mkdir -p bin

clean:
	rm -rf bin

.PHONY: all clean
