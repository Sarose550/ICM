# ICM Equity Computation — Makefile
#
# Usage:
#   make                    # serial build for current device
#   make parallel           # OpenMP build
#   make DEVICE=zen4        # build for a different device
#   make test               # quick verify
#   make bench              # full benchmark grid

DEVICE ?= m3_max

CC = gcc
CFLAGS = -O3 -march=native -Wall -Wno-unused-variable -Wno-unused-function
INCLUDES = -Isrc -Idevices/$(DEVICE)
LDFLAGS = -lfftw3 -lm
NVCC ?= nvcc
CUDA_ARCH ?= sm_100
CUDA_FLAGS = -O3 -std=c++17 -arch=$(CUDA_ARCH)
CUDA_LIBS = -lcufft -lcudart
CUFFTDX_INC ?=
ifeq ($(strip $(CUFFTDX_INC)),)
CUFFTDX_INCLUDE_DIR := $(firstword $(wildcard /opt/nvidia/mathdx/nvidia/mathdx/*/include /opt/nvidia/mathdx/nvidia-mathdx-*/nvidia/mathdx/*/include))
ifneq ($(strip $(CUFFTDX_INCLUDE_DIR)),)
CUFFTDX_INC := -I$(CUFFTDX_INCLUDE_DIR)
endif
endif
BUILD_DIR = build

UNAME := $(shell uname)

ifeq ($(UNAME),Darwin)
  # macOS: Homebrew paths + Accelerate (vvexp)
  INCLUDES += -I/opt/homebrew/include
  LDFLAGS  := -L/opt/homebrew/lib $(LDFLAGS) -framework Accelerate
  OMP_CFLAGS  = -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
  OMP_LDFLAGS = -L/opt/homebrew/opt/libomp/lib -lomp -lfftw3_threads
else
  # Linux: native OpenMP, system FFTW, dlopen for MKL dual dispatch
  LDFLAGS += -ldl
  OMP_CFLAGS  = -fopenmp
  OMP_LDFLAGS = -lfftw3_threads -lpthread
endif

# Zen 4 override: use -march=znver4 instead of -march=native if preferred
ifeq ($(DEVICE),zen4)
  CFLAGS := -O3 -march=znver4 -Wall -Wno-unused-variable -Wno-unused-function
endif

SRC = bench/bench.c
OUT = bench_grid
CPU_REF_OBJ = $(BUILD_DIR)/icm_cpu_ref.o

.PHONY: all parallel test bench calibrate bench_gpu bench_gpu_fused calibrate_gpu heatmap_gpu push_limit_gpu validate_planner_gpu campaign_b200 clean

all:
	$(CC) $(CFLAGS) $(INCLUDES) -o $(OUT) $(SRC) $(LDFLAGS)

parallel:
	$(CC) $(CFLAGS) $(OMP_CFLAGS) $(INCLUDES) -o $(OUT) $(SRC) $(LDFLAGS) $(OMP_LDFLAGS)

test: all
	./$(OUT) quick

bench: all
	nice -20 ./$(OUT) quick

calibrate:
	$(CC) $(CFLAGS) $(INCLUDES) -o calibrate tools/calibrate.c $(LDFLAGS)
	@echo "Run: ./calibrate (then copy fft_config.h + fftw_wisdom.dat to devices/$(DEVICE)/)"

$(CPU_REF_OBJ): src/icm.c src/icm.h devices/$(DEVICE)/fft_config.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c src/icm.c -o $(CPU_REF_OBJ)

bench_gpu: bench/bench_gpu.cu src/icm_gpu.cu src/icm_gpu.h devices/b200/gpu_fft_config.h $(CPU_REF_OBJ)
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) -Idevices/b200 -o $@ bench/bench_gpu.cu src/icm_gpu.cu $(CPU_REF_OBJ) $(LDFLAGS) $(CUDA_LIBS)

bench_gpu_fused: bench/bench_gpu.cu src/icm_gpu.cu src/icm_gpu.h devices/b200/gpu_fft_config.h $(CPU_REF_OBJ)
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) -Idevices/b200 $(CUFFTDX_INC) -DUSE_CUFFTDX -DICM_REQUIRE_CUFFTDX -DCUFFTDX_DISABLE_CUTLASS_DEPENDENCY -o $@ bench/bench_gpu.cu src/icm_gpu.cu $(CPU_REF_OBJ) $(LDFLAGS) $(CUDA_LIBS)

calibrate_gpu: tools/calibrate_gpu.cu src/icm_gpu.cu src/icm_gpu.h devices/b200/gpu_fft_config.h
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) -Idevices/b200 $(CUFFTDX_INC) -DUSE_CUFFTDX -DICM_REQUIRE_CUFFTDX -DCUFFTDX_DISABLE_CUTLASS_DEPENDENCY -o $@ tools/calibrate_gpu.cu src/icm_gpu.cu $(CUDA_LIBS)

heatmap_gpu: tools/heatmap_gpu.cu src/icm_gpu.cu src/icm_gpu.h devices/b200/gpu_fft_config.h
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) -Idevices/b200 $(CUFFTDX_INC) -DUSE_CUFFTDX -DICM_REQUIRE_CUFFTDX -DCUFFTDX_DISABLE_CUTLASS_DEPENDENCY -o $@ tools/heatmap_gpu.cu src/icm_gpu.cu $(CUDA_LIBS)

push_limit_gpu: tools/push_limit_gpu.cu src/icm_gpu.cu src/icm_gpu.h devices/b200/gpu_fft_config.h
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) -Idevices/b200 $(CUFFTDX_INC) -DUSE_CUFFTDX -DICM_REQUIRE_CUFFTDX -DCUFFTDX_DISABLE_CUTLASS_DEPENDENCY -o $@ tools/push_limit_gpu.cu src/icm_gpu.cu $(CUDA_LIBS)

validate_planner_gpu: tools/validate_planner_gpu.cu src/icm_gpu.cu src/icm_gpu.h devices/b200/gpu_fft_config.h
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) -Idevices/b200 $(CUFFTDX_INC) -DUSE_CUFFTDX -DICM_REQUIRE_CUFFTDX -DCUFFTDX_DISABLE_CUTLASS_DEPENDENCY -o $@ tools/validate_planner_gpu.cu src/icm_gpu.cu $(CUDA_LIBS)

campaign_b200: bench_gpu_fused calibrate_gpu heatmap_gpu push_limit_gpu validate_planner_gpu
	bash tools/run_b200_campaign.sh

clean:
	rm -f $(OUT) calibrate bench_gpu bench_gpu_fused calibrate_gpu heatmap_gpu push_limit_gpu validate_planner_gpu
	rm -rf $(BUILD_DIR)
