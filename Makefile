# ICM Equity Computation — Makefile
#
# Usage:
#   make                    # serial build (bench_grid)
#   make parallel           # OpenMP build (bench_grid)
#   make DEVICE=zen4        # build for a different device
#   make test               # quick verify
#   make bench              # full benchmark grid
#   make libicm.a           # build the static library
#   make libicm             # build the shared library (.so/.dylib)
#   make contour_1s         # contour sweep tool (serial)
#   make contour_1s_par     # contour sweep tool (parallel)

DEVICE ?= m3_pro

.DEFAULT_GOAL := all

CC = gcc
CFLAGS = -O3 -march=native -Wall
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
  # macOS: Homebrew paths + Accelerate (vDSP, vvexp)
  BREW_PREFIX := $(shell brew --prefix 2>/dev/null)
  ifeq ($(BREW_PREFIX),)
    BREW_PREFIX := /opt/homebrew
  endif
  INCLUDES += -I$(BREW_PREFIX)/include
  LDFLAGS  := -L$(BREW_PREFIX)/lib $(LDFLAGS) -framework Accelerate
  OMP_CFLAGS  = -Xpreprocessor -fopenmp -I$(BREW_PREFIX)/opt/libomp/include
  OMP_LDFLAGS = -L$(BREW_PREFIX)/opt/libomp/lib -lomp -lfftw3_threads
else
  # Linux: native OpenMP, system FFTW, dlopen for MKL dual dispatch
  LDFLAGS += -ldl -lmvec
  OMP_CFLAGS  = -fopenmp
  OMP_LDFLAGS = -lfftw3_threads -lpthread
  # Auto-detect AOCL-FFTW
  ifneq ($(wildcard /usr/local/aocl-fftw/lib/libfftw3.so),)
    INCLUDES += -I/usr/local/aocl-fftw/include
    LDFLAGS  := -L/usr/local/aocl-fftw/lib -Wl,-rpath,/usr/local/aocl-fftw/lib $(LDFLAGS)
  endif
endif

# Zen 4 override: use -march=znver4
ifeq ($(DEVICE),zen4)
  CFLAGS := -O3 -march=znver4 -Wall -Wno-unused-variable -Wno-unused-function
endif

SRC = bench/bench.c
OUT = bench_grid

# ── Library ─────────────────────────────────────────────────────

LIBICM = $(BUILD_DIR)/libicm.a
LIBICM_OBJ = $(BUILD_DIR)/icm.o
LIBICM_OMP_OBJ = $(BUILD_DIR)/icm_omp.o

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(LIBICM_OBJ): src/icm.c src/icm.h src/linear_batched_impl.inc devices/$(DEVICE)/fft_config.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c src/icm.c -o $@

$(LIBICM): $(LIBICM_OBJ)
	ar rcs $@ $^

# Shared library
ifeq ($(UNAME),Darwin)
  SHARED_EXT = dylib
  SHARED_FLAGS = -dynamiclib -install_name @rpath/libicm.$(SHARED_EXT)
else
  SHARED_EXT = so
  SHARED_FLAGS = -shared
endif

LIBICM_SHARED = $(BUILD_DIR)/libicm.$(SHARED_EXT)

$(BUILD_DIR)/icm_shared.o: src/icm.c src/icm.h src/linear_batched_impl.inc devices/$(DEVICE)/fft_config.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -fPIC $(INCLUDES) -c src/icm.c -o $@

$(LIBICM_SHARED): $(BUILD_DIR)/icm_shared.o
	$(CC) $(SHARED_FLAGS) -o $@ $^ $(LDFLAGS)

libicm: $(LIBICM_SHARED)

libicm.dylib: $(LIBICM_SHARED)

libicm.so: $(LIBICM_SHARED)

# OpenMP variant
$(LIBICM_OMP_OBJ): src/icm.c src/icm.h src/linear_batched_impl.inc devices/$(DEVICE)/fft_config.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(OMP_CFLAGS) $(INCLUDES) -c src/icm.c -o $@

$(BUILD_DIR)/libicm_omp.a: $(LIBICM_OMP_OBJ)
	ar rcs $@ $^

libicm.a: $(LIBICM)

# ── Bench grid (includes icm.c directly for profiling access) ──

.PHONY: all parallel test bench calibrate clean libicm.a libicm libicm.dylib libicm.so

all:
	$(CC) $(CFLAGS) $(INCLUDES) -o $(OUT) $(SRC) $(LDFLAGS)

parallel:
	$(CC) $(CFLAGS) $(OMP_CFLAGS) $(INCLUDES) -o $(OUT) $(SRC) $(LDFLAGS) $(OMP_LDFLAGS)

test: all
	./$(OUT) quick

bench: all
	nice -20 ./$(OUT) quick

# ── Tools (link against libicm.a) ──────────────────────────────

contour_1s: $(LIBICM)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tools/contour_1s.c $(LIBICM) $(LDFLAGS)

contour_1s_par: $(BUILD_DIR)/libicm_omp.a
	$(CC) $(CFLAGS) $(OMP_CFLAGS) $(INCLUDES) -o $@ tools/contour_1s.c $(BUILD_DIR)/libicm_omp.a $(LDFLAGS) $(OMP_LDFLAGS)

calibrate:
	$(CC) $(CFLAGS) $(INCLUDES) -o calibrate tools/calibrate.c $(LDFLAGS)
	@echo "Run: ./calibrate (then copy fft_config.h + fftw_wisdom.dat to devices/$(DEVICE)/)"

# ── Regenerate results/ data for tools/plot_contour.py ──────────
# Requires devices/$(DEVICE)/ to already be calibrated (see "Calibrating
# for a New Device" in README.md). Writes stable (undated) files directly
# into results/, overwriting any previous run -- plot_contour.py picks up
# these files by name (via find_latest's mtime tiebreak), no renaming or
# copying needed. DEVICE=zen4 must be run on Zen4 hardware; DEVICE=m3_pro
# on Apple Silicon. Git history is the record of prior runs; file names are
# not used for versioning.
# plot_contour.py's DEVICE_CONFIGS use "m3pro" (no underscore) in filenames
# even though the build device is "m3_pro" -- match that convention here.
RESULTS_TAG := $(subst m3_pro,m3pro,$(DEVICE))
# NOTE: `all` and `parallel` both build to the same $(OUT) binary
# (bench_grid). Both used to be listed as prerequisites here, which Make
# builds BEFORE the recipe body runs -- so by the time `./$(OUT) >
# ..._serial.txt` executed, `parallel`'s OpenMP-enabled build had already
# silently overwritten the serial one, meaning every "serial" results file
# this target ever produced was actually generated by the OpenMP-enabled
# binary (defaulting to OMP_NUM_THREADS_DEFAULT threads, since the serial
# invocation doesn't set OMP_NUM_THREADS itself). Fixed by rebuilding
# explicitly inside the recipe body, immediately before each binary's use,
# instead of relying on prerequisite ordering.
results-refresh: contour_1s contour_1s_par
	mkdir -p results
	$(MAKE) all
	./$(OUT) > results/bench_grid_$(RESULTS_TAG)_serial.txt
	$(MAKE) parallel
	OMP_NUM_THREADS=$${OMP_NUM_THREADS:-$$(sysctl -n hw.ncpu 2>/dev/null || nproc)} \
	    ./$(OUT) > results/bench_grid_$(RESULTS_TAG)_parallel.txt
	./contour_1s --contour > results/contour_$(RESULTS_TAG)_serial_q256.csv
	OMP_NUM_THREADS=$${OMP_NUM_THREADS:-$$(sysctl -n hw.ncpu 2>/dev/null || nproc)} \
	    ./contour_1s_par --contour > results/contour_$(RESULTS_TAG)_parallel_q256.csv
	python3 tools/plot_contour.py --device $(DEVICE)
	@echo "Refreshed results/ for DEVICE=$(DEVICE)."

.PHONY: results-refresh

# ── GPU targets ─────────────────────────────────────────────────

# CPU reference object for GPU benchmarks (cross-check against CPU results)
CPU_REF_OBJ = $(BUILD_DIR)/icm_cpu_ref.o

$(CPU_REF_OBJ): src/icm.c src/icm.h devices/$(DEVICE)/fft_config.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c src/icm.c -o $@

CUFFTDX_FLAGS = $(CUFFTDX_INC) -DUSE_CUFFTDX -DICM_REQUIRE_CUFFTDX -DCUFFTDX_DISABLE_CUTLASS_DEPENDENCY
VKFFT_FLAGS = $(if $(VKFFT_INC),$(VKFFT_INC) -DUSE_VKFFT)
VKFFT_LIBS = $(if $(VKFFT_INC),-lnvrtc -lcuda)
GPU_INCLUDES = $(INCLUDES) -Idevices/b200

# ── Multi-file GPU compilation (separate compilation + device linking) ──
GPU_SRCS = src/gpu/gpu_kernels.cu src/gpu/gpu_plan.cu src/gpu/gpu_exec.cu src/gpu/gpu_api.cu
GPU_OBJS = $(patsubst src/gpu/%.cu,$(BUILD_DIR)/gpu_%.o,$(GPU_SRCS))
GPU_HDRS = src/gpu/gpu_internal.h src/icm_gpu.h devices/b200/gpu_fft_config.h

$(BUILD_DIR)/gpu_%.o: src/gpu/%.cu $(GPU_HDRS) | $(BUILD_DIR)
	$(NVCC) $(CUDA_FLAGS) $(GPU_INCLUDES) -Isrc/gpu -dc -o $@ $<

$(BUILD_DIR)/gpu_%_fused.o: src/gpu/%.cu $(GPU_HDRS) | $(BUILD_DIR)
	$(NVCC) $(CUDA_FLAGS) $(GPU_INCLUDES) -Isrc/gpu $(CUFFTDX_FLAGS) $(VKFFT_FLAGS) -dc -o $@ $<

GPU_OBJS_PLAIN = $(patsubst src/gpu/%.cu,$(BUILD_DIR)/gpu_%.o,$(GPU_SRCS))
GPU_OBJS_FUSED = $(patsubst src/gpu/%.cu,$(BUILD_DIR)/gpu_%_fused.o,$(GPU_SRCS))

$(BUILD_DIR)/gpu_dlink.o: $(GPU_OBJS_PLAIN) | $(BUILD_DIR)
	$(NVCC) $(CUDA_FLAGS) -dlink -o $@ $(GPU_OBJS_PLAIN) $(CUDA_LIBS)

$(BUILD_DIR)/gpu_dlink_fused.o: $(GPU_OBJS_FUSED) | $(BUILD_DIR)
	$(NVCC) $(CUDA_FLAGS) -dlink -o $@ $(GPU_OBJS_FUSED) $(CUDA_LIBS)

bench_gpu: bench/bench_gpu.cu $(GPU_OBJS_PLAIN) $(BUILD_DIR)/gpu_dlink.o $(CPU_REF_OBJ)
	$(NVCC) $(CUDA_FLAGS) $(GPU_INCLUDES) -Isrc/gpu -dc -o $(BUILD_DIR)/bench_gpu.o bench/bench_gpu.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $(BUILD_DIR)/bench_gpu.o $(GPU_OBJS_PLAIN) $(BUILD_DIR)/gpu_dlink.o $(CPU_REF_OBJ) $(LDFLAGS) $(CUDA_LIBS)

bench_gpu_fused: bench/bench_gpu.cu $(GPU_OBJS_FUSED) $(BUILD_DIR)/gpu_dlink_fused.o $(CPU_REF_OBJ)
	$(NVCC) $(CUDA_FLAGS) $(GPU_INCLUDES) -Isrc/gpu $(CUFFTDX_FLAGS) $(VKFFT_FLAGS) -dc -o $(BUILD_DIR)/bench_gpu_fused.o bench/bench_gpu.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $(BUILD_DIR)/bench_gpu_fused.o $(GPU_OBJS_FUSED) $(BUILD_DIR)/gpu_dlink_fused.o $(CPU_REF_OBJ) $(LDFLAGS) $(CUDA_LIBS) $(VKFFT_LIBS)

calibrate_gpu: tools/calibrate_gpu.cu $(GPU_OBJS_FUSED) $(BUILD_DIR)/gpu_dlink_fused.o
	$(NVCC) $(CUDA_FLAGS) $(GPU_INCLUDES) -Isrc/gpu $(CUFFTDX_FLAGS) $(VKFFT_FLAGS) -dc -o $(BUILD_DIR)/calibrate_gpu.o tools/calibrate_gpu.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $(BUILD_DIR)/calibrate_gpu.o $(GPU_OBJS_FUSED) $(BUILD_DIR)/gpu_dlink_fused.o $(CUDA_LIBS) $(VKFFT_LIBS)

heatmap_gpu: tools/heatmap_gpu.cu $(GPU_OBJS_FUSED) $(BUILD_DIR)/gpu_dlink_fused.o
	$(NVCC) $(CUDA_FLAGS) $(GPU_INCLUDES) -Isrc/gpu $(CUFFTDX_FLAGS) $(VKFFT_FLAGS) -dc -o $(BUILD_DIR)/heatmap_gpu.o tools/heatmap_gpu.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $(BUILD_DIR)/heatmap_gpu.o $(GPU_OBJS_FUSED) $(BUILD_DIR)/gpu_dlink_fused.o $(CUDA_LIBS) $(VKFFT_LIBS)

push_limit_gpu: tools/push_limit_gpu.cu $(GPU_OBJS_FUSED) $(BUILD_DIR)/gpu_dlink_fused.o
	$(NVCC) $(CUDA_FLAGS) $(GPU_INCLUDES) -Isrc/gpu $(CUFFTDX_FLAGS) $(VKFFT_FLAGS) -dc -o $(BUILD_DIR)/push_limit_gpu.o tools/push_limit_gpu.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $(BUILD_DIR)/push_limit_gpu.o $(GPU_OBJS_FUSED) $(BUILD_DIR)/gpu_dlink_fused.o $(CUDA_LIBS) $(VKFFT_LIBS)

validate_planner_gpu: tools/validate_planner_gpu.cu $(GPU_OBJS_FUSED) $(BUILD_DIR)/gpu_dlink_fused.o
	$(NVCC) $(CUDA_FLAGS) $(GPU_INCLUDES) -Isrc/gpu $(CUFFTDX_FLAGS) $(VKFFT_FLAGS) -dc -o $(BUILD_DIR)/validate_planner_gpu.o tools/validate_planner_gpu.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $(BUILD_DIR)/validate_planner_gpu.o $(GPU_OBJS_FUSED) $(BUILD_DIR)/gpu_dlink_fused.o $(CUDA_LIBS) $(VKFFT_LIBS)

gpu_sample_plans: tools/gpu_sample_plans.cu $(GPU_OBJS_FUSED) $(BUILD_DIR)/gpu_dlink_fused.o
	$(NVCC) $(CUDA_FLAGS) $(GPU_INCLUDES) -Isrc/gpu $(CUFFTDX_FLAGS) $(VKFFT_FLAGS) -dc -o $(BUILD_DIR)/gpu_sample_plans.o tools/gpu_sample_plans.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $(BUILD_DIR)/gpu_sample_plans.o $(GPU_OBJS_FUSED) $(BUILD_DIR)/gpu_dlink_fused.o $(CUDA_LIBS) $(VKFFT_LIBS)

gpu_phase_profile: tools/gpu_phase_profile.cu $(GPU_OBJS_FUSED) $(BUILD_DIR)/gpu_dlink_fused.o
	$(NVCC) $(CUDA_FLAGS) $(GPU_INCLUDES) -Isrc/gpu $(CUFFTDX_FLAGS) $(VKFFT_FLAGS) -dc -o $(BUILD_DIR)/gpu_phase_profile.o tools/gpu_phase_profile.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $(BUILD_DIR)/gpu_phase_profile.o $(GPU_OBJS_FUSED) $(BUILD_DIR)/gpu_dlink_fused.o $(CUDA_LIBS) $(VKFFT_LIBS)

test_gpu_cost_model: tools/test_gpu_cost_model.cu $(GPU_OBJS_FUSED) $(BUILD_DIR)/gpu_dlink_fused.o
	$(NVCC) $(CUDA_FLAGS) $(GPU_INCLUDES) -Isrc/gpu $(CUFFTDX_FLAGS) $(VKFFT_FLAGS) -dc -o $(BUILD_DIR)/test_gpu_cost_model.o tools/test_gpu_cost_model.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $(BUILD_DIR)/test_gpu_cost_model.o $(GPU_OBJS_FUSED) $(BUILD_DIR)/gpu_dlink_fused.o $(CUDA_LIBS) $(VKFFT_LIBS)

test_cpu_cost_model: tools/test_cpu_cost_model.c src/icm.c src/icm.h devices/$(DEVICE)/fft_config.h
	# -Wno-unused-function: this tool only exercises a subset of icm.c
	# (no naive-engine path), so some functions used elsewhere in the
	# codebase are legitimately unreferenced from this translation unit.
	$(CC) $(CFLAGS) -Wno-unused-function $(INCLUDES) -o $@ tools/test_cpu_cost_model.c $(LDFLAGS)

.PHONY: bench_gpu bench_gpu_fused calibrate_gpu heatmap_gpu push_limit_gpu validate_planner_gpu test_gpu_cost_model test_cpu_cost_model campaign_b200

campaign_b200: bench_gpu_fused calibrate_gpu heatmap_gpu push_limit_gpu validate_planner_gpu
	bash tools/run_b200_campaign.sh

# ── Clean ───────────────────────────────────────────────────────

clean:
	rm -f $(OUT) calibrate contour_1s contour_1s_par accuracy_bench
	rm -f bench_gpu bench_gpu_fused calibrate_gpu heatmap_gpu push_limit_gpu validate_planner_gpu test_gpu_cost_model test_cpu_cost_model
	rm -rf $(BUILD_DIR)
	rm -rf python/*.egg-info python/build python/dist
