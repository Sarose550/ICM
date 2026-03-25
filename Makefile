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

UNAME := $(shell uname)

ifeq ($(UNAME),Darwin)
  # macOS: Homebrew paths + Accelerate (vvexp)
  INCLUDES += -I/opt/homebrew/include
  LDFLAGS  := -L/opt/homebrew/lib $(LDFLAGS) -framework Accelerate
  OMP_CFLAGS  = -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
  OMP_LDFLAGS = -L/opt/homebrew/opt/libomp/lib -lomp -lfftw3_threads
else
  # Linux: native OpenMP, system FFTW
  OMP_CFLAGS  = -fopenmp
  OMP_LDFLAGS = -lfftw3_threads -lpthread
endif

# Zen 4 override: use -march=znver4 instead of -march=native if preferred
ifeq ($(DEVICE),zen4)
  CFLAGS := -O3 -march=znver4 -Wall -Wno-unused-variable -Wno-unused-function
endif

SRC = bench/bench.c
OUT = bench_grid

.PHONY: all parallel test bench calibrate clean

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

clean:
	rm -f $(OUT) calibrate
