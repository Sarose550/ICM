# ICM Calculator — Makefile
#
# Targets:
#   make               AVX2 full-n benchmark → bench_avx2
#   make avx512        AVX2 + AVX-512 → bench_avx512
#   make topk          Top-k benchmark → bench_topk
#   make calibrate     B-parameter profiler → calibrate_B
#   make all           Build everything available
#   make test          Quick correctness check
#   make clean         Remove build artifacts

CC       = gcc
CFLAGS   = -O3 -march=native -mavx2 -mfma -Wall -Wextra
LDFLAGS  = -lm
AVX512   = -mavx512f -mavx512dq

COMMON   = icm_common.c icm_detect.c
AVX2     = icm_avx2.c
AVX512_F = icm_avx512.c
TOPK     = icm_topk.c

# ── CPU targets ──────────────────────────────────────────────
.PHONY: default avx512 topk calibrate all test plot clean

default: bench_avx2

bench_avx2: bench.c $(COMMON) $(AVX2) icm.h
	$(CC) $(CFLAGS) -o $@ bench.c $(COMMON) $(AVX2) $(LDFLAGS)

avx512: bench_avx512

bench_avx512: bench.c $(COMMON) $(AVX2) $(AVX512_F) icm.h
	$(CC) $(CFLAGS) $(AVX512) -o $@ bench.c $(COMMON) $(AVX2) $(AVX512_F) $(LDFLAGS)

topk: bench_topk

bench_topk: bench_topk.c $(COMMON) $(AVX2) $(TOPK) icm.h
	$(CC) $(CFLAGS) -o $@ bench_topk.c $(COMMON) $(AVX2) $(TOPK) $(LDFLAGS)

calibrate: calibrate_B

calibrate_B: calibrate_B.c
	$(CC) $(CFLAGS) -o $@ calibrate_B.c $(LDFLAGS)

# ── Convenience ──────────────────────────────────────────────

all: default topk calibrate
	-$(MAKE) avx512 2>/dev/null || true

test: bench_avx2 bench_topk
	@echo "=== Full-n quick test ==="
	./bench_avx2 --quick 2>&1 | tail -10
	@echo ""
	@echo "=== Top-k quick test ==="
	./bench_topk 512 256 2>&1 | head -20

# Run B-parameter calibration and analysis on this CPU
calibrate-analyze: calibrate_B
	@echo "Running calibration (M1 + M4)..."
	./calibrate_B 1 > calibration.csv
	./calibrate_B 4 >> calibration.csv
	@echo "Analyzing..."
	python3 analyze_calibration.py calibration.csv

plot:
	python3 plot_results.py

clean:
	rm -f bench_avx2 bench_avx512 bench_topk calibrate_B
	rm -f cpu_*.csv calibration.csv *.png
