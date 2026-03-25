/* Zen 4 (Ryzen 7950X) FFT configuration — NOT YET CALIBRATED.
 *
 * To generate this file:
 *   1. Build the calibration tool on the Zen 4 machine:
 *        gcc -O3 -march=znver4 -o calibrate tools/calibrate.c -lfftw3 -lm
 *   2. Run calibration (pin to one core, quiet machine):
 *        taskset -c 0 nice -20 ./calibrate
 *   3. Copy outputs:
 *        cp fft_config.h devices/zen4/fft_config.h
 *        cp fftw_wisdom.dat devices/zen4/fftw_wisdom.dat
 *   4. Build and measure platform constants:
 *        make DEVICE=zen4
 *        ./bench_grid profile
 *   5. Update the #define constants in this file based on profile output:
 *        - FMA_NS: from schoolbook benchmark (expect ~0.06-0.08 with AVX-512)
 *        - FFT_OVERHEAD_NS: from FFT overhead measurement
 *        - PAIRED_CACHED_CORR_RATIO: from phase split measurement
 *        - INDEP_PAIR_RATIO: from phase split measurement
 *   6. Rebuild and run crossover sweep:
 *        make DEVICE=zen4
 *        ./bench_grid crossover
 *   7. Update dispatch constants in icm.c if crossover changed.
 *   8. Verify and benchmark:
 *        ./bench_grid verify
 *        ./bench_grid
 *
 * See OPTIMIZATION_GUIDE.md for detailed tuning instructions.
 */
#error "Zen 4 fft_config.h not yet calibrated. Run: gcc -O3 -march=znver4 -o calibrate tools/calibrate.c -lfftw3 -lm && taskset -c 0 nice -20 ./calibrate"
