/*
 * Lattice — 3D Numerical Relativity
 * Per-kernel profiling via clock_gettime (CLOCK_MONOTONIC).
 *
 * Usage:
 *   TIMER_START(rhs);
 *   backend_compute_rhs(...);
 *   double ms;
 *   TIMER_STOP(rhs, ms);
 *   printf("RHS: %.2f ms\n", ms);
 */

#ifndef LATTICE_TIMER_H
#define LATTICE_TIMER_H

#include <time.h>

#define TIMER_START(name) \
    struct timespec _timer_##name##_start; \
    clock_gettime(CLOCK_MONOTONIC, &_timer_##name##_start)

#define TIMER_STOP(name, elapsed_ms) do { \
    struct timespec _timer_##name##_end; \
    clock_gettime(CLOCK_MONOTONIC, &_timer_##name##_end); \
    (elapsed_ms) = (_timer_##name##_end.tv_sec - _timer_##name##_start.tv_sec) * 1.0e3 \
                 + (_timer_##name##_end.tv_nsec - _timer_##name##_start.tv_nsec) * 1.0e-6; \
} while(0)

/* Accumulating timer for repeated measurements */
typedef struct {
    double total_ms;
    int    count;
} timer_accum_t;

static inline void timer_accum_reset(timer_accum_t *t)
{
    t->total_ms = 0.0;
    t->count = 0;
}

static inline void timer_accum_add(timer_accum_t *t, double ms)
{
    t->total_ms += ms;
    t->count++;
}

static inline double timer_accum_avg(const timer_accum_t *t)
{
    return t->count > 0 ? t->total_ms / t->count : 0.0;
}

#endif /* LATTICE_TIMER_H */
