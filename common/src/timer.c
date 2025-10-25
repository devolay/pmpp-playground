#define _POSIX_C_SOURCE 199309L  // enable clock_gettime on most systems

#include "timer.h"
#include <time.h>
#include <stdint.h>

static uint64_t ns_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

void pmpp_timer_start(pmpp_timer_t* t) {
    t->start_ns = ns_now();
}

void pmpp_timer_stop(pmpp_timer_t* t) {
    t->stop_ns = ns_now();
}

double pmpp_timer_ms(const pmpp_timer_t* t) {
    return (double)(t->stop_ns - t->start_ns) / 1.0e6;
}
