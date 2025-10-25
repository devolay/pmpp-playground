#pragma once
#include <stdint.h>

typedef struct {
    uint64_t start_ns;
    uint64_t stop_ns;
} pmpp_timer_t;

void pmpp_timer_start(pmpp_timer_t* t);
void pmpp_timer_stop(pmpp_timer_t* t);
double pmpp_timer_ms(const pmpp_timer_t* t);