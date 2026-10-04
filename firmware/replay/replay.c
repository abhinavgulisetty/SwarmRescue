/* Freestanding conformance harness; never a whole-robot RAM/timing claim. */
#include <stdint.h>
#include <limits.h>
#include "log_lmdp.h"
#include "deadline.h"
#include "fixtures.h"

static int32_t current[REPLAY_MAX_N], following[REPLAY_MAX_N];
static int32_t fast_time[REPLAY_MAX_N], previous[REPLAY_MAX_N], route[REPLAY_MAX_N];
static int64_t fast_cost[REPLAY_MAX_N];
static uint8_t visited[REPLAY_MAX_N];
static ember_deadline_label labels[REPLAY_MAX_LABELS];

#ifdef __arm__
extern uint32_t _stack_top, _stack_bottom, _bss_start, _bss_end;
extern uint32_t _data_start, _data_end, _data_load;
static void semihost(uint32_t operation, const void *argument) {
    register uint32_t r0 __asm__("r0") = operation;
    register const void *r1 __asm__("r1") = argument;
    __asm__ volatile("bkpt 0xab" : "+r"(r0) : "r"(r1) : "memory");
}
static void write_text(const char *text) { semihost(4, text); }
static void finish(uint32_t code) {
    uint32_t parameters[2] = {0x20026, code};
    semihost(0x20, parameters);
    for (;;) {}
}
static volatile uint32_t *canary_limit;
static void start_canary(void) {
    uint32_t sp;
    __asm__ volatile("mov %0, sp" : "=r"(sp));
    canary_limit = (volatile uint32_t *)(sp - 128);
    for (volatile uint32_t *p = &_stack_bottom; p < canary_limit; ++p)
        *p = 0xa5a5a5a5;
}
static uint32_t stack_high_water(void) {
    volatile uint32_t *p = &_stack_bottom;
    while (p < canary_limit && *p == 0xa5a5a5a5) ++p;
    return (uint32_t)((uintptr_t)&_stack_top - (uintptr_t)p);
}
void Reset_Handler(void);
__attribute__((section(".vectors"),used))
const uintptr_t vectors[] = {(uintptr_t)&_stack_top, (uintptr_t)Reset_Handler};
#else
#include <stdio.h>
static void write_text(const char *text) { fputs(text, stdout); }
static void start_canary(void) {}
static uint32_t stack_high_water(void) { return 0; }
#endif

static void write_uint(uint32_t value) {
    char text[12]; unsigned int cursor = 10;
    text[11] = 0;
    do { text[cursor--] = (char)('0' + value % 10); value /= 10; } while (value);
    write_text(&text[cursor + 1]);
}

static int replay(void) {
    start_canary();
    uint32_t failures = 0, comparisons = 0;
    for (unsigned int c = 0; c < REPLAY_LOG_CASES; ++c) {
        const log_fixture *f = &log_cases[c];
        for (int32_t i = 0; i < f->n; ++i) current[i] = f->exits[i] ? 0 : INT32_MAX;
        int fixed = 0;
        for (int32_t sweep = 0; sweep < f->max_sweeps; ++sweep) {
            if (ember_log_q16_sweep(f->n, f->edges, f->rows, f->columns,
                    f->passive, f->cost, current, f->exits, f->deaths, following)) {
                failures++; break;
            }
            fixed = 1;
            for (int32_t i = 0; i < f->n; ++i) {
                if (current[i] != following[i]) fixed = 0;
                current[i] = following[i];
            }
            if (fixed) break;
        }
        if (!fixed) failures++;
        for (int32_t i = 0; i < f->n; ++i) {
            comparisons += 2;
            failures += current[i] != f->expected[i];
            failures += ember_log_q16_next(i, f->n, f->rows, f->columns,
                            f->passive, current) != f->actions[i];
        }
    }
    for (unsigned int c = 0; c < REPLAY_DEADLINE_CASES; ++c) {
        const deadline_fixture *f = &deadline_cases[c];
        ember_deadline_result result;
        int status = ember_deadline_route(f->n, f->edges, f->rows, f->columns,
            f->times, f->cost, f->exits, 0, f->deadline, f->capacity,
            labels, fast_time, fast_cost, previous, visited, route, &result);
        comparisons += 8;
        failures += status != 0;
        if (status) continue;
        failures += result.travel_steps != f->time;
        failures += result.objective_q16 != f->objective;
        failures += result.route_length != f->route_length;
        failures += result.feasible != f->feasible;
        failures += result.optimal != f->optimal;
        failures += result.resource_fallback != f->fallback;
        failures += result.labels_used != f->labels_used;
        for (int32_t i = 0; i < f->route_length; ++i) {
            comparisons++;
            failures += route[i] != f->route[i];
        }
        comparisons++;
        failures += ember_deadline_accept(f->n, f->edges, f->rows, f->columns,
            f->times, f->exits, 0, f->deadline, route, result.route_length) != f->feasible;
    }
    uint32_t stack = stack_high_water();
    write_text("{\"failures\":"); write_uint(failures);
    write_text(",\"comparisons\":"); write_uint(comparisons);
    write_text(",\"log_cases\":"); write_uint(REPLAY_LOG_CASES);
    write_text(",\"deadline_cases\":"); write_uint(REPLAY_DEADLINE_CASES);
    write_text(",\"observed_stack_high_water_bytes\":"); write_uint(stack);
    write_text("}\n");
    return failures ? 1 : 0;
}

#ifdef __arm__
void Reset_Handler(void) {
    uint32_t *source = &_data_load;
    for (uint32_t *p = &_data_start; p < &_data_end; ++p) *p = *source++;
    for (uint32_t *p = &_bss_start; p < &_bss_end; ++p) *p = 0;
    finish((uint32_t)replay());
}
#else
int main(void) { return replay(); }
#endif
