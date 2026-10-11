#ifndef EMBER_DEADLINE_EXACT_H
#define EMBER_DEADLINE_EXACT_H
#include <stddef.h>
#include <stdint.h>
typedef struct {
  int64_t cost, lower_bound, travel;
  uint64_t peak;
  uint32_t route_length, feasible, optimal, labels, states, runs, method,
      expanded;
} ember_deadline_stats;
/* mode 0: DP hybrid; 1: full DP; 2: corridor DP; 3: lazy A*;
   mode 4: unit-clock/truncated certificates; small DP or suffix-pruned A*
   followed by exact DP. No hidden malloc. */
int ember_deadline_exact(uint32_t n, const uint32_t *rows, const uint32_t *to,
                         const uint32_t *duration, const int64_t *weight,
                         const uint32_t *reverse_rows,
                         const uint32_t *reverse_to,
                         const uint32_t *reverse_duration,
                         const int64_t *reverse_weight, const uint8_t *goals,
                         uint32_t start, uint32_t deadline,
                         uint32_t label_limit, uint32_t mode, uint8_t *arena,
                         size_t capacity, size_t offset, uint32_t *route,
                         ember_deadline_stats *stats);
#endif
