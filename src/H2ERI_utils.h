#ifndef __H2ERI_UTILS_H__
#define __H2ERI_UTILS_H__

#include "H2ERI_typedef.h"

static inline void atomic_add_f64(volatile double *global_addr, double addend)
{
    uint64_t expected_value, new_value;
    do {
        double old_value = *global_addr;
        double tmp_value;
        #ifdef __INTEL_COMPILER
        expected_value = _castf64_u64(old_value);
        new_value      = _castf64_u64(old_value + addend);
        #else
        expected_value = *((uint64_t *) &old_value);
        tmp_value      = old_value + addend;
        new_value      = *((uint64_t *) &tmp_value);
        #endif
    } while (!__sync_bool_compare_and_swap((volatile uint64_t *) global_addr, expected_value, new_value));
}

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

#endif
