#pragma once
/**
 * \brief a simple version of top-k
 *
 * \tparam T now support (signed, unsigned)(char, short, int, long long) and
 * (double, float, float16)
 * \
 * \warning origin data **is not reserved**! If you want to keep the origin
 * data, store it some wherer else. \note result topk is unordered.
 */
void radix_select(unsigned int *d_data, int n, unsigned int *result, int topk);

#include "radix_select.inl"