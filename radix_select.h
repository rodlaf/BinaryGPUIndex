#pragma once
/**
 * \brief a simple version of top-k
 * 
 * \tparam T now support (signed, unsigned)(char, short, int, long long) and (double, float, float16)
 * \
 * \warning origin data **is not reserved**! If you want to keep the origin data, store it some wherer else.
 * \note result topk is unordered.
 */
template <typename T>
void radix_select(T *d_data, int n, T *result, int topk);

/**
 * \brief a detail version of top-k
 *
 *
 * you may manage memory yourself with d_data1 same size as d_data, d_params
 * with 90*(256 + 256 * 90 + sizeof(T) * 8 + 10) * sizeof(unsigned), d_limits:2
 * * sizeof(U)
 *
 * \tparam T support (signed, unsigned)(char, short, int, long long), (double,
 * float, float 16) is not supported, reinterpertion cast them to long long, int
 * and short \tprarm U should be coresponding unsigned type of T (unsigned)
 * (char, short, int, long long)
 */
template <typename T, typename U>
void radix_select_detail(T *d_data, int n, T *d_data1, T *d_data2, int topk, unsigned *d_params, U *d_limits);


#include "radix_select.inl"