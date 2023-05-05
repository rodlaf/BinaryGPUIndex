#pragma once

typedef unsigned int uint32_cu;

void radix_select(uint32_cu *values, int *keys, int numValues, int k,
                       uint32_cu *kSmallestValues, int *kSmallestKeys); 

#include "radix_select.inl"