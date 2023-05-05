#pragma once

void radix_select(unsigned *values, unsigned *keys, int numValues, int k,
                  unsigned *kSmallestValues, unsigned *kSmallestKeys); 

#include "radix_select.inl"