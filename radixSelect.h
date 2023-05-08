#pragma once

/*
  Copy the smallest k values and their corresponding keys into kSmallestValues
  and kSmallestKeys. Two pointers to allocated memory must be passed as 
  working memory of type unsigned integer and of size 
  (numValues * sizeof(unsigned)).

  All memory that is passed must be located on device.
*/
void radixSelect(unsigned *values, unsigned *keys, int numValues, int k,
                  unsigned *kSmallestValues, unsigned *kSmallestKeys, 
                  unsigned *workingMem1, unsigned *workingMem2); 

#include "radixSelect.inl"