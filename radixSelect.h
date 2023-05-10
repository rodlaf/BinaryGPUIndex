#pragma once

/*
  Copy the smallest k values and their corresponding keys into kSmallestValues
  and kSmallestKeys. Two pointers to allocated memory must be passed as 
  working memory of type unsigned integer and of size 
  (numValues * sizeof(unsigned)).

  All memory that is passed must be located on device.

  Note: It is possible to make this method retrieve the keys' corresponding
  vectors too. A new argument would be introduced with a name like 
  "passengerValues" of any type. This would be possible by the addition of two 
  more calls to thrust::copy_if. It is also worth mentioning here that if 
  the keys are made to be sequential and representative of indexes in an array,
  retrieval of vectors can be made easily much quicker.
*/
void radixSelect(unsigned *values, unsigned *keys, int numValues, int k,
                  unsigned *kSmallestValues, unsigned *kSmallestKeys, 
                  unsigned *workingMem1, unsigned *workingMem2); 

#include "radixSelect.cu"