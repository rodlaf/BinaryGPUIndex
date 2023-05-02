typedef unsigned int uint32_cu;

/**
 * Collect histogram of values into 256 bins representing 8 bits within integers
 *
 * @param d_data integers to be collected
 * @param d_total 
 * @param histogram
 * @param prefix
 */
__global__ void collect_histogram(uint32_cu *d_data, uint32_cu *d_total,
                                  uint32_cu *histogram, uint32_cu *prefix,
                                  int mask) {
  __shared__ int s_histogram[256];

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int id = threadIdx.x;

  if (id < 256) {
    s_histogram[id] = 0;
  }

  __syncthreads();

  int n = d_total[mask + 8];
  int m = d_total[mask + 9];

  while (idx < n) {
    if (idx >= m) {
      uint32_cu data = *(uint32_cu *)&d_data[idx];

      uint32_cu bin = (data >> mask) & 0xff;
      atomicAdd(&s_histogram[bin], 1);
    }
    idx += blockDim.x * gridDim.x;
  }

  __syncthreads();

  if (id < 256) {
    prefix[id + 256 * blockIdx.x] = atomicAdd(&histogram[id], s_histogram[id]);
  }
}

/**
 * Set limit
 *
 * @param histogram
 * @param topk
 * @param limits
 * @param d_total
 * @param mask
 */
__global__ void set_limit(uint32_cu *histogram, int topk, uint32_cu *limits,
                          uint32_cu *d_total, int mask) {

  int i = 0;

  uint32_cu m = d_total[mask + 9];
  uint32_cu total = 0;
  uint32_cu old_total = 0;

  while (i < 256) // TODO: use atomic function to unroll this loop
  {
    // NOTICE: if we change the logical in here, we may pervent data lost, while
    // the speed lost is huge
    total += histogram[i];
    histogram[i] = old_total;

    /// if find the pivot in histogram [i, i + 1]:
    if (total >= topk - m || i == 255) // TODO: whether is >= or >
    {
      limits[1] =
          limits[1] - ((static_cast<uint32_cu>(0xff - i)
                        << mask)); ///> upper bound of rest numbers (value)
      //  limits[0] = limits[0] + ((static_cast<uint32_cu>(i) << mask)); ///>
      //  lower bound ... useless in program.
      /// numbers rest is in address [lower bound, upperbound)
      d_total[mask] =
          total + m; ///> upper bound of rest (undetermined) numbers (address)
      d_total[mask + 1] =
          old_total + m; ///> lower bound of rest numbers (address)
      break;
    }
    old_total = total;
    i++;
  }
}

/**
 * Relocate
 *
 * @param d_data
 * @param d_data2
 * @param d_total 
 * @param prefix
 * @param limits
 * @param histogram
 * @param mask
 */
__global__ void relocation(uint32_cu *d_data, uint32_cu *d_data2,
                           uint32_cu *d_total, uint32_cu *prefix,
                           uint32_cu *limits, uint32_cu *histogram,
                           int mask) {
  __shared__ uint32_cu s_histogram[256];
  uint32_cu upper = limits[1];

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int id = threadIdx.x;

  uint32_cu n = d_total[mask + 8]; // load last time upper bond
  uint32_cu m = d_total[mask + 9]; // load last time lower bond

  if (id < 256) {
    s_histogram[id] = prefix[id + 256 * blockIdx.x] + histogram[id] + m;
  }

  __syncthreads();

  while (idx < n) {
    uint32_cu data = *(uint32_cu *)&d_data[idx];

    if (idx < m) {
      d_data2[idx] = data;
    } else {
      if (data <= upper) {
        uint32_cu bin = (data >> mask) & 0xff;
        int index = atomicAdd(&s_histogram[bin], 1);
        d_data2[index] = data;
      }
    }

    idx += blockDim.x * gridDim.x;
  }
}

/**
 * Assign
 *
 * @param x
 * @param n
 * @param value
 */
__global__ void assign(uint32_cu *x, int n, uint32_cu value) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  while (idx < n) {
    x[idx] = value;
    idx += blockDim.x * gridDim.x;
  }
}

/**
 * Run radix select
 *
 * @param d_data
 * @param n
 * @param result
 * @param topk
 */
void radix_select(uint32_cu *d_data, int n, uint32_cu *result, int topk) {
  uint32_cu *d_data1, *d_data2, *d_limits, *histogram, *prefix, *d_total;

  cudaMalloc(&d_data1, n * sizeof(uint32_cu));
  cudaMalloc(&d_data2, n * sizeof(uint32_cu));
  cudaMalloc(&d_limits, 2 * sizeof(uint32_cu));

  cudaMalloc(&histogram, 256 * sizeof(uint32_cu));
  cudaMalloc(&prefix, 256 * 90 * sizeof(uint32_cu));
  cudaMalloc(&d_total, (sizeof(uint32_cu) * 8 + 10) * sizeof(uint32_cu));

  assign<<<1, 1>>>(d_total + sizeof(uint32_cu) * 8, 1, (uint32_cu)n);
  assign<<<1, 1>>>(d_total + sizeof(uint32_cu) * 8 + 1, 1, (uint32_cu)0);
  assign<<<1, 1>>>(d_limits + 1, 1, 0);

  for (int mask = sizeof(uint32_cu) * 8 - 8; mask >= 0; mask -= 8) {
    assign<<<1, 256>>>(histogram, 256, (uint32_cu)0);

    if (mask == sizeof(uint32_cu) * 8 - 8)
      collect_histogram<<<90, 1024>>>(d_data, d_total, histogram, prefix, mask);
    else
      collect_histogram<<<90, 1024>>>(d_data1, d_total, histogram, prefix,
                                      mask);

    set_limit<<<1, 1>>>(histogram, topk, d_limits, d_total, mask);

    if (mask == sizeof(uint32_cu) * 8 - 8)
      relocation<<<90, 1024>>>(d_data, d_data2, d_total, prefix, d_limits,
                               histogram, mask);
    else
      relocation<<<90, 1024>>>(d_data1, d_data2, d_total, prefix, d_limits,
                               histogram, mask);

    uint32_cu *temp = d_data1;
    d_data1 = d_data2;
    d_data2 = temp;
  }

  cudaMemcpy(result, d_data1, topk * sizeof(uint32_cu),
             cudaMemcpyDeviceToHost);

  cudaFree(d_data1);
  cudaFree(d_data2);
  cudaFree(d_limits);
}
