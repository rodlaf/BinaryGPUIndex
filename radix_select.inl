__global__ void collect_histogram(unsigned int *d_data, unsigned *d_total,
                                  unsigned *histogram, unsigned *prefix,
                                  int mask);

__global__ void set_limit(unsigned *histogram, int topk, unsigned int *limits,
                          unsigned *d_total, int mask);

__global__ void relocation(unsigned int *d_data, unsigned int *d_data2,
                           unsigned *d_total, unsigned *prefix,
                           unsigned int *limits, unsigned *histogram, int mask);

__global__ void assign(unsigned int *x, int n, unsigned int value);

void radix_select(unsigned int *d_data, int n, unsigned int *result, int topk) {
  unsigned int *d_data1, *d_data2, *d_limits;
  cudaMalloc(&d_data1, n * sizeof(unsigned int));
  cudaMalloc(&d_data2, n * sizeof(unsigned int));
  cudaMalloc(&d_limits, 2 * sizeof(unsigned int));
  unsigned *d_params;
  cudaMalloc(&d_params, (256 + 256 * 90 + sizeof(unsigned int) * 8 + 10) *
                            sizeof(unsigned));

  unsigned *histogram = d_params; // 256
  unsigned *prefix = histogram + 256;
  unsigned *d_total = prefix + 256 * 90;
  assign<<<1, 1>>>(d_total + sizeof(unsigned int) * 8, 1, (unsigned)n);
  assign<<<1, 1>>>(d_total + sizeof(unsigned int) * 8 + 1, 1, (unsigned)0);
  assign<<<1, 1>>>(d_limits + 1, 1, 0);

  for (int mask = sizeof(unsigned int) * 8 - 8; mask >= 0; mask -= 8) {
    assign<<<1, 256>>>(histogram, 256, (unsigned)0);
    if (mask == sizeof(unsigned int) * 8 - 8)
      collect_histogram<<<90, 1024>>>(d_data, d_total, histogram, prefix, mask);
    else
      collect_histogram<<<90, 1024>>>(d_data1, d_total, histogram, prefix,
                                      mask);

    set_limit<<<1, 1>>>(histogram, topk, d_limits, d_total, mask);

    if (mask == sizeof(unsigned int) * 8 - 8)
      relocation<<<90, 1024>>>(d_data, d_data2, d_total, prefix, d_limits,
                               histogram, mask);
    else
      relocation<<<90, 1024>>>(d_data1, d_data2, d_total, prefix, d_limits,
                               histogram, mask);

    unsigned int *temp = d_data1;
    d_data1 = d_data2;
    d_data2 = temp;
  }

  cudaMemcpy(result, d_data1, topk * sizeof(unsigned int),
             cudaMemcpyDeviceToHost);

  cudaFree(d_data1);
  cudaFree(d_data2);
  cudaFree(d_params);
  cudaFree(d_limits);
}

__global__ void collect_histogram(unsigned int *d_data, unsigned *d_total,
                                  unsigned *histogram, unsigned *prefix,
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
      unsigned int data = *(unsigned int *)&d_data[idx];

      unsigned bin = (data >> mask) & 0xff;
      atomicAdd(&s_histogram[bin], 1);
    }
    idx += blockDim.x * gridDim.x;
  }
  __syncthreads();
  if (id < 256) {
    prefix[id + 256 * blockIdx.x] = atomicAdd(&histogram[id], s_histogram[id]);
  }
}

__global__ void set_limit(unsigned *histogram, int topk, unsigned int *limits,
                          unsigned *d_total, int mask) {

  int i = 0;
  unsigned m =
      d_total[mask + 9]; ///> LAST TIME lower bound of numbers of small set
  unsigned total = 0;    ///> accumulater
  unsigned old_total = 0;

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
          limits[1] - ((static_cast<unsigned int>(0xff - i)
                        << mask)); ///> upper bound of rest numbers (value)
      //  limits[0] = limits[0] + ((static_cast<unsigned int>(i) << mask)); ///>
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

__global__ void relocation(unsigned int *d_data, unsigned int *d_data2,
                           unsigned *d_total, unsigned *prefix,
                           unsigned int *limits, unsigned *histogram,
                           int mask) {
  __shared__ unsigned s_histogram[256];
  unsigned int upper = limits[1];
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int id = threadIdx.x;

  unsigned n = d_total[mask + 8]; // load last time upper bond
  unsigned m = d_total[mask + 9]; // load last time lower bond
  if (id < 256) {
    s_histogram[id] = prefix[id + 256 * blockIdx.x] + histogram[id] + m;
  }
  __syncthreads();
  while (idx < n) {
    unsigned int data = *(unsigned int *)&d_data[idx];
    if (idx < m) {
      d_data2[idx] = data;
    } else {
      if (data <= upper) {
        unsigned bin = (data >> mask) & 0xff;
        int index = atomicAdd(&s_histogram[bin], 1);
        d_data2[index] = data;
      }
    }
    idx += blockDim.x * gridDim.x;
  }
}

__global__ void assign(unsigned int *x, int n, unsigned int value) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  while (idx < n) {
    x[idx] = value;
    idx += blockDim.x * gridDim.x;
  }
}
