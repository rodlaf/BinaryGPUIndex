typedef unsigned int uint32_cu;

__global__ void collect_histogram(uint32_cu *d_data, uint32_cu *d_total,
                                  uint32_cu *histogram, uint32_cu *prefix,
                                  int mask) {
  __shared__ int s_histogram[256];

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int id = threadIdx.x;

  if (id < 256)
    s_histogram[id] = 0;

  __syncthreads();

  int n = d_total[mask + 8];
  int m = d_total[mask + 9];

  while (idx < n) {
    if (idx >= m) {
      uint32_cu data = d_data[idx];
      uint32_cu bin = (data >> mask) & 0xff;

      atomicAdd(&s_histogram[bin], 1);
    }

    idx += blockDim.x * gridDim.x;
  }

  __syncthreads();

  if (id < 256)
    prefix[id + 256 * blockIdx.x] = atomicAdd(&histogram[id], s_histogram[id]);
}

__global__ void set_limit(uint32_cu *histogram, int topk, uint32_cu *limit,
                          uint32_cu *d_total, int mask) {

  int i = 0;

  uint32_cu m = d_total[mask + 9];
  uint32_cu total = 0;
  uint32_cu old_total = 0;

  while (i < 256) {
    total += histogram[i];
    histogram[i] = old_total;

    if (total >= topk - m || i == 255) {
      *limit -= ((static_cast<uint32_cu>(0xff - i) << mask));

      d_total[mask] = total + m;
      d_total[mask + 1] = old_total + m;

      break;
    }

    old_total = total;
    i++;
  }
}

__global__ void relocation(uint32_cu *d_data, uint32_cu *d_data2,
                           uint32_cu *d_total, uint32_cu *prefix,
                           uint32_cu *limit, uint32_cu *histogram, int mask) {
  __shared__ uint32_cu s_histogram[256];
  uint32_cu upper = *limit;

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int id = threadIdx.x;

  uint32_cu n = d_total[mask + 8];
  uint32_cu m = d_total[mask + 9];

  if (id < 256) {
    s_histogram[id] = prefix[id + 256 * blockIdx.x] + histogram[id] + m;
  }

  __syncthreads();

  while (idx < n) {
    uint32_cu data = d_data[idx];

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

__global__ void assign(uint32_cu *x, int n, uint32_cu value) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  while (idx < n) {
    x[idx] = value;
    idx += blockDim.x * gridDim.x;
  }
}

void radix_select(uint32_cu *d_data, int n, uint32_cu *result, int topk) {
  uint32_cu *d_data1, *d_data2, *d_limit, *histogram, *prefix, *d_total;

  cudaMalloc(&d_data1, n * sizeof(uint32_cu));
  cudaMalloc(&d_data2, n * sizeof(uint32_cu));
  cudaMalloc(&d_limit, sizeof(uint32_cu));

  cudaMalloc(&histogram, 256 * sizeof(uint32_cu));
  cudaMalloc(&prefix, 256 * 90 * sizeof(uint32_cu));
  cudaMalloc(&d_total, (sizeof(uint32_cu) * 8 + 10) * sizeof(uint32_cu));

  assign<<<1, 1>>>(d_total + sizeof(uint32_cu) * 8, 1, (uint32_cu)n);
  assign<<<1, 1>>>(d_total + sizeof(uint32_cu) * 8 + 1, 1, (uint32_cu)0);
  assign<<<1, 1>>>(d_limit, 1, 0);

  // iterate over segments of 8 bits
  for (int mask = sizeof(uint32_cu) * 8 - 8; mask >= 0; mask -= 8) {
    assign<<<1, 256>>>(histogram, 256, (uint32_cu)0);

    if (mask == sizeof(uint32_cu) * 8 - 8)
      collect_histogram<<<90, 1024>>>(d_data, d_total, histogram, prefix, mask);
    else
      collect_histogram<<<90, 1024>>>(d_data1, d_total, histogram, prefix,
                                      mask);

    set_limit<<<1, 1>>>(histogram, topk, d_limit, d_total, mask);

    if (mask == sizeof(uint32_cu) * 8 - 8)
      relocation<<<90, 1024>>>(d_data, d_data2, d_total, prefix, d_limit,
                               histogram, mask);
    else
      relocation<<<90, 1024>>>(d_data1, d_data2, d_total, prefix, d_limit,
                               histogram, mask);

    uint32_cu *temp = d_data1;
    d_data1 = d_data2;
    d_data2 = temp;
  }

  cudaMemcpy(result, d_data1, topk * sizeof(uint32_cu), cudaMemcpyDeviceToHost);

  cudaFree(d_data1);
  cudaFree(d_data2);
  cudaFree(d_limit);
}
