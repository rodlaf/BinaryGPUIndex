
template <typename T, typename U>
static __global__ void collect_histogram(T *d_data, unsigned *d_total, unsigned *histogram, unsigned *prefix, int mask);

template <typename T, typename U>
static __global__ void set_limit(unsigned *histogram, int topk, U *limits, unsigned *d_total, int mask);

template <typename T, typename U>
static __global__ void relocation(T *d_data, T *d_data2, unsigned *d_total, unsigned *prefix, U *limits, unsigned *histogram, int mask);

template <typename T>
static __host__ __device__ __forceinline__ bool is_signed();

template <typename T>
static __global__ void assign(T *x, int n, T value);


template <typename T>
void radix_select(T *d_data, int n, T *result, int topk)
{
    T *d_data1, *d_data2, *d_limits;
    cudaMalloc(&d_data1, n * sizeof(T));
    cudaMalloc(&d_data2, n * sizeof(T));
    cudaMalloc(&d_limits, 2 * sizeof(T));
    unsigned *d_params;
    cudaMalloc(&d_params, (256 + 256 * 90 + sizeof(T) * 8 + 10) * sizeof(unsigned));
    if (is_signed<T>())
    {
        if (sizeof(T) == 1)
            radix_select_detail<signed char, unsigned char>((signed char *)d_data, n, (signed char *)d_data1, (signed char *)d_data2, topk, d_params, (unsigned char *)d_limits);
        if (sizeof(T) == 2)
            radix_select_detail<signed short, unsigned short>((signed short *)d_data, n, (signed short *)d_data1, (signed short *)d_data2, topk, d_params, (unsigned short *)d_limits);
        if (sizeof(T) == 4)
            radix_select_detail<signed int, unsigned int>((signed int *)d_data, n, (signed int *)d_data1, (signed int *)d_data2, topk, d_params, (unsigned int *)d_limits);
        if (sizeof(T) == 8)
            radix_select_detail<signed long long, unsigned long long>((signed long long *)d_data, n, (signed long long *)d_data1, (signed long long *)d_data2, topk, d_params, (unsigned long long *)d_limits);
    }
    else
    {
        if (sizeof(T) == 1)
            radix_select_detail<unsigned char, unsigned char>((unsigned char *)d_data, n, (unsigned char *)d_data1, (unsigned char *)d_data2, topk, d_params, (unsigned char *)d_limits);
        if (sizeof(T) == 2)
            radix_select_detail<unsigned short, unsigned short>((unsigned short *)d_data, n, (unsigned short *)d_data1, (unsigned short *)d_data2, topk, d_params, (unsigned short *)d_limits);
        if (sizeof(T) == 4)
            radix_select_detail<unsigned int, unsigned int>((unsigned int *)d_data, n, (unsigned int *)d_data1, (unsigned int *)d_data2, topk, d_params, (unsigned int *)d_limits);
        if (sizeof(T) == 8)
            radix_select_detail<unsigned long long, unsigned long long>((unsigned long long *)d_data, n, (unsigned long long *)d_data1, (unsigned long long *)d_data2, topk, d_params, (unsigned long long *)d_limits);
    }
    if (sizeof(T) % 2)
    {
        cudaMemcpy(result, d_data2, topk * sizeof(T), cudaMemcpyDeviceToHost);
    }
    else
    {
        cudaMemcpy(result, d_data1, topk * sizeof(T), cudaMemcpyDeviceToHost);
    }
    cudaFree(d_data1);
    cudaFree(d_data2);
    cudaFree(d_params);
    cudaFree(d_limits);
}

template <typename T>
static __host__ __device__ __forceinline__ bool is_signed()
{
    return T(-1) < T(0);
}

template <typename T, typename U>
void radix_select_detail(T *d_data, int n, T *d_data1, T *d_data2, int topk, unsigned *d_params, U *d_limits)
{
    // printf("radixselcct_detail\n");
    unsigned *histogram = d_params; // 256
    unsigned *prefix = histogram + 256;
    unsigned *d_total = prefix + 256 * 90;
    assign<<<1, 1>>>(d_total + sizeof(T) * 8, 1, (unsigned)n);
    assign<<<1, 1>>>(d_total + sizeof(T) * 8 + 1, 1, (unsigned)0);
    assign<U><<<1, 1>>>(d_limits + 1, 1, (U)~U(0)); // TODO: limits is of type T
    for (int mask = sizeof(T) * 8 - 8; mask >= 0; mask -= 8)
    {
        assign<unsigned><<<1, 256>>>(histogram, 256, (unsigned)0);
        if (mask == sizeof(T) * 8 - 8)
            collect_histogram<T, U><<<90, 1024>>>(d_data, d_total, histogram, prefix, mask);
        else
            collect_histogram<T, U><<<90, 1024>>>(d_data1, d_total, histogram, prefix, mask);
        set_limit<T, U><<<1, 1>>>(histogram, topk, d_limits, d_total, mask);
        if (mask == sizeof(T) * 8 - 8)
            relocation<T, U><<<90, 1024>>>(d_data, d_data2, d_total, prefix, d_limits, histogram, mask);
        else
            relocation<T, U><<<90, 1024>>>(d_data1, d_data2, d_total, prefix, d_limits, histogram, mask);
        //view<<<1, 32>>>(d_data2, n);
        T *temp = d_data1;
        d_data1 = d_data2;
        d_data2 = temp;
        //break;
    }
}

template <typename T, typename U>
static __global__ void collect_histogram(T *d_data, unsigned *d_total, unsigned *histogram, unsigned *prefix, int mask)
{
    __shared__ int s_histogram[256];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int id = threadIdx.x;
    if (id < 256)
    {
        s_histogram[id] = 0;
    }
    __syncthreads();
    int n = d_total[mask + 8];
    int m = d_total[mask + 9];
    while (idx < n)
    {
        if (idx >= m)
        {
            U data = *(U *)&d_data[idx];
            if (is_signed<T>())
            {
                if (mask == sizeof(T) * 8 - 8)
                {
                    data ^= ~(U(1) << (sizeof(U) * 8 - 1));
                    data = ~data;
                }
            }
            unsigned bin = (data >> mask) & 0xff;
            atomicAdd(&s_histogram[bin], 1);
        }
        idx += blockDim.x * gridDim.x;
    }
    __syncthreads();
    if (id < 256)
    {
        prefix[id + 256 * blockIdx.x] = atomicAdd(&histogram[id], s_histogram[id]);
    }
}

template <typename T, typename U>
static __global__ void set_limit(unsigned *histogram, int topk, U *limits, unsigned *d_total, int mask)
{

    int i = 0;
    unsigned m = d_total[mask + 9]; ///> LAST TIME lower bound of numbers of small set
    unsigned total = 0;             ///> accumulater
    unsigned old_total = 0;

    while (i < 256) // TODO: use atomic function to unroll this loop
    {
        // NOTICE: if we change the logical in here, we may pervent data lost, while the speed lost is huge
        total += histogram[i];
        histogram[i] = old_total;
        /// if find the pivot in histogram [i, i + 1]:
        if (total >= topk - m || i == 255) //TODO: whether is >= or >
        {
            limits[1] = limits[1] - ((static_cast<U>(0xff - i) << mask)); ///> upper bound of rest numbers (value)
            //  limits[0] = limits[0] + ((static_cast<T>(i) << mask));  ///> lower bound ... useless in program.
            /// numbers rest is in address [lower bound, upperbound)
            d_total[mask] = total + m;         ///> upper bound of rest (undetermined) numbers (address)
            d_total[mask + 1] = old_total + m; ///> lower bound of rest numbers (address)
            break;
        }
        old_total = total;
        i++;
    }
    // printf("%d:%d,%d,%d,bin%d\n", __LINE__, total + m, old_total + m, limits[1], i);
}
template <typename T, typename U>
static __global__ void relocation(T *d_data, T *d_data2, unsigned *d_total, unsigned *prefix, U *limits, unsigned *histogram, int mask)
{
    __shared__ unsigned s_histogram[256];
    U upper = limits[1];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int id = threadIdx.x;

    unsigned n = d_total[mask + 8]; // load last time upper bond
    unsigned m = d_total[mask + 9]; // load last time lower bond
    if (id < 256)
    {
        s_histogram[id] = prefix[id + 256 * blockIdx.x] + histogram[id] + m;
    }
    __syncthreads();
    while (idx < n)
    {
        U data = *(U *)&d_data[idx];
        if (idx < m)
        {
            // TODO: if topk is huge, we do not need to move data back and forth.
            if (is_signed<T>())
            {
                if (mask == 0)
                {
                    data ^= ~(U(1) << (sizeof(U) * 8 - 1));
                    data = ~data;
                }
            }
            d_data2[idx] = data;
        }
        else
        {
            if (is_signed<T>())
            {
                if (mask == sizeof(T) * 8 - 8)
                {
                    data ^= ~(U(1) << (sizeof(U) * 8 - 1));
                    data = ~data;
                }
                if (data <= upper)
                {
                    unsigned bin = (data >> mask) & 0xff;
                    int index = atomicAdd(&s_histogram[bin], 1);
                    if (mask == 0)
                    {
                        data ^= ~(U(1) << (sizeof(U) * 8 - 1));
                        data = ~data;
                    }
                    d_data2[index] = data;
                }
            }
            else
            {
                if (data <= upper)
                {
                    unsigned bin = (data >> mask) & 0xff;
                    int index = atomicAdd(&s_histogram[bin], 1);
                    d_data2[index] = data;
                    // printf("%d: %d, %d, %d, %d\n", __LINE__, index, idx, bin, histogram[bin]);
                }
            }
        }
        idx += blockDim.x * gridDim.x;
    }
}

template <typename T>
static __global__ void assign(T *x, int n, T value)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    while (idx < n)
    {
        x[idx] = value;
        idx += blockDim.x * gridDim.x;
    }
}
