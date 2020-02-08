#include <gtest/gtest.h>
#include <ATen/ATen.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <ATen/cuda/CUDAContext.h>

using namespace at::native::memory;
__managed__ double4 buffer1[1024];
__managed__ double4 buffer2[1024];

void reset_buffers() {
  for (int i = 0; i < 1024; i++) {
    buffer1[i].x = i;
    buffer1[i].y = i + 0.1;
    buffer1[i].z = i + 0.2;
    buffer1[i].w = i + 0.3;

    buffer2[2].x = -i;
    buffer2[2].y = -(i + 0.1);
    buffer2[2].z = -(i + 0.2);
    buffer2[2].w = -(i + 0.3);
  }
}

TEST(TestVectorizedMemoryAccess, CanVectorizeUpTo) {
  char *ptr = reinterpret_cast<char *>(buffer1);

  ASSERT_EQ(can_vectorize_up_to<bool>(ptr), 4);
  ASSERT_EQ(can_vectorize_up_to<int8_t>(ptr), 4);
  ASSERT_EQ(can_vectorize_up_to<int16_t>(ptr), 4);
  ASSERT_EQ(can_vectorize_up_to<int>(ptr), 4);
  ASSERT_EQ(can_vectorize_up_to<int64_t>(ptr), 4);

  ASSERT_EQ(can_vectorize_up_to<bool>(ptr + 1), 1);
  ASSERT_EQ(can_vectorize_up_to<int8_t>(ptr + 1), 1);

  ASSERT_EQ(can_vectorize_up_to<bool>(ptr + 2), 2);
  ASSERT_EQ(can_vectorize_up_to<int8_t>(ptr + 2), 2);
  ASSERT_EQ(can_vectorize_up_to<int16_t>(ptr + 2), 1);

  ASSERT_EQ(can_vectorize_up_to<bool>(ptr + 4), 4);
  ASSERT_EQ(can_vectorize_up_to<int8_t>(ptr + 4), 4);
  ASSERT_EQ(can_vectorize_up_to<int16_t>(ptr + 4), 2);
  ASSERT_EQ(can_vectorize_up_to<int>(ptr + 4), 1);

  ASSERT_EQ(can_vectorize_up_to<bool>(ptr + 8), 4);
  ASSERT_EQ(can_vectorize_up_to<int8_t>(ptr + 8), 4);
  ASSERT_EQ(can_vectorize_up_to<int16_t>(ptr + 8), 4);
  ASSERT_EQ(can_vectorize_up_to<int>(ptr + 8), 2);
  ASSERT_EQ(can_vectorize_up_to<int64_t>(ptr + 8), 1);
}

// The following kernel copy values by using vectorized policies
// defined in `ATen/native/cuda/MemoryAccess.cuh`
template <typename scalar_t, int vec_size>
__global__ void vectorized_copy(scalar_t *dst, scalar_t *src) {
  using vectorized = policies<64, 4>::vectorized<vec_size>;
  auto policy = vectorized();
  scalar_t buf[vectorized::thread_work_size];
  auto accessor = [&](int index) -> scalar_t & { return buf[index]; };
  policy.load(accessor, src + 256 * blockIdx.x);
  policy.store(accessor, dst + 256 * blockIdx.x);
}

TEST(TestVectorizedMemoryAccess, CopyKernel) {
  if (!at::cuda::is_available()) {
    return;
  }

  double *b1 = reinterpret_cast<double *>(buffer1);
  double *b2 = reinterpret_cast<double *>(buffer2);

  // vec4 copy
  reset_buffers();
  cudaDeviceSynchronize();
  vectorized_copy<double, 4><<<16, 64>>>(b2, b1);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  for (int i = 0; i < 1024; i++) {
    ASSERT_EQ(buffer1[i].x, buffer2[i].x);
    ASSERT_EQ(buffer1[i].y, buffer2[i].y);
    ASSERT_EQ(buffer1[i].z, buffer2[i].z);
    ASSERT_EQ(buffer1[i].w, buffer2[i].w);
  }

  // vec2 copy
  reset_buffers();
  cudaDeviceSynchronize();
  vectorized_copy<double, 2><<<16, 64>>>(b2, b1);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  for (int i = 0; i < 1024; i++) {
    ASSERT_EQ(buffer1[i].x, buffer2[i].x);
    ASSERT_EQ(buffer1[i].y, buffer2[i].y);
    ASSERT_EQ(buffer1[i].z, buffer2[i].z);
    ASSERT_EQ(buffer1[i].w, buffer2[i].w);
  }

  // vec1 copy
  reset_buffers();
  cudaDeviceSynchronize();
  vectorized_copy<double, 1><<<16, 64>>>(b2, b1);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  for (int i = 0; i < 1024; i++) {
    ASSERT_EQ(buffer1[i].x, buffer2[i].x);
    ASSERT_EQ(buffer1[i].y, buffer2[i].y);
    ASSERT_EQ(buffer1[i].z, buffer2[i].z);
    ASSERT_EQ(buffer1[i].w, buffer2[i].w);
  }

  // unaligned
  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 16; j++) {
      b1 = reinterpret_cast<double *>(reinterpret_cast<char *>(buffer1) + i);
      b2 = reinterpret_cast<double *>(reinterpret_cast<char *>(buffer2) + j);
      cudaGetLastError();
      cudaDeviceSynchronize();
      vectorized_copy<double, 4><<<1, 64>>>(b2, b1);
      cudaDeviceSynchronize();
      auto err = cudaGetLastError();
      if (i % 16 == 0 && j % 16 == 0) {
        ASSERT_EQ(err, cudaSuccess);
      } else {
        ASSERT_EQ(err, cudaErrorMisalignedAddress);
      }
    }
  }
}
