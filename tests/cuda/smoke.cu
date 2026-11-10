// CUDA toolchain smoke test: one kernel, checked on the host.
#include <cstdio>
#include <vector>

__global__ void iota_squared(int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = i * i;
}

int main() {
  constexpr int n = 1000;
  int* d = nullptr;
  if (cudaMalloc(&d, n * sizeof(int)) != cudaSuccess) {
    std::printf("cudaMalloc failed (no GPU?)\n");
    return 1;
  }
  iota_squared<<<(n + 255) / 256, 256>>>(d, n);
  std::vector<int> h(n);
  cudaError_t err = cudaMemcpy(h.data(), d, n * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d);
  if (err != cudaSuccess) {
    std::printf("CUDA error: %s\n", cudaGetErrorString(err));
    return 1;
  }
  for (int i = 0; i < n; ++i)
    if (h[i] != i * i) {
      std::printf("mismatch at %d\n", i);
      return 1;
    }
  std::printf("cuda smoke: ok\n");
  return 0;
}
