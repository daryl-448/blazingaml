#include "runtime/widen.hpp"

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace blz {

namespace {

// dst[i] = src[i] over [a, b), streaming stores on 32-byte aligned runs of 8 cells.
template <class T>
void widen_range(const T* src, std::uint32_t* dst, std::uint64_t a, std::uint64_t b) {
  std::uint64_t i = a;
#if defined(__AVX2__)
  while (i < b && (reinterpret_cast<std::uintptr_t>(dst + i) & 31) != 0) {
    dst[i] = src[i];
    ++i;
  }
  for (; i + 8 <= b; i += 8) {
    __m256i y;
    if constexpr (sizeof(T) == 1) {
      y = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + i)));
    } else {
      y = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i)));
    }
    _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + i), y);
  }
  _mm_sfence();  // the streaming stores are visible before the team's barrier
#endif
  for (; i < b; ++i) dst[i] = src[i];
}

template <class T>
void widen(const T* src, std::uint32_t* dst, std::uint64_t n) {
  // One contiguous share per thread; small inputs on one thread.
  constexpr std::uint64_t kMinPerThread = 1 << 16;
  const int threads = int(std::min<std::uint64_t>(std::uint64_t(omp_get_max_threads()), std::max<std::uint64_t>(n / kMinPerThread, 1)));
#pragma omp parallel num_threads(threads)
  {
    const std::uint64_t t = std::uint64_t(omp_get_thread_num()), nt = std::uint64_t(omp_get_num_threads());
    widen_range(src, dst, n * t / nt, n * (t + 1) / nt);
  }
}

}  // namespace

void widen_cells(const void* src, std::uint32_t width, std::uint32_t* dst, std::uint64_t n) {
  if (width == 1) widen(static_cast<const std::uint8_t*>(src), dst, n);
  else if (width == 2) widen(static_cast<const std::uint16_t*>(src), dst, n);
  else throw std::invalid_argument("widen_cells: width is 1 or 2");
}

}  // namespace blz
