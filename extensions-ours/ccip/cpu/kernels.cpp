// Written with AI assistance (Claude Code).
#include "ccip/cpu/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace gufo::models::ccip::cpu {
namespace {
constexpr int kRowBlock = 4;
constexpr int kColBlock = 256;
}  // namespace

void Gemm(const float* a, const float* b, const float* bias, float* c, int m, int n, int k,
          bool accumulate) {
  const int row_blocks = (m + kRowBlock - 1) / kRowBlock;
  const int col_blocks = (n + kColBlock - 1) / kColBlock;
#pragma omp parallel for schedule(static)
  for (int t = 0; t < row_blocks * col_blocks; ++t) {
    const int rb = t / col_blocks;
    const int cb = t % col_blocks;
    const int i0 = rb * kRowBlock;
    const int i1 = std::min(m, i0 + kRowBlock);
    const int j0 = cb * kColBlock;
    const int j1 = std::min(n, j0 + kColBlock);
    const int nj = j1 - j0;
    alignas(64) float acc[kRowBlock][kColBlock];
    for (int r = 0; r < i1 - i0; ++r) {
      float* dst = acc[r];
      if (accumulate) {
        std::memcpy(dst, c + static_cast<std::size_t>(i0 + r) * n + j0, sizeof(float) * nj);
      } else if (bias != nullptr) {
        std::memcpy(dst, bias + j0, sizeof(float) * nj);
      } else {
        std::fill(dst, dst + nj, 0.0f);
      }
    }
    if (i1 - i0 == kRowBlock) {
      const float* a0 = a + static_cast<std::size_t>(i0) * k;
      const float* a1 = a0 + k;
      const float* a2 = a1 + k;
      const float* a3 = a2 + k;
      for (int p = 0; p < k; ++p) {
        const float* brow = b + static_cast<std::size_t>(p) * n + j0;
        const float v0 = a0[p], v1 = a1[p], v2 = a2[p], v3 = a3[p];
        float* c0 = acc[0];
        float* c1 = acc[1];
        float* c2 = acc[2];
        float* c3 = acc[3];
#pragma omp simd
        for (int j = 0; j < nj; ++j) {
          const float bv = brow[j];
          c0[j] += v0 * bv;
          c1[j] += v1 * bv;
          c2[j] += v2 * bv;
          c3[j] += v3 * bv;
        }
      }
    } else {
      for (int r = 0; r < i1 - i0; ++r) {
        const float* ar = a + static_cast<std::size_t>(i0 + r) * k;
        float* cr = acc[r];
        for (int p = 0; p < k; ++p) {
          const float* brow = b + static_cast<std::size_t>(p) * n + j0;
          const float v = ar[p];
#pragma omp simd
          for (int j = 0; j < nj; ++j) cr[j] += v * brow[j];
        }
      }
    }
    for (int r = 0; r < i1 - i0; ++r) {
      std::memcpy(c + static_cast<std::size_t>(i0 + r) * n + j0, acc[r], sizeof(float) * nj);
    }
  }
}

void GemmBT(const float* a, const float* b, const float* bias, float* c, int m, int n, int k) {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < m; ++i) {
    const float* ar = a + static_cast<std::size_t>(i) * k;
    for (int j = 0; j < n; ++j) {
      const float* br = b + static_cast<std::size_t>(j) * k;
      float s = 0.0f;
#pragma omp simd reduction(+ : s)
      for (int p = 0; p < k; ++p) s += ar[p] * br[p];
      c[static_cast<std::size_t>(i) * n + j] = s + (bias != nullptr ? bias[j] : 0.0f);
    }
  }
}

void LayerNorm(const float* x, const float* weight, float* y, int rows, int dim, float eps) {
#pragma omp parallel for schedule(static)
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + static_cast<std::size_t>(r) * dim;
    float* yr = y + static_cast<std::size_t>(r) * dim;
    double mean = 0.0;
    for (int i = 0; i < dim; ++i) mean += xr[i];
    mean /= dim;
    double var = 0.0;
    for (int i = 0; i < dim; ++i) {
      const double d = xr[i] - mean;
      var += d * d;
    }
    var /= dim;
    const float inv = static_cast<float>(1.0 / std::sqrt(var + eps));
    const float m = static_cast<float>(mean);
    for (int i = 0; i < dim; ++i) yr[i] = (xr[i] - m) * inv * weight[i];
  }
}

void StarRelu(float* x, std::size_t n, float scale, float bias) {
  const auto count = static_cast<std::ptrdiff_t>(n);
#pragma omp parallel for simd schedule(static)
  for (std::ptrdiff_t i = 0; i < count; ++i) {
    const float r = x[i] > 0.0f ? x[i] : 0.0f;
    x[i] = scale * (r * r) + bias;
  }
}

void SoftmaxRows(float* x, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    float* xr = x + static_cast<std::size_t>(r) * cols;
    float mx = xr[0];
    for (int i = 1; i < cols; ++i) mx = std::max(mx, xr[i]);
    float sum = 0.0f;
    for (int i = 0; i < cols; ++i) {
      xr[i] = std::exp(xr[i] - mx);
      sum += xr[i];
    }
    const float inv = 1.0f / sum;
    for (int i = 0; i < cols; ++i) xr[i] *= inv;
  }
}

void DepthwiseConvNhwc(const float* x, const float* w, float* y, int h, int wd, int c,
                       int kernel, int pad) {
  // Weight [c, 1, k, k] -> per tap [k*k][c] for contiguous channel loops.
  std::vector<float> wt(static_cast<std::size_t>(kernel) * kernel * c);
  for (int ch = 0; ch < c; ++ch) {
    for (int t = 0; t < kernel * kernel; ++t) {
      wt[static_cast<std::size_t>(t) * c + ch] = w[static_cast<std::size_t>(ch) * kernel * kernel + t];
    }
  }
#pragma omp parallel for schedule(static)
  for (int oy = 0; oy < h; ++oy) {
    std::vector<float> acc(static_cast<std::size_t>(c));
    for (int ox = 0; ox < wd; ++ox) {
      std::fill(acc.begin(), acc.end(), 0.0f);
      for (int ky = 0; ky < kernel; ++ky) {
        const int iy = oy + ky - pad;
        if (iy < 0 || iy >= h) continue;
        for (int kx = 0; kx < kernel; ++kx) {
          const int ix = ox + kx - pad;
          if (ix < 0 || ix >= wd) continue;
          const float* xp = x + (static_cast<std::size_t>(iy) * wd + ix) * c;
          const float* wp = wt.data() + static_cast<std::size_t>(ky * kernel + kx) * c;
#pragma omp simd
          for (int ch = 0; ch < c; ++ch) acc[ch] += xp[ch] * wp[ch];
        }
      }
      std::memcpy(y + (static_cast<std::size_t>(oy) * wd + ox) * c, acc.data(), sizeof(float) * c);
    }
  }
}

void ConvToNhwc(const float* x, bool chw_input, int h, int w, int cin, const float* weight,
                const float* bias, int cout, int kernel, int stride, int pad, float* y,
                int* out_h, int* out_w) {
  const int oh = (h + 2 * pad - kernel) / stride + 1;
  const int ow = (w + 2 * pad - kernel) / stride + 1;
  const int kdim = cin * kernel * kernel;
  // Patches [oh*ow, cin*k*k] in weight order (in, ky, kx).
  std::vector<float> cols(static_cast<std::size_t>(oh) * ow * kdim);
#pragma omp parallel for schedule(static)
  for (int oy = 0; oy < oh; ++oy) {
    for (int ox = 0; ox < ow; ++ox) {
      float* dst = cols.data() + (static_cast<std::size_t>(oy) * ow + ox) * kdim;
      for (int ci = 0; ci < cin; ++ci) {
        for (int ky = 0; ky < kernel; ++ky) {
          const int iy = oy * stride + ky - pad;
          for (int kx = 0; kx < kernel; ++kx) {
            const int ix = ox * stride + kx - pad;
            float v = 0.0f;
            if (iy >= 0 && iy < h && ix >= 0 && ix < w) {
              v = chw_input ? x[(static_cast<std::size_t>(ci) * h + iy) * w + ix]
                            : x[(static_cast<std::size_t>(iy) * w + ix) * cin + ci];
            }
            *dst++ = v;
          }
        }
      }
    }
  }
  GemmBT(cols.data(), weight, bias, y, oh * ow, cout, kdim);
  *out_h = oh;
  *out_w = ow;
}

// ------------------------------------------------------------------ fast OCR kernels
namespace {

constexpr int kNr = 16;  // packed B panel width (two AVX2 vectors)
constexpr int kMr = 6;   // rows per register tile

int MaxThreads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

// rows (1..6) x 16 tile. acc starts at init16 (16 column values) or 0, then
// acc = fma(a[r][p], panel[p][j], acc) for p ascending, then + row_bias[r].
template <int R>
inline void MicroTile(const float* a, std::size_t lda, const float* panel, int k,
                      const float* init16, const float* row_bias, float* c, std::size_t ldc,
                      int cols) {
  __m256 acc[R][2];
  const __m256 i0 = init16 != nullptr ? _mm256_loadu_ps(init16) : _mm256_setzero_ps();
  const __m256 i1 = init16 != nullptr ? _mm256_loadu_ps(init16 + 8) : _mm256_setzero_ps();
  for (int r = 0; r < R; ++r) {
    acc[r][0] = i0;
    acc[r][1] = i1;
  }
  const float* ar[R];
  for (int r = 0; r < R; ++r) ar[r] = a + static_cast<std::size_t>(r) * lda;
  for (int p = 0; p < k; ++p) {
    const __m256 b0 = _mm256_loadu_ps(panel + static_cast<std::size_t>(p) * kNr);
    const __m256 b1 = _mm256_loadu_ps(panel + static_cast<std::size_t>(p) * kNr + 8);
    for (int r = 0; r < R; ++r) {
      const __m256 av = _mm256_broadcast_ss(ar[r] + p);
      acc[r][0] = _mm256_fmadd_ps(av, b0, acc[r][0]);
      acc[r][1] = _mm256_fmadd_ps(av, b1, acc[r][1]);
    }
  }
  for (int r = 0; r < R; ++r) {
    if (row_bias != nullptr) {
      const __m256 bv = _mm256_set1_ps(row_bias[r]);
      acc[r][0] = _mm256_add_ps(acc[r][0], bv);
      acc[r][1] = _mm256_add_ps(acc[r][1], bv);
    }
    float* o = c + static_cast<std::size_t>(r) * ldc;
    if (cols == kNr) {
      _mm256_storeu_ps(o, acc[r][0]);
      _mm256_storeu_ps(o + 8, acc[r][1]);
    } else {
      alignas(32) float tmp[kNr];
      _mm256_store_ps(tmp, acc[r][0]);
      _mm256_store_ps(tmp + 8, acc[r][1]);
      std::memcpy(o, tmp, sizeof(float) * static_cast<std::size_t>(cols));
    }
  }
}

void Tile(int rows, const float* a, std::size_t lda, const float* panel, int k,
          const float* init16, const float* row_bias, float* c, std::size_t ldc, int cols) {
  switch (rows) {
    case 6: MicroTile<6>(a, lda, panel, k, init16, row_bias, c, ldc, cols); break;
    case 5: MicroTile<5>(a, lda, panel, k, init16, row_bias, c, ldc, cols); break;
    case 4: MicroTile<4>(a, lda, panel, k, init16, row_bias, c, ldc, cols); break;
    case 3: MicroTile<3>(a, lda, panel, k, init16, row_bias, c, ldc, cols); break;
    case 2: MicroTile<2>(a, lda, panel, k, init16, row_bias, c, ldc, cols); break;
    default: MicroTile<1>(a, lda, panel, k, init16, row_bias, c, ldc, cols); break;
  }
}

std::vector<float>& PackBuffer(std::size_t n) {
  thread_local std::vector<float> buf;
  if (buf.size() < n) buf.resize(n);
  return buf;
}

// C[m, n] = A[m, k] * B (+ col_bias as the start value, + row_bias after the
// sum). B comes from pack(first_col, panels, dst), which writes `panels`
// panels [k][16] (zero padded past n) starting at column first_col.
template <class Pack>
void GemmDriver(const float* a, int m, int n, int k, const float* col_bias,
                const float* row_bias, float* c, const Pack& pack) {
  if (m <= 0 || n <= 0) return;
  const int threads = MaxThreads();
  const int panels = (n + kNr - 1) / kNr;
  const int row_tiles = (m + kMr - 1) / kMr;
  // Panels per task: enough tasks for balance, packed chunk <= ~512 KiB.
  const int cap = std::max(1, (512 * 1024 / 4) / (std::max(k, 1) * kNr));
  const int chunk = std::clamp(panels / (4 * threads), 1, std::min(16, cap));
  const int chunks = (panels + chunk - 1) / chunk;
  // Few column chunks: split the rows too (each row group packs its own B).
  int row_groups = 1;
  if (chunks < 4 * threads)
    row_groups = std::clamp((4 * threads + chunks - 1) / chunks, 1, row_tiles);
  const int tiles_per_group = (row_tiles + row_groups - 1) / row_groups;
  row_groups = (row_tiles + tiles_per_group - 1) / tiles_per_group;
  const int tasks = chunks * row_groups;
  const double work = static_cast<double>(m) * n * k;
  // Column bias padded to whole panels (the last panel may be partial).
  std::vector<float> bias_pad;
  if (col_bias != nullptr && n % kNr != 0) {
    bias_pad.assign(static_cast<std::size_t>(panels) * kNr, 0.0f);
    std::memcpy(bias_pad.data(), col_bias, sizeof(float) * static_cast<std::size_t>(n));
    col_bias = bias_pad.data();
  }
#pragma omp parallel for schedule(dynamic, 1) if (tasks > 1 && work > 262144.0)
  for (int t = 0; t < tasks; ++t) {
    const int ch = t / row_groups;
    const int rg = t % row_groups;
    const int p0 = ch * chunk;
    const int np = std::min(chunk, panels - p0);
    std::vector<float>& buf = PackBuffer(static_cast<std::size_t>(np) * k * kNr);
    pack(p0 * kNr, np, buf.data());
    const int r_begin = rg * tiles_per_group * kMr;
    const int r_end = std::min(m, r_begin + tiles_per_group * kMr);
    for (int r0 = r_begin; r0 < r_end; r0 += kMr) {
      const int rows = std::min(kMr, r_end - r0);
      for (int q = 0; q < np; ++q) {
        const int j0 = (p0 + q) * kNr;
        const int cols = std::min(kNr, n - j0);
        Tile(rows, a + static_cast<std::size_t>(r0) * k, static_cast<std::size_t>(k),
             buf.data() + static_cast<std::size_t>(q) * k * kNr, k,
             col_bias != nullptr ? col_bias + j0 : nullptr,
             row_bias != nullptr ? row_bias + r0 : nullptr,
             c + static_cast<std::size_t>(r0) * n + j0, static_cast<std::size_t>(n), cols);
      }
    }
  }
}

// Packs the rows of a row-major [k, n] matrix (row stride ldb) into panels.
// Row outer, panels inner: each B row is read as one contiguous run.
void PackRows(const float* b, std::size_t ldb, int n, int k, int j0, int np, float* dst) {
  const int full = std::min(np, (n - j0) / kNr);  // panels without padding
  const std::size_t pstride = static_cast<std::size_t>(k) * kNr;
  for (int p = 0; p < k; ++p) {
    const float* s = b + static_cast<std::size_t>(p) * ldb + j0;
    float* d = dst + static_cast<std::size_t>(p) * kNr;
    int q = 0;
    for (; q < full; ++q) {
      _mm256_storeu_ps(d + q * pstride, _mm256_loadu_ps(s + q * kNr));
      _mm256_storeu_ps(d + q * pstride + 8, _mm256_loadu_ps(s + q * kNr + 8));
    }
    for (; q < np; ++q) {
      const int cols = std::min(kNr, n - (j0 + q * kNr));
      float* dp = d + q * pstride;
      int j = 0;
      for (; j < cols; ++j) dp[j] = s[q * kNr + j];
      for (; j < kNr; ++j) dp[j] = 0.0f;
    }
  }
}

}  // namespace

void GemmFast(const float* a, const float* b, const float* col_bias, float* c, int m, int n,
              int k) {
  GemmDriver(a, m, n, k, col_bias, nullptr, c, [&](int j0, int np, float* dst) {
    PackRows(b, static_cast<std::size_t>(n), n, k, j0, np, dst);
  });
}

void ConvNchwFast(const float* x, int cin, int h, int w, const float* weight, const float* bias,
                  int cout, int kh, int kw, int sh, int sw, int pt, int pl, int oh, int ow,
                  float* y) {
  const int n = oh * ow;
  const int k = cin * kh * kw;
  if (kh == 1 && kw == 1 && sh == 1 && sw == 1 && pt == 0 && pl == 0 && oh == h && ow == w) {
    GemmDriver(weight, cout, n, k, nullptr, bias, y, [&](int j0, int np, float* dst) {
      PackRows(x, static_cast<std::size_t>(n), n, k, j0, np, dst);
    });
    return;
  }
  const std::size_t plane = static_cast<std::size_t>(h) * w;
  GemmDriver(weight, cout, n, k, nullptr, bias, y, [&](int j0, int np, float* dst) {
    // Per panel: input origin of its 16 pixels and whether they are 16
    // neighbours in one output row (stride 1), so a tap reads a contiguous run.
    int iy0[16][kNr], ix0[16][kNr];
    bool row_run[16];
    for (int q = 0; q < np; ++q) {
      const int jq = j0 + q * kNr;
      const int cols = std::min(kNr, n - jq);
      for (int j = 0; j < kNr; ++j) {
        if (j < cols) {
          const int idx = jq + j;
          iy0[q][j] = (idx / ow) * sh - pt;
          ix0[q][j] = (idx % ow) * sw - pl;
        } else {
          iy0[q][j] = -(1 << 28);  // padding column: always outside
          ix0[q][j] = 0;
        }
      }
      row_run[q] = cols == kNr && iy0[q][0] == iy0[q][kNr - 1] && sw == 1;
    }
    const std::size_t pstride = static_cast<std::size_t>(k) * kNr;
    // B row (ci, ky, kx) outer, panels inner: neighbouring panels read
    // neighbouring inputs.
    int p = 0;
    for (int ci = 0; ci < cin; ++ci) {
      const float* xc = x + static_cast<std::size_t>(ci) * plane;
      for (int ky = 0; ky < kh; ++ky) {
        for (int kx = 0; kx < kw; ++kx, ++p) {
          for (int q = 0; q < np; ++q) {
            float* dp = dst + q * pstride + static_cast<std::size_t>(p) * kNr;
            if (row_run[q]) {
              const int iy = iy0[q][0] + ky;
              const int ix = ix0[q][0] + kx;
              if (iy < 0 || iy >= h) {
                _mm256_storeu_ps(dp, _mm256_setzero_ps());
                _mm256_storeu_ps(dp + 8, _mm256_setzero_ps());
                continue;
              }
              const float* xr = xc + static_cast<std::size_t>(iy) * w;
              if (ix >= 0 && ix + kNr <= w) {
                _mm256_storeu_ps(dp, _mm256_loadu_ps(xr + ix));
                _mm256_storeu_ps(dp + 8, _mm256_loadu_ps(xr + ix + 8));
                continue;
              }
              for (int j = 0; j < kNr; ++j) {
                const int xx = ix + j;
                dp[j] = (xx >= 0 && xx < w) ? xr[xx] : 0.0f;
              }
              continue;
            }
            for (int j = 0; j < kNr; ++j) {
              const int iy = iy0[q][j] + ky;
              const int ix = ix0[q][j] + kx;
              dp[j] = (iy >= 0 && iy < h && ix >= 0 && ix < w)
                          ? xc[static_cast<std::size_t>(iy) * w + ix]
                          : 0.0f;
            }
          }
        }
      }
    }
  });
}

void DepthwiseNchwFast(const float* x, int c, int h, int w, const float* weight,
                       const float* bias, int kh, int kw, int sh, int sw, int pt, int pl, int oh,
                       int ow, float* y) {
#pragma clang fp contract(off)
  const int threads = MaxThreads();
  const int blocks = std::clamp((8 * threads + c - 1) / c, 1, std::max(oh, 1));
  const int rows_per = (oh + blocks - 1) / blocks;
  const int tasks = c * blocks;
  // Output columns whose taps all lie inside the row (ox * sw - pl >= 0 and
  // ox * sw - pl + kw - 1 <= w - 1); the stride-2 vector loads read one float
  // further, so one less there.
  const int ox_lo = std::min(ow, (pl + sw - 1) / sw);
  const int last = sw == 2 ? w - kw + pl - 1 : w - kw + pl;
  const int ox_hi = std::clamp(last >= 0 ? last / sw + 1 : 0, ox_lo, ow);
  const double work = static_cast<double>(c) * oh * ow * kh * kw;
#pragma omp parallel for schedule(dynamic, 1) if (tasks > 1 && work > 131072.0)
  for (int t = 0; t < tasks; ++t) {
    const int ch = t / blocks;
    const int oy_begin = (t % blocks) * rows_per;
    const int oy_end = std::min(oh, oy_begin + rows_per);
    const float* xp = x + static_cast<std::size_t>(ch) * h * w;
    const float* wp = weight + static_cast<std::size_t>(ch) * kh * kw;
    const float bv = bias != nullptr ? bias[ch] : 0.0f;
    for (int oy = oy_begin; oy < oy_end; ++oy) {
      float* yr = y + (static_cast<std::size_t>(ch) * oh + oy) * ow;
      auto scalar = [&](int ox) {
        float acc = bv;
        for (int ky = 0; ky < kh; ++ky) {
          const int iy = oy * sh - pt + ky;
          if (iy < 0 || iy >= h) continue;
          const float* xr = xp + static_cast<std::size_t>(iy) * w;
          for (int kx = 0; kx < kw; ++kx) {
            const int ix = ox * sw - pl + kx;
            if (ix < 0 || ix >= w) continue;
            const float prod = wp[ky * kw + kx] * xr[ix];
            acc = acc + prod;
          }
        }
        yr[ox] = acc;
      };
      int ox = 0;
      for (; ox < ox_lo; ++ox) scalar(ox);
      if (sw == 1 || sw == 2) {
        for (; ox + 8 <= ox_hi; ox += 8) {
          __m256 acc = _mm256_set1_ps(bv);
          for (int ky = 0; ky < kh; ++ky) {
            const int iy = oy * sh - pt + ky;
            if (iy < 0 || iy >= h) continue;
            const float* xr = xp + static_cast<std::size_t>(iy) * w + (ox * sw - pl);
            for (int kx = 0; kx < kw; ++kx) {
              __m256 v;
              if (sw == 1) {
                v = _mm256_loadu_ps(xr + kx);
              } else {
                const __m256 lo = _mm256_loadu_ps(xr + kx);
                const __m256 hi = _mm256_loadu_ps(xr + kx + 8);
                const __m256 ev = _mm256_shuffle_ps(lo, hi, _MM_SHUFFLE(2, 0, 2, 0));
                v = _mm256_castpd_ps(
                    _mm256_permute4x64_pd(_mm256_castps_pd(ev), _MM_SHUFFLE(3, 1, 2, 0)));
              }
              acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_set1_ps(wp[ky * kw + kx]), v));
            }
          }
          _mm256_storeu_ps(yr + ox, acc);
        }
      }
      for (; ox < ow; ++ox) scalar(ox);
    }
  }
}

}  // namespace gufo::models::ccip::cpu
