// Written with AI assistance (Claude Code).
// See animate_host.hpp. Shapes and formulas follow
// comfy/ldm/wan/model_animate.py (FaceEncoder, CausalConv1d, EqualLinear,
// Direction, after_patch_embedding).
#include "src/models/wan22/animate_host.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace gufo::models::wan22 {
namespace {

// y[t, o] = b[o] + sum_{c, k} w[o, c, k] * x[stride * t + k - 2, c] with
// the time axis padded on the left by replicating frame 0 (kernel 3).
std::vector<float> CausalConv(const std::vector<float>& x, int frames,
                              int in, const float* weight, const float* bias,
                              int out, int stride, int* out_frames) {
  const int t_out = (frames + 2 - 3) / stride + 1;
  *out_frames = t_out;
  std::vector<float> y(static_cast<std::size_t>(t_out) * out);
  std::vector<float> window(static_cast<std::size_t>(in) * 3);
  for (int t = 0; t < t_out; ++t) {
    // window[c * 3 + k] = x[max(0, stride * t + k - 2), c]
    for (int k = 0; k < 3; ++k) {
      const int source = std::max(0, stride * t + k - 2);
      const float* row = x.data() + static_cast<std::size_t>(source) * in;
      for (int c = 0; c < in; ++c)
        window[static_cast<std::size_t>(c) * 3 + k] = row[c];
    }
    float* target = y.data() + static_cast<std::size_t>(t) * out;
    for (int o = 0; o < out; ++o) {
      const float* w = weight + static_cast<std::size_t>(o) * in * 3;
      double sum = 0;
      for (std::size_t i = 0; i < window.size(); ++i)
        sum += static_cast<double>(w[i]) * window[i];
      target[o] = static_cast<float>(sum + bias[o]);
    }
  }
  return y;
}

// LayerNorm (no affine, eps 1e-6) then SiLU over rows of `width`.
void NormSilu(std::vector<float>* x, int width) {
  const std::size_t rows = x->size() / width;
  for (std::size_t r = 0; r < rows; ++r) {
    float* v = x->data() + r * width;
    double mean = 0;
    for (int i = 0; i < width; ++i)
      mean += v[i];
    mean /= width;
    double var = 0;
    for (int i = 0; i < width; ++i)
      var += (v[i] - mean) * (v[i] - mean);
    var /= width;
    const double rstd = 1.0 / std::sqrt(var + 1e-6);
    for (int i = 0; i < width; ++i) {
      const double y = (v[i] - mean) * rstd;
      v[i] = static_cast<float>(y / (1.0 + std::exp(-y)));
    }
  }
}

}  // namespace

void HouseholderQ(const std::vector<double>& matrix, int rows, int columns,
                  std::vector<double>* q) {
  const int m = rows, n = columns;
  std::vector<double> a = matrix;  // row-major [m, n]
  std::vector<double> tau(n, 0.0);
  const auto at = [&](int r, int c) -> double& {
    return a[static_cast<std::size_t>(r) * n + c];
  };
  // dgeqr2: Householder vectors below the diagonal (v[i] = 1 implied).
  for (int i = 0; i < n; ++i) {
    double xnorm = 0;
    for (int r = i + 1; r < m; ++r)
      xnorm += at(r, i) * at(r, i);
    xnorm = std::sqrt(xnorm);
    const double alpha = at(i, i);
    if (xnorm == 0.0) {
      tau[i] = 0.0;
      continue;
    }
    const double beta = -std::copysign(std::hypot(alpha, xnorm), alpha);
    tau[i] = (beta - alpha) / beta;
    const double scale = 1.0 / (alpha - beta);
    for (int r = i + 1; r < m; ++r)
      at(r, i) *= scale;
    at(i, i) = beta;
    // A[i:, i+1:] -= tau v (v^T A[i:, i+1:]).
    for (int c = i + 1; c < n; ++c) {
      double dot = at(i, c);
      for (int r = i + 1; r < m; ++r)
        dot += at(r, i) * at(r, c);
      dot *= tau[i];
      at(i, c) -= dot;
      for (int r = i + 1; r < m; ++r)
        at(r, c) -= dot * at(r, i);
    }
  }
  // dorg2r: Q = H_0 ... H_{n-1} I[:, :n].
  q->assign(static_cast<std::size_t>(m) * n, 0.0);
  for (int i = 0; i < n; ++i)
    (*q)[static_cast<std::size_t>(i) * n + i] = 1.0;
  for (int i = n - 1; i >= 0; --i) {
    if (tau[i] == 0.0)
      continue;
    for (int c = 0; c < n; ++c) {
      double dot = (*q)[static_cast<std::size_t>(i) * n + c];
      for (int r = i + 1; r < m; ++r)
        dot += at(r, i) * (*q)[static_cast<std::size_t>(r) * n + c];
      dot *= tau[i];
      (*q)[static_cast<std::size_t>(i) * n + c] -= dot;
      for (int r = i + 1; r < m; ++r)
        (*q)[static_cast<std::size_t>(r) * n + c] -= dot * at(r, i);
    }
  }
}

void SetMotionDirections(std::span<const float> direction, MotionHead* head) {
  const int m = head->width, n = head->motion;
  std::vector<double> matrix(static_cast<std::size_t>(m) * n);
  for (int r = 0; r < m; ++r)
    for (int c = 0; c < n; ++c)
      matrix[static_cast<std::size_t>(r) * n + c] =
          direction[static_cast<std::size_t>(r) * n + c] +
          (r == c ? 1e-8 : 0.0);
  std::vector<double> q;
  HouseholderQ(matrix, m, n, &q);
  head->q.resize(q.size());
  std::transform(q.begin(), q.end(), head->q.begin(),
                 [](double v) { return static_cast<float>(v); });
}

std::vector<float> MotionVector(const MotionHead& head,
                                std::span<const float> feature) {
  std::vector<float> x(feature.begin(), feature.end());
  for (std::size_t l = 0; l < head.weights.size(); ++l) {
    const std::size_t in = x.size();
    const std::size_t out = head.biases[l].size();
    std::vector<float> y(out);
    for (std::size_t o = 0; o < out; ++o) {
      double sum = head.biases[l][o];
      for (std::size_t i = 0; i < in; ++i)
        sum += static_cast<double>(head.weights[l][o * in + i]) * x[i];
      y[o] = static_cast<float>(sum);
    }
    x = std::move(y);
  }
  // sum_j x[j] * Q[:, j].
  std::vector<float> motion(head.width, 0.0F);
  for (int r = 0; r < head.width; ++r) {
    double sum = 0;
    for (int j = 0; j < head.motion; ++j)
      sum += static_cast<double>(x[j]) *
             head.q[static_cast<std::size_t>(r) * head.motion + j];
    motion[r] = static_cast<float>(sum);
  }
  return motion;
}

int FaceEncoderFrames(int frames) {
  if (frames < 1)
    return 0;
  const int once = (frames - 1) / 2 + 1;
  return (once - 1) / 2 + 1;
}

std::vector<float> RunFaceEncoder(const FaceEncoderWeights& w,
                                  std::span<const float> motion, int frames,
                                  int dit_frames) {
  const int in = w.in_dim, heads = w.heads, ch = w.channels, width = w.width;
  const std::size_t wide = static_cast<std::size_t>(heads) * ch;
  if (frames < 1 || dit_frames < 1 ||
      motion.size() != static_cast<std::size_t>(frames) * in ||
      w.conv1_weight.size() != wide * in * 3 || w.conv1_bias.size() != wide ||
      w.conv2_weight.size() != static_cast<std::size_t>(ch) * ch * 3 ||
      w.conv3_weight.size() != w.conv2_weight.size() ||
      w.conv2_bias.size() != static_cast<std::size_t>(ch) ||
      w.conv3_bias.size() != static_cast<std::size_t>(ch) ||
      w.out_weight.size() != static_cast<std::size_t>(width) * ch ||
      w.out_bias.size() != static_cast<std::size_t>(width) ||
      w.padding.size() != static_cast<std::size_t>(width))
    return {};
  const std::vector<float> x(motion.begin(), motion.end());
  int t1 = 0;
  const std::vector<float> y1 =
      CausalConv(x, frames, in, w.conv1_weight.data(), w.conv1_bias.data(),
                 static_cast<int>(wide), 1, &t1);
  const int tokens = heads + 1;
  const int encoded = FaceEncoderFrames(frames);
  std::vector<float> out(static_cast<std::size_t>(dit_frames) * tokens * width,
                         0.0F);
  for (int n = 0; n < heads; ++n) {
    // "b (n c) t -> (b n) t c": head n reads channels n * ch .. + ch.
    std::vector<float> h(static_cast<std::size_t>(t1) * ch);
    for (int t = 0; t < t1; ++t)
      std::copy_n(y1.begin() + static_cast<std::size_t>(t) * wide +
                      static_cast<std::size_t>(n) * ch,
                  ch, h.begin() + static_cast<std::size_t>(t) * ch);
    NormSilu(&h, ch);
    int t2 = 0, t3 = 0;
    std::vector<float> h2 = CausalConv(h, t1, ch, w.conv2_weight.data(),
                                       w.conv2_bias.data(), ch, 2, &t2);
    NormSilu(&h2, ch);
    std::vector<float> h3 = CausalConv(h2, t2, ch, w.conv3_weight.data(),
                                       w.conv3_bias.data(), ch, 2, &t3);
    NormSilu(&h3, ch);
    for (int t = 0; t < t3 && t + 1 < dit_frames; ++t) {
      float* target =
          out.data() +
          (static_cast<std::size_t>(t + 1) * tokens + n) * width;
      const float* row = h3.data() + static_cast<std::size_t>(t) * ch;
      for (int o = 0; o < width; ++o) {
        const float* wo = w.out_weight.data() + static_cast<std::size_t>(o) * ch;
        double sum = w.out_bias[o];
        for (int c = 0; c < ch; ++c)
          sum += static_cast<double>(wo[c]) * row[c];
        target[o] = static_cast<float>(sum);
      }
    }
  }
  for (int t = 0; t < encoded && t + 1 < dit_frames; ++t)
    std::copy(w.padding.begin(), w.padding.end(),
              out.begin() +
                  (static_cast<std::size_t>(t + 1) * tokens + heads) * width);
  return out;
}

}  // namespace gufo::models::wan22
