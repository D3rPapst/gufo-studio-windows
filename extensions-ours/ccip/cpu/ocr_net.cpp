// Written with AI assistance (Claude Code).
#include "ccip/cpu/ocr_net.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <gufo/ext/json.hpp>

#include "ccip/cpu/kernels.hpp"
#include "ccip/tensors.hpp"

namespace gufo::models::ccip::cpu {

std::int64_t NetTensor::numel() const {
  std::int64_t n = 1;
  for (std::int64_t d : shape) n *= d;
  return n;
}

namespace {

using Shape = std::vector<std::int64_t>;

[[noreturn]] void Fail(const std::string& what) { throw std::runtime_error(what); }

enum class Op {
  kAdd, kAveragePool, kBatchNorm, kCast, kClip, kConcat, kConv, kConvTranspose, kDiv,
  kGlobalAveragePool, kHardSigmoid, kHardSwish, kMatMul, kMul, kPow, kReduceMean, kRelu, kReshape,
  kResize, kShape, kSigmoid, kSlice, kSoftmax, kSqrt, kSqueeze, kSub, kTranspose, kUnsqueeze,
};

Op ParseOp(const std::string& s) {
  static const std::map<std::string, Op> ops = {
      {"Add", Op::kAdd}, {"AveragePool", Op::kAveragePool},
      {"BatchNormalization", Op::kBatchNorm}, {"Cast", Op::kCast}, {"Clip", Op::kClip},
      {"Concat", Op::kConcat}, {"Conv", Op::kConv}, {"ConvTranspose", Op::kConvTranspose},
      {"Div", Op::kDiv}, {"GlobalAveragePool", Op::kGlobalAveragePool},
      {"HardSigmoid", Op::kHardSigmoid}, {"HardSwish", Op::kHardSwish},
      {"MatMul", Op::kMatMul}, {"Mul", Op::kMul},
      {"Pow", Op::kPow}, {"ReduceMean", Op::kReduceMean}, {"Relu", Op::kRelu},
      {"Reshape", Op::kReshape}, {"Resize", Op::kResize}, {"Shape", Op::kShape},
      {"Sigmoid", Op::kSigmoid}, {"Slice", Op::kSlice}, {"Softmax", Op::kSoftmax},
      {"Sqrt", Op::kSqrt}, {"Squeeze", Op::kSqueeze}, {"Sub", Op::kSub},
      {"Transpose", Op::kTranspose}, {"Unsqueeze", Op::kUnsqueeze},
  };
  auto it = ops.find(s);
  if (it == ops.end()) Fail("unsupported op " + s);
  return it->second;
}

struct Attrs {
  std::map<std::string, std::vector<std::int64_t>> ints;
  std::map<std::string, double> floats;
  std::map<std::string, std::string> strings;
  std::int64_t Int(const std::string& k, std::int64_t def) const {
    auto it = ints.find(k);
    return it == ints.end() || it->second.size() != 1 ? def : it->second[0];
  }
  std::vector<std::int64_t> List(const std::string& k, std::vector<std::int64_t> def) const {
    auto it = ints.find(k);
    return it == ints.end() ? def : it->second;
  }
  double Float(const std::string& k, double def) const {
    auto it = floats.find(k);
    return it == floats.end() ? def : it->second;
  }
};

constexpr int kNone = std::numeric_limits<int>::min();

struct Node {
  Op op{};
  std::string name;          // first output (error messages)
  std::vector<int> in;       // run slot (>= 0), constant -(index + 1), or kNone
  std::vector<int> out;
  std::vector<int> release;  // run slots whose last use is this node
  Attrs attrs;
};

std::size_t Sz(std::int64_t v) { return static_cast<std::size_t>(v); }

// A run of consecutive elementwise nodes whose operands are run slots of one
// common shape or scalar constants, evaluated tile by tile in one pass: every
// element goes through the same float operations in the same order as node
// by node, only the intermediate tensors stay in small per-thread tiles.
struct FusedArg {
  int reg{-1};        // >= 0: tile register; < 0: the constant `value`
  float value{0.0f};
};
struct FusedOp {
  Op op{};
  FusedArg a, b;      // b: second operand of binary ops
  float p0{0.0f}, p1{0.0f};  // Clip lo/hi, HardSigmoid alpha/beta
};
struct FusedRun {
  std::size_t first{0}, last{0};  // node range [first, last]
  std::vector<int> ext;           // run slot of register r < ext.size()
  std::vector<FusedOp> ops;       // node first + i writes register ext.size() + i
  std::vector<std::pair<int, int>> outputs;  // (register, run slot) kept in memory
  std::vector<int> release;       // run slots whose last use is inside the run
};

Shape Strides(const Shape& s) {
  Shape st(s.size(), 1);
  for (int d = static_cast<int>(s.size()) - 2; d >= 0; --d) st[Sz(d)] = st[Sz(d) + 1] * s[Sz(d) + 1];
  return st;
}

// Buffers of released run slots, handed out again by FloatTensor during the
// same Run: large tensors are not freed and faulted in again for every node.
// Holds at most kMaxFree buffers; the smallest goes first when it is full.
class BufferPool {
public:
  std::vector<float> Take(std::size_t n) {
    std::size_t best = free_.size();
    for (std::size_t k = 0; k < free_.size(); ++k)
      if (free_[k].capacity() >= n &&
          (best == free_.size() || free_[k].capacity() < free_[best].capacity()))
        best = k;
    if (best == free_.size()) return std::vector<float>(n);
    std::vector<float> v = std::move(free_[best]);
    free_.erase(free_.begin() + static_cast<std::ptrdiff_t>(best));
    v.resize(n);  // only grows (and zero fills) when size < n <= capacity
    return v;
  }
  void Give(std::vector<float>&& v) {
    if (v.capacity() < kMinFloats || v.capacity() > kMaxFloats) return;
    free_.push_back(std::move(v));
    std::size_t total = 0;
    for (const auto& f : free_) total += f.capacity();
    while (free_.size() > kMaxFree || (free_.size() > 1 && total > kMaxTotalFloats)) {
      auto smallest = std::min_element(free_.begin(), free_.end(), [](const auto& a, const auto& b) {
        return a.capacity() < b.capacity();
      });
      total -= smallest->capacity();
      free_.erase(smallest);
    }
  }

private:
  // Measured on a 1920x1080 frame, 3x pass: without the pool 2.35 s and
  // 1.1 GiB peak, unbounded 1.8 s and 2.8 GiB, these limits 2.0 s and 1.5 GiB.
  static constexpr std::size_t kMaxFree = 8;
  static constexpr std::size_t kMinFloats = 4096;
  static constexpr std::size_t kMaxFloats = std::size_t{32} << 20;       // 128 MiB
  static constexpr std::size_t kMaxTotalFloats = std::size_t{128} << 20;  // 512 MiB
  std::vector<std::vector<float>> free_;
};

thread_local BufferPool* tls_pool = nullptr;

// Puts a run slot's buffer back into the pool (or frees it).
void Recycle(NetTensor& t) {
  if (tls_pool != nullptr && !t.is_int) tls_pool->Give(std::move(t.f));
  t = NetTensor{};
}

NetTensor FloatTensor(Shape shape) {
  NetTensor t;
  t.shape = std::move(shape);
  if (tls_pool != nullptr) {
    t.f = tls_pool->Take(Sz(t.numel()));
  } else {
    t.f.resize(Sz(t.numel()));
  }
  return t;
}

// ------------------------------------------------------------------ elementwise
enum class Bin { kAdd, kSub, kMul, kDiv, kPow };

// Elements per parallel work item of the elementwise loops.
constexpr std::int64_t kChunk = 16384;

template <Bin op>
inline float ApplyT(float a, float b) {
  if constexpr (op == Bin::kAdd) return a + b;
  if constexpr (op == Bin::kSub) return a - b;
  if constexpr (op == Bin::kMul) return a * b;
  if constexpr (op == Bin::kDiv) return a / b;
  if constexpr (op == Bin::kPow) return b == 2.0f ? a * a : std::pow(a, b);
  return 0.0f;
}

// One broadcast loop: out[o * inner + j] = op(a[oa(o) + j * ia], b[ob(o) + j * ib]).
// Work is split into (outer row, chunk of the row) items so that one long row
// (scalar broadcasts) runs on all threads as well. `out` may alias an input
// that has the output shape (each element is read before it is written).
template <Bin op>
void BinaryLoop(const float* pa, const float* pb, float* po, const Shape& dims, const Shape& st_a,
                const Shape& st_b, std::int64_t total) {
  const std::int64_t inner = dims.back();
  const std::int64_t ia = st_a.back(), ib = st_b.back();
  const std::int64_t outer = total / inner;
  const std::int64_t pieces = (inner + kChunk - 1) / kChunk;
  const std::int64_t items = outer * pieces;
  const int nd = static_cast<int>(dims.size()) - 1;
#pragma omp parallel for schedule(static) if (items > 1 && total > 32768)
  for (std::int64_t it = 0; it < items; ++it) {
    const std::int64_t o = it / pieces;
    const std::int64_t j0 = (it % pieces) * kChunk;
    const std::int64_t j1 = std::min(inner, j0 + kChunk);
    std::int64_t rem = o, oa = 0, ob = 0;
    for (int d = nd - 1; d >= 0; --d) {
      const std::int64_t idx = rem % dims[Sz(d)];
      rem /= dims[Sz(d)];
      oa += idx * st_a[Sz(d)];
      ob += idx * st_b[Sz(d)];
    }
    const float* xa = pa + oa;
    const float* xb = pb + ob;
    float* y = po + o * inner;
    if (ia == 1 && ib == 1) {
      for (std::int64_t j = j0; j < j1; ++j) y[j] = ApplyT<op>(xa[j], xb[j]);
    } else if (ia == 1 && ib == 0) {
      const float v = xb[0];
      for (std::int64_t j = j0; j < j1; ++j) y[j] = ApplyT<op>(xa[j], v);
    } else if (ia == 0 && ib == 1) {
      const float v = xa[0];
      for (std::int64_t j = j0; j < j1; ++j) y[j] = ApplyT<op>(v, xb[j]);
    } else {
      for (std::int64_t j = j0; j < j1; ++j) y[j] = ApplyT<op>(xa[j * ia], xb[j * ib]);
    }
  }
}

// ONNX HardSwish: x * max(0, min(1, x / 6 + 1 / 2)) (alpha 1/6 as float, like onnxruntime).
inline float HardSwish(float v) {
  return v * std::max(0.0f, std::min(1.0f, (1.0f / 6.0f) * v + 0.5f));
}

// y[q] = op(a, b) over one tile; a null pointer means the scalar value.
template <Bin op>
inline void FusedBinary(const float* xa, float va, const float* xb, float vb, float* y, std::int64_t len) {
  if (xa != nullptr && xb != nullptr) {
    for (std::int64_t q = 0; q < len; ++q) y[q] = ApplyT<op>(xa[q], xb[q]);
  } else if (xa != nullptr) {
    for (std::int64_t q = 0; q < len; ++q) y[q] = ApplyT<op>(xa[q], vb);
  } else {
    for (std::int64_t q = 0; q < len; ++q) y[q] = ApplyT<op>(va, xb[q]);
  }
}

// One fused op over one tile (a null operand pointer means its scalar).
// Built for AVX2 (no FMA, so every multiply and add rounds as node by node).
__attribute__((target("avx2"))) void FusedOpTile(const FusedOp& op, const float* xa,
                                                 const float* xb, float* y, std::int64_t len) {
  switch (op.op) {
    case Op::kAdd: FusedBinary<Bin::kAdd>(xa, op.a.value, xb, op.b.value, y, len); break;
    case Op::kSub: FusedBinary<Bin::kSub>(xa, op.a.value, xb, op.b.value, y, len); break;
    case Op::kMul: FusedBinary<Bin::kMul>(xa, op.a.value, xb, op.b.value, y, len); break;
    case Op::kDiv: FusedBinary<Bin::kDiv>(xa, op.a.value, xb, op.b.value, y, len); break;
    case Op::kPow: FusedBinary<Bin::kPow>(xa, op.a.value, xb, op.b.value, y, len); break;
    case Op::kRelu:
      for (std::int64_t q = 0; q < len; ++q) y[q] = xa[q] > 0.0f ? xa[q] : 0.0f;
      break;
    case Op::kSigmoid:
      for (std::int64_t q = 0; q < len; ++q) y[q] = 1.0f / (1.0f + std::exp(-xa[q]));
      break;
    case Op::kSqrt:
      for (std::int64_t q = 0; q < len; ++q) y[q] = std::sqrt(xa[q]);
      break;
    case Op::kHardSigmoid: {
      const float alpha = op.p0, beta = op.p1;
      for (std::int64_t q = 0; q < len; ++q)
        y[q] = std::max(0.0f, std::min(1.0f, alpha * xa[q] + beta));
      break;
    }
    case Op::kClip: {
      const float lo = op.p0, hi = op.p1;
      for (std::int64_t q = 0; q < len; ++q) y[q] = std::min(hi, std::max(lo, xa[q]));
      break;
    }
    case Op::kHardSwish:
      for (std::int64_t q = 0; q < len; ++q) y[q] = HardSwish(xa[q]);
      break;
    default:
      break;
  }
}

// `reuse_a` / `reuse_b`: the input's run slot may be overwritten (last use).
NetTensor Binary(Bin op, const NetTensor& a, const NetTensor& b, NetTensor* reuse_a,
                 NetTensor* reuse_b) {
  if (a.is_int || b.is_int) Fail("elementwise ops on int tensors are not implemented");
  const std::size_t rank = std::max(a.shape.size(), b.shape.size());
  Shape sa(rank, 1), sb(rank, 1), so(rank, 1);
  std::copy(a.shape.begin(), a.shape.end(), sa.begin() + static_cast<std::ptrdiff_t>(rank - a.shape.size()));
  std::copy(b.shape.begin(), b.shape.end(), sb.begin() + static_cast<std::ptrdiff_t>(rank - b.shape.size()));
  for (std::size_t d = 0; d < rank; ++d) {
    if (sa[d] != sb[d] && sa[d] != 1 && sb[d] != 1) Fail("broadcast mismatch");
    so[d] = (sa[d] == 0 || sb[d] == 0) ? 0 : std::max(sa[d], sb[d]);
  }
  const float* pa = a.f.data();
  const float* pb = b.f.data();
  NetTensor out;
  if (reuse_a != nullptr && sa == so) {
    out = std::move(*reuse_a);
  } else if (reuse_b != nullptr && sb == so) {
    out = std::move(*reuse_b);
  } else {
    out = FloatTensor(so);
  }
  out.shape = so;
  if (out.f.empty()) return out;
  // Element strides (0 on broadcast dims); merge dims that stay contiguous for
  // both inputs so the inner loop is long.
  Shape ta = Strides(sa), tb = Strides(sb);
  for (std::size_t d = 0; d < rank; ++d) {
    if (sa[d] == 1) ta[d] = 0;
    if (sb[d] == 1) tb[d] = 0;
  }
  Shape dims, st_a, st_b;
  for (std::size_t d = 0; d < rank; ++d) {
    if (so[d] == 1) continue;
    if (!dims.empty() && st_a.back() == ta[d] * so[d] && st_b.back() == tb[d] * so[d]) {
      dims.back() *= so[d];
      st_a.back() = ta[d];
      st_b.back() = tb[d];
    } else {
      dims.push_back(so[d]);
      st_a.push_back(ta[d]);
      st_b.push_back(tb[d]);
    }
  }
  if (dims.empty()) {
    dims = {1};
    st_a = {0};
    st_b = {0};
  }
  const std::int64_t total = static_cast<std::int64_t>(out.f.size());
  float* po = out.f.data();
  switch (op) {
    case Bin::kAdd: BinaryLoop<Bin::kAdd>(pa, pb, po, dims, st_a, st_b, total); break;
    case Bin::kSub: BinaryLoop<Bin::kSub>(pa, pb, po, dims, st_a, st_b, total); break;
    case Bin::kMul: BinaryLoop<Bin::kMul>(pa, pb, po, dims, st_a, st_b, total); break;
    case Bin::kDiv: BinaryLoop<Bin::kDiv>(pa, pb, po, dims, st_a, st_b, total); break;
    case Bin::kPow: BinaryLoop<Bin::kPow>(pa, pb, po, dims, st_a, st_b, total); break;
  }
  return out;
}

template <typename F>
NetTensor Unary(const NetTensor& x, F fn, NetTensor* reuse) {
  if (x.is_int) Fail("unary op on an int tensor");
  const float* px = x.f.data();
  NetTensor out;
  if (reuse != nullptr) {
    out = std::move(*reuse);
  } else {
    out = FloatTensor(x.shape);
  }
  const std::int64_t n = static_cast<std::int64_t>(out.f.size());
  float* po = out.f.data();
#pragma omp parallel for schedule(static) if (n > 65536)
  for (std::int64_t c = 0; c < n; c += kChunk) {
    const std::int64_t e = std::min(n, c + kChunk);
    for (std::int64_t k = c; k < e; ++k) po[k] = fn(px[k]);
  }
  return out;
}

// ------------------------------------------------------------------ conv
struct ConvParams {
  int group{1};
  int kh{1}, kw{1};
  int sh{1}, sw{1};
  int pt{0}, pl{0}, pb{0}, pr{0};
};

ConvParams ReadConv(const Attrs& at, const Shape& wshape) {
  ConvParams p;
  p.group = static_cast<int>(at.Int("group", 1));
  const auto k = at.List("kernel_shape", {wshape[2], wshape[3]});
  p.kh = static_cast<int>(k[0]);
  p.kw = static_cast<int>(k[1]);
  const auto s = at.List("strides", {1, 1});
  p.sh = static_cast<int>(s[0]);
  p.sw = static_cast<int>(s[1]);
  const auto pads = at.List("pads", {0, 0, 0, 0});
  p.pt = static_cast<int>(pads[0]);
  p.pl = static_cast<int>(pads[1]);
  p.pb = static_cast<int>(pads[2]);
  p.pr = static_cast<int>(pads[3]);
  return p;
}

NetTensor Conv(const NetTensor& x, const NetTensor& w, const NetTensor* b, const Attrs& at) {
  if (x.shape.size() != 4 || w.shape.size() != 4) Fail("Conv expects 4-D tensors");
  const ConvParams p = ReadConv(at, w.shape);
  const int n = static_cast<int>(x.shape[0]);
  const int c = static_cast<int>(x.shape[1]);
  const int h = static_cast<int>(x.shape[2]);
  const int wd = static_cast<int>(x.shape[3]);
  const int m = static_cast<int>(w.shape[0]);
  const int cg = static_cast<int>(w.shape[1]);
  if (cg * p.group != c || m % p.group != 0 || w.shape[2] != p.kh || w.shape[3] != p.kw)
    Fail("Conv weight shape mismatch");
  const int oh = (h + p.pt + p.pb - p.kh) / p.sh + 1;
  const int ow = (wd + p.pl + p.pr - p.kw) / p.sw + 1;
  NetTensor out = FloatTensor({n, m, oh, ow});
  const std::size_t ohw = Sz(oh) * Sz(ow);
  const float* bias = b != nullptr ? b->f.data() : nullptr;
  const int mg = m / p.group;
  if (cg == 1 && mg == 1) {
    // Depthwise: one output plane per input channel.
    for (int bn = 0; bn < n; ++bn)
      DepthwiseNchwFast(x.f.data() + Sz(bn) * Sz(c) * Sz(h) * Sz(wd), m, h, wd, w.f.data(), bias,
                        p.kh, p.kw, p.sh, p.sw, p.pt, p.pl, oh, ow,
                        out.f.data() + Sz(bn) * Sz(m) * ohw);
    return out;
  }
  if (p.group == 1) {
    for (int bn = 0; bn < n; ++bn)
      ConvNchwFast(x.f.data() + Sz(bn) * Sz(c) * Sz(h) * Sz(wd), c, h, wd, w.f.data(), bias, m,
                   p.kh, p.kw, p.sh, p.sw, p.pt, p.pl, oh, ow,
                   out.f.data() + Sz(bn) * Sz(m) * ohw);
    return out;
  }
  const int kk = cg * p.kh * p.kw;
  const bool pointwise = p.kh == 1 && p.kw == 1 && p.sh == 1 && p.sw == 1 && p.pt == 0 &&
                         p.pl == 0 && p.pb == 0 && p.pr == 0;
  std::vector<float> col;
  if (!pointwise) col.resize(Sz(kk) * ohw);
  for (int bn = 0; bn < n; ++bn) {
    for (int g = 0; g < p.group; ++g) {
      const float* xg = x.f.data() + (Sz(bn) * Sz(c) + Sz(g) * Sz(cg)) * Sz(h) * Sz(wd);
      const float* src = xg;
      if (!pointwise) {
#pragma omp parallel for schedule(static)
        for (int r = 0; r < kk; ++r) {
          const int ci = r / (p.kh * p.kw);
          const int ky = (r / p.kw) % p.kh;
          const int kx = r % p.kw;
          float* dst = col.data() + Sz(r) * ohw;
          const float* xc = xg + Sz(ci) * Sz(h) * Sz(wd);
          for (int oy = 0; oy < oh; ++oy) {
            const int iy = oy * p.sh - p.pt + ky;
            float* drow = dst + Sz(oy) * Sz(ow);
            if (iy < 0 || iy >= h) {
              std::fill(drow, drow + ow, 0.0f);
              continue;
            }
            const float* xrow = xc + Sz(iy) * Sz(wd);
            for (int ox = 0; ox < ow; ++ox) {
              const int ix = ox * p.sw - p.pl + kx;
              drow[ox] = (ix >= 0 && ix < wd) ? xrow[ix] : 0.0f;
            }
          }
        }
        src = col.data();
      }
      float* yg = out.f.data() + (Sz(bn) * Sz(m) + Sz(g) * Sz(mg)) * ohw;
      Gemm(w.f.data() + Sz(g) * Sz(mg) * Sz(kk), src, nullptr, yg, mg, static_cast<int>(ohw), kk);
      if (bias != nullptr) {
#pragma omp parallel for schedule(static)
        for (int o = 0; o < mg; ++o) {
          float* yr = yg + Sz(o) * ohw;
          const float bv = bias[g * mg + o];
          for (std::size_t j = 0; j < ohw; ++j) yr[j] += bv;
        }
      }
    }
  }
  return out;
}

NetTensor ConvTranspose(const NetTensor& x, const NetTensor& w, const NetTensor* b,
                        const Attrs& at) {
  const ConvParams p = ReadConv(at, w.shape);
  if (p.group != 1) Fail("ConvTranspose with groups is not implemented");
  const int n = static_cast<int>(x.shape[0]);
  const int ci = static_cast<int>(x.shape[1]);
  const int h = static_cast<int>(x.shape[2]);
  const int wd = static_cast<int>(x.shape[3]);
  if (w.shape[0] != ci) Fail("ConvTranspose weight shape mismatch");
  const int co = static_cast<int>(w.shape[1]);
  const int oh = (h - 1) * p.sh - p.pt - p.pb + p.kh;
  const int ow = (wd - 1) * p.sw - p.pl - p.pr + p.kw;
  NetTensor out = FloatTensor({n, co, oh, ow});
  const int rows = co * p.kh * p.kw;
  // wt[rows, ci] = w[ci, rows]^T
  std::vector<float> wt(Sz(rows) * Sz(ci));
  for (int i = 0; i < ci; ++i)
    for (int r = 0; r < rows; ++r) wt[Sz(r) * Sz(ci) + Sz(i)] = w.f[Sz(i) * Sz(rows) + Sz(r)];
  const std::size_t hw = Sz(h) * Sz(wd);
  NetTensor col_t = FloatTensor({rows, static_cast<std::int64_t>(hw)});
  std::vector<float>& col = col_t.f;
  for (int bn = 0; bn < n; ++bn) {
    GemmFast(wt.data(), x.f.data() + Sz(bn) * Sz(ci) * hw, nullptr, col.data(), rows,
             static_cast<int>(hw), ci);
    float* y = out.f.data() + Sz(bn) * Sz(co) * Sz(oh) * Sz(ow);
#pragma omp parallel for schedule(static)
    for (std::int64_t task = 0; task < static_cast<std::int64_t>(co) * oh; ++task) {
      const int o = static_cast<int>(task / oh);
      const int yy = static_cast<int>(task % oh);
      float* yr = y + (Sz(o) * Sz(oh) + Sz(yy)) * Sz(ow);
      const float bv = b != nullptr ? b->f[Sz(o)] : 0.0f;
      std::fill(yr, yr + ow, bv);
      for (int ky = 0; ky < p.kh; ++ky) {
        const int t = yy + p.pt - ky;  // = iy * sh
        if (t < 0 || t % p.sh != 0) continue;
        const int iy = t / p.sh;
        if (iy >= h) continue;
        for (int kx = 0; kx < p.kw; ++kx) {
          const float* cr = col.data() + (Sz(o) * Sz(p.kh) * Sz(p.kw) + Sz(ky * p.kw + kx)) * hw +
                            Sz(iy) * Sz(wd);
          for (int ix = 0; ix < wd; ++ix) {
            const int xx = ix * p.sw - p.pl + kx;
            if (xx < 0 || xx >= ow) continue;
            yr[xx] += cr[ix];
          }
        }
      }
    }
  }
  return out;
}

NetTensor BatchNorm(const NetTensor& x, const NetTensor& scale, const NetTensor& bias,
                    const NetTensor& mean, const NetTensor& var, double eps, NetTensor* reuse) {
  const std::int64_t n = x.shape[0], c = x.shape[1];
  const std::int64_t inner = x.numel() / (n * c);
  const float* px = x.f.data();
  NetTensor out = reuse != nullptr ? std::move(*reuse) : FloatTensor(x.shape);
  std::vector<float> s(Sz(c)), t(Sz(c));
  for (std::int64_t ch = 0; ch < c; ++ch) {
    const float inv = 1.0f / std::sqrt(var.f[Sz(ch)] + static_cast<float>(eps));
    s[Sz(ch)] = scale.f[Sz(ch)] * inv;
    t[Sz(ch)] = bias.f[Sz(ch)] - mean.f[Sz(ch)] * s[Sz(ch)];
  }
  const std::int64_t pieces = (inner + kChunk - 1) / kChunk;
#pragma omp parallel for schedule(static)
  for (std::int64_t it = 0; it < n * c * pieces; ++it) {
    const std::int64_t r = it / pieces;
    const std::int64_t j0 = (it % pieces) * kChunk;
    const std::int64_t j1 = std::min(inner, j0 + kChunk);
    const float sv = s[Sz(r % c)], tv = t[Sz(r % c)];
    const float* xp = px + Sz(r) * Sz(inner);
    float* yp = out.f.data() + Sz(r) * Sz(inner);
    for (std::int64_t j = j0; j < j1; ++j) yp[j] = xp[j] * sv + tv;
  }
  return out;
}

NetTensor GlobalAveragePool(const NetTensor& x) {
  const std::int64_t n = x.shape[0], c = x.shape[1];
  const std::int64_t inner = x.numel() / (n * c);
  NetTensor out = FloatTensor({n, c, 1, 1});
#pragma omp parallel for schedule(static) if (n * c * inner > 65536)
  for (std::int64_t r = 0; r < n * c; ++r) {
    const float* xp = x.f.data() + Sz(r) * Sz(inner);
    double s = 0.0;
    for (std::int64_t j = 0; j < inner; ++j) s += xp[j];
    out.f[Sz(r)] = static_cast<float>(s / static_cast<double>(inner));
  }
  return out;
}

NetTensor AveragePool(const NetTensor& x, const Attrs& at) {
  const auto k = at.List("kernel_shape", {});
  const auto s = at.List("strides", {1, 1});
  if (k.size() != 2 || x.shape.size() != 4) Fail("AveragePool expects 2-D kernels");
  const std::int64_t n = x.shape[0], c = x.shape[1], h = x.shape[2], w = x.shape[3];
  const std::int64_t oh = (h - k[0]) / s[0] + 1, ow = (w - k[1]) / s[1] + 1;
  NetTensor out = FloatTensor({n, c, oh, ow});
  const float inv = 1.0f / static_cast<float>(k[0] * k[1]);
  for (std::int64_t r = 0; r < n * c; ++r) {
    const float* xp = x.f.data() + Sz(r) * Sz(h) * Sz(w);
    float* yp = out.f.data() + Sz(r) * Sz(oh) * Sz(ow);
    for (std::int64_t oy = 0; oy < oh; ++oy)
      for (std::int64_t ox = 0; ox < ow; ++ox) {
        float acc = 0.0f;
        for (std::int64_t ky = 0; ky < k[0]; ++ky)
          for (std::int64_t kx = 0; kx < k[1]; ++kx)
            acc += xp[Sz(oy * s[0] + ky) * Sz(w) + Sz(ox * s[1] + kx)];
        yp[Sz(oy) * Sz(ow) + Sz(ox)] = acc * inv;
      }
  }
  return out;
}

NetTensor Resize(const NetTensor& x, const NetTensor* scales, const NetTensor* sizes) {
  const std::size_t rank = x.shape.size();
  Shape os(rank);
  std::vector<double> sc(rank, 1.0);
  if (sizes != nullptr && sizes->numel() == static_cast<std::int64_t>(rank)) {
    for (std::size_t d = 0; d < rank; ++d) {
      os[d] = sizes->is_int ? sizes->i[d] : static_cast<std::int64_t>(sizes->f[d]);
      sc[d] = static_cast<double>(os[d]) / static_cast<double>(x.shape[d]);
    }
  } else if (scales != nullptr && scales->numel() == static_cast<std::int64_t>(rank)) {
    for (std::size_t d = 0; d < rank; ++d) {
      sc[d] = scales->f[d];
      os[d] = static_cast<std::int64_t>(static_cast<double>(x.shape[d]) * sc[d]);
    }
  } else {
    Fail("Resize needs scales or sizes");
  }
  if (rank != 4 || os[0] != x.shape[0] || os[1] != x.shape[1]) Fail("Resize only over H, W");
  NetTensor out = FloatTensor(os);
  const std::int64_t h = x.shape[2], w = x.shape[3], oh = os[2], ow = os[3];
  std::vector<std::int64_t> ys(Sz(oh)), xs(Sz(ow));
  // coordinate_transformation_mode asymmetric, nearest_mode floor.
  for (std::int64_t y = 0; y < oh; ++y)
    ys[Sz(y)] = std::min<std::int64_t>(
        h - 1, static_cast<std::int64_t>(std::floor(static_cast<double>(y) / sc[2])));
  for (std::int64_t xx = 0; xx < ow; ++xx)
    xs[Sz(xx)] = std::min<std::int64_t>(
        w - 1, static_cast<std::int64_t>(std::floor(static_cast<double>(xx) / sc[3])));
  const std::int64_t planes = os[0] * os[1];
#pragma omp parallel for schedule(static)
  for (std::int64_t task = 0; task < planes * oh; ++task) {
    const std::int64_t r = task / oh, y = task % oh;
    const float* xrow = x.f.data() + Sz(r) * Sz(h) * Sz(w) + Sz(ys[Sz(y)]) * Sz(w);
    float* yrow = out.f.data() + (Sz(r) * Sz(oh) + Sz(y)) * Sz(ow);
    for (std::int64_t xx = 0; xx < ow; ++xx) yrow[xx] = xrow[xs[Sz(xx)]];
  }
  return out;
}

std::int64_t Axis(std::int64_t a, std::size_t rank) {
  if (a < 0) a += static_cast<std::int64_t>(rank);
  if (a < 0 || a >= static_cast<std::int64_t>(rank)) Fail("axis out of range");
  return a;
}

NetTensor Concat(const std::vector<const NetTensor*>& xs, std::int64_t axis_attr) {
  const NetTensor& first = *xs[0];
  const std::size_t rank = first.shape.size();
  const std::int64_t axis = Axis(axis_attr, rank);
  NetTensor out;
  out.is_int = first.is_int;
  out.shape = first.shape;
  out.shape[Sz(axis)] = 0;
  for (const NetTensor* t : xs) {
    if (t->is_int != out.is_int || t->shape.size() != rank) Fail("Concat input mismatch");
    out.shape[Sz(axis)] += t->shape[Sz(axis)];
  }
  std::int64_t outer = 1;
  for (std::int64_t d = 0; d < axis; ++d) outer *= out.shape[Sz(d)];
  std::int64_t inner = 1;
  for (std::size_t d = Sz(axis) + 1; d < rank; ++d) inner *= out.shape[d];
  if (out.is_int)
    out.i.resize(Sz(out.numel()));
  else
    out.f.resize(Sz(out.numel()));
  std::int64_t offset = 0;
  const std::int64_t row = out.shape[Sz(axis)] * inner;
  for (const NetTensor* t : xs) {
    const std::int64_t chunk = t->shape[Sz(axis)] * inner;
    if (out.is_int) {
      for (std::int64_t o = 0; o < outer; ++o)
        std::copy_n(t->i.data() + o * chunk, chunk, out.i.data() + o * row + offset);
    } else {
      const std::int64_t pieces = (chunk + kChunk - 1) / kChunk;
#pragma omp parallel for schedule(static) if (outer * chunk > 65536)
      for (std::int64_t it = 0; it < outer * pieces; ++it) {
        const std::int64_t o = it / pieces;
        const std::int64_t j0 = (it % pieces) * kChunk;
        const std::int64_t len = std::min(kChunk, chunk - j0);
        std::copy_n(t->f.data() + o * chunk + j0, len, out.f.data() + o * row + offset + j0);
      }
    }
    offset += chunk;
  }
  return out;
}

std::vector<std::int64_t> IntValues(const NetTensor& t) {
  if (t.is_int) return t.i;
  std::vector<std::int64_t> v;
  for (float f : t.f) v.push_back(static_cast<std::int64_t>(f));
  return v;
}

NetTensor Slice(const NetTensor& x, const NetTensor& starts, const NetTensor* ends_t,
                const NetTensor* axes_t, const NetTensor* steps_t) {
  if (ends_t == nullptr) Fail("Slice without ends");
  const std::size_t rank = x.shape.size();
  const auto st = IntValues(starts);
  const auto en = IntValues(*ends_t);
  std::vector<std::int64_t> ax, sp;
  if (axes_t != nullptr) {
    ax = IntValues(*axes_t);
  } else {
    for (std::size_t k = 0; k < st.size(); ++k) ax.push_back(static_cast<std::int64_t>(k));
  }
  if (steps_t != nullptr) sp = IntValues(*steps_t);
  else sp.assign(st.size(), 1);
  std::vector<std::int64_t> b(rank, 0), step(rank, 1);
  Shape os = x.shape;
  for (std::size_t k = 0; k < st.size(); ++k) {
    const std::int64_t a = Axis(ax[k], rank);
    const std::int64_t dim = x.shape[Sz(a)];
    if (sp[k] <= 0) Fail("Slice with step <= 0 is not implemented");
    std::int64_t s = st[k], e = en[k];
    if (s < 0) s += dim;
    if (e < 0) e += dim;
    s = std::clamp<std::int64_t>(s, 0, dim);
    e = std::clamp<std::int64_t>(e, 0, dim);
    b[Sz(a)] = s;
    step[Sz(a)] = sp[k];
    os[Sz(a)] = e > s ? (e - s + sp[k] - 1) / sp[k] : 0;
  }
  NetTensor out;
  out.is_int = x.is_int;
  out.shape = os;
  const std::int64_t n = out.numel();
  if (out.is_int) out.i.resize(Sz(n));
  else out.f.resize(Sz(n));
  const Shape xs = Strides(x.shape);
  if (!out.is_int && rank > 0 && step[rank - 1] == 1 && n > 0) {
    // Last axis unit step: copy whole rows.
    const std::int64_t inner = os[rank - 1];
    const std::int64_t rows = n / inner;
    for (std::int64_t r = 0; r < rows; ++r) {
      std::int64_t rem = r, src = b[rank - 1];
      for (int d = static_cast<int>(rank) - 2; d >= 0; --d) {
        const std::int64_t v = rem % os[Sz(d)];
        rem /= os[Sz(d)];
        src += (b[Sz(d)] + v * step[Sz(d)]) * xs[Sz(d)];
      }
      std::copy_n(x.f.data() + src, inner, out.f.data() + r * inner);
    }
    return out;
  }
  for (std::int64_t idx = 0; idx < n; ++idx) {
    std::int64_t rem = idx, src = 0;
    for (int d = static_cast<int>(rank) - 1; d >= 0; --d) {
      const std::int64_t v = rem % os[Sz(d)];
      rem /= os[Sz(d)];
      src += (b[Sz(d)] + v * step[Sz(d)]) * xs[Sz(d)];
    }
    if (out.is_int) out.i[Sz(idx)] = x.i[Sz(src)];
    else out.f[Sz(idx)] = x.f[Sz(src)];
  }
  return out;
}

NetTensor Transpose(const NetTensor& x, const std::vector<std::int64_t>& perm_attr) {
  const std::size_t rank = x.shape.size();
  if (x.is_int) Fail("Transpose on int tensors is not implemented");
  std::vector<std::int64_t> perm = perm_attr;
  if (perm.empty())
    for (std::size_t d = 0; d < rank; ++d) perm.push_back(static_cast<std::int64_t>(rank - 1 - d));
  if (perm.size() != rank) Fail("Transpose perm rank");
  Shape os(rank);
  for (std::size_t d = 0; d < rank; ++d) os[d] = x.shape[Sz(perm[d])];
  const Shape xs = Strides(x.shape);
  Shape src_stride(rank);
  for (std::size_t d = 0; d < rank; ++d) src_stride[d] = xs[Sz(perm[d])];
  NetTensor out = FloatTensor(os);
  const std::int64_t n = out.numel();
  if (n == 0) return out;
  const std::int64_t inner = os[rank - 1];
  const std::int64_t is = src_stride[rank - 1];
  const std::int64_t outer = n / inner;
#pragma omp parallel for schedule(static) if (n > 65536)
  for (std::int64_t o = 0; o < outer; ++o) {
    std::int64_t rem = o, src = 0;
    for (int d = static_cast<int>(rank) - 2; d >= 0; --d) {
      const std::int64_t v = rem % os[Sz(d)];
      rem /= os[Sz(d)];
      src += v * src_stride[Sz(d)];
    }
    float* y = out.f.data() + o * inner;
    const float* xp = x.f.data() + src;
    for (std::int64_t j = 0; j < inner; ++j) y[j] = xp[j * is];
  }
  return out;
}

NetTensor Reshape(const NetTensor& x, const NetTensor& shape_t) {
  const auto req = IntValues(shape_t);
  Shape os(req.size());
  std::int64_t known = 1;
  int infer = -1;
  for (std::size_t d = 0; d < req.size(); ++d) {
    if (req[d] == 0) {
      if (d >= x.shape.size()) Fail("Reshape 0 beyond the input rank");
      os[d] = x.shape[d];
    } else if (req[d] == -1) {
      if (infer >= 0) Fail("Reshape with two -1");
      infer = static_cast<int>(d);
      os[d] = 1;
      continue;
    } else {
      os[d] = req[d];
    }
    known *= os[d];
  }
  if (infer >= 0) os[Sz(infer)] = known == 0 ? 0 : x.numel() / known;
  NetTensor out;
  out.is_int = x.is_int;
  out.shape = os;
  if (out.numel() != x.numel()) Fail("Reshape element count mismatch");
  out.f = x.f;
  out.i = x.i;
  return out;
}

NetTensor MatMul(const NetTensor& a, const NetTensor& b) {
  if (a.shape.size() < 2 || b.shape.size() < 2) Fail("MatMul expects rank >= 2");
  const std::int64_t m = a.shape[a.shape.size() - 2], k = a.shape.back();
  const std::int64_t k2 = b.shape[b.shape.size() - 2], n = b.shape.back();
  if (k != k2) Fail("MatMul inner dims");
  Shape os = a.shape;
  os.back() = n;
  NetTensor out = FloatTensor(os);
  if (b.shape.size() == 2) {
    const std::int64_t rows = a.numel() / k;
    GemmFast(a.f.data(), b.f.data(), nullptr, out.f.data(), static_cast<int>(rows),
             static_cast<int>(n), static_cast<int>(k));
    return out;
  }
  if (a.shape.size() != b.shape.size() ||
      !std::equal(a.shape.begin(), a.shape.end() - 2, b.shape.begin()))
    Fail("MatMul batch broadcast is not implemented");
  const std::int64_t batches = a.numel() / (m * k);
  for (std::int64_t bt = 0; bt < batches; ++bt)
    GemmFast(a.f.data() + bt * m * k, b.f.data() + bt * k * n, nullptr, out.f.data() + bt * m * n,
             static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
  return out;
}

NetTensor Softmax(const NetTensor& x, std::int64_t axis_attr) {
  // opset < 13: softmax over the flattened dims [axis, rank).
  const std::int64_t axis = Axis(axis_attr, x.shape.size());
  std::int64_t inner = 1;
  for (std::size_t d = Sz(axis); d < x.shape.size(); ++d) inner *= x.shape[d];
  const std::int64_t rows = x.numel() / inner;
  NetTensor out = FloatTensor(x.shape);
#pragma omp parallel for schedule(static) if (rows > 1 && rows * inner > 32768)
  for (std::int64_t r = 0; r < rows; ++r) {
    const float* xp = x.f.data() + r * inner;
    float* yp = out.f.data() + r * inner;
    float mx = xp[0];
    for (std::int64_t j = 1; j < inner; ++j) mx = std::max(mx, xp[j]);
    float sum = 0.0f;
    for (std::int64_t j = 0; j < inner; ++j) {
      yp[j] = std::exp(xp[j] - mx);
      sum += yp[j];
    }
    for (std::int64_t j = 0; j < inner; ++j) yp[j] /= sum;
  }
  return out;
}

NetTensor ReduceMeanLast(const NetTensor& x, const Attrs& at) {
  const auto axes = at.List("axes", {-1});
  if (axes.size() != 1 ||
      Axis(axes[0], x.shape.size()) != static_cast<std::int64_t>(x.shape.size()) - 1)
    Fail("ReduceMean only over the last axis");
  const bool keep = at.Int("keepdims", 1) != 0;
  const std::int64_t inner = x.shape.back();
  const std::int64_t rows = x.numel() / inner;
  Shape os = x.shape;
  if (keep) os.back() = 1;
  else os.pop_back();
  NetTensor out = FloatTensor(os);
  for (std::int64_t r = 0; r < rows; ++r) {
    const float* xp = x.f.data() + r * inner;
    float s = 0.0f;
    for (std::int64_t j = 0; j < inner; ++j) s += xp[j];
    out.f[Sz(r)] = s / static_cast<float>(inner);
  }
  return out;
}

NetTensor Squeeze(const NetTensor& x, const std::vector<std::int64_t>& axes) {
  NetTensor out = x;
  Shape os;
  for (std::size_t d = 0; d < x.shape.size(); ++d) {
    bool drop = axes.empty() && x.shape[d] == 1;
    for (std::int64_t a : axes)
      if (Axis(a, x.shape.size()) == static_cast<std::int64_t>(d)) drop = true;
    if (drop && x.shape[d] != 1) Fail("Squeeze of a dim != 1");
    if (!drop) os.push_back(x.shape[d]);
  }
  out.shape = os;
  return out;
}

NetTensor Unsqueeze(const NetTensor& x, const std::vector<std::int64_t>& axes) {
  const std::size_t rank = x.shape.size() + axes.size();
  std::vector<bool> inserted(rank, false);
  for (std::int64_t a : axes) {
    const std::int64_t d = Axis(a, rank);
    if (inserted[Sz(d)]) Fail("Unsqueeze: repeated axis");
    inserted[Sz(d)] = true;
  }
  NetTensor out = x;
  Shape os;
  std::size_t src = 0;
  for (std::size_t d = 0; d < rank; ++d) os.push_back(inserted[d] ? 1 : x.shape[src++]);
  out.shape = os;
  return out;
}

class CpuOcrNet final : public OcrNet {
public:
  bool Init(const std::filesystem::path& dir, const std::string& stem, std::string* error) {
    TensorFile weights;
    if (!weights.Load(dir / (stem + ".safetensors"), error)) return false;
    std::ifstream in(dir / (stem + ".json"), std::ios::binary);
    if (!in) {
      if (error) *error = "missing " + (dir / (stem + ".json")).string();
      return false;
    }
    std::stringstream ss;
    ss << in.rdbuf();
    try {
      const json::Value arch = json::parse(ss.str());
      std::unordered_map<std::string, int> names;  // run slots (>= 0), constants (< 0)
      for (const auto& [name, t] : weights.all()) {
        NetTensor c;
        c.shape = t.shape;
        c.f = t.f32;
        if (c.f.size() != Sz(c.numel())) Fail("tensor " + name + " is not F32");
        consts_.push_back(std::move(c));
        names[name] = -static_cast<int>(consts_.size());
      }
      if (const json::Value* ic = arch.find("int_constants"); ic != nullptr) {
        for (const auto& [name, t] : ic->members()) {
          NetTensor c;
          c.is_int = true;
          for (const auto& d : t.find("shape")->items())
            c.shape.push_back(static_cast<std::int64_t>(d.as_double()));
          for (const auto& v : t.find("data")->items()) {
            const double dv = v.as_double();
            c.i.push_back(dv >= 9.2e18    ? std::numeric_limits<std::int64_t>::max()
                          : dv <= -9.2e18 ? std::numeric_limits<std::int64_t>::min()
                                          : static_cast<std::int64_t>(dv));
          }
          if (c.i.size() != Sz(c.numel())) Fail("int constant " + name + " size");
          consts_.push_back(std::move(c));
          names[name] = -static_cast<int>(consts_.size());
        }
      }
      if (const json::Value* op = arch.find("opset"); op != nullptr && op->is_number())
        opset_ = static_cast<std::int64_t>(op->as_double());
      const std::string input = arch.member_str("input");
      const std::string output = arch.member_str("output");
      input_ = slots_++;
      names[input] = input_;
      for (const auto& nj : arch.find("nodes")->items()) {
        Node node;
        node.op = ParseOp(nj.member_str("op"));
        for (const auto& i : nj.find("inputs")->items()) {
          if (i.str().empty()) {
            node.in.push_back(kNone);
            continue;
          }
          auto it = names.find(i.str());
          if (it == names.end()) Fail("input " + i.str() + " is not produced");
          node.in.push_back(it->second);
        }
        const auto& outs = nj.find("outputs")->items();
        if (outs.size() != 1) Fail("nodes with several outputs are not implemented");
        node.name = outs[0].str();
        names[node.name] = slots_;
        node.out.push_back(slots_++);
        if (const json::Value* at = nj.find("attrs"); at != nullptr) {
          for (const auto& [key, v] : at->members()) {
            if (v.is_string()) {
              node.attrs.strings[key] = v.str();
            } else if (v.is_array()) {
              std::vector<std::int64_t> l;
              for (const auto& e : v.items()) l.push_back(static_cast<std::int64_t>(e.as_double()));
              node.attrs.ints[key] = l;
            } else if (v.is_number()) {
              const double dv = v.as_double();
              node.attrs.floats[key] = dv;
              if (std::floor(dv) == dv) node.attrs.ints[key] = {static_cast<std::int64_t>(dv)};
            }
          }
        }
        nodes_.push_back(std::move(node));
      }
      auto it = names.find(output);
      if (it == names.end() || it->second < 0) Fail("output " + output + " is not produced");
      output_ = it->second;
      // Free every run slot after its last use.
      std::vector<int> last(Sz(slots_), -1);
      for (std::size_t k = 0; k < nodes_.size(); ++k)
        for (int s : nodes_[k].in)
          if (s >= 0) last[Sz(s)] = static_cast<int>(k);
      for (int s = 0; s < slots_; ++s)
        if (last[Sz(s)] >= 0 && s != output_) nodes_[Sz(last[Sz(s)])].release.push_back(s);
      FindFusedRuns(last);
    } catch (const std::exception& e) {
      if (error) *error = "ocr model " + stem + ": " + e.what();
      return false;
    }
    return true;
  }

  NetTensor Run(NetTensor input) const override {
    BufferPool pool;
    BufferPool* const outer_pool = tls_pool;
    tls_pool = &pool;
    struct Restore {
      BufferPool* p;
      ~Restore() { tls_pool = p; }
    } restore{outer_pool};
    std::vector<NetTensor> slots(Sz(slots_));
    slots[Sz(input_)] = std::move(input);
    std::size_t next_run = 0;
    for (std::size_t ni = 0; ni < nodes_.size(); ++ni) {
      while (next_run < runs_.size() && runs_[next_run].first < ni) ++next_run;
      if (next_run < runs_.size() && runs_[next_run].first == ni &&
          ExecFused(runs_[next_run], slots)) {
        ni = runs_[next_run].last;
        continue;
      }
      const Node& n = nodes_[ni];
      NetTensor y;
      try {
        y = Eval(n, slots);
      } catch (const std::exception& e) {
        Fail(std::string(e.what()) + " (node " + n.name + ")");
      }
      slots[Sz(n.out[0])] = std::move(y);
      for (int s : n.release) Recycle(slots[Sz(s)]);
    }
    return std::move(slots[Sz(output_)]);
  }

private:
  // Scalar float constant (one element) behind input id, or nullptr.
  const NetTensor* ScalarConst(int id) const {
    if (id >= 0 || id == kNone) return nullptr;
    const NetTensor& t = consts_[Sz(-id - 1)];
    return (!t.is_int && t.f.size() == 1) ? &t : nullptr;
  }

  bool Fusable(const Node& n) const {
    auto operand = [&](std::size_t k) {
      return k < n.in.size() && (n.in[k] >= 0 || ScalarConst(n.in[k]) != nullptr);
    };
    switch (n.op) {
      case Op::kAdd: case Op::kSub: case Op::kMul: case Op::kDiv:
        return n.in.size() == 2 && operand(0) && operand(1) && (n.in[0] >= 0 || n.in[1] >= 0);
      case Op::kPow:
        return n.in.size() == 2 && n.in[0] >= 0 && ScalarConst(n.in[1]) != nullptr;
      case Op::kRelu: case Op::kSigmoid: case Op::kSqrt: case Op::kHardSigmoid: case Op::kHardSwish:
        return n.in.size() == 1 && n.in[0] >= 0;
      case Op::kClip:
        if (n.in.empty() || n.in[0] < 0) return false;
        for (std::size_t k = 1; k < n.in.size(); ++k)
          if (n.in[k] != kNone && ScalarConst(n.in[k]) == nullptr) return false;
        return true;
      default:
        return false;
    }
  }

  void FindFusedRuns(const std::vector<int>& last) {
    for (std::size_t k = 0; k < nodes_.size(); ++k) {
      if (!Fusable(nodes_[k])) continue;
      std::size_t j = k;
      while (j + 1 < nodes_.size() && Fusable(nodes_[j + 1])) ++j;
      if (j > k) {
        FusedRun run;
        run.first = k;
        run.last = j;
        std::unordered_map<int, int> reg;  // run slot -> register
        // Externals first (registers 0..), then one register per node.
        for (std::size_t i = k; i <= j; ++i)
          for (int id : nodes_[i].in) {
            if (id < 0 || reg.count(id) != 0) continue;
            bool produced = false;
            for (std::size_t q = k; q < i; ++q) produced = produced || nodes_[q].out[0] == id;
            if (!produced) {
              reg[id] = static_cast<int>(run.ext.size());
              run.ext.push_back(id);
            }
          }
        const int base = static_cast<int>(run.ext.size());
        for (std::size_t i = k; i <= j; ++i) {
          const Node& n = nodes_[i];
          auto arg = [&](std::size_t q) {
            FusedArg a;
            if (q >= n.in.size() || n.in[q] == kNone) return a;
            if (n.in[q] >= 0) {
              a.reg = reg.at(n.in[q]);
            } else {
              a.value = ScalarConst(n.in[q])->f[0];
            }
            return a;
          };
          FusedOp op;
          op.op = n.op;
          op.a = arg(0);
          if (n.op == Op::kClip) {
            op.p0 = -std::numeric_limits<float>::infinity();
            op.p1 = std::numeric_limits<float>::infinity();
            if (n.in.size() > 1 && n.in[1] != kNone) op.p0 = ScalarConst(n.in[1])->f[0];
            else op.p0 = static_cast<float>(n.attrs.Float("min", op.p0));
            if (n.in.size() > 2 && n.in[2] != kNone) op.p1 = ScalarConst(n.in[2])->f[0];
            else op.p1 = static_cast<float>(n.attrs.Float("max", op.p1));
          } else if (n.op == Op::kHardSigmoid) {
            op.p0 = static_cast<float>(n.attrs.Float("alpha", 0.2));
            op.p1 = static_cast<float>(n.attrs.Float("beta", 0.5));
          } else if (n.in.size() == 2) {
            op.b = arg(1);
          }
          run.ops.push_back(op);
          const int out = n.out[0];
          reg[out] = base + static_cast<int>(i - k);
          if (out == output_ || last[Sz(out)] > static_cast<int>(j))
            run.outputs.push_back({reg[out], out});
          for (int r : n.release) run.release.push_back(r);
        }
        if (!run.outputs.empty()) runs_.push_back(std::move(run));
      }
      k = j;
    }
  }

  // Runs a fused run; false (nothing changed) when the slot inputs differ in
  // shape, then the nodes run one by one.
  bool ExecFused(const FusedRun& run, std::vector<NetTensor>& s) const {
    const NetTensor& first = s[Sz(run.ext[0])];
    for (int id : run.ext) {
      const NetTensor& t = s[Sz(id)];
      if (t.is_int || t.shape != first.shape || t.f.size() != first.f.size()) return false;
    }
    const Shape shape = first.shape;
    const std::int64_t n = static_cast<std::int64_t>(first.f.size());
    std::vector<const float*> src;
    for (int id : run.ext) src.push_back(s[Sz(id)].f.data());
    // Output buffers: reuse an external whose last use is inside the run.
    std::vector<NetTensor> outs(run.outputs.size());
    std::vector<bool> taken(run.ext.size(), false);
    for (std::size_t o = 0; o < run.outputs.size(); ++o) {
      for (std::size_t e = 0; e < run.ext.size(); ++e) {
        if (taken[e]) continue;
        if (std::find(run.release.begin(), run.release.end(), run.ext[e]) == run.release.end())
          continue;
        taken[e] = true;
        outs[o] = std::move(s[Sz(run.ext[e])]);
        break;
      }
      if (outs[o].f.size() != Sz(n)) outs[o] = FloatTensor(shape);
      outs[o].shape = shape;
    }
    std::vector<float*> dst;
    for (auto& t : outs) dst.push_back(t.f.data());
    constexpr std::int64_t kTile = 1024;
    const std::int64_t tiles = (n + kTile - 1) / kTile;
    const int ext_regs = static_cast<int>(run.ext.size());
    const std::size_t op_regs = run.ops.size();
#pragma omp parallel if (n > 32768)
    {
      thread_local std::vector<float> buf;
      thread_local std::vector<const float*> regs;
      if (buf.size() < op_regs * Sz(kTile)) buf.resize(op_regs * Sz(kTile));
      regs.resize(Sz(ext_regs) + op_regs);
#pragma omp for schedule(static)
      for (std::int64_t t = 0; t < tiles; ++t) {
        const std::int64_t off = t * kTile;
        const std::int64_t len = std::min(kTile, n - off);
        for (int e = 0; e < ext_regs; ++e) regs[Sz(e)] = src[Sz(e)] + off;
        for (std::size_t i = 0; i < op_regs; ++i) {
          const FusedOp& op = run.ops[i];
          float* y = buf.data() + i * Sz(kTile);
          regs[Sz(ext_regs) + i] = y;
          const float* xa = op.a.reg >= 0 ? regs[Sz(op.a.reg)] : nullptr;
          const float* xb = op.b.reg >= 0 ? regs[Sz(op.b.reg)] : nullptr;
          FusedOpTile(op, xa, xb, y, len);
        }
        for (std::size_t o = 0; o < run.outputs.size(); ++o)
          std::memcpy(dst[o] + off, regs[Sz(run.outputs[o].first)], sizeof(float) * Sz(len));
      }
    }
    for (std::size_t o = 0; o < run.outputs.size(); ++o)
      s[Sz(run.outputs[o].second)] = std::move(outs[o]);
    for (int r : run.release) {
      bool kept = false;
      for (const auto& o : run.outputs) kept = kept || o.second == r;
      if (!kept) Recycle(s[Sz(r)]);
    }
    return true;
  }

  const NetTensor& Get(const std::vector<NetTensor>& slots, int id) const {
    if (id >= 0) return slots[Sz(id)];
    if (id == kNone) Fail("missing input");
    return consts_[Sz(-id - 1)];
  }
  const NetTensor* Opt(const std::vector<NetTensor>& slots, const Node& n, std::size_t k) const {
    if (k >= n.in.size() || n.in[k] == kNone) return nullptr;
    return &Get(slots, n.in[k]);
  }

  // Run slot of input k when this node is its last use (it may be
  // overwritten), else nullptr.
  NetTensor* Reusable(const Node& n, std::vector<NetTensor>& s, std::size_t k) const {
    if (k >= n.in.size() || n.in[k] < 0) return nullptr;
    const int id = n.in[k];
    if (std::find(n.release.begin(), n.release.end(), id) == n.release.end()) return nullptr;
    NetTensor& t = s[Sz(id)];
    return t.is_int ? nullptr : &t;
  }

  NetTensor Eval(const Node& n, std::vector<NetTensor>& s) const {
    const NetTensor& x = Get(s, n.in[0]);
    auto in = [&](std::size_t k) -> const NetTensor& { return Get(s, n.in.at(k)); };
    NetTensor* r0 = Reusable(n, s, 0);
    auto bin = [&](Bin op) { return Binary(op, x, in(1), r0, Reusable(n, s, 1)); };
    switch (n.op) {
      case Op::kAdd: return bin(Bin::kAdd);
      case Op::kSub: return bin(Bin::kSub);
      case Op::kMul: return bin(Bin::kMul);
      case Op::kDiv: return bin(Bin::kDiv);
      case Op::kPow: return bin(Bin::kPow);
      case Op::kRelu: return Unary(x, [](float v) { return v > 0.0f ? v : 0.0f; }, r0);
      case Op::kSigmoid:
        return Unary(x, [](float v) { return 1.0f / (1.0f + std::exp(-v)); }, r0);
      case Op::kSqrt: return Unary(x, [](float v) { return std::sqrt(v); }, r0);
      case Op::kHardSwish: return Unary(x, [](float v) { return HardSwish(v); }, r0);
      case Op::kHardSigmoid: {
        const float alpha = static_cast<float>(n.attrs.Float("alpha", 0.2));
        const float beta = static_cast<float>(n.attrs.Float("beta", 0.5));
        return Unary(x, [alpha, beta](float v) {
          return std::max(0.0f, std::min(1.0f, alpha * v + beta));
        }, r0);
      }
      case Op::kClip: {
        float lo = -std::numeric_limits<float>::infinity();
        float hi = std::numeric_limits<float>::infinity();
        if (const NetTensor* t = Opt(s, n, 1)) lo = t->f.at(0);
        else lo = static_cast<float>(n.attrs.Float("min", lo));
        if (const NetTensor* t = Opt(s, n, 2)) hi = t->f.at(0);
        else hi = static_cast<float>(n.attrs.Float("max", hi));
        return Unary(x, [lo, hi](float v) { return std::min(hi, std::max(lo, v)); }, r0);
      }
      case Op::kConv: return Conv(x, in(1), Opt(s, n, 2), n.attrs);
      case Op::kConvTranspose: return ConvTranspose(x, in(1), Opt(s, n, 2), n.attrs);
      case Op::kBatchNorm:
        return BatchNorm(x, in(1), in(2), in(3), in(4), n.attrs.Float("epsilon", 1e-5), r0);
      case Op::kGlobalAveragePool: return GlobalAveragePool(x);
      case Op::kAveragePool: return AveragePool(x, n.attrs);
      case Op::kResize: return Resize(x, Opt(s, n, 2), Opt(s, n, 3));
      case Op::kConcat: {
        std::vector<const NetTensor*> xs;
        for (int id : n.in) xs.push_back(&Get(s, id));
        return Concat(xs, n.attrs.Int("axis", 0));
      }
      case Op::kShape: {
        NetTensor t;
        t.is_int = true;
        t.shape = {static_cast<std::int64_t>(x.shape.size())};
        t.i = x.shape;
        return t;
      }
      case Op::kCast: {
        const std::int64_t to = n.attrs.Int("to", 1);
        NetTensor y;
        y.shape = x.shape;
        if (to == 1) {
          if (!x.is_int) return x;
          for (std::int64_t v : x.i) y.f.push_back(static_cast<float>(v));
          return y;
        }
        y.is_int = true;
        if (x.is_int) {
          y.i = x.i;
          if (to == 6)
            for (auto& v : y.i) v = static_cast<std::int32_t>(v);
        } else {
          for (float v : x.f) y.i.push_back(static_cast<std::int64_t>(v));
        }
        return y;
      }
      case Op::kSlice: return Slice(x, in(1), Opt(s, n, 2), Opt(s, n, 3), Opt(s, n, 4));
      // opset >= 13: the axes are an int input instead of an attribute.
      case Op::kSqueeze: return Squeeze(x, AxesOf(n, s));
      case Op::kUnsqueeze: return Unsqueeze(x, AxesOf(n, s));
      case Op::kReshape: return Reshape(x, in(1));
      case Op::kTranspose: return Transpose(x, n.attrs.List("perm", {}));
      case Op::kReduceMean: return ReduceMeanLast(x, n.attrs);
      case Op::kMatMul: return MatMul(x, in(1));
      case Op::kSoftmax: {
        // opset 13 changed the default axis to -1 and the meaning to one
        // axis; both agree when it is the last one, which is all that runs.
        const std::int64_t axis = n.attrs.Int("axis", opset_ >= 13 ? -1 : 1);
        if (opset_ >= 13 && Axis(axis, x.shape.size()) != static_cast<std::int64_t>(x.shape.size()) - 1)
          Fail("Softmax over a non-last axis (opset >= 13) is not implemented");
        return Softmax(x, axis);
      }
    }
    Fail("unhandled op");
  }

  // Squeeze / Unsqueeze axes: the second (int) input when present, else the attribute.
  std::vector<std::int64_t> AxesOf(const Node& n, const std::vector<NetTensor>& s) const {
    if (const NetTensor* t = Opt(s, n, 1)) {
      if (!t->is_int) Fail("axes must be an int tensor");
      return t->i;
    }
    return n.attrs.List("axes", {});
  }

  std::vector<NetTensor> consts_;
  std::vector<Node> nodes_;
  std::int64_t opset_{11};
  std::vector<FusedRun> runs_;
  int slots_{0};
  int input_{0};
  int output_{0};
};

}  // namespace

std::unique_ptr<OcrNet> OcrNet::Load(const std::filesystem::path& dir, const std::string& stem,
                                     std::string* error) {
  auto net = std::make_unique<CpuOcrNet>();
  if (!net->Init(dir, stem, error)) return nullptr;
  return net;
}

}  // namespace gufo::models::ccip::cpu
