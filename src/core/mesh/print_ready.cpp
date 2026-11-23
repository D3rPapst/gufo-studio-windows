// Print-ready meshes ("druckfertig"), see print_ready.hpp.
// AI-assisted (Claude Code), 2026-11-23.
#include "src/core/mesh/print_ready.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <thread>

#include "src/core/json.hpp"
#include "src/models/triposg/marching_cubes.hpp"

namespace gufo::mesh {
namespace {

using Clock = std::chrono::steady_clock;

int thread_count(int requested) {
    if (requested > 0) return requested;
    const unsigned hw = std::thread::hardware_concurrency();
    return static_cast<int>(std::clamp<unsigned>(hw == 0 ? 4u : hw, 1u, 16u));
}

// f(begin, end) over [0, n) in chunks, on up to `threads` threads (the caller is one of them).
template <class F>
void parallel_for(std::size_t n, int threads, std::size_t chunk, F&& f) {
    if (n == 0) return;
    chunk = std::max<std::size_t>(1, chunk);
    const std::size_t chunks = (n + chunk - 1) / chunk;
    const int t = static_cast<int>(std::min<std::size_t>(static_cast<std::size_t>(std::max(1, threads)), chunks));
    std::atomic<std::size_t> next{0};
    auto work = [&] {
        for (;;) {
            const std::size_t c = next.fetch_add(1, std::memory_order_relaxed);
            if (c >= chunks) return;
            const std::size_t b = c * chunk;
            f(b, std::min(n, b + chunk));
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(static_cast<std::size_t>(t > 1 ? t - 1 : 0));
    for (int i = 1; i < t; ++i) pool.emplace_back(work);
    work();
    for (auto& th : pool) th.join();
}

struct Vec3 {
    double x, y, z;
};
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator*(Vec3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
inline double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// Closest point on triangle abc to p (Ericson, Real-Time Collision Detection 5.1.5);
// returns the point and its barycentric weights (wa, wb, wc).
Vec3 closest_on_triangle(Vec3 p, Vec3 a, Vec3 b, Vec3 c, double w[3]) {
    const Vec3 ab = b - a, ac = c - a, ap = p - a;
    const double d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0 && d2 <= 0) { w[0] = 1; w[1] = 0; w[2] = 0; return a; }
    const Vec3 bp = p - b;
    const double d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0 && d4 <= d3) { w[0] = 0; w[1] = 1; w[2] = 0; return b; }
    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) {
        const double v = d1 / (d1 - d3);
        w[0] = 1 - v; w[1] = v; w[2] = 0;
        return a + ab * v;
    }
    const Vec3 cp = p - c;
    const double d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0 && d5 <= d6) { w[0] = 0; w[1] = 0; w[2] = 1; return c; }
    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) {
        const double v = d2 / (d2 - d6);
        w[0] = 1 - v; w[1] = 0; w[2] = v;
        return a + ac * v;
    }
    const double va = d3 * d6 - d5 * d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
        const double v = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        w[0] = 0; w[1] = 1 - v; w[2] = v;
        return b + (c - b) * v;
    }
    const double denom = 1.0 / (va + vb + vc);
    const double v = vb * denom, u = vc * denom;
    w[0] = 1 - v - u; w[1] = v; w[2] = u;
    return a + ab * v + ac * u;
}

// Uniform grid of the source triangles (CSR) for nearest-surface queries.
class SurfaceGrid {
public:
    SurfaceGrid(const std::vector<float>& pos, const std::vector<std::uint32_t>& idx, const double lo[3], const double hi[3], double cell)
        : pos_(pos), idx_(idx), h_(cell) {
        for (int a = 0; a < 3; ++a) {
            lo_[a] = lo[a] - cell;
            n_[a] = static_cast<int>(std::floor((hi[a] + cell - lo_[a]) / cell)) + 1;
        }
        const std::size_t cells = static_cast<std::size_t>(n_[0]) * n_[1] * n_[2];
        start_.assign(cells + 1, 0);
        const std::size_t faces = idx.size() / 3;
        auto range = [&](std::size_t f, int r0[3], int r1[3]) {
            for (int a = 0; a < 3; ++a) {
                double mn = std::numeric_limits<double>::max(), mx = -mn;
                for (int k = 0; k < 3; ++k) {
                    const double v = pos[idx[f * 3 + k] * 3 + a];
                    mn = std::min(mn, v); mx = std::max(mx, v);
                }
                r0[a] = std::clamp(static_cast<int>(std::floor((mn - lo_[a]) / h_)), 0, n_[a] - 1);
                r1[a] = std::clamp(static_cast<int>(std::floor((mx - lo_[a]) / h_)), 0, n_[a] - 1);
            }
        };
        for (std::size_t f = 0; f < faces; ++f) {
            int r0[3], r1[3];
            range(f, r0, r1);
            for (int z = r0[2]; z <= r1[2]; ++z)
                for (int y = r0[1]; y <= r1[1]; ++y)
                    for (int x = r0[0]; x <= r1[0]; ++x) ++start_[cell_index(x, y, z) + 1];
        }
        for (std::size_t c = 0; c < cells; ++c) start_[c + 1] += start_[c];
        items_.resize(start_[cells]);
        std::vector<std::uint32_t> cursor(start_.begin(), start_.end() - 1);
        for (std::size_t f = 0; f < faces; ++f) {
            int r0[3], r1[3];
            range(f, r0, r1);
            for (int z = r0[2]; z <= r1[2]; ++z)
                for (int y = r0[1]; y <= r1[1]; ++y)
                    for (int x = r0[0]; x <= r1[0]; ++x) items_[cursor[cell_index(x, y, z)]++] = static_cast<std::uint32_t>(f);
        }
    }

    std::size_t bytes() const { return start_.size() * 4 + items_.size() * 4; }

    // Nearest surface point within max_rings cells (Chebyshev) of p's cell; max_rings < 0 = unbounded.
    // Returns false when nothing was found.
    bool nearest(Vec3 p, int max_rings, Vec3* out, std::uint32_t* face, double w[3], double* dist) const {
        const int c[3] = {cell_of(p.x, 0), cell_of(p.y, 1), cell_of(p.z, 2)};
        const int limit = max_rings < 0 ? std::max({n_[0], n_[1], n_[2]}) : max_rings;
        double best = std::numeric_limits<double>::max();
        bool found = false;
        for (int r = 0; r <= limit; ++r) {
            for (int z = c[2] - r; z <= c[2] + r; ++z) {
                if (z < 0 || z >= n_[2]) continue;
                for (int y = c[1] - r; y <= c[1] + r; ++y) {
                    if (y < 0 || y >= n_[1]) continue;
                    // Shell only: inner rows visit just their two end cells.
                    const bool full = r == 0 || std::abs(z - c[2]) == r || std::abs(y - c[1]) == r;
                    const int x_step = full ? 1 : 2 * r;
                    for (int x = c[0] - r; x <= c[0] + r; x += x_step) {
                        if (x < 0 || x >= n_[0]) continue;
                        const std::size_t ci = cell_index(x, y, z);
                        for (std::uint32_t k = start_[ci]; k < start_[ci + 1]; ++k) {
                            const std::uint32_t f = items_[k];
                            double ww[3];
                            const Vec3 q = closest_on_triangle(p, vertex(idx_[f * 3]), vertex(idx_[f * 3 + 1]), vertex(idx_[f * 3 + 2]), ww);
                            const Vec3 d = q - p;
                            const double d2 = dot(d, d);
                            if (d2 < best) {
                                best = d2; *out = q; *face = f;
                                w[0] = ww[0]; w[1] = ww[1]; w[2] = ww[2];
                                found = true;
                            }
                        }
                    }
                }
            }
            // Every triangle not seen yet lies at least r * h away from p.
            if (found && std::sqrt(best) <= r * h_) break;
        }
        if (found) *dist = std::sqrt(best);
        return found;
    }

private:
    Vec3 vertex(std::uint32_t v) const { return {pos_[v * 3], pos_[v * 3 + 1], pos_[v * 3 + 2]}; }
    int cell_of(double v, int a) const { return std::clamp(static_cast<int>(std::floor((v - lo_[a]) / h_)), 0, n_[a] - 1); }
    std::size_t cell_index(int x, int y, int z) const { return (static_cast<std::size_t>(z) * n_[1] + y) * n_[0] + x; }

    const std::vector<float>& pos_;
    const std::vector<std::uint32_t>& idx_;
    double lo_[3];
    int n_[3];
    double h_;
    std::vector<std::uint32_t> start_, items_;
};

// Union-find over vertices; pieces connected through shared vertices of faces.
struct DisjointSet {
    std::vector<std::uint32_t> parent;
    explicit DisjointSet(std::size_t n) : parent(n) { std::iota(parent.begin(), parent.end(), 0u); }
    std::uint32_t find(std::uint32_t v) {
        while (parent[v] != v) { parent[v] = parent[parent[v]]; v = parent[v]; }
        return v;
    }
    void unite(std::uint32_t a, std::uint32_t b) {
        a = find(a); b = find(b);
        if (a != b) parent[std::max(a, b)] = std::min(a, b);
    }
};

// Keeps the faces of the face-connected piece with most faces; returns the number of pieces.
std::size_t keep_largest_piece(PrintMesh* m) {
    const std::size_t nv = m->vertex_count(), nf = m->face_count();
    if (nf == 0) return 0;
    DisjointSet ds(nv);
    for (std::size_t f = 0; f < nf; ++f) {
        ds.unite(m->indices[f * 3], m->indices[f * 3 + 1]);
        ds.unite(m->indices[f * 3], m->indices[f * 3 + 2]);
    }
    std::vector<std::uint32_t> count(nv, 0);
    for (std::size_t f = 0; f < nf; ++f) ++count[ds.find(m->indices[f * 3])];
    std::size_t pieces = 0;
    std::uint32_t best = 0;
    for (std::size_t v = 0; v < nv; ++v) {
        if (count[v] > 0) ++pieces;
        if (count[v] > count[best]) best = static_cast<std::uint32_t>(v);
    }
    if (pieces <= 1) return pieces;
    std::vector<std::uint32_t> remap(nv, UINT32_MAX);
    PrintMesh out;
    for (std::size_t f = 0; f < nf; ++f) {
        if (ds.find(m->indices[f * 3]) != best) continue;
        for (int k = 0; k < 3; ++k) {
            const std::uint32_t v = m->indices[f * 3 + k];
            if (remap[v] == UINT32_MAX) {
                remap[v] = static_cast<std::uint32_t>(out.vertex_count());
                out.positions.insert(out.positions.end(), m->positions.begin() + v * 3, m->positions.begin() + v * 3 + 3);
                if (!m->colors.empty()) out.colors.insert(out.colors.end(), m->colors.begin() + v * 4, m->colors.begin() + v * 4 + 4);
            }
            out.indices.push_back(remap[v]);
        }
    }
    *m = std::move(out);
    return pieces;
}

// Voxel volume, x fastest: index = (z * ny + y) * nx + x.
struct Dims {
    std::size_t nx, ny, nz;
    std::size_t size() const { return nx * ny * nz; }
    std::size_t at(std::size_t x, std::size_t y, std::size_t z) const { return (z * ny + y) * nx + x; }
};

// One step of 6-neighbourhood dilation (grow) or erosion (shrink, outside counts as empty).
void morph_step(const Dims& d, const std::uint8_t* in, std::uint8_t* out, bool grow, int threads) {
    const std::size_t sx = 1, sy = d.nx, sz = d.nx * d.ny;
    parallel_for(d.nz, threads, 4, [&](std::size_t z0, std::size_t z1) {
        for (std::size_t z = z0; z < z1; ++z)
            for (std::size_t y = 0; y < d.ny; ++y) {
                const std::size_t row = d.at(0, y, z);
                for (std::size_t x = 0; x < d.nx; ++x) {
                    const std::size_t i = row + x;
                    if (grow) {
                        std::uint8_t v = in[i];
                        if (!v) {
                            v = (x > 0 && in[i - sx]) || (x + 1 < d.nx && in[i + sx]) || (y > 0 && in[i - sy]) ||
                                (y + 1 < d.ny && in[i + sy]) || (z > 0 && in[i - sz]) || (z + 1 < d.nz && in[i + sz]);
                        }
                        out[i] = v ? 1 : 0;
                    } else {
                        bool v = in[i] != 0;
                        if (v) {
                            v = x > 0 && in[i - sx] && x + 1 < d.nx && in[i + sx] && y > 0 && in[i - sy] && y + 1 < d.ny &&
                                in[i + sy] && z > 0 && in[i - sz] && z + 1 < d.nz && in[i + sz];
                        }
                        out[i] = v ? 1 : 0;
                    }
                }
            }
    });
}

// 2-D hole filling of one w x h image (in may equal out): background not 4-connected to the border
// becomes solid. Returns the solid pixel count.
std::size_t fill_holes_2d(const std::uint8_t* in, std::uint8_t* out, std::size_t w, std::size_t h, std::vector<std::uint8_t>* seen_buf,
                          std::vector<std::uint32_t>* stack) {
    std::vector<std::uint8_t>& seen = *seen_buf;
    seen.assign(w * h, 0);
    stack->clear();
    auto push = [&](std::size_t i) {
        if (!in[i] && !seen[i]) { seen[i] = 1; stack->push_back(static_cast<std::uint32_t>(i)); }
    };
    for (std::size_t x = 0; x < w; ++x) { push(x); push((h - 1) * w + x); }
    for (std::size_t y = 0; y < h; ++y) { push(y * w); push(y * w + w - 1); }
    while (!stack->empty()) {
        const std::size_t i = stack->back();
        stack->pop_back();
        const std::size_t x = i % w, y = i / w;
        if (x > 0) push(i - 1);
        if (x + 1 < w) push(i + 1);
        if (y > 0) push(i - w);
        if (y + 1 < h) push(i + w);
    }
    std::size_t n = 0;
    for (std::size_t i = 0; i < w * h; ++i) {
        out[i] = (in[i] || !seen[i]) ? 1 : 0;
        n += out[i];
    }
    return n;
}

// Squared Euclidean distance of every pixel to the nearest pixel with mask == target
// (Felzenszwalb & Huttenlocher, separable); "infinite" when there is none.
void squared_distance_2d(const std::uint8_t* mask, std::uint8_t target, std::size_t w, std::size_t h, std::vector<double>* out) {
    const double inf = 1e20;
    std::vector<double>& d = *out;
    d.assign(w * h, inf);
    for (std::size_t i = 0; i < w * h; ++i)
        if (mask[i] == target) d[i] = 0;
    std::vector<double> f(std::max(w, h)), z(std::max(w, h) + 1), res(std::max(w, h));
    std::vector<int> v(std::max(w, h));
    auto pass = [&](std::size_t n) {   // 1-D lower envelope of parabolas over f[0..n) -> res
        auto cut = [&](int q, int r) {
            return ((f[static_cast<std::size_t>(q)] + double(q) * q) - (f[static_cast<std::size_t>(r)] + double(r) * r)) / (2.0 * q - 2.0 * r);
        };
        std::size_t k = 0;
        v[0] = 0; z[0] = -inf; z[1] = inf;
        for (int q = 1; q < static_cast<int>(n); ++q) {
            double sq = cut(q, v[k]);
            while (sq <= z[k]) { --k; sq = cut(q, v[k]); }   // z[0] = -inf stops at k = 0
            ++k;
            v[k] = q; z[k] = sq; z[k + 1] = inf;
        }
        k = 0;
        for (int q = 0; q < static_cast<int>(n); ++q) {
            while (z[k + 1] < q) ++k;
            const int r = v[k];
            res[static_cast<std::size_t>(q)] = double(q - r) * (q - r) + f[static_cast<std::size_t>(r)];
        }
    };
    for (std::size_t x = 0; x < w; ++x) {
        for (std::size_t y = 0; y < h; ++y) f[y] = d[y * w + x];
        pass(h);
        for (std::size_t y = 0; y < h; ++y) d[y * w + x] = res[y];
    }
    for (std::size_t y = 0; y < h; ++y) {
        for (std::size_t x = 0; x < w; ++x) f[x] = d[y * w + x];
        pass(w);
        for (std::size_t x = 0; x < w; ++x) d[y * w + x] = res[x];
    }
}

struct SliceStats {
    std::size_t open = 0, patched = 0;
};

// Step 3: fill holes per Z slice (an open hem becomes solid). A slice whose ring is open at one
// spot (a robe lapel at chest height) stays a thin hollow wall that the smoothing later removes,
// leaving a crack through the body: every slice whose filled area A[z] is below 0.6 x the running
// median over 41 slices gets a 2-D closing with a growing disk (r = 2, 4, 6, 9, 12, 16, 22) before
// the fill, until A >= 0.8 x median, ORed with the raw slice (reparatur.py, 23.11.).
SliceStats fill_slices(const Dims& d, std::uint8_t* vol, std::uint8_t* raw, bool patch_open, int threads) {
    const std::size_t plane = d.nx * d.ny;
    std::copy(vol, vol + d.size(), raw);
    std::vector<double> area(d.nz, 0.0);
    parallel_for(d.nz, threads, 2, [&](std::size_t z0, std::size_t z1) {
        std::vector<std::uint8_t> seen;
        std::vector<std::uint32_t> stack;
        for (std::size_t z = z0; z < z1; ++z) {
            const std::uint8_t* r = raw + z * plane;
            if (std::none_of(r, r + plane, [](std::uint8_t v) { return v != 0; })) continue;
            area[z] = static_cast<double>(fill_holes_2d(r, vol + z * plane, d.nx, d.ny, &seen, &stack));
        }
    });
    SliceStats st;
    if (!patch_open) return st;
    std::vector<std::size_t> bad;
    std::vector<double> expected(d.nz);
    {
        std::vector<double> window;
        const long n = static_cast<long>(d.nz);
        for (long z = 0; z < n; ++z) {   // scipy median_filter(size = 41, mode = nearest)
            window.clear();
            for (long k = z - 20; k <= z + 20; ++k) window.push_back(area[static_cast<std::size_t>(std::clamp(k, 0l, n - 1))]);
            std::nth_element(window.begin(), window.begin() + 20, window.end());
            expected[static_cast<std::size_t>(z)] = window[20];
            if (area[static_cast<std::size_t>(z)] > 0 && area[static_cast<std::size_t>(z)] < 0.6 * window[20]) bad.push_back(static_cast<std::size_t>(z));
        }
    }
    st.open = bad.size();
    std::atomic<std::size_t> patched{0};
    parallel_for(bad.size(), threads, 1, [&](std::size_t b0, std::size_t b1) {
        std::vector<std::uint8_t> padded, closed, filled, seen;
        std::vector<std::uint32_t> stack;
        std::vector<double> dist;
        for (std::size_t b = b0; b < b1; ++b) {
            const std::size_t z = bad[b];
            const std::uint8_t* r = raw + z * plane;
            for (int rad : {2, 4, 6, 9, 12, 16, 22}) {
                const std::size_t pad = static_cast<std::size_t>(rad) + 1, w = d.nx + 2 * pad, h = d.ny + 2 * pad;
                padded.assign(w * h, 0);
                for (std::size_t y = 0; y < d.ny; ++y) std::copy(r + y * d.nx, r + (y + 1) * d.nx, padded.begin() + (y + pad) * w + pad);
                // closing = dilation then erosion with the disk x^2 + y^2 <= r^2
                const double r2 = static_cast<double>(rad) * rad;
                squared_distance_2d(padded.data(), 1, w, h, &dist);
                closed.resize(w * h);
                for (std::size_t i = 0; i < w * h; ++i) closed[i] = dist[i] <= r2 ? 1 : 0;
                squared_distance_2d(closed.data(), 0, w, h, &dist);
                for (std::size_t i = 0; i < w * h; ++i) closed[i] = dist[i] > r2 ? 1 : 0;
                filled.resize(w * h);
                fill_holes_2d(closed.data(), filled.data(), w, h, &seen, &stack);
                std::size_t count = 0;
                for (std::size_t y = 0; y < d.ny; ++y)
                    for (std::size_t x = 0; x < d.nx; ++x) count += filled[(y + pad) * w + x + pad];
                if (static_cast<double>(count) >= 0.8 * expected[z]) {
                    std::uint8_t* out = vol + z * plane;
                    for (std::size_t y = 0; y < d.ny; ++y)
                        for (std::size_t x = 0; x < d.nx; ++x) out[y * d.nx + x] = (filled[(y + pad) * w + x + pad] || r[y * d.nx + x]) ? 1 : 0;
                    ++patched;
                    break;
                }
            }
        }
    });
    st.patched = patched.load();
    return st;
}

// Breadth-first flood over voxels equal to `from` (6-neighbourhood), set to `to`; level by level
// so the frontier stays a thin shell. Seeds must already hold `to`.
std::size_t flood(const Dims& d, std::uint8_t* vol, std::vector<std::uint32_t> frontier, std::uint8_t from, std::uint8_t to) {
    std::size_t count = frontier.size();
    std::vector<std::uint32_t> next;
    const std::size_t sy = d.nx, sz = d.nx * d.ny;
    while (!frontier.empty()) {
        next.clear();
        for (const std::uint32_t i : frontier) {
            const std::size_t x = i % d.nx, y = (i / sy) % d.ny, z = i / sz;
            auto visit = [&](std::size_t j) {
                if (vol[j] == from) { vol[j] = to; next.push_back(static_cast<std::uint32_t>(j)); }
            };
            if (x > 0) visit(i - 1);
            if (x + 1 < d.nx) visit(i + 1);
            if (y > 0) visit(i - sy);
            if (y + 1 < d.ny) visit(i + sy);
            if (z > 0) visit(i - sz);
            if (z + 1 < d.nz) visit(i + sz);
        }
        count += next.size();
        frontier.swap(next);
    }
    return count;
}

// 3-D hole filling: background reachable from the volume border stays empty, the rest is solid.
void fill_holes_3d(const Dims& d, std::uint8_t* vol) {
    std::vector<std::uint32_t> seeds;
    for (std::size_t z = 0; z < d.nz; ++z)
        for (std::size_t y = 0; y < d.ny; ++y)
            for (std::size_t x = 0; x < d.nx; ++x) {
                const bool border = x == 0 || y == 0 || z == 0 || x + 1 == d.nx || y + 1 == d.ny || z + 1 == d.nz;
                if (!border) { x = d.nx - 2; continue; }   // jump to the last column of this row
                const std::size_t i = d.at(x, y, z);
                if (vol[i] == 0) { vol[i] = 2; seeds.push_back(static_cast<std::uint32_t>(i)); }
            }
    flood(d, vol, std::move(seeds), 0, 2);
    for (std::size_t i = 0; i < d.size(); ++i) vol[i] = vol[i] == 2 ? 0 : 1;
}

// Keeps the largest 6-connected solid component; returns the number of components.
std::size_t keep_largest_component(const Dims& d, std::uint8_t* vol) {
    std::size_t components = 0, best_size = 0;
    std::uint32_t best_seed = 0;
    for (std::size_t i = 0; i < d.size(); ++i) {
        if (vol[i] != 1) continue;
        vol[i] = 3;
        const std::size_t n = flood(d, vol, {static_cast<std::uint32_t>(i)}, 1, 3);
        ++components;
        if (n > best_size) { best_size = n; best_seed = static_cast<std::uint32_t>(i); }
    }
    if (components == 0) return 0;
    vol[best_seed] = 4;
    flood(d, vol, {best_seed}, 3, 4);
    for (std::size_t i = 0; i < d.size(); ++i) vol[i] = vol[i] == 4 ? 1 : 0;
    return components;
}

// scipy.ndimage.gaussian_filter (truncate 4, mode reflect), separable, in place.
void gaussian(const Dims& d, float* v, float sigma, int threads) {
    const int r = static_cast<int>(4.0 * sigma + 0.5);
    if (r <= 0) return;
    std::vector<float> w(static_cast<std::size_t>(2 * r + 1));
    double sum = 0;
    for (int k = -r; k <= r; ++k) sum += w[static_cast<std::size_t>(k + r)] = static_cast<float>(std::exp(-0.5 * k * k / (static_cast<double>(sigma) * sigma)));
    for (auto& x : w) x = static_cast<float>(x / sum);
    auto reflect = [](long i, long n) {
        while (i < 0 || i >= n) i = i < 0 ? -i - 1 : 2 * n - i - 1;
        return static_cast<std::size_t>(i);
    };
    const long nx = static_cast<long>(d.nx), ny = static_cast<long>(d.ny), nz = static_cast<long>(d.nz);
    // x: per row
    parallel_for(d.nz, threads, 4, [&](std::size_t z0, std::size_t z1) {
        std::vector<float> buf(d.nx);
        for (std::size_t z = z0; z < z1; ++z)
            for (std::size_t y = 0; y < d.ny; ++y) {
                float* row = v + d.at(0, y, z);
                std::copy(row, row + d.nx, buf.begin());
                for (long x = 0; x < nx; ++x) {
                    float acc = 0;
                    for (int k = -r; k <= r; ++k) acc += w[static_cast<std::size_t>(k + r)] * buf[reflect(x + k, nx)];
                    row[x] = acc;
                }
            }
    });
    // y: per slice
    parallel_for(d.nz, threads, 2, [&](std::size_t z0, std::size_t z1) {
        std::vector<float> buf(d.nx * d.ny);
        for (std::size_t z = z0; z < z1; ++z) {
            float* s = v + d.at(0, 0, z);
            std::copy(s, s + d.nx * d.ny, buf.begin());
            for (long y = 0; y < ny; ++y) {
                float* out = s + static_cast<std::size_t>(y) * d.nx;
                std::fill(out, out + d.nx, 0.f);
                for (int k = -r; k <= r; ++k) {
                    const float* in = buf.data() + reflect(y + k, ny) * d.nx;
                    const float wk = w[static_cast<std::size_t>(k + r)];
                    for (std::size_t x = 0; x < d.nx; ++x) out[x] += wk * in[x];
                }
            }
        }
    });
    // z: per y, the x-z plane gathered into a buffer
    parallel_for(d.ny, threads, 2, [&](std::size_t y0, std::size_t y1) {
        std::vector<float> buf(d.nx * d.nz);
        for (std::size_t y = y0; y < y1; ++y) {
            for (std::size_t z = 0; z < d.nz; ++z) std::copy(v + d.at(0, y, z), v + d.at(0, y, z) + d.nx, buf.begin() + z * d.nx);
            for (long z = 0; z < nz; ++z) {
                float* out = v + d.at(0, y, static_cast<std::size_t>(z));
                std::fill(out, out + d.nx, 0.f);
                for (int k = -r; k <= r; ++k) {
                    const float* in = buf.data() + reflect(z + k, nz) * d.nx;
                    const float wk = w[static_cast<std::size_t>(k + r)];
                    for (std::size_t x = 0; x < d.nx; ++x) out[x] += wk * in[x];
                }
            }
        }
    });
}

// scipy.ndimage.zoom(order=1): output n' = round(n * zoom), input coordinate = out * (n - 1) / (n' - 1).
std::vector<float> zoom_linear(const Dims& d, const float* v, float zoom, Dims* od, double scale[3], int threads) {
    const std::size_t n[3] = {d.nx, d.ny, d.nz};
    std::size_t o[3];
    std::vector<std::uint32_t> i0[3];
    std::vector<float> fr[3];
    for (int a = 0; a < 3; ++a) {
        o[a] = std::max<std::size_t>(2, static_cast<std::size_t>(std::llround(static_cast<double>(n[a]) * zoom)));
        scale[a] = static_cast<double>(n[a] - 1) / static_cast<double>(o[a] - 1);
        i0[a].resize(o[a]);
        fr[a].resize(o[a]);
        for (std::size_t k = 0; k < o[a]; ++k) {
            const double c = std::min(static_cast<double>(n[a] - 1), k * scale[a]);
            std::size_t b = static_cast<std::size_t>(std::floor(c));
            if (b + 1 >= n[a]) b = n[a] - 2;
            i0[a][k] = static_cast<std::uint32_t>(b);
            fr[a][k] = static_cast<float>(c - static_cast<double>(b));
        }
    }
    *od = Dims{o[0], o[1], o[2]};
    std::vector<float> out(od->size());
    parallel_for(o[2], threads, 4, [&](std::size_t z0, std::size_t z1) {
        for (std::size_t z = z0; z < z1; ++z) {
            const std::size_t bz = i0[2][z];
            const float tz = fr[2][z];
            for (std::size_t y = 0; y < o[1]; ++y) {
                const std::size_t by = i0[1][y];
                const float ty = fr[1][y];
                const float* p00 = v + d.at(0, by, bz);
                const float* p10 = v + d.at(0, by + 1, bz);
                const float* p01 = v + d.at(0, by, bz + 1);
                const float* p11 = v + d.at(0, by + 1, bz + 1);
                float* dst = out.data() + od->at(0, y, z);
                for (std::size_t x = 0; x < o[0]; ++x) {
                    const std::size_t bx = i0[0][x];
                    const float tx = fr[0][x];
                    auto lx = [&](const float* p) { return p[bx] + (p[bx + 1] - p[bx]) * tx; };
                    const float a = lx(p00) + (lx(p10) - lx(p00)) * ty;
                    const float b = lx(p01) + (lx(p11) - lx(p01)) * ty;
                    dst[x] = a + (b - a) * tz;
                }
            }
        }
    });
    return out;
}

// Equal-weight umbrella Laplacian, trimesh.smoothing.filter_taubin(lamb = 0.5, nu = 0.5).
void taubin(PrintMesh* m, int iterations, int threads) {
    const std::size_t nv = m->vertex_count(), nf = m->face_count();
    std::vector<std::uint32_t> start(nv + 1, 0);
    for (std::size_t f = 0; f < nf; ++f)
        for (int k = 0; k < 3; ++k) start[m->indices[f * 3 + k] + 1] += 2;
    for (std::size_t v = 0; v < nv; ++v) start[v + 1] += start[v];
    std::vector<std::uint32_t> nb(start[nv]);
    {
        std::vector<std::uint32_t> cur(start.begin(), start.end() - 1);
        for (std::size_t f = 0; f < nf; ++f)
            for (int k = 0; k < 3; ++k) {
                const std::uint32_t a = m->indices[f * 3 + k], b = m->indices[f * 3 + (k + 1) % 3], c = m->indices[f * 3 + (k + 2) % 3];
                nb[cur[a]++] = b;
                nb[cur[a]++] = c;
            }
    }
    // Unique neighbours per vertex (compacted in place, new offsets in `first`).
    std::vector<std::uint32_t> first(nv + 1, 0);
    {
        std::size_t w = 0;
        for (std::size_t v = 0; v < nv; ++v) {
            auto b = nb.begin() + start[v], e = nb.begin() + start[v + 1];
            std::sort(b, e);
            const auto u = std::unique(b, e);
            first[v] = static_cast<std::uint32_t>(w);
            for (auto it = b; it != u; ++it) nb[w++] = *it;
        }
        first[nv] = static_cast<std::uint32_t>(w);
        nb.resize(w);
        nb.shrink_to_fit();
    }
    std::vector<float> next(m->positions.size());
    for (int it = 0; it < iterations; ++it) {
        const float factor = it % 2 == 0 ? 0.5f : -0.5f;
        const std::vector<float>& cur = m->positions;
        parallel_for(nv, threads, 4096, [&](std::size_t v0, std::size_t v1) {
            for (std::size_t v = v0; v < v1; ++v) {
                const std::uint32_t b = first[v], e = first[v + 1];
                for (int a = 0; a < 3; ++a) {
                    const float self = cur[v * 3 + a];
                    if (b == e) { next[v * 3 + a] = self; continue; }
                    double s = 0;
                    for (std::uint32_t k = b; k < e; ++k) s += cur[nb[k] * 3 + a];
                    const float avg = static_cast<float>(s / (e - b));
                    next[v * 3 + a] = self + factor * (avg - self);
                }
            }
        });
        m->positions.swap(next);
    }
}

void bounds(const std::vector<float>& pos, double lo[3], double hi[3]) {
    for (int a = 0; a < 3; ++a) { lo[a] = std::numeric_limits<double>::max(); hi[a] = -lo[a]; }
    for (std::size_t i = 0; i + 2 < pos.size(); i += 3)
        for (int a = 0; a < 3; ++a) { lo[a] = std::min<double>(lo[a], pos[i + a]); hi[a] = std::max<double>(hi[a], pos[i + a]); }
}

struct MemoryTrack {
    std::size_t now = 0, peak = 0;
    void add(std::size_t b) { now += b; peak = std::max(peak, now); }
    void sub(std::size_t b) { now -= std::min(now, b); }
};

}  // namespace

std::size_t count_open_edges(const PrintMesh& mesh) {
    // Directed edges a->b with a < b against the reversed ones b->a: watertight and consistently
    // oriented when both sorted lists are identical and free of duplicates.
    std::vector<std::uint64_t> fwd, bwd;
    fwd.reserve(mesh.indices.size() / 2 + 1);
    bwd.reserve(mesh.indices.size() / 2 + 1);
    for (std::size_t f = 0; f < mesh.face_count(); ++f)
        for (int k = 0; k < 3; ++k) {
            const std::uint64_t a = mesh.indices[f * 3 + k], b = mesh.indices[f * 3 + (k + 1) % 3];
            if (a < b) fwd.push_back(a << 32 | b);
            else bwd.push_back(b << 32 | a);
        }
    std::sort(fwd.begin(), fwd.end());
    std::sort(bwd.begin(), bwd.end());
    std::size_t bad = 0, i = 0, j = 0;
    while (i < fwd.size() || j < bwd.size()) {
        if (j >= bwd.size() || (i < fwd.size() && fwd[i] < bwd[j])) { ++bad; ++i; continue; }
        if (i >= fwd.size() || bwd[j] < fwd[i]) { ++bad; ++j; continue; }
        const std::uint64_t key = fwd[i];
        std::size_t ni = 0, nj = 0;
        while (i < fwd.size() && fwd[i] == key) { ++i; ++ni; }
        while (j < bwd.size() && bwd[j] == key) { ++j; ++nj; }
        if (ni != 1 || nj != 1) bad += std::max(ni, nj);
    }
    return bad;
}

std::size_t count_bodies(const PrintMesh& mesh) {
    DisjointSet ds(mesh.vertex_count());
    for (std::size_t f = 0; f < mesh.face_count(); ++f) {
        ds.unite(mesh.indices[f * 3], mesh.indices[f * 3 + 1]);
        ds.unite(mesh.indices[f * 3], mesh.indices[f * 3 + 2]);
    }
    std::vector<std::uint8_t> root(mesh.vertex_count(), 0);
    std::size_t n = 0;
    for (std::size_t f = 0; f < mesh.face_count(); ++f) {
        const std::uint32_t r = ds.find(mesh.indices[f * 3]);
        if (!root[r]) { root[r] = 1; ++n; }
    }
    return n;
}

MeshCheck check_mesh(const PrintMesh& mesh) {
    // Undirected edge key (low << 32 | high) shifted left by one, direction (low -> high) in bit 0.
    std::vector<std::uint64_t> e;
    e.reserve(mesh.indices.size());
    for (std::size_t f = 0; f < mesh.face_count(); ++f)
        for (int k = 0; k < 3; ++k) {
            const std::uint64_t a = mesh.indices[f * 3 + k], b = mesh.indices[f * 3 + (k + 1) % 3];
            const std::uint64_t lo = std::min(a, b), hi = std::max(a, b);
            e.push_back(((lo << 32 | hi) << 1) | (a < b ? 1u : 0u));
        }
    std::sort(e.begin(), e.end());
    MeshCheck c;
    for (std::size_t i = 0; i < e.size();) {
        std::size_t j = i, forward = 0;
        while (j < e.size() && (e[j] >> 1) == (e[i] >> 1)) { forward += e[j] & 1u; ++j; }
        const std::size_t n = j - i;
        if (n == 1) ++c.boundary_edges;
        else if (n > 2) ++c.nonmanifold_edges;
        else if (forward != 1) ++c.inconsistent_edges;
        i = j;
    }
    c.watertight = c.boundary_edges == 0 && c.nonmanifold_edges == 0;
    c.winding_consistent = c.inconsistent_edges == 0;
    c.volume = signed_volume(mesh);
    c.bodies = count_bodies(mesh);
    return c;
}

std::string check_line(const MeshCheck& c) {
    auto yn = [](bool b) { return b ? std::string("yes") : std::string("no"); };
    return "watertight " + yn(c.watertight) + ", boundary edges " + std::to_string(c.boundary_edges) + ", non-manifold edges " +
           std::to_string(c.nonmanifold_edges) + ", winding consistent " + yn(c.winding_consistent) + ", volume > 0 " + yn(c.volume > 0) +
           ", bodies " + std::to_string(c.bodies);
}

double signed_volume(const PrintMesh& mesh) {
    double vol = 0;
    const auto& p = mesh.positions;
    for (std::size_t f = 0; f < mesh.face_count(); ++f) {
        const std::size_t a = mesh.indices[f * 3] * 3ull, b = mesh.indices[f * 3 + 1] * 3ull, c = mesh.indices[f * 3 + 2] * 3ull;
        const double cx = static_cast<double>(p[b + 1]) * p[c + 2] - static_cast<double>(p[b + 2]) * p[c + 1];
        const double cy = static_cast<double>(p[b + 2]) * p[c] - static_cast<double>(p[b]) * p[c + 2];
        const double cz = static_cast<double>(p[b]) * p[c + 1] - static_cast<double>(p[b + 1]) * p[c];
        vol += p[a] * cx + p[a + 1] * cy + p[a + 2] * cz;
    }
    return vol / 6.0;
}

bool make_print_ready(const PrintMesh& source, const PrintOptions& o, PrintResult* result, std::string* error) {
    const auto t_start = Clock::now();
    auto fail = [&](const std::string& message) {
        if (error) *error = message;
        return false;
    };
    auto cancelled = [&] { return o.cancel != nullptr && o.cancel->load(); };
    auto step = [&](int k) {
        if (o.progress) o.progress(k, 10);
    };
    const int threads = thread_count(o.threads);
    PrintStats& st = result->stats;
    st = PrintStats{};
    MemoryTrack mem;
    if (source.face_count() == 0 || source.vertex_count() == 0) return fail("the mesh is empty");
    if (o.grid < 16 || o.grid > 2048) return fail("grid must be 16..2048");
    if (!source.colors.empty() && source.colors.size() != source.vertex_count() * 4) return fail("colours do not match the vertices");
    for (std::uint32_t i : source.indices)
        if (i >= source.vertex_count()) return fail("face index out of range");

    // Z up: glTF (x, y, z) -> (x, -z, y).
    std::vector<float> src(source.positions.size());
    for (std::size_t v = 0; v < source.vertex_count(); ++v) {
        src[v * 3] = source.positions[v * 3];
        src[v * 3 + 1] = -source.positions[v * 3 + 2];
        src[v * 3 + 2] = source.positions[v * 3 + 1];
    }
    mem.add(src.size() * 4);
    double lo[3], hi[3];
    bounds(src, lo, hi);
    const double extent = std::max({hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]});
    if (!(extent > 0) || !std::isfinite(extent)) return fail("the mesh has no extent");
    double p = extent / o.grid;
    const int R = std::max(0, o.close_radius);
    const int margin = 4 + std::max(0, R - 2);   // reference: 4 empty voxels before the surface
    Dims d{};
    auto make_dims = [&] {
        std::size_t n[3];
        for (int a = 0; a < 3; ++a) n[a] = static_cast<std::size_t>(std::floor((hi[a] - lo[a]) / p)) + 2 * margin + 1;
        d = Dims{n[0], n[1], n[2]};
    };
    make_dims();
    // Compact objects (a chair, a cup) would fill the whole cube: coarser voxels keep the memory bound.
    const std::size_t max_voxels = o.max_voxels > 0 ? o.max_voxels : (1ull << 31) - 1;
    for (int k = 0; k < 64 && d.size() > max_voxels; ++k) {
        p *= std::max(1.01, std::cbrt(static_cast<double>(d.size()) / static_cast<double>(max_voxels)));
        make_dims();
    }
    if (d.size() >= (1ull << 31)) return fail("voxel grid too large");
    st.dims[0] = static_cast<int>(d.nx); st.dims[1] = static_cast<int>(d.ny); st.dims[2] = static_cast<int>(d.nz);
    st.voxel = p;

    // 1. surface -> voxels: every triangle sampled with a spacing of at most half a voxel
    std::vector<std::uint8_t> vol(d.size(), 0);
    mem.add(vol.size());
    parallel_for(source.face_count(), threads, 4096, [&](std::size_t f0, std::size_t f1) {
        for (std::size_t f = f0; f < f1; ++f) {
            double c[3][3];
            for (int k = 0; k < 3; ++k)
                for (int a = 0; a < 3; ++a) c[k][a] = (src[source.indices[f * 3 + k] * 3 + a] - lo[a]) / p + margin;
            double longest = 0;
            for (int k = 0; k < 3; ++k) {
                const double* a = c[k];
                const double* b = c[(k + 1) % 3];
                longest = std::max(longest, std::sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2])));
            }
            const int n = std::max(1, static_cast<int>(std::ceil(longest / 0.5)));
            for (int i = 0; i <= n; ++i)
                for (int j = 0; i + j <= n; ++j) {
                    const double u = static_cast<double>(i) / n, w = static_cast<double>(j) / n;
                    std::size_t q[3];
                    for (int a = 0; a < 3; ++a) {
                        const double x = c[0][a] + (c[1][a] - c[0][a]) * u + (c[2][a] - c[0][a]) * w;
                        q[a] = static_cast<std::size_t>(std::max(0.0, std::floor(x)));
                    }
                    q[0] = std::min(q[0], d.nx - 1); q[1] = std::min(q[1], d.ny - 1); q[2] = std::min(q[2], d.nz - 1);
                    std::atomic_ref<std::uint8_t>(vol[d.at(q[0], q[1], q[2])]).store(1, std::memory_order_relaxed);
                }
        }
    });
    step(1);
    if (cancelled()) return fail("cancelled");

    // 2. dilation
    std::vector<std::uint8_t> tmp(d.size());
    mem.add(tmp.size());
    for (int k = 0; k < R; ++k) { morph_step(d, vol.data(), tmp.data(), true, threads); vol.swap(tmp); }
    step(2);
    // 3. per Z slice hole filling (open hems), open rings patched (tmp holds the raw slices)
    if (o.fill_slices) {
        const SliceStats ss = fill_slices(d, vol.data(), tmp.data(), o.patch_open_slices, threads);
        st.open_slices = ss.open;
        st.patched_slices = ss.patched;
    }
    step(3);
    if (cancelled()) return fail("cancelled");
    // 4. 3-D hole filling
    fill_holes_3d(d, vol.data());
    step(4);
    // 5. erosion
    for (int k = 0; k < R; ++k) { morph_step(d, vol.data(), tmp.data(), false, threads); vol.swap(tmp); }
    std::vector<std::uint8_t>().swap(tmp);
    mem.sub(d.size());
    step(5);
    if (cancelled()) return fail("cancelled");
    // 6. largest component
    st.components_before = keep_largest_component(d, vol.data());
    if (st.components_before == 0) return fail("no solid left after closing");
    step(6);

    // 7. float volume padded by one voxel, Gaussian, zoom, marching cubes at 0.5
    const Dims fd{d.nx + 2, d.ny + 2, d.nz + 2};
    std::vector<float> field(fd.size(), 0.f);
    mem.add(field.size() * 4);
    parallel_for(d.nz, threads, 8, [&](std::size_t z0, std::size_t z1) {
        for (std::size_t z = z0; z < z1; ++z)
            for (std::size_t y = 0; y < d.ny; ++y) {
                const std::uint8_t* in = vol.data() + d.at(0, y, z);
                float* out = field.data() + fd.at(1, y + 1, z + 1);
                for (std::size_t x = 0; x < d.nx; ++x) out[x] = in[x] ? 1.f : 0.f;
            }
    });
    std::vector<std::uint8_t>().swap(vol);
    mem.sub(d.size());
    if (o.sigma > 0) gaussian(fd, field.data(), o.sigma, threads);
    if (cancelled()) return fail("cancelled");
    Dims zd = fd;
    double zscale[3] = {1, 1, 1};
    if (o.zoom > 0 && std::fabs(o.zoom - 1.f) > 1e-6f) {
        mem.add(static_cast<std::size_t>(fd.size() * 4.0 * o.zoom * o.zoom * o.zoom));
        std::vector<float> z = zoom_linear(fd, field.data(), o.zoom, &zd, zscale, threads);
        std::vector<float>().swap(field);
        mem.sub(fd.size() * 4);
        field.swap(z);
    }
    PrintMesh m;
    {
        // ScalarGrid is z fastest: grid axes (ix, iy, iz) = our (z, y, x).
        triposg::ScalarGrid sg;
        sg.values = field.data();
        sg.nx = zd.nz; sg.ny = zd.ny; sg.nz = zd.nx;
        sg.min[0] = sg.min[1] = sg.min[2] = 0.f;
        sg.max[0] = static_cast<float>(zd.nz - 1); sg.max[1] = static_cast<float>(zd.ny - 1); sg.max[2] = static_cast<float>(zd.nx - 1);
        triposg::MarchingCubesOptions mo;
        mo.inside_positive = true;   // 1 inside: the field is negated, so the iso level is -0.5
        mo.iso_level = -0.5f;
        triposg::Mesh mc = triposg::marching_cubes(sg, mo);
        std::vector<float>().swap(field);
        mem.sub(zd.size() * 4);
        m.positions.resize(mc.vertices.size() * 3);
        const double off = 1.0 + margin;   // padding + margin
        for (std::size_t v = 0; v < mc.vertices.size(); ++v) {
            const double gx = mc.vertices[v].z * zscale[0], gy = mc.vertices[v].y * zscale[1], gz = mc.vertices[v].x * zscale[2];
            // Voxel i covers [i - off, i - off + 1) voxels from lo: its centre is at + 0.5.
            m.positions[v * 3] = static_cast<float>((gx - off + 0.5) * p + lo[0]);
            m.positions[v * 3 + 1] = static_cast<float>((gy - off + 0.5) * p + lo[1]);
            m.positions[v * 3 + 2] = static_cast<float>((gz - off + 0.5) * p + lo[2]);
        }
        std::vector<triposg::MeshVertex>().swap(mc.vertices);
        // Swapping x and z mirrors the frame: reverse the winding.
        m.indices.resize(mc.faces.size() * 3);
        for (std::size_t f = 0; f < mc.faces.size(); ++f) {
            m.indices[f * 3] = mc.faces[f][0];
            m.indices[f * 3 + 1] = mc.faces[f][2];
            m.indices[f * 3 + 2] = mc.faces[f][1];
        }
    }
    mem.add(m.positions.size() * 4 + m.indices.size() * 4);
    st.pieces_after_mc = keep_largest_piece(&m);
    if (m.face_count() == 0) return fail("marching cubes found no surface");
    if (signed_volume(m) < 0)
        for (std::size_t f = 0; f < m.face_count(); ++f) std::swap(m.indices[f * 3 + 1], m.indices[f * 3 + 2]);
    step(7);
    if (cancelled()) return fail("cancelled");

    // 8. snap onto the original surface
    const double cell = std::max(o.snap, 1.0f) * p;
    SurfaceGrid surface(src, source.indices, lo, hi, cell);
    mem.add(surface.bytes());
    if (o.snap > 0) {
        std::atomic<std::size_t> moved{0};
        const double limit = o.snap * p;
        parallel_for(m.vertex_count(), threads, 2048, [&](std::size_t v0, std::size_t v1) {
            std::size_t local = 0;
            for (std::size_t v = v0; v < v1; ++v) {
                const Vec3 q{m.positions[v * 3], m.positions[v * 3 + 1], m.positions[v * 3 + 2]};
                Vec3 hit;
                std::uint32_t face;
                double w[3], dist;
                if (surface.nearest(q, 1, &hit, &face, w, &dist) && dist < limit) {
                    m.positions[v * 3] = static_cast<float>(hit.x);
                    m.positions[v * 3 + 1] = static_cast<float>(hit.y);
                    m.positions[v * 3 + 2] = static_cast<float>(hit.z);
                    ++local;
                }
            }
            moved += local;
        });
        st.snapped_fraction = static_cast<double>(moved.load()) / static_cast<double>(m.vertex_count());
    }
    step(8);
    if (cancelled()) return fail("cancelled");
    // 9. Taubin
    taubin(&m, std::max(0, o.taubin), threads);
    step(9);
    // 10. colours from the original surface
    if (!source.colors.empty()) {
        m.colors.resize(m.vertex_count() * 4);
        parallel_for(m.vertex_count(), threads, 2048, [&](std::size_t v0, std::size_t v1) {
            for (std::size_t v = v0; v < v1; ++v) {
                const Vec3 q{m.positions[v * 3], m.positions[v * 3 + 1], m.positions[v * 3 + 2]};
                Vec3 hit;
                std::uint32_t face = 0;
                double w[3] = {1, 0, 0}, dist;
                std::uint8_t* out = m.colors.data() + v * 4;
                if (!surface.nearest(q, -1, &hit, &face, w, &dist)) { out[0] = out[1] = out[2] = 128; out[3] = 255; continue; }
                for (int c = 0; c < 4; ++c) {
                    double s = 0;
                    for (int k = 0; k < 3; ++k) s += w[k] * source.colors[source.indices[face * 3 + k] * 4 + c];
                    out[c] = static_cast<std::uint8_t>(std::clamp(std::lround(s), 0l, 255l));
                }
            }
        });
    }
    st.volume_model = signed_volume(m);
    step(10);

    // Feet at z = 0, centred in X / Y, millimetres.
    double mlo[3], mhi[3];
    bounds(m.positions, mlo, mhi);
    const double h = mhi[2] - mlo[2];
    double target = 0;
    if (o.height_mm > 0) target = o.height_mm;
    else if (o.eye_height_mm > 0 && o.eye_ratio > 0) target = o.eye_height_mm / o.eye_ratio;
    const double s = target > 0 && h > 0 ? target / h : 1.0;
    const double cx = (mlo[0] + mhi[0]) / 2, cy = (mlo[1] + mhi[1]) / 2;
    for (std::size_t v = 0; v < m.vertex_count(); ++v) {
        m.positions[v * 3] = static_cast<float>((m.positions[v * 3] - cx) * s);
        m.positions[v * 3 + 1] = static_cast<float>((m.positions[v * 3 + 1] - cy) * s);
        m.positions[v * 3 + 2] = static_cast<float>((m.positions[v * 3 + 2] - mlo[2]) * s);
    }
    st.scale = s;
    for (int a = 0; a < 3; ++a) st.size[a] = (mhi[a] - mlo[a]) * s;
    st.faces = m.face_count();
    st.vertices = m.vertex_count();
    st.check = check_mesh(m);
    st.open_edges = count_open_edges(m);
    st.bodies = st.check.bodies;
    st.volume = st.check.volume;
    st.peak_bytes = mem.peak;
    st.seconds = std::chrono::duration<double>(Clock::now() - t_start).count();
    result->mesh = std::move(m);
    return true;
}

std::vector<std::uint8_t> write_stl(const PrintMesh& mesh, const std::string& header) {
    const std::size_t nf = mesh.face_count();
    std::vector<std::uint8_t> out(84 + nf * 50, 0);
    std::memcpy(out.data(), header.data(), std::min<std::size_t>(header.size(), 80));
    const std::uint32_t count = static_cast<std::uint32_t>(nf);
    std::memcpy(out.data() + 80, &count, 4);
    std::uint8_t* w = out.data() + 84;
    for (std::size_t f = 0; f < nf; ++f, w += 50) {
        float v[3][3];
        for (int k = 0; k < 3; ++k)
            for (int a = 0; a < 3; ++a) v[k][a] = mesh.positions[mesh.indices[f * 3 + k] * 3 + a];
        const double ux = v[1][0] - v[0][0], uy = v[1][1] - v[0][1], uz = v[1][2] - v[0][2];
        const double wx = v[2][0] - v[0][0], wy = v[2][1] - v[0][1], wz = v[2][2] - v[0][2];
        double n[3] = {uy * wz - uz * wy, uz * wx - ux * wz, ux * wy - uy * wx};
        const double len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        float nf3[3] = {0, 0, 0};
        if (len > 0)
            for (int a = 0; a < 3; ++a) nf3[a] = static_cast<float>(n[a] / len);
        std::memcpy(w, nf3, 12);
        std::memcpy(w + 12, v, 36);
    }
    return out;
}

std::vector<std::uint8_t> write_print_glb(const PrintMesh& mesh, double metres_per_unit, const std::string& generator) {
    const std::size_t nv = mesh.vertex_count(), nf = mesh.face_count();
    const bool colours = mesh.colors.size() == nv * 4 && nv > 0;
    // glTF: Y up -> (x, z, -y) * metres_per_unit.
    std::vector<float> pos(nv * 3), nrm(nv * 3, 0.f);
    float mn[3] = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
    float mx[3] = {-mn[0], -mn[1], -mn[2]};
    for (std::size_t v = 0; v < nv; ++v) {
        const float s = static_cast<float>(metres_per_unit);
        pos[v * 3] = mesh.positions[v * 3] * s;
        pos[v * 3 + 1] = mesh.positions[v * 3 + 2] * s;
        pos[v * 3 + 2] = -mesh.positions[v * 3 + 1] * s;
        for (int a = 0; a < 3; ++a) { mn[a] = std::min(mn[a], pos[v * 3 + a]); mx[a] = std::max(mx[a], pos[v * 3 + a]); }
    }
    for (std::size_t f = 0; f < nf; ++f) {   // area-weighted vertex normals
        const std::uint32_t i[3] = {mesh.indices[f * 3], mesh.indices[f * 3 + 1], mesh.indices[f * 3 + 2]};
        const float* a = &pos[i[0] * 3];
        const float* b = &pos[i[1] * 3];
        const float* c = &pos[i[2] * 3];
        const float u[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]}, w[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
        const float n[3] = {u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]};
        for (int k = 0; k < 3; ++k)
            for (int e = 0; e < 3; ++e) nrm[i[k] * 3 + e] += n[e];
    }
    for (std::size_t v = 0; v < nv; ++v) {
        float* n = &nrm[v * 3];
        const float len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (len > 0) { n[0] /= len; n[1] /= len; n[2] /= len; } else { n[1] = 1.f; }
    }
    std::vector<std::uint8_t> bin;
    auto append = [&](const void* data, std::size_t bytes) {
        const std::size_t off = bin.size();
        bin.resize(off + bytes);
        std::memcpy(bin.data() + off, data, bytes);
        while (bin.size() % 4) bin.push_back(0);
        return off;
    };
    const std::size_t off_pos = append(pos.data(), pos.size() * 4);
    const std::size_t off_nrm = append(nrm.data(), nrm.size() * 4);
    const std::size_t off_col = colours ? append(mesh.colors.data(), mesh.colors.size()) : 0;
    const std::size_t off_idx = append(mesh.indices.data(), mesh.indices.size() * 4);
    json::Value root = json::Value::object();
    json::Value asset = json::Value::object();
    asset["version"] = "2.0";
    asset["generator"] = generator;
    root["asset"] = asset;
    root["scene"] = 0;
    json::Value scene = json::Value::object(), nodes = json::Value::array();
    nodes.push_back(0);
    scene["nodes"] = nodes;
    json::Value scenes = json::Value::array();
    scenes.push_back(scene);
    root["scenes"] = scenes;
    json::Value node = json::Value::object();
    node["mesh"] = 0;
    node["name"] = "druckfertig";
    json::Value node_list = json::Value::array();
    node_list.push_back(node);
    root["nodes"] = node_list;
    json::Value views = json::Value::array(), accessors = json::Value::array();
    auto view = [&](std::size_t offset, std::size_t length, int target) {
        json::Value v = json::Value::object();
        v["buffer"] = 0;
        v["byteOffset"] = static_cast<unsigned long long>(offset);
        v["byteLength"] = static_cast<unsigned long long>(length);
        v["target"] = target;
        views.push_back(v);
        return static_cast<int>(views.size() - 1);
    };
    auto accessor = [&](int bv, int component, std::size_t count, const char* type, bool normalized, bool with_bounds = false) {
        json::Value a = json::Value::object();
        a["bufferView"] = bv;
        a["componentType"] = component;
        a["count"] = static_cast<unsigned long long>(count);
        a["type"] = type;
        if (normalized) a["normalized"] = true;
        if (with_bounds) {   // POSITION needs min / max
            json::Value amin = json::Value::array(), amax = json::Value::array();
            for (int k = 0; k < 3; ++k) { amin.push_back(static_cast<double>(nv ? mn[k] : 0.f)); amax.push_back(static_cast<double>(nv ? mx[k] : 0.f)); }
            a["min"] = amin;
            a["max"] = amax;
        }
        accessors.push_back(a);
        return static_cast<int>(accessors.size() - 1);
    };
    const int a_pos = accessor(view(off_pos, pos.size() * 4, 34962), 5126, nv, "VEC3", false, true);
    const int a_nrm = accessor(view(off_nrm, nrm.size() * 4, 34962), 5126, nv, "VEC3", false);
    const int a_col = colours ? accessor(view(off_col, mesh.colors.size(), 34962), 5121, nv, "VEC4", true) : -1;
    const int a_idx = accessor(view(off_idx, mesh.indices.size() * 4, 34963), 5125, mesh.indices.size(), "SCALAR", false);
    root["bufferViews"] = views;
    root["accessors"] = accessors;
    json::Value buffer = json::Value::object();
    buffer["byteLength"] = static_cast<unsigned long long>(bin.size());
    json::Value buffers = json::Value::array();
    buffers.push_back(buffer);
    root["buffers"] = buffers;
    json::Value material = json::Value::object(), pbr = json::Value::object();
    pbr["metallicFactor"] = 0.0;
    pbr["roughnessFactor"] = 0.9;
    material["pbrMetallicRoughness"] = pbr;
    // Snapping leaves ~0.2 % tiny folds; single-sided they show as see-through specks (reparatur.py, 23.11.).
    material["doubleSided"] = true;
    material["name"] = "druckfertig";
    json::Value materials = json::Value::array();
    materials.push_back(material);
    root["materials"] = materials;
    json::Value attributes = json::Value::object();
    attributes["POSITION"] = a_pos;
    attributes["NORMAL"] = a_nrm;
    if (a_col >= 0) attributes["COLOR_0"] = a_col;
    json::Value prim = json::Value::object();
    prim["attributes"] = attributes;
    prim["indices"] = a_idx;
    prim["material"] = 0;
    prim["mode"] = 4;
    json::Value prims = json::Value::array();
    prims.push_back(prim);
    json::Value gmesh = json::Value::object();
    gmesh["primitives"] = prims;
    gmesh["name"] = "druckfertig";
    json::Value meshes = json::Value::array();
    meshes.push_back(gmesh);
    root["meshes"] = meshes;
    std::string text = root.dump();
    while (text.size() % 4) text.push_back(' ');
    std::vector<std::uint8_t> out;
    const std::uint32_t total = static_cast<std::uint32_t>(12 + 8 + text.size() + 8 + bin.size());
    auto u32 = [&](std::uint32_t v) {
        for (int k = 0; k < 4; ++k) out.push_back(static_cast<std::uint8_t>(v >> (8 * k)));
    };
    out.reserve(total);
    u32(0x46546C67u);   // "glTF"
    u32(2);
    u32(total);
    u32(static_cast<std::uint32_t>(text.size()));
    u32(0x4E4F534Au);   // "JSON"
    out.insert(out.end(), text.begin(), text.end());
    u32(static_cast<std::uint32_t>(bin.size()));
    u32(0x004E4942u);   // "BIN\0"
    out.insert(out.end(), bin.begin(), bin.end());
    return out;
}

bool read_glb_mesh(std::span<const std::uint8_t> bytes, PrintMesh* mesh, std::string* error) {
    auto fail = [&](const std::string& m) {
        if (error) *error = m;
        return false;
    };
    auto u32 = [&](std::size_t off) {
        std::uint32_t v;
        std::memcpy(&v, bytes.data() + off, 4);
        return v;
    };
    if (bytes.size() < 20 || u32(0) != 0x46546C67u) return fail("not a GLB");
    const std::uint32_t json_len = u32(12);
    if (u32(16) != 0x4E4F534Au || 20ull + json_len > bytes.size()) return fail("GLB without JSON chunk");
    json::Value root;
    try {
        root = json::parse(std::string_view(reinterpret_cast<const char*>(bytes.data() + 20), json_len));
    } catch (const std::exception& e) {
        return fail(std::string("GLB JSON: ") + e.what());
    }
    std::span<const std::uint8_t> bin;
    const std::size_t bin_head = 20ull + json_len;
    if (bin_head + 8 <= bytes.size() && u32(bin_head + 4) == 0x004E4942u) {
        const std::size_t len = u32(bin_head);
        if (bin_head + 8 + len > bytes.size()) return fail("GLB BIN chunk truncated");
        bin = bytes.subspan(bin_head + 8, len);
    }
    const json::Value* meshes = root.find("meshes");
    const json::Value* accessors = root.find("accessors");
    const json::Value* views = root.find("bufferViews");
    if (!meshes || !meshes->is_array() || meshes->empty() || !accessors || !views) return fail("GLB without a mesh");
    // Accessor -> (pointer, stride, count, component type, components, normalized)
    struct Acc {
        const std::uint8_t* data = nullptr;
        std::size_t stride = 0, count = 0;
        int component = 0, width = 0;
    };
    auto get = [&](double index, Acc* out) -> bool {
        const std::size_t ai = static_cast<std::size_t>(index);
        if (ai >= accessors->items().size()) return false;
        const json::Value& a = accessors->items()[ai];
        const std::size_t vi = a.member_size("bufferView", SIZE_MAX);
        if (vi >= views->items().size()) return false;
        const json::Value& v = views->items()[vi];
        if (v.member_size("buffer", 0) != 0) return false;
        const std::string type = a.member_str("type");
        out->width = type == "SCALAR" ? 1 : type == "VEC2" ? 2 : type == "VEC3" ? 3 : type == "VEC4" ? 4 : 0;
        out->component = static_cast<int>(a.member_double("componentType", 0));
        out->count = a.member_size("count", 0);
        const int csize = out->component == 5126 || out->component == 5125 ? 4 : out->component == 5123 || out->component == 5122 ? 2 : 1;
        out->stride = v.member_size("byteStride", 0);
        if (out->stride == 0) out->stride = static_cast<std::size_t>(csize * out->width);
        const std::size_t start = v.member_size("byteOffset", 0) + a.member_size("byteOffset", 0);
        const std::size_t need = out->count == 0 ? 0 : (out->count - 1) * out->stride + static_cast<std::size_t>(csize * out->width);
        if (out->width == 0 || start + need > bin.size() || start + need > v.member_size("byteOffset", 0) + v.member_size("byteLength", 0)) return false;
        out->data = bin.data() + start;
        return true;
    };
    auto component = [](const Acc& a, std::size_t i, int k) -> double {
        const std::uint8_t* p = a.data + i * a.stride;
        switch (a.component) {
            case 5126: { float f; std::memcpy(&f, p + k * 4, 4); return f; }
            case 5125: { std::uint32_t u; std::memcpy(&u, p + k * 4, 4); return u; }
            case 5123: { std::uint16_t u; std::memcpy(&u, p + k * 2, 2); return u; }
            case 5121: return p[k];
            default: return 0;
        }
    };
    *mesh = PrintMesh{};
    bool any_colour = false;
    const json::Value* prims_v = meshes->items()[0].find("primitives");
    if (!prims_v || !prims_v->is_array()) return fail("GLB mesh without primitives");
    for (const json::Value& prim : prims_v->items()) {
        if (prim.member_double("mode", 4) != 4) continue;
        const json::Value* attrs = prim.find("attributes");
        if (!attrs || !attrs->find("POSITION")) continue;
        Acc pa, ia, ca;
        if (!get(attrs->member_double("POSITION", -1), &pa) || pa.component != 5126 || pa.width != 3) return fail("unsupported POSITION accessor");
        const bool has_col = attrs->find("COLOR_0") && get(attrs->member_double("COLOR_0", -1), &ca) && ca.width >= 3;
        const std::size_t base = mesh->vertex_count();
        for (std::size_t i = 0; i < pa.count; ++i)
            for (int k = 0; k < 3; ++k) mesh->positions.push_back(static_cast<float>(component(pa, i, k)));
        if (has_col && !any_colour) { mesh->colors.assign(base * 4, 255); any_colour = true; }
        if (any_colour) {
            for (std::size_t i = 0; i < pa.count; ++i)
                for (int k = 0; k < 4; ++k) {
                    double c = 1.0;
                    if (has_col && k < ca.width) {
                        c = component(ca, i, k);
                        if (ca.component == 5121) c /= 255.0;
                        else if (ca.component == 5123) c /= 65535.0;
                    }
                    mesh->colors.push_back(static_cast<std::uint8_t>(std::clamp(std::lround(c * 255.0), 0l, 255l)));
                }
        }
        if (prim.find("indices")) {
            if (!get(prim.member_double("indices", -1), &ia) || ia.width != 1) return fail("unsupported index accessor");
            for (std::size_t i = 0; i < ia.count; ++i) {
                const std::uint32_t v = static_cast<std::uint32_t>(component(ia, i, 0));
                if (v >= pa.count) return fail("index out of range");
                mesh->indices.push_back(static_cast<std::uint32_t>(base) + v);
            }
        } else {
            for (std::size_t i = 0; i + 2 < pa.count; i += 3)
                for (int k = 0; k < 3; ++k) mesh->indices.push_back(static_cast<std::uint32_t>(base + i + k));
        }
    }
    if (mesh->face_count() == 0) return fail("GLB has no triangles");
    return true;
}

}  // namespace gufo::mesh
