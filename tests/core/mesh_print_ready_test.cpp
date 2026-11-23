// Print-ready meshes (src/core/mesh/print_ready.cpp) on a small synthetic open mesh: a tube open at
// the bottom (robe hem) with a vertical slit, a cap on top and a thin finger sticking out. The result
// must be one watertight body, solid inside (slit and hem closed), keep the finger and its colour,
// and be scaled exactly. CPU only, no model.
// AI-assisted (Claude Code), 2026-11-23.
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numbers>
#include <span>
#include <string>
#include <vector>

#include "src/core/mesh/print_ready.hpp"

using gufo::mesh::PrintMesh;

namespace {

constexpr double kRadius = 0.2, kHeight = 0.8, kFingerLength = 0.2, kFingerThick = 0.015, kFingerY = 0.6;
constexpr int kSegments = 96, kRings = 40;

std::uint32_t add_vertex(PrintMesh* m, double x, double y, double z, std::uint8_t r, std::uint8_t g, std::uint8_t b) {
    m->positions.insert(m->positions.end(), {static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)});
    m->colors.insert(m->colors.end(), {r, g, b, 255});
    return static_cast<std::uint32_t>(m->vertex_count() - 1);
}

void add_face(PrintMesh* m, std::uint32_t a, std::uint32_t b, std::uint32_t c) { m->indices.insert(m->indices.end(), {a, b, c}); }

// glTF frame (Y up). Red tube without bottom, one segment column missing for y in [0.2, 0.5]
// (the slit, ~2.6 voxels wide at grid 160), a cap on top, a blue closed box as finger along +x.
PrintMesh make_figure() {
    PrintMesh m;
    std::vector<std::uint32_t> ring((kRings + 1) * kSegments);
    for (int j = 0; j <= kRings; ++j)
        for (int i = 0; i < kSegments; ++i) {
            const double a = 2 * std::numbers::pi * i / kSegments;
            ring[j * kSegments + i] = add_vertex(&m, kRadius * std::cos(a), kHeight * j / kRings, kRadius * std::sin(a), 200, 30, 30);
        }
    const int slit = kSegments / 4;   // column facing +z
    for (int j = 0; j < kRings; ++j) {
        const double y = kHeight * (j + 0.5) / kRings;
        for (int i = 0; i < kSegments; ++i) {
            if (i == slit && y > 0.2 && y < 0.5) continue;
            const int n = (i + 1) % kSegments;
            const std::uint32_t a = ring[j * kSegments + i], b = ring[j * kSegments + n], c = ring[(j + 1) * kSegments + n], d = ring[(j + 1) * kSegments + i];
            add_face(&m, a, c, b);
            add_face(&m, a, d, c);
        }
    }
    const std::uint32_t top = add_vertex(&m, 0, kHeight, 0, 200, 30, 30);
    for (int i = 0; i < kSegments; ++i) add_face(&m, ring[kRings * kSegments + i], top, ring[kRings * kSegments + (i + 1) % kSegments]);
    // Finger: box x in [r - 0.02, r + L], y around kFingerY, z around 0.
    const double x0 = kRadius - 0.02, x1 = kRadius + kFingerLength, h = kFingerThick / 2;
    std::uint32_t c[8];
    for (int k = 0; k < 8; ++k)
        c[k] = add_vertex(&m, (k & 1) ? x1 : x0, kFingerY + ((k & 2) ? h : -h), (k & 4) ? h : -h, 20, 40, 220);
    const int quads[6][4] = {{0, 2, 3, 1}, {4, 5, 7, 6}, {0, 1, 5, 4}, {2, 6, 7, 3}, {0, 4, 6, 2}, {1, 3, 7, 5}};
    for (const auto& q : quads) {
        add_face(&m, c[q[0]], c[q[1]], c[q[2]]);
        add_face(&m, c[q[0]], c[q[2]], c[q[3]]);
    }
    return m;
}

// Robe with an open lapel: a tube open at the bottom whose wall has a gap of `gap_columns` segment
// columns (~2.6 voxels each at grid 160) between y0 and y1 - wider than the 2-voxel closing, so the
// per-slice fill fails there (the v1 crack across the chest, 23.11.).
PrintMesh make_robe(int gap_columns, double y0, double y1) {
    PrintMesh m;
    std::vector<std::uint32_t> ring((kRings + 1) * kSegments);
    for (int j = 0; j <= kRings; ++j)
        for (int i = 0; i < kSegments; ++i) {
            const double a = 2 * std::numbers::pi * i / kSegments;
            ring[j * kSegments + i] = add_vertex(&m, kRadius * std::cos(a), kHeight * j / kRings, kRadius * std::sin(a), 150, 120, 90);
        }
    const int first = kSegments / 4 - gap_columns / 2;   // gap facing +z (output -y)
    for (int j = 0; j < kRings; ++j) {
        const double y = kHeight * (j + 0.5) / kRings;
        for (int i = 0; i < kSegments; ++i) {
            if (i >= first && i < first + gap_columns && y > y0 && y < y1) continue;
            const int n = (i + 1) % kSegments;
            add_face(&m, ring[j * kSegments + i], ring[(j + 1) * kSegments + n], ring[j * kSegments + n]);
            add_face(&m, ring[j * kSegments + i], ring[(j + 1) * kSegments + i], ring[(j + 1) * kSegments + n]);
        }
    }
    const std::uint32_t top = add_vertex(&m, 0, kHeight, 0, 150, 120, 90);
    for (int i = 0; i < kSegments; ++i) add_face(&m, ring[kRings * kSegments + i], top, ring[kRings * kSegments + (i + 1) % kSegments]);
    return m;
}

// Ray parity along a slightly tilted +x direction.
bool inside(const PrintMesh& m, double px, double py, double pz) {
    const double d[3] = {1.0, 1.3e-3, 2.1e-3};
    int hits = 0;
    for (std::size_t f = 0; f < m.face_count(); ++f) {
        const float* a = &m.positions[m.indices[f * 3] * 3];
        const float* b = &m.positions[m.indices[f * 3 + 1] * 3];
        const float* c = &m.positions[m.indices[f * 3 + 2] * 3];
        const double e1[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]}, e2[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
        const double h[3] = {d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2], d[0] * e2[1] - d[1] * e2[0]};
        const double det = e1[0] * h[0] + e1[1] * h[1] + e1[2] * h[2];
        if (std::fabs(det) < 1e-18) continue;
        const double s[3] = {px - a[0], py - a[1], pz - a[2]};
        const double u = (s[0] * h[0] + s[1] * h[1] + s[2] * h[2]) / det;
        if (u < 0 || u > 1) continue;
        const double q[3] = {s[1] * e1[2] - s[2] * e1[1], s[2] * e1[0] - s[0] * e1[2], s[0] * e1[1] - s[1] * e1[0]};
        const double v = (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) / det;
        if (v < 0 || u + v > 1) continue;
        const double t = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) / det;
        if (t > 0) ++hits;
    }
    return hits % 2 == 1;
}

// Ray through the front of a solid along +y (output frame): first surface hit and whether every
// sample point between the entry and the far side lies inside (no gap, no crack).
bool solid_along_y(const PrintMesh& m, double x, double z, double y_from, double y_to, int samples) {
    for (int k = 0; k <= samples; ++k) {
        const double y = y_from + (y_to - y_from) * k / samples;
        if (!inside(m, x, y, z)) return false;
    }
    return true;
}

void bbox(const PrintMesh& m, double lo[3], double hi[3]) {
    for (int a = 0; a < 3; ++a) { lo[a] = 1e30; hi[a] = -1e30; }
    for (std::size_t v = 0; v < m.vertex_count(); ++v)
        for (int a = 0; a < 3; ++a) { lo[a] = std::min(lo[a], static_cast<double>(m.positions[v * 3 + a])); hi[a] = std::max(hi[a], static_cast<double>(m.positions[v * 3 + a])); }
}

#define CHECK(cond)                                                                 \
    do {                                                                            \
        if (!(cond)) {                                                              \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
            return 1;                                                               \
        }                                                                           \
    } while (0)

}  // namespace

int main() {
    const PrintMesh src = make_figure();
    CHECK(gufo::mesh::count_open_edges(src) > 0);   // the input is open (hem, slit)

    // 1. unscaled run: topology, solid interior, finger, colours
    gufo::mesh::PrintOptions o;
    o.grid = 160;
    gufo::mesh::PrintResult r;
    std::string error;
    int last_step = 0;
    o.progress = [&](int done, int total) { assert(total == 10 && done >= last_step); last_step = done; };
    CHECK(gufo::mesh::make_print_ready(src, o, &r, &error));
    const auto& st = r.stats;
    std::printf("grid %dx%dx%d voxel %.5f, pieces before %zu, after MC %zu, snapped %.1f %%, faces %zu, open edges %zu, bodies %zu, volume %.5f, %.2f s, peak %.1f MB\n",
                st.dims[0], st.dims[1], st.dims[2], st.voxel, st.components_before, st.pieces_after_mc, st.snapped_fraction * 100, st.faces,
                st.open_edges, st.bodies, st.volume, st.seconds, st.peak_bytes / 1048576.0);
    CHECK(last_step == 10);
    CHECK(st.open_edges == 0);                       // watertight
    CHECK(st.bodies == 1);                           // one body
    CHECK(gufo::mesh::count_open_edges(r.mesh) == 0);
    CHECK(st.volume > 0);
    CHECK(std::fabs(st.scale - 1.0) < 1e-12);
    CHECK(st.snapped_fraction > 0.3);
    // output check report
    CHECK(st.check.watertight && st.check.boundary_edges == 0 && st.check.nonmanifold_edges == 0);
    CHECK(st.check.winding_consistent && st.check.volume > 0 && st.check.bodies == 1);
    std::printf("check: %s\n", gufo::mesh::check_line(st.check).c_str());
    CHECK(gufo::mesh::check_line(st.check) ==
          "watertight yes, boundary edges 0, non-manifold edges 0, winding consistent yes, volume > 0 yes, bodies 1");
    {   // the check finds an open, a flipped and a non-manifold mesh
        const auto open = gufo::mesh::check_mesh(src);
        CHECK(!open.watertight && open.boundary_edges > 0);
        PrintMesh flipped = r.mesh;
        std::swap(flipped.indices[1], flipped.indices[2]);
        const auto f = gufo::mesh::check_mesh(flipped);
        CHECK(f.watertight && !f.winding_consistent && f.inconsistent_edges == 3);
    }
    // Solid cylinder + finger; smoothing rounds the rim a little.
    const double solid = std::numbers::pi * kRadius * kRadius * kHeight + kFingerLength * kFingerThick * kFingerThick;
    std::printf("volume %.5f, solid cylinder + finger %.5f (%.2f %%)\n", st.volume, solid, 100.0 * (st.volume / solid - 1));
    CHECK(std::fabs(st.volume / solid - 1) < 0.05);
    double lo[3], hi[3];
    bbox(r.mesh, lo, hi);
    CHECK(std::fabs(lo[2]) < 1e-6);                  // feet at z = 0
    CHECK(std::fabs(hi[2] - kHeight) < 0.01);        // height kept (unscaled)
    CHECK(std::fabs(lo[1] + hi[1]) < 1e-5);          // centred in Y
    CHECK(std::fabs(lo[0] + hi[0]) < 1e-5);          // centred in X
    // Input x of an output point: output x = x_in - cx with the tube side at x_in = -r.
    const double cx = -kRadius - lo[0];
    // Output frame: (x_in - cx, -z_in, y_in).
    CHECK(inside(r.mesh, 0 - cx, 0, 0.35));          // tube centre at slit height: solid (slit closed)
    CHECK(inside(r.mesh, 0 - cx, -0.9 * kRadius, 0.35));   // in the slit's direction, just inside the wall
    CHECK(inside(r.mesh, 0 - cx, 0, 0.03));          // just above the open hem: solid
    CHECK(inside(r.mesh, kRadius + kFingerLength * 0.7 - cx, 0, kFingerY));   // finger survives
    CHECK(!inside(r.mesh, kRadius + kFingerLength * 0.7 - cx, 0, kFingerY + 0.03));
    CHECK(hi[0] - lo[0] > 2 * kRadius + 0.95 * kFingerLength);   // finger tip kept
    // Colour: the vertex furthest out along x is on the blue finger.
    std::size_t tip = 0;
    for (std::size_t v = 0; v < r.mesh.vertex_count(); ++v)
        if (r.mesh.positions[v * 3] > r.mesh.positions[tip * 3]) tip = v;
    CHECK(r.mesh.colors.size() == r.mesh.vertex_count() * 4);
    CHECK(r.mesh.colors[tip * 4 + 2] > 180 && r.mesh.colors[tip * 4] < 60);

    // 2. writers: binary STL and GLB round trip
    const auto stl = gufo::mesh::write_stl(r.mesh);
    CHECK(stl.size() == 84 + 50 * r.mesh.face_count());
    std::uint32_t stl_faces = 0;
    std::memcpy(&stl_faces, stl.data() + 80, 4);
    CHECK(stl_faces == r.mesh.face_count());
    const auto glb = gufo::mesh::write_print_glb(r.mesh, 1.0);
    PrintMesh back;
    CHECK(gufo::mesh::read_glb_mesh(std::span<const std::uint8_t>(glb.data(), glb.size()), &back, &error));
    CHECK(back.face_count() == r.mesh.face_count() && back.vertex_count() == r.mesh.vertex_count());
    CHECK(back.colors.size() == back.vertex_count() * 4);
    {   // material is double sided (folds left by the snapping must not show as see-through specks)
        std::uint32_t json_len = 0;
        std::memcpy(&json_len, glb.data() + 12, 4);
        const std::string json(reinterpret_cast<const char*>(glb.data() + 20), json_len);
        CHECK(json.find("\"doubleSided\":true") != std::string::npos);
    }
    CHECK(gufo::mesh::count_open_edges(back) == 0);
    // GLB is Y up: (x, z, -y).
    CHECK(std::fabs(back.positions[tip * 3 + 1] - r.mesh.positions[tip * 3 + 2]) < 1e-6);
    CHECK(std::fabs(back.positions[tip * 3 + 2] + r.mesh.positions[tip * 3 + 1]) < 1e-6);
    CHECK(std::fabs(gufo::mesh::signed_volume(back) - st.volume) < 1e-6 * std::max(1.0, st.volume) + 1e-9);

    // 3. scaling: total height and eye height (eyes at 93 %)
    gufo::mesh::PrintOptions oh = o;
    oh.progress = nullptr;
    oh.height_mm = 50;
    gufo::mesh::PrintResult rh;
    CHECK(gufo::mesh::make_print_ready(src, oh, &rh, &error));
    bbox(rh.mesh, lo, hi);
    std::printf("height 50 mm: size %.4f x %.4f x %.4f mm, scale %.4f\n", rh.stats.size[0], rh.stats.size[1], rh.stats.size[2], rh.stats.scale);
    CHECK(std::fabs(hi[2] - lo[2] - 50.0) < 1e-3);
    CHECK(std::fabs(lo[2]) < 1e-5);
    CHECK(std::fabs(rh.stats.size[2] - 50.0) < 1e-9);
    CHECK(std::fabs(rh.stats.volume / (st.volume * std::pow(rh.stats.scale, 3)) - 1) < 1e-4);
    gufo::mesh::PrintOptions oe = oh;
    oe.height_mm = 0;
    oe.eye_height_mm = 28;
    gufo::mesh::PrintResult re;
    CHECK(gufo::mesh::make_print_ready(src, oe, &re, &error));
    bbox(re.mesh, lo, hi);
    std::printf("eye height 28 mm: total %.4f mm (28 / 0.93 = %.4f)\n", hi[2] - lo[2], 28 / 0.93);
    CHECK(std::fabs(hi[2] - lo[2] - 28.0 / 0.93) < 1e-3);
    CHECK(re.stats.open_edges == 0 && re.stats.bodies == 1);

    // 4. open lapel: the ring is open over 0.40..0.48 by ~10 voxels (wider than the closing); the
    //    open slices are found and closed, a ray through the front finds no gap (v1 had a crack)
    {
        const PrintMesh robe = make_robe(4, 0.40, 0.48);
        gufo::mesh::PrintOptions orobe;
        orobe.grid = 160;
        gufo::mesh::PrintResult fixed, crack;
        CHECK(gufo::mesh::make_print_ready(robe, orobe, &fixed, &error));
        std::printf("open lapel: open slices %zu, patched %zu, %s\n", fixed.stats.open_slices, fixed.stats.patched_slices,
                    gufo::mesh::check_line(fixed.stats.check).c_str());
        CHECK(fixed.stats.open_slices >= 10 && fixed.stats.patched_slices == fixed.stats.open_slices);
        CHECK(fixed.stats.check.watertight && fixed.stats.check.bodies == 1 && fixed.stats.check.winding_consistent);
        double rlo[3], rhi[3];
        bbox(fixed.mesh, rlo, rhi);
        // output frame: (x, -z_in, y_in), centred; the gap faces output -y. Rays at the gap heights
        // and at the heights next to it, from just inside the front wall to just inside the back.
        bool all_solid = true;
        for (double zz : {0.41, 0.43, 0.44, 0.45, 0.47, 0.35, 0.55})
            for (double xx : {-0.05, 0.0, 0.05})
                all_solid = all_solid && solid_along_y(fixed.mesh, xx, zz, -0.85 * kRadius, 0.85 * kRadius, 40);
        CHECK(all_solid);
        const double solid = std::numbers::pi * kRadius * kRadius * kHeight;
        std::printf("open lapel: volume %.5f, solid cylinder %.5f (%.2f %%)\n", fixed.stats.volume, solid, 100.0 * (fixed.stats.volume / solid - 1));
        CHECK(std::fabs(fixed.stats.volume / solid - 1) < 0.05);
        // Without the patch (v1) the same robe has the crack: the ray through the front hits a gap.
        orobe.patch_open_slices = false;
        CHECK(gufo::mesh::make_print_ready(robe, orobe, &crack, &error));
        bool crack_found = false;
        for (double zz : {0.43, 0.44, 0.45}) crack_found = crack_found || !solid_along_y(crack.mesh, 0.0, zz, -0.85 * kRadius, 0.85 * kRadius, 40);
        std::printf("open lapel without patch: crack %s, volume %.5f\n", crack_found ? "found" : "not found", crack.stats.volume);
        CHECK(crack_found);
    }

    // 5. a compact object (closed cube) at grid 720 stays inside the voxel budget
    {
        PrintMesh cube;
        std::uint32_t c[8];
        for (int k = 0; k < 8; ++k) c[k] = add_vertex(&cube, (k & 1) ? 0.5 : -0.5, (k & 2) ? 1.0 : 0.0, (k & 4) ? 0.5 : -0.5, 90, 90, 90);
        const int quads[6][4] = {{0, 2, 3, 1}, {4, 5, 7, 6}, {0, 1, 5, 4}, {2, 6, 7, 3}, {0, 4, 6, 2}, {1, 3, 7, 5}};
        for (const auto& q : quads) {
            add_face(&cube, c[q[0]], c[q[1]], c[q[2]]);
            add_face(&cube, c[q[0]], c[q[2]], c[q[3]]);
        }
        CHECK(gufo::mesh::count_open_edges(cube) == 0);
        gufo::mesh::PrintOptions ob;
        ob.max_voxels = 2u << 20;
        ob.height_mm = 20;
        gufo::mesh::PrintResult rb;
        CHECK(gufo::mesh::make_print_ready(cube, ob, &rb, &error));
        const double voxels = static_cast<double>(rb.stats.dims[0]) * rb.stats.dims[1] * rb.stats.dims[2];
        std::printf("cube: grid %dx%dx%d (%.2f M voxels), volume %.1f mm^3 (20^3 = 8000), open edges %zu\n", rb.stats.dims[0], rb.stats.dims[1],
                    rb.stats.dims[2], voxels / 1048576.0, rb.stats.volume, rb.stats.open_edges);
        CHECK(voxels <= 2.0 * 1048576.0);
        CHECK(rb.stats.open_edges == 0 && rb.stats.bodies == 1);
        CHECK(std::fabs(rb.stats.size[2] - 20.0) < 1e-9);
        CHECK(rb.stats.volume > 0.9 * 8000 && rb.stats.volume < 1.02 * 8000);   // corners rounded by the smoothing
    }

    // 6. errors: empty mesh, cancel
    gufo::mesh::PrintResult rx;
    CHECK(!gufo::mesh::make_print_ready(PrintMesh{}, o, &rx, &error));
    std::atomic<bool> cancel{true};
    gufo::mesh::PrintOptions oc = oh;
    oc.cancel = &cancel;
    CHECK(!gufo::mesh::make_print_ready(src, oc, &rx, &error));
    CHECK(error == "cancelled");
    PrintMesh bad = src;
    bad.indices[0] = 1u << 30;
    CHECK(!gufo::mesh::make_print_ready(bad, o, &rx, &error));

    std::printf("mesh_print_ready_test: ok\n");
    return 0;
}
