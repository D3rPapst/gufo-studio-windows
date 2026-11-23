// Print-ready meshes ("druckfertig") for the 3D generators (TRELLIS.2, TripoSG): one watertight,
// smooth body in millimetres for miniature printing, made from an open generator mesh.
//
// Port of the proven Python reference D:\claude\gufo-arbeit\lars-pose\reparatur.py (production
// parameters 720 1.0 6 1.3 2.5) and stl_miniatur.py:
//   1. sample the surface densely into a voxel grid (long axis `grid` cells, voxel p)
//   2. dilate `close_radius` voxels (6-neighbourhood)
//   3. fill holes per Z slice (an open robe hem becomes solid); slices whose ring is open at one
//      spot (lapel) and whose filled area is < 0.6 x the 41-slice running median get a 2-D closing
//      with a growing disk (r = 2 .. 22) until the area is >= 0.8 x median (no crack through the body)
//   4. fill holes in 3D
//   5. erode `close_radius` voxels
//   6. keep the largest 6-connected component
//   7. Gaussian sigma (voxels), linear zoom, marching cubes at 0.5, keep the largest piece
//   8. snap every vertex within `snap` voxels onto the ORIGINAL surface (face / finger detail)
//   9. Taubin smoothing (`taubin` iterations, equal weights, lambda = mu = 0.5 as trimesh)
//  10. vertex colours from the original surface (closest point, barycentric)
// then Z up, feet at z = 0, centred in X / Y, scaled to millimetres: either a total height or an
// eye height with eyes at `eye_ratio` (default 93 %) of the body height.
// Deviations from the reference: the surface is rasterised per triangle (no random samples) and
// step 8 projects onto the exact nearest surface point (the limit of dense sampling).
// CPU only, no dependencies but the C++ standard library; parallel with std::thread.
// AI-assisted (Claude Code), 2026-11-23.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace gufo::mesh {

// Indexed triangle mesh: positions xyz, indices (counter-clockwise seen from outside), optional
// per-vertex colours RGBA8 (empty = none).
struct PrintMesh {
    std::vector<float> positions;          // [vertices * 3]
    std::vector<std::uint32_t> indices;    // [faces * 3]
    std::vector<std::uint8_t> colors;      // [vertices * 4] or empty
    std::size_t vertex_count() const { return positions.size() / 3; }
    std::size_t face_count() const { return indices.size() / 3; }
};

struct PrintOptions {
    // Input frame: glTF (Y up). The result (`PrintResult::mesh`) is Z up in millimetres.
    int grid = 720;              // cells along the longest axis
    // Upper bound of the voxel grid (memory: ~14 bytes per voxel at the peak with zoom 1.3);
    // compact objects get coarser voxels, a 1024 TRELLIS.2 figure (~41 M voxels) keeps 720.
    std::size_t max_voxels = 48u << 20;
    int close_radius = 2;        // dilation / erosion in voxels
    bool fill_slices = true;     // step 3
    bool patch_open_slices = true;   // step 3: close open rings (lapel) by growing 2-D closing
    float sigma = 1.0f;          // Gaussian in voxels (0 = off)
    float zoom = 1.3f;           // linear resampling before marching cubes
    float snap = 2.5f;           // voxels (0 = off)
    int taubin = 6;              // iterations
    // Scale: height_mm > 0 wins; else eye_height_mm > 0 with eye_ratio; else unscaled (model units).
    double height_mm = 0;
    double eye_height_mm = 0;
    double eye_ratio = 0.93;
    int threads = 0;             // 0 = hardware threads, at most 16
    const std::atomic<bool>* cancel = nullptr;
    std::function<void(int completed, int total)> progress;   // steps done of 10
};

// Output check of a print mesh (the "check line" of the job result).
struct MeshCheck {
    std::size_t boundary_edges = 0;      // edges used by one face
    std::size_t nonmanifold_edges = 0;   // edges used by more than two faces
    std::size_t inconsistent_edges = 0;  // two faces, same direction (winding flipped)
    bool watertight = false;             // no boundary and no non-manifold edges
    bool winding_consistent = false;
    double volume = 0;
    std::size_t bodies = 0;
};

struct PrintStats {
    int dims[3] = {0, 0, 0};     // voxel grid (Z up)
    double voxel = 0;            // voxel size in input units (larger than extent / grid when max_voxels bound)
    std::size_t open_slices = 0, patched_slices = 0;   // step 3: open rings found / closed
    std::size_t components_before = 0;   // voxel pieces before step 6
    std::size_t pieces_after_mc = 0;
    double snapped_fraction = 0;          // vertices moved onto the original surface
    std::size_t faces = 0, vertices = 0;
    std::size_t open_edges = 0;           // 0 = watertight (every edge twice, opposite directions)
    std::size_t bodies = 0;
    double volume = 0;                    // in output units (mm^3 when scaled)
    double volume_model = 0;              // in input units^3 (before scaling)
    double size[3] = {0, 0, 0};           // output extents x, y, z (mm when scaled)
    double scale = 1;                     // output units per input unit
    double seconds = 0;
    std::size_t peak_bytes = 0;           // largest tracked working set of this function (estimate)
    MeshCheck check;                      // output check of the result
};

struct PrintResult {
    PrintMesh mesh;   // Z up, feet at z = 0, centred in X / Y, mm (when scaled)
    PrintStats stats;
};

// Runs the whole pipeline. Returns false with *error on bad input, a cancel or no solid left;
// a result that is not watertight is returned with stats.open_edges > 0 (callers report it).
bool make_print_ready(const PrintMesh& source, const PrintOptions& options, PrintResult* result,
                      std::string* error);

// Binary STL of a Z-up mesh in its own units (mm for a print result); per-face normals.
std::vector<std::uint8_t> write_stl(const PrintMesh& mesh, const std::string& header = "gufo druckfertig");

// GLB with COLOR_0 (normalised UNSIGNED_BYTE) of a Z-up millimetre mesh, written in glTF
// convention: Y up and metres (x, z, -y) / 1000 so it imports at its real size; the material is
// doubleSided (tiny folds left by the snapping would otherwise show as see-through specks).
std::vector<std::uint8_t> write_print_glb(const PrintMesh& mesh, double metres_per_unit = 0.001,
                                          const std::string& generator = "gufo druckfertig");

// Reads all triangle primitives of the first mesh of a GLB: FLOAT positions, UNSIGNED_BYTE /
// SHORT / INT indices (or none), optional COLOR_0 (FLOAT or normalised UNSIGNED_BYTE / SHORT).
// Node transforms are ignored (the gufo writers emit none).
bool read_glb_mesh(std::span<const std::uint8_t> bytes, PrintMesh* mesh, std::string* error);

// Topology helpers (also used by the tests).
std::size_t count_open_edges(const PrintMesh& mesh);   // edges not used exactly once each way
std::size_t count_bodies(const PrintMesh& mesh);       // face-connected pieces
double signed_volume(const PrintMesh& mesh);
MeshCheck check_mesh(const PrintMesh& mesh);
// "watertight yes, boundary edges 0, non-manifold edges 0, winding consistent yes, volume > 0 yes, bodies 1"
std::string check_line(const MeshCheck& check);

}  // namespace gufo::mesh
