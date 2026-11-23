// gufo-mesh-print: the print-ready step (print_ready.hpp) on a GLB file, for measurements and
// comparisons with the Python reference (reparatur.py + stl_miniatur.py).
//   gufo-mesh-print in.glb out-prefix [--height-mm H | --eye-mm E [--eye-ratio 0.93]]
//                   [--grid 720] [--sigma 1] [--zoom 1.3] [--snap 2.5] [--taubin 6] [--threads N]
// Writes <out-prefix>.stl (binary, mm, Z up) and <out-prefix>.glb (Y up, metres, vertex colours)
// and prints one JSON line with the statistics, the wall time and the process memory.
// AI-assisted (Claude Code), 2026-11-23.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#endif

#include "src/core/mesh/print_ready.hpp"

namespace {

struct Memory {
    double private_mb = 0, peak_private_mb = 0, peak_working_set_mb = 0;
};

Memory memory() {
    Memory m;
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX c{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&c), sizeof(c))) {
        m.private_mb = c.PrivateUsage / 1048576.0;
        m.peak_private_mb = c.PeakPagefileUsage / 1048576.0;
        m.peak_working_set_mb = c.PeakWorkingSetSize / 1048576.0;
    }
#endif
    return m;
}

bool write_file(const std::string& path, const std::vector<std::uint8_t>& data) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    return static_cast<bool>(out);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: gufo-mesh-print in.glb out-prefix [--height-mm H | --eye-mm E [--eye-ratio R]] [--grid N] [--sigma S]\n"
                     "                       [--zoom Z] [--snap V] [--taubin K] [--threads T]\n");
        return 2;
    }
    gufo::mesh::PrintOptions o;
    for (int i = 3; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        const double v = std::atof(argv[i + 1]);
        if (k == "--height-mm") o.height_mm = v;
        else if (k == "--eye-mm") o.eye_height_mm = v;
        else if (k == "--eye-ratio") o.eye_ratio = v;
        else if (k == "--grid") o.grid = static_cast<int>(v);
        else if (k == "--sigma") o.sigma = static_cast<float>(v);
        else if (k == "--zoom") o.zoom = static_cast<float>(v);
        else if (k == "--snap") o.snap = static_cast<float>(v);
        else if (k == "--taubin") o.taubin = static_cast<int>(v);
        else if (k == "--threads") o.threads = static_cast<int>(v);
        else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); return 2; }
    }
    std::vector<std::uint8_t> bytes;
    {
        std::ifstream in(argv[1], std::ios::binary);
        if (!in) { std::fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
        bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    gufo::mesh::PrintMesh src;
    std::string error;
    if (!gufo::mesh::read_glb_mesh(bytes, &src, &error)) { std::fprintf(stderr, "%s\n", error.c_str()); return 1; }
    std::vector<std::uint8_t>().swap(bytes);
    const Memory before = memory();
    const auto t0 = std::chrono::steady_clock::now();
    o.progress = [&](int done, int total) {
        std::fprintf(stderr, "step %d/%d %.1f s\n", done, total, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    };
    gufo::mesh::PrintResult r;
    if (!gufo::mesh::make_print_ready(src, o, &r, &error)) { std::fprintf(stderr, "print step failed: %s\n", error.c_str()); return 1; }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const Memory after = memory();
    const std::string prefix = argv[2];
    const auto t1 = std::chrono::steady_clock::now();
    const bool ok = write_file(prefix + ".stl", gufo::mesh::write_stl(r.mesh)) &&
                    write_file(prefix + ".glb", gufo::mesh::write_print_glb(r.mesh, r.stats.scale != 1.0 ? 0.001 : 1.0));
    const double seconds_write = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    const auto& s = r.stats;
    std::printf("{\"source_vertices\":%zu,\"source_faces\":%zu,\"source_colours\":%s,\"grid\":[%d,%d,%d],\"voxel\":%.8f,"
                "\"components_before\":%zu,\"pieces_after_mc\":%zu,\"snapped\":%.4f,\"faces\":%zu,\"vertices\":%zu,\"open_edges\":%zu,"
                "\"bodies\":%zu,\"volume_model\":%.9f,\"volume\":%.6f,\"size\":[%.4f,%.4f,%.4f],\"scale\":%.6f,\"seconds\":%.2f,"
                "\"seconds_write\":%.2f,\"tracked_peak_mb\":%.1f,\"private_mb_before\":%.1f,\"peak_private_mb\":%.1f,"
                "\"peak_working_set_mb\":%.1f,\"written\":%s}\n",
                src.vertex_count(), src.face_count(), src.colors.empty() ? "false" : "true", s.dims[0], s.dims[1], s.dims[2], s.voxel,
                s.components_before, s.pieces_after_mc, s.snapped_fraction, s.faces, s.vertices, s.open_edges, s.bodies, s.volume_model,
                s.volume, s.size[0], s.size[1], s.size[2], s.scale, seconds, seconds_write, s.peak_bytes / 1048576.0, before.private_mb,
                after.peak_private_mb, after.peak_working_set_mb, ok ? "true" : "false");
    std::printf("check: %s; open slices %zu, patched %zu\n", gufo::mesh::check_line(s.check).c_str(), s.open_slices, s.patched_slices);
    return ok ? 0 : 1;
}
