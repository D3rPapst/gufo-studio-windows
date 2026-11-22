// TRELLIS.2 cascade ("1024_cascade" / "1536_cascade", Trellis2ImageTo3DPipeline.
// sample_shape_slat_cascade): the 512 shape latent is decoded by the shape decoder's four
// subdivision levels (shape_slat_decoder.upsample(slat, 4) = the voxel list of the last
// level on the 512 grid); those voxels are quantised to the token grid of the high
// resolution, hr / 16 per axis:
//     q = int((c + 0.5) / 512 * (hr / 16))      (exact here: = floor((2c + 1) * g / 1024))
// and deduplicated (torch.unique(dim=0) -> lexicographic x, y, z). While the token count
// is >= max_tokens (49152) and hr > 1024, hr drops by 128 (1536 -> 1408 -> 1280 -> 1152 ->
// 1024); at 1024 the list is taken whatever its size. The shape / texture flows 1024 then
// run on these tokens and the decoders produce the mesh on the hr grid.
// AI-assisted (Claude Code), 2026-11-22.
#pragma once

#include <algorithm>
#include <tuple>
#include <vector>

#include "src/models/trellis2/sparse_conv.hpp"

namespace gufo::trellis2 {

constexpr int kCascadeLowResolution = 512;
constexpr int kCascadeMaxTokens = 49152;

// Pipeline types of the original: 512 (no cascade), 1024_cascade, 1536_cascade.
inline bool is_supported_resolution(int r) { return r == 512 || r == 1024 || r == 1536; }

// Quantised, sorted, unique token coordinates for the high resolution `hr` (multiple of 16).
inline std::vector<Coord> cascade_quantise(const std::vector<Coord>& fine, int lr_resolution, int hr) {
    const long long g = hr / 16, den = 2LL * lr_resolution;
    std::vector<Coord> out;
    out.reserve(fine.size() / 4 + 1);
    for (const Coord& c : fine)
        out.push_back({static_cast<int>((2LL * c.x + 1) * g / den), static_cast<int>((2LL * c.y + 1) * g / den),
                       static_cast<int>((2LL * c.z + 1) * g / den)});
    std::sort(out.begin(), out.end(), [](const Coord& a, const Coord& b) { return std::tie(a.x, a.y, a.z) < std::tie(b.x, b.y, b.z); });
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

// The token list and the resolution actually used (*resolution) for target 1024 or 1536.
inline std::vector<Coord> cascade_coords(const std::vector<Coord>& fine, int target, int max_tokens, int* resolution,
                                         int lr_resolution = kCascadeLowResolution) {
    int hr = target;
    std::vector<Coord> coords;
    for (;;) {
        coords = cascade_quantise(fine, lr_resolution, hr);
        if (static_cast<long long>(coords.size()) < max_tokens || hr == 1024) break;
        hr -= 128;
    }
    if (resolution) *resolution = hr;
    return coords;
}

}  // namespace gufo::trellis2
