// CPU tests for src/models/trellis2/cascade.hpp: token quantisation of the 1024 / 1536
// cascade against tools/trellis2/referenz/multiview_ref.py (numpy float32 of the original
// formula ((c + 0.5) / 512 * (hr // 16)).int() + unique), the token cap loop and the
// exactness of the integer form on the whole 512 grid.
// AI-assisted (Claude Code), 2026-11-22.
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "src/models/trellis2/cascade.hpp"

namespace tr = gufo::trellis2;

namespace {
int failures = 0;
void Expect(bool c, const std::string& what) {
    if (!c) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}
long long Sum(const std::vector<tr::Coord>& c) {
    long long s = 0;
    for (const auto& v : c) s += v.x + v.y + v.z;
    return s;
}
}  // namespace

int main() {
    std::vector<tr::Coord> fine;
    for (int i = 0; i < 6000; ++i) fine.push_back({(i * 37) % 512, (i * 101 + 7) % 512, (i * 13 + i / 7) % 512});

    int hr = 0;
    auto c = tr::cascade_coords(fine, 1024, tr::kCascadeMaxTokens, &hr);
    Expect(hr == 1024 && c.size() == 3584 && Sum(c) == 338688, "1024: tokens and coordinate sum = reference");
    Expect(c.front() == tr::Coord{0, 0, 0} && c.back() == tr::Coord{63, 63, 57}, "1024: sorted first / last = reference");
    c = tr::cascade_coords(fine, 1536, tr::kCascadeMaxTokens, &hr);
    Expect(hr == 1536 && c.size() == 3584 && Sum(c) == 510592, "1536: tokens and coordinate sum = reference");
    Expect(c.front() == tr::Coord{0, 1, 0} && c.back() == tr::Coord{95, 84, 90}, "1536: sorted first / last = reference");
    c = tr::cascade_coords(fine, 1536, 3500, &hr);
    Expect(hr == 1024 && c.size() == 3584 && Sum(c) == 338688, "cap 3500: 1536 -> 1408 -> 1280 -> 1152 -> 1024 (reference)");
    c = tr::cascade_coords(fine, 1024, 10, &hr);
    Expect(hr == 1024 && c.size() == 3584, "1024 ignores the cap (as the original)");

    // the cap loop stops at the first resolution under the cap
    std::vector<tr::Coord> dense;
    for (int x = 0; x < 512; x += 2)
        for (int y = 0; y < 512; y += 16)
            for (int z = 0; z < 512; z += 16) dense.push_back({x, y, z});
    const auto n1536 = tr::cascade_quantise(dense, 512, 1536).size();
    const auto n1408 = tr::cascade_quantise(dense, 512, 1408).size();
    c = tr::cascade_coords(dense, 1536, static_cast<int>(n1536), &hr);
    Expect(hr == 1408 && c.size() == n1408, "cap = tokens at 1536 -> one step down to 1408");
    c = tr::cascade_coords(dense, 1536, static_cast<int>(n1536) + 1, &hr);
    Expect(hr == 1536 && c.size() == n1536, "cap above the tokens keeps 1536");

    // integer form == float32 formula of the original for every coordinate and resolution
    bool exact = true;
    for (int hr_res : {1024, 1152, 1280, 1408, 1536}) {
        const float g = static_cast<float>(hr_res / 16);
        for (int v = 0; v < 512; ++v) {
            const int ref = static_cast<int>((static_cast<float>(v) + 0.5f) / 512.f * g);
            const auto q = tr::cascade_quantise({{v, v, v}}, 512, hr_res);
            exact = exact && q.size() == 1 && q[0].x == ref && ref < hr_res / 16;
        }
    }
    Expect(exact, "integer quantisation equals the float32 formula, tokens < hr/16");
    Expect(tr::is_supported_resolution(512) && tr::is_supported_resolution(1024) && tr::is_supported_resolution(1536) &&
               !tr::is_supported_resolution(256) && !tr::is_supported_resolution(2048),
           "pipeline resolutions");

    if (failures == 0) std::cout << "trellis2 cascade: all tests passed\n";
    return failures == 0 ? 0 : 1;
}
