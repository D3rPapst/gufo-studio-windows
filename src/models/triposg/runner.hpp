// TripoSG: HIP-free interface of the image -> GLB generator for the server
// (gufo serve objekt). CreateObjektRunner lives in the HIP library; the CPU-only
// build links a stub that reports the missing GPU path.
// AI-assisted (Claude Code), 2026-11-12.
#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "src/core/mesh/print_ready.hpp"

namespace gufo::triposg {

// A further view of the same object (TRELLIS.2 multi-image conditioning).
struct ObjektView {
    std::vector<std::uint8_t> rgba;  // packed RGBA8
    int width = 0, height = 0;
};

struct ObjektRequest {
    std::vector<std::uint8_t> rgba;  // packed RGBA8 (the first / only view)
    int width = 0, height = 0;
    std::vector<ObjektView> more_views;   // views 2..n (`images`), empty for one image
    std::string multi_mode = "stochastic";   // several views: stochastic | multidiffusion
    int steps = 50;
    float guidance = 7.f;
    std::uint64_t seed = 42;
    int resolution = 512;
    int max_faces = 200000;
    bool preview = true;
    bool texture = false;   // TRELLIS.2: texture flow + decoder (vertex colours)
    int texture_size = 1024;   // TRELLIS.2 with texture: 0 = vertex colours, else baked PBR textures of this size
    // Print-ready result ("druckfertig", src/core/mesh/print_ready.hpp): the runner hands the final
    // mesh back in ObjektResult::print_source, the service closes, smooths and scales it.
    bool print_ready = false;
    double print_height_mm = 0;        // total height (wins when > 0)
    double print_eye_height_mm = 0;    // sole to eyes, eyes at print_eye_ratio of the height
    double print_eye_ratio = 0.93;
    std::string print_format = "both";   // stl | glb | both
};

struct ObjektResult {
    std::vector<std::uint8_t> glb, preview_png;
    std::size_t vertices = 0, faces = 0, faces_before_decimation = 0;
    float bbox_min[3] = {0, 0, 0}, bbox_max[3] = {0, 0, 0};
    double volume = 0;
    bool closed = false;
    double seconds_encode = 0, seconds_denoise = 0, seconds_decode = 0, seconds_mesh = 0;
    int resolution = 0;       // grid actually used (TRELLIS.2 1536 may step down under the token cap; 0 = as requested)
    std::size_t tokens = 0;   // TRELLIS.2: tokens of the last shape flow
    // With request.print_ready: the exported mesh (glTF frame, Y up) with vertex colours when the
    // model has them; the service consumes and clears it.
    gufo::mesh::PrintMesh print_source;
};

using ObjektProgress = std::function<void(const std::string& phase, int completed, int total)>;

class ObjektRunner {
public:
    virtual ~ObjektRunner() = default;
    virtual bool Run(const ObjektRequest& request, const ObjektProgress& progress, const std::atomic<bool>* cancel,
                     ObjektResult* result, std::string* error) = 0;
    virtual std::size_t weight_bytes() const = 0;
    // Sampler default when the request omits `steps` (TripoSG 50, TRELLIS.2 12).
    virtual int default_steps() const { return 50; }
    virtual std::string capability_note() const { return "shape"; }
    // Attribution the model's licences ask for, shown in /v1/models ("" = none).
    virtual std::string attribution() const { return ""; }
    // Accepted `resolution` values (TripoSG: 256 / 512; TRELLIS.2 adds 1024 / 1536 when the
    // 1024 flows are installed) and the most views one request may carry (`images`).
    virtual std::vector<int> resolutions() const { return {256, 512}; }
    virtual int max_views() const { return 1; }
};

// Loads the converted triposg.safetensors onto the GPU (HIP build) or fails with *error.
std::unique_ptr<ObjektRunner> CreateObjektRunner(const std::filesystem::path& weights, std::string* error);

}  // namespace gufo::triposg
