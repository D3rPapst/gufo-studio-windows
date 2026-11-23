// TRELLIS.2 pipeline (see pipeline.hpp). HIP build only.
// AI-assisted (Claude Code), 2026-11-12.
#include "src/models/trellis2/pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>

#include "src/models/trellis2/cascade.hpp"
#include "src/models/trellis2/config.hpp"
#include "src/models/trellis2/dinov3.hpp"
#include "src/models/trellis2/dit.hpp"
#include "src/models/trellis2/dual_grid_mesh.hpp"
#include "src/models/trellis2/glb_writer.hpp"
#include "src/models/trellis2/mesh_post.hpp"
#include "src/models/trellis2/mesh_preview.hpp"
#include "src/models/trellis2/png.hpp"
#include "src/models/trellis2/texture_bake.hpp"
#include "src/models/trellis2/preprocess.hpp"
#include "src/models/trellis2/scheduler.hpp"
#include "src/models/trellis2/sparse_conv.hpp"
#include "src/models/trellis2/sparse_dec.hpp"
#include "src/models/trellis2/ss_dec.hpp"

namespace gufo::trellis2 {
namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}
bool cancelled(const std::atomic<bool>* cancel) { return cancel != nullptr && cancel->load(); }

std::vector<float> gaussian(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> v(n);
    for (auto& x : v) x = nd(rng);
    return v;
}

// Euler flow with interval CFG and x0 rescale over one or several views (flow_sampler.hpp);
// x in place. `extra` columns (texture flow: the normalised shape latent) are appended to x
// for every forward and never updated.
bool run_flow(DitHip& dit, std::vector<float>& x, int channels, const std::vector<float>* extra, int extra_channels,
              const std::vector<Coord>& coords, const std::vector<const float*>& conds, const float* zero_cond, int cond_tokens,
              const SamplerParams& p, int steps, MultiMode mode, const std::atomic<bool>* cancel,
              const std::function<void(int, int)>& tick, std::string* error) {
    const FlowForward forward = [&](const float* in, const float* cond, float t_model, std::vector<float>* v, std::string* err) {
        return dit.Forward(in, coords, cond, cond_tokens, t_model, v, nullptr, err);
    };
    FlowLoopHooks hooks;
    hooks.cancelled = [cancel] { return cancelled(cancel); };
    hooks.tick = tick;
    return sample_flow(forward, x, static_cast<int>(coords.size()), channels, extra, extra_channels, conds, zero_cond, p, steps, mode,
                       hooks, error);
}

// Diagnostics: one stderr line per stage (worker log), cheap. GUFO_TRELLIS2_DEBUG_DIR also saves
// the preprocessed square of every view as PNG (what DINOv3 sees, before the final resize).
void diag(const std::string& line) {
    std::fprintf(stderr, "trellis2: %s\n", line.c_str());
    std::fflush(stderr);
}
std::string stats_text(const std::vector<float>& v, std::size_t stride = 0, std::size_t column = 0) {
    double sum = 0, sq = 0;
    float lo = 0, hi = 0;
    std::size_t n = 0;
    for (std::size_t i = stride ? column : 0; i < v.size(); i += stride ? stride : 1) {
        const float x = v[i];
        if (n == 0) lo = hi = x;
        lo = std::min(lo, x); hi = std::max(hi, x);
        sum += x; sq += static_cast<double>(x) * x; ++n;
    }
    if (n == 0) return "n=0";
    char buf[160];
    std::snprintf(buf, sizeof buf, "n=%zu mean=%.4g rms=%.4g min=%.4g max=%.4g", n, sum / n, std::sqrt(sq / n), lo, hi);
    return buf;
}
std::string debug_dir() {
    const char* d = std::getenv("GUFO_TRELLIS2_DEBUG_DIR");
    return d ? std::string(d) : std::string();
}

std::vector<const float*> pointers(const std::vector<std::vector<float>>& v) {
    std::vector<const float*> out;
    for (const auto& e : v) out.push_back(e.data());
    return out;
}

// Colours per mesh vertex from the per-voxel attributes [N, 6] (base colour 3, metallic,
// roughness, alpha) by voxel lookup; returns RGBA [vertices * 4] and the mean metallic /
// roughness of the used voxels.
std::vector<float> voxel_colours(const Mesh& mesh, const VoxelAttrSampler& sampler, float* metallic, float* roughness) {
    std::vector<float> out(mesh.vertices.size() * 4, 1.f);
    double m_sum = 0, r_sum = 0;
    std::size_t used = 0;
    for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
        const MeshVertex& v = mesh.vertices[i];
        float a[16];
        if (!sampler.Sample(v.x, v.y, v.z, a)) { out[i * 4] = out[i * 4 + 1] = out[i * 4 + 2] = 0.5f; continue; }
        out[i * 4] = a[0]; out[i * 4 + 1] = a[1]; out[i * 4 + 2] = a[2]; out[i * 4 + 3] = 1.f;
        m_sum += a[3]; r_sum += a[4];
        ++used;
    }
    if (used) { *metallic = static_cast<float>(m_sum / used); *roughness = static_cast<float>(r_sum / used); }
    return out;
}

}  // namespace

struct Pipeline::Impl {
    Dinov3Weights dino_weights;
    std::unique_ptr<Dinov3Hip> dino;
    std::unique_ptr<DitHip> ss_flow, shape_flow, tex_flow;
    SsDecWeights ss_dec_weights;
    std::unique_ptr<SsDecHip> ss_dec;
    std::unique_ptr<SparseDecHip> shape_dec, tex_dec;
    std::size_t weight_bytes = 0;
    // 1024 / 1536 cascade: the 1024 flows, loaded on the first such request.
    std::filesystem::path dir;
    bool with_texture = false, hq_available = false;
    std::unique_ptr<DitHip> shape_flow_hq, tex_flow_hq;

    bool LoadHighResolution(std::string* error) {
        if (shape_flow_hq && (!with_texture || tex_flow_hq)) return true;
        auto file = [&](const char* part) { return dir / (std::string("trellis2-") + part + ".safetensors"); };
        if (!shape_flow_hq) {
            DitShape s;
            s.in_channels = 32; s.out_channels = 32;
            shape_flow_hq = DitHip::Create(file("shape_flow_1024"), s, error);
            if (!shape_flow_hq) return false;
        }
        if (with_texture && !tex_flow_hq) {
            DitShape s;
            s.in_channels = 64; s.out_channels = 32;
            tex_flow_hq = DitHip::Create(file("tex_flow_1024"), s, error);
            if (!tex_flow_hq) return false;
        }
        return true;
    }
};

Pipeline::Pipeline(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Pipeline::~Pipeline() = default;
std::size_t Pipeline::weight_bytes() const { return impl_->weight_bytes; }
bool Pipeline::has_texture() const { return impl_->tex_flow != nullptr && impl_->tex_dec != nullptr; }
bool Pipeline::has_high_resolution() const { return impl_->hq_available; }

std::unique_ptr<Pipeline> Pipeline::Load(const std::filesystem::path& dir, bool with_texture, std::string* error) {
    auto impl = std::make_unique<Impl>();
    Impl& m = *impl;
    auto file = [&](const char* part) { return dir / (std::string("trellis2-") + part + ".safetensors"); };
    if (!Dinov3Weights::Load(file("dinov3"), &m.dino_weights, error)) return nullptr;
    m.dino = Dinov3Hip::Create(m.dino_weights, PipelineConfig::image_cond_512, error);
    if (!m.dino) return nullptr;
    m.weight_bytes += m.dino->device_bytes();
    m.dino_weights = Dinov3Weights();   // host copy no longer needed
    DitShape ss_shape;
    ss_shape.in_channels = 8; ss_shape.out_channels = 8;
    m.ss_flow = DitHip::Create(file("ss_flow"), ss_shape, error);
    if (!m.ss_flow) return nullptr;
    if (!SsDecWeights::Load(file("ss_dec"), &m.ss_dec_weights, error)) return nullptr;
    m.ss_dec = SsDecHip::Create(m.ss_dec_weights, error);
    if (!m.ss_dec) return nullptr;
    m.weight_bytes += m.ss_dec->device_bytes();
    m.ss_dec_weights = SsDecWeights();
    DitShape shape_shape;
    shape_shape.in_channels = 32; shape_shape.out_channels = 32;
    m.shape_flow = DitHip::Create(file("shape_flow_512"), shape_shape, error);
    if (!m.shape_flow) return nullptr;
    SparseDecShape dec_shape;
    m.shape_dec = SparseDecHip::Create(file("shape_dec"), dec_shape, error);
    if (!m.shape_dec) return nullptr;
    if (with_texture) {
        DitShape tex_shape;
        tex_shape.in_channels = 64; tex_shape.out_channels = 32;
        m.tex_flow = DitHip::Create(file("tex_flow_512"), tex_shape, error);
        if (!m.tex_flow) return nullptr;
        SparseDecShape tex_dec_shape;
        tex_dec_shape.outputs = SparseDecConfig::tex_outputs;
        tex_dec_shape.pred_subdiv = false;
        m.tex_dec = SparseDecHip::Create(file("tex_dec"), tex_dec_shape, error);
        if (!m.tex_dec) return nullptr;
    }
    m.dir = dir;
    m.with_texture = with_texture;
    std::error_code ec;
    m.hq_available = std::filesystem::is_regular_file(file("shape_flow_1024"), ec) &&
                     (!with_texture || std::filesystem::is_regular_file(file("tex_flow_1024"), ec));
    return std::unique_ptr<Pipeline>(new Pipeline(std::move(impl)));
}

bool Pipeline::Generate(const std::uint8_t* rgba, int width, int height, const GenerateOptions& o, const GenerateProgress& progress,
                        const std::atomic<bool>* cancel, GenerateResult* result, std::string* error) {
    return Generate(std::vector<ImageView>{{rgba, width, height}}, o, progress, cancel, result, error);
}

bool Pipeline::Generate(const std::vector<ImageView>& views, const GenerateOptions& o, const GenerateProgress& progress,
                        const std::atomic<bool>* cancel, GenerateResult* result, std::string* error) {
    Impl& m = *impl_;
    auto report = [&](const std::string& phase, int done, int total) { if (progress) progress(phase, done, total); };
    auto fail_cancel = [&]() { if (error) *error = "cancelled"; return false; };
    const int steps = o.steps > 0 ? o.steps : 12;
    if (!is_supported_resolution(o.resolution)) {
        if (error) *error = "TRELLIS.2: resolution must be 512, 1024 or 1536";
        return false;
    }
    if (views.empty() || views.size() > static_cast<std::size_t>(kMaxViews)) {
        if (error) *error = "TRELLIS.2: 1 to " + std::to_string(kMaxViews) + " views";
        return false;
    }
    const bool cascade = o.resolution > 512;
    if (cascade) {
        if (!m.hq_available) {
            if (error) *error = "TRELLIS.2: resolution " + std::to_string(o.resolution) + " needs trellis2-shape_flow_1024" +
                                (m.with_texture ? std::string(" and trellis2-tex_flow_1024") : std::string()) + ".safetensors in " + m.dir.string();
            return false;
        }
        report("load", 0, 1);
        if (!m.LoadHighResolution(error)) return false;
        report("load", 1, 1);
    }
    const int V = static_cast<int>(views.size());
    result->views = V;
    // 1. encode: every view at 512 px, for the cascade also at 1024 px (each view cropped to its own alpha box)
    auto t0 = Clock::now();
    const int encodes = V * (cascade ? 2 : 1);
    report("encode", 0, encodes);
    std::vector<std::vector<float>> cond512(static_cast<std::size_t>(V)), cond1024(cascade ? static_cast<std::size_t>(V) : 0);
    int done = 0;
    for (int image : {512, 1024}) {
        if (image == 1024 && !cascade) break;
        for (int v = 0; v < V; ++v) {
            Preprocessed pre;
            const ImageView& view = views[static_cast<std::size_t>(v)];
            if (!preprocess_rgba(view.rgba, view.width, view.height, image, &pre, error)) {
                if (error && V > 1) *error = "view " + std::to_string(v + 1) + ": " + *error;
                return false;
            }
            auto& out = image == 512 ? cond512[static_cast<std::size_t>(v)] : cond1024[static_cast<std::size_t>(v)];
            if (!m.dino->Encode(pre.pixels.data(), image, &out, error)) return false;
            diag("view " + std::to_string(v + 1) + "/" + std::to_string(V) + " " + std::to_string(view.width) + "x" + std::to_string(view.height) +
                 " bbox " + std::to_string(pre.bbox[0]) + "," + std::to_string(pre.bbox[1]) + " " + std::to_string(pre.bbox[2]) + "x" +
                 std::to_string(pre.bbox[3]) + " square " + std::to_string(pre.square_size) + " -> " + std::to_string(image) + " px; pixels " +
                 stats_text(pre.pixels) + "; tokens " + stats_text(out));
            if (const std::string dir = debug_dir(); !dir.empty() && !pre.square.empty()) {
                const auto png = encode_png_rgb(pre.square.data(), pre.square_size, pre.square_size);
                std::ofstream f(std::filesystem::path(dir) / ("view" + std::to_string(v + 1) + "-" + std::to_string(image) + ".png"), std::ios::binary);
                f.write(reinterpret_cast<const char*>(png.data()), static_cast<std::streamsize>(png.size()));
            }
            report("encode", ++done, encodes);
            if (cancelled(cancel)) return fail_cancel();
        }
    }
    const int M = DinoV3Config::tokens(512);
    const int M_hq = DinoV3Config::tokens(1024);
    const std::vector<float> zero_cond(static_cast<std::size_t>(M) * DinoV3Config::width, 0.f);
    const std::vector<float> zero_cond_hq(cascade ? static_cast<std::size_t>(M_hq) * DinoV3Config::width : 0, 0.f);
    const std::vector<const float*> conds = pointers(cond512), conds_hq = pointers(cond1024);
    result->seconds_encode = seconds_since(t0);

    // 2. sparse structure (32^3 tokens for every pipeline type, as the original)
    t0 = Clock::now();
    const int R = SsDecConfig::latent_res, Cs = 8, Ns = R * R * R;
    std::vector<float> noise_cm = gaussian(static_cast<std::size_t>(Cs) * Ns, o.seed);
    std::vector<float> x(static_cast<std::size_t>(Ns) * Cs);
    for (int c = 0; c < Cs; ++c)
        for (int v = 0; v < Ns; ++v) x[static_cast<std::size_t>(v) * Cs + c] = noise_cm[static_cast<std::size_t>(c) * Ns + v];
    const std::vector<Coord> grid = dense_grid_coords(R);
    if (!run_flow(*m.ss_flow, x, Cs, nullptr, 0, grid, conds, zero_cond.data(), M, kSsSampler, steps, o.multi_mode, cancel,
                  [&](int d, int t) { report("structure", d, t); }, error))
        return false;
    std::vector<float> logits;
    if (!m.ss_dec->Forward(x.data(), &logits, error)) return false;
    std::vector<Coord> coords = occupancy_to_coords(logits.data(), 64, 32);
    result->structure_voxels = coords.size();
    result->seconds_structure = seconds_since(t0);
    diag("structure: latent " + stats_text(x) + "; logits " + stats_text(logits) + "; tokens(32^3) " + std::to_string(coords.size()));
    if (coords.empty()) { if (error) *error = "no active voxels in the sparse structure"; return false; }
    if (cancelled(cancel)) return fail_cancel();

    // 3. shape flow 512 (the whole shape for 512, the low-resolution stage of the cascade)
    t0 = Clock::now();
    const int Cl = 32;
    int N = static_cast<int>(coords.size());
    std::vector<float> slat = gaussian(static_cast<std::size_t>(N) * Cl, o.seed + 1);
    if (!run_flow(*m.shape_flow, slat, Cl, nullptr, 0, coords, conds, zero_cond.data(), M, kShapeSampler, steps, o.multi_mode, cancel,
                  [&](int d, int t) { report("shape", d, t); }, error))
        return false;
    auto denormalise = [&](std::vector<float>& s) {
        for (std::size_t i = 0; i < s.size() / Cl; ++i)
            for (int c = 0; c < Cl; ++c) s[i * Cl + static_cast<std::size_t>(c)] = s[i * Cl + static_cast<std::size_t>(c)] * kShapeStd[c] + kShapeMean[c];
    };
    diag("shape 512: normalised latent " + stats_text(slat));
    std::vector<float> slat_norm = slat;   // normalised (for the texture flow), slat -> denormalised
    denormalise(slat);
    int res = 512;
    if (cascade) {
        // 3b. upsample: the shape decoder's subdivision levels give the voxels on the 512 grid
        report("upsample", 0, 1);
        SparseDecResult up;
        if (!m.shape_dec->Forward(slat.data(), coords, &up, error)) return false;
        std::vector<float>().swap(up.outputs);
        up.masks.clear();
        coords = cascade_coords(up.coords, o.resolution, o.max_tokens > 0 ? o.max_tokens : kCascadeMaxTokens, &res);
        diag("cascade: " + std::to_string(up.coords.size()) + " voxels on the 512 grid -> " + std::to_string(coords.size()) + " tokens at " +
             std::to_string(res));
        std::vector<Coord>().swap(up.coords);
        report("upsample", 1, 1);
        if (cancelled(cancel)) return fail_cancel();
        // 3c. shape flow 1024 on the high-resolution tokens, DINOv3 tokens of the 1024 px views
        N = static_cast<int>(coords.size());
        slat = gaussian(static_cast<std::size_t>(N) * Cl, o.seed + 3);
        if (!run_flow(*m.shape_flow_hq, slat, Cl, nullptr, 0, coords, conds_hq, zero_cond_hq.data(), M_hq, kShapeSampler, steps,
                      o.multi_mode, cancel, [&](int d, int t) { report("shape_hq", d, t); }, error))
            return false;
        slat_norm = slat;
        denormalise(slat);
    }
    result->resolution = res;
    result->tokens = coords.size();
    result->seconds_shape = seconds_since(t0);
    if (cancelled(cancel)) return fail_cancel();

    // 4. shape decoder (16x the token grid: 512 / 1024 / 1536)
    t0 = Clock::now();
    report("decode", 0, 1);
    SparseDecResult dec;
    if (!m.shape_dec->Forward(slat.data(), coords, &dec, error)) return false;
    result->shape_voxels = dec.coords.size();
    result->seconds_decode = seconds_since(t0);
    {
        std::string levels;
        for (const auto& mk : dec.masks) {
            std::size_t ones = 0;
            for (auto b : mk) ones += b != 0;
            levels += " " + std::to_string(mk.size() / 8) + "->" + std::to_string(ones);
        }
        std::size_t crossed = 0;
        for (std::size_t i = 0; i < dec.coords.size(); ++i)
            for (int a = 0; a < 3; ++a) crossed += dec.outputs[i * 7 + 3 + static_cast<std::size_t>(a)] > 0.f;
        diag("decode: levels" + levels + "; voxels " + std::to_string(dec.coords.size()) + "; edge flags " + std::to_string(crossed) +
             "; vertex logit " + stats_text(dec.outputs, 7, 0) + "; edge logit " + stats_text(dec.outputs, 7, 3) + "; weight " +
             stats_text(dec.outputs, 7, 6));
    }
    report("decode", 1, 1);
    if (cancelled(cancel)) return fail_cancel();
    // 5. texture (optional): attributes per voxel of the final level
    std::vector<float> attrs;   // [M, 6] base colour, metallic, roughness, alpha
    if (o.texture && has_texture()) {
        t0 = Clock::now();
        std::vector<float> tex = gaussian(static_cast<std::size_t>(N) * Cl, o.seed + 2);
        DitHip& tex_flow = cascade ? *m.tex_flow_hq : *m.tex_flow;
        if (!run_flow(tex_flow, tex, Cl, &slat_norm, Cl, coords, cascade ? conds_hq : conds, cascade ? zero_cond_hq.data() : zero_cond.data(),
                      cascade ? M_hq : M, kTexSampler, steps, o.multi_mode, cancel, [&](int d, int t) { report("texture", d, t); }, error))
            return false;
        for (int i = 0; i < N; ++i)
            for (int c = 0; c < Cl; ++c) tex[static_cast<std::size_t>(i) * Cl + c] = tex[static_cast<std::size_t>(i) * Cl + c] * kTexStd[c] + kTexMean[c];
        SparseDecOptions guided;
        guided.guide_masks = &dec.masks;
        SparseDecResult tdec;
        if (!m.tex_dec->Forward(tex.data(), coords, &tdec, error, guided)) return false;
        if (tdec.coords != dec.coords) { if (error) *error = "texture decoder voxels differ from the shape decoder"; return false; }
        attrs = std::move(tdec.outputs);
        for (auto& v : attrs) v = std::clamp(v * 0.5f + 0.5f, 0.f, 1.f);
        result->seconds_texture = seconds_since(t0);
        if (cancelled(cancel)) return fail_cancel();
    }

    // 6. mesh
    t0 = Clock::now();
    report("mesh", 0, 3);
    std::vector<float> dual, wgt;
    std::vector<std::uint8_t> inter;
    decode_shape_outputs(dec.outputs.data(), dec.coords.size(), &dual, &inter, &wgt);
    DualGridInput in;
    in.coords = &dec.coords; in.dual = dual.data(); in.intersected = inter.data(); in.split_weight = wgt.data();
    in.resolution = res;
    DualGridStats st;
    Mesh mesh = dual_grid_to_mesh(in, &st);
    result->faces_before_decimation = mesh.faces.size();
    report("mesh", 1, 3);
    result->seam_edges = st.seam_edges;
    // stray bubbles: 512 faces at 512, scaled with the surface (x4 at 1024, x9 at 1536)
    result->small_components_removed = remove_small_components(&mesh, static_cast<std::size_t>(512.0 * (res / 512.0) * (res / 512.0)));
    const UnifyStats un = unify_orientation(&mesh);
    result->flipped_faces = un.flipped_faces;
    HoleFillStats hf = fill_holes(&mesh, o.hole_perimeter);
    result->holes_filled = hf.filled;
    result->open_loops = hf.skipped_large + hf.skipped_nonmanifold;
    hf.skipped_perimeters.resize(std::min<std::size_t>(hf.skipped_perimeters.size(), 5));
    result->largest_open_loops = std::move(hf.skipped_perimeters);
    diag("mesh: raw " + std::to_string(result->faces_before_decimation) + " faces, seam edges " + std::to_string(st.seam_edges) +
         ", small components removed " + std::to_string(result->small_components_removed) + ", after cleanup " +
         std::to_string(mesh.faces.size()) + " faces, holes filled " + std::to_string(hf.filled));
    { std::vector<float>().swap(dual); std::vector<float>().swap(wgt); std::vector<std::uint8_t>().swap(inter); }   // host RAM
    std::vector<float>().swap(dec.outputs);
    report("mesh", 2, 3);
    if (o.max_faces > 0 && mesh.faces.size() > static_cast<std::size_t>(o.max_faces))
        mesh = decimate_mesh(mesh, static_cast<std::size_t>(o.max_faces));
    // Colours after decimation, in the z-up voxel frame: trilinear voxel attributes per vertex
    // (seam caps take the nearest active voxel within 8 voxels); with texture_size > 0 the mesh is
    // unwrapped (xatlas) and the attributes are baked into PBR textures instead.
    std::vector<float> colours;
    float metallic = 0.f, roughness = 0.9f;
    std::unique_ptr<VoxelAttrSampler> sampler;
    if (!attrs.empty()) sampler = std::make_unique<VoxelAttrSampler>(dec.coords, attrs, 6, res, 8);
    const MeshStats ms = mesh_stats(to_y_up(mesh));   // topology of the closed mesh, before seam duplication
    UvMesh uvm;
    BakedTextures baked;
    bool textured = false;
    if (sampler && o.texture_size > 0) {
        const auto t_bake = Clock::now();
        std::string unwrap_error;
        if (unwrap_uv(mesh, o.texture_size, &uvm, &unwrap_error)) {
            baked = bake_textures(uvm, *sampler, o.texture_size);
            mesh = std::move(uvm.mesh);
            textured = true;
            result->uv_charts = uvm.charts;
            result->texture_size = static_cast<std::size_t>(o.texture_size);
            result->unsampled_texels = baked.unsampled_texels;
            result->texel_coverage = static_cast<double>(baked.covered_texels) / (static_cast<double>(o.texture_size) * static_cast<double>(o.texture_size));
        }
        result->seconds_bake = seconds_since(t_bake);
    }
    if (sampler) colours = voxel_colours(mesh, *sampler, &metallic, &roughness);
    mesh = to_y_up(mesh);
    result->vertices = mesh.vertices.size();
    result->faces = mesh.faces.size();
    result->open_edges = ms.open_edges;
    result->closed = ms.closed;
    result->volume = ms.volume;
    for (int a = 0; a < 3; ++a) { result->bbox_min[a] = ms.bbox_min[a]; result->bbox_max[a] = ms.bbox_max[a]; }
    result->seconds_mesh = seconds_since(t0);
    report("mesh", 3, 3);
    if (cancelled(cancel)) return fail_cancel();

    // 7. export
    report("export", 0, 1);
    GlbWriteOptions go;
    go.generator = "gufo trellis2";
    go.mesh_name = "trellis2";
    std::vector<std::uint8_t> base_png, mr_png;
    if (textured) {
        base_png = encode_png_rgba(baked.base_color.data(), baked.size, baked.size);
        mr_png = encode_png_rgb(baked.metallic_roughness.data(), baked.size, baked.size);
        go.texcoords = &uvm.uv;
        go.base_color_png = &base_png;
        go.metallic_roughness_png = &mr_png;
    } else if (!colours.empty()) {
        go.vertex_colors = &colours;
        go.metallic_factor = metallic;
        go.roughness_factor = roughness;
    }
    result->glb = write_glb(mesh, go);
    if (o.preview) {
        PreviewOptions po;
        po.size = o.preview_size;
        if (!colours.empty()) po.vertex_colors = &colours;
        int pw, ph;
        const auto rgb = render_preview(mesh, po, &pw, &ph);
        result->preview_png = encode_png_rgb(rgb.data(), pw, ph);
    }
    report("export", 1, 1);
    return true;
}

}  // namespace gufo::trellis2
