// TRELLIS.2: ObjektRunner on top of Pipeline (HIP library).
// AI-assisted (Claude Code), 2026-11-12.
#include "src/models/trellis2/runner.hpp"

#include <algorithm>

#include "src/models/trellis2/pipeline.hpp"

namespace gufo::trellis2 {
namespace {

class PipelineRunner final : public triposg::ObjektRunner {
public:
    explicit PipelineRunner(std::unique_ptr<Pipeline> pipeline) : pipeline_(std::move(pipeline)) {}
    bool Run(const triposg::ObjektRequest& r, const triposg::ObjektProgress& progress, const std::atomic<bool>* cancel,
             triposg::ObjektResult* out, std::string* error) override {
        GenerateOptions o;
        o.steps = r.steps;
        o.seed = r.seed;
        // 512 (256 requests are served at 512), 1024 = 1024_cascade, 1536 = 1536_cascade
        o.resolution = r.resolution > 512 ? r.resolution : 512;
        if (!parse_multi_mode(r.multi_mode, &o.multi_mode)) {
            if (error) *error = "multi_mode must be stochastic or multidiffusion";
            return false;
        }
        std::vector<ImageView> views{{r.rgba.data(), r.width, r.height}};
        for (const auto& v : r.more_views) views.push_back({v.rgba.data(), v.width, v.height});
        o.max_faces = r.max_faces;
        o.preview = r.preview;
        o.texture = r.texture && pipeline_->has_texture();
        o.texture_size = o.texture ? std::clamp(r.texture_size, 0, 4096) : 0;
        o.keep_mesh = r.print_ready;
        GenerateResult g;
        if (!pipeline_->Generate(views, o, progress, cancel, &g, error)) return false;
        out->glb = std::move(g.glb);
        out->preview_png = std::move(g.preview_png);
        out->vertices = g.vertices;
        out->faces = g.faces;
        out->faces_before_decimation = g.faces_before_decimation;
        for (int a = 0; a < 3; ++a) { out->bbox_min[a] = g.bbox_min[a]; out->bbox_max[a] = g.bbox_max[a]; }
        out->volume = g.volume;
        out->closed = g.closed;
        out->seconds_encode = g.seconds_encode;
        out->seconds_denoise = g.seconds_structure + g.seconds_shape + g.seconds_texture;
        out->seconds_decode = g.seconds_decode;
        out->seconds_mesh = g.seconds_mesh;
        out->resolution = g.resolution;
        out->tokens = g.tokens;
        out->print_source = std::move(g.mesh);
        return true;
    }
    std::size_t weight_bytes() const override { return pipeline_->weight_bytes(); }
    int default_steps() const override { return 12; }
    std::vector<int> resolutions() const override {
        if (pipeline_->has_high_resolution()) return {256, 512, 1024, 1536};
        return {256, 512};
    }
    int max_views() const override { return Pipeline::kMaxViews; }
    std::string capability_note() const override { return pipeline_->has_texture() ? "shape+pbr-textures" : "shape"; }
    // TRELLIS.2 encodes the image with DINOv3 (Meta, DINOv3 License).
    std::string attribution() const override { return "TRELLIS.2 (Microsoft, MIT). Built with DINOv3 (Meta, DINOv3 License)."; }

private:
    std::unique_ptr<Pipeline> pipeline_;
};

}  // namespace

std::unique_ptr<triposg::ObjektRunner> CreateObjektRunner(const std::filesystem::path& dir, bool with_texture, std::string* error) {
    auto pipeline = Pipeline::Load(dir, with_texture, error);
    if (!pipeline) return nullptr;
    return std::make_unique<PipelineRunner>(std::move(pipeline));
}

}  // namespace gufo::trellis2
