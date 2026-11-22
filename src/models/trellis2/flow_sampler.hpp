// TRELLIS.2: the Euler flow loop of every stage (sparse structure, shape, texture) with
// one or several image conditions, independent of the GPU model behind `forward`:
//   * one view: v = model(x, cond); inside the guidance interval CFG with the zero
//     condition (scheduler.hpp apply_guidance) -- unchanged from the single-image path;
//   * several views, mode "stochastic" (TRELLIS v1 run_multi_image, microsoft/TRELLIS.2
//     PR #104): step i uses view i % views only -- the cost of one view;
//   * several views, mode "multidiffusion" (PR #104): the conditional predictions of all
//     views are averaged (summed in view order, then divided by the count) and the
//     averaged prediction takes the place of the positive one in CFG and its x0 rescale;
//     the negative pass runs once -- views + 1 model calls per guided step.
// The view counter starts at 0 for every flow (PR #104 keeps one counter per sampler
// across the two cascade shape flows; with 12 steps and 2..4 views both agree).
// AI-assisted (Claude Code), 2026-11-22.
#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/trellis2/scheduler.hpp"

namespace gufo::trellis2 {

enum class MultiMode { kStochastic, kMultiDiffusion };

inline const char* multi_mode_name(MultiMode m) { return m == MultiMode::kMultiDiffusion ? "multidiffusion" : "stochastic"; }
inline bool parse_multi_mode(std::string_view s, MultiMode* out) {
    if (s == "stochastic") { *out = MultiMode::kStochastic; return true; }
    if (s == "multidiffusion") { *out = MultiMode::kMultiDiffusion; return true; }
    return false;
}

// View used by step `step` in the stochastic mode.
inline int stochastic_view(int step, int views) { return views > 0 ? step % views : 0; }

// Model call: input rows [N, channels + extra_channels] (extra = the texture flow's shape
// latent), one condition (token rows), velocity out [N, channels].
using FlowForward = std::function<bool(const float* input, const float* cond, float t_model, std::vector<float>* v, std::string* error)>;

struct FlowLoopHooks {
    std::function<bool()> cancelled;                 // optional
    std::function<void(int done, int total)> tick;   // optional
};

// x [N, channels] in place: Euler from t = 1 to 0 over `steps` (rescaled times of p).
inline bool sample_flow(const FlowForward& forward, std::vector<float>& x, int n, int channels, const std::vector<float>* extra,
                        int extra_channels, const std::vector<const float*>& conds, const float* zero_cond, const SamplerParams& p,
                        int steps, MultiMode mode, const FlowLoopHooks& hooks, std::string* error) {
    if (conds.empty()) { if (error) *error = "no image condition"; return false; }
    const int views = static_cast<int>(conds.size());
    const std::vector<float> times = flow_times(steps, p.rescale_t);
    std::vector<float> input;
    auto make_input = [&]() -> const float* {
        if (!extra) return x.data();
        const std::size_t w = static_cast<std::size_t>(channels + extra_channels);
        input.resize(static_cast<std::size_t>(n) * w);
        for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
            for (int c = 0; c < channels; ++c) input[i * w + static_cast<std::size_t>(c)] = x[i * static_cast<std::size_t>(channels) + static_cast<std::size_t>(c)];
            for (int c = 0; c < extra_channels; ++c)
                input[i * w + static_cast<std::size_t>(channels + c)] = (*extra)[i * static_cast<std::size_t>(extra_channels) + static_cast<std::size_t>(c)];
        }
        return input.data();
    };
    for (int i = 0; i < steps; ++i) {
        if (hooks.cancelled && hooks.cancelled()) { if (error) *error = "cancelled"; return false; }
        const float t = times[static_cast<std::size_t>(i)], t_prev = times[static_cast<std::size_t>(i) + 1];
        const float tm = model_timestep(t);
        const float* in = make_input();
        std::vector<float> pos, neg;
        if (views == 1 || mode == MultiMode::kStochastic) {
            if (!forward(in, conds[static_cast<std::size_t>(stochastic_view(i, views))], tm, &pos, error)) return false;
        } else {
            std::vector<float> one;
            for (int k = 0; k < views; ++k) {
                if (!forward(in, conds[static_cast<std::size_t>(k)], tm, k == 0 ? &pos : &one, error)) return false;
                if (k > 0) for (std::size_t e = 0; e < pos.size(); ++e) pos[e] += one[e];
            }
            const float count = static_cast<float>(views);
            for (auto& v : pos) v /= count;
        }
        if (guidance_active(t, p)) {
            if (!forward(in, zero_cond, tm, &neg, error)) return false;
            apply_guidance(&pos, neg, x, t, p);
        }
        euler_step(x, pos, t, t_prev);
        if (hooks.tick) hooks.tick(i + 1, steps);
    }
    return true;
}

}  // namespace gufo::trellis2
