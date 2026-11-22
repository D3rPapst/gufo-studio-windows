// CPU tests for src/models/trellis2/flow_sampler.hpp (single view, stochastic and
// multidiffusion multi-view sampling) against tools/trellis2/referenz/multiview_ref.py
// (numpy float64 re-implementation of FlowEulerGuidanceIntervalSampler + PR #104), on a
// toy model; plus the view schedule and the mode names.
// AI-assisted (Claude Code), 2026-11-22.
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "src/models/trellis2/flow_sampler.hpp"

namespace tr = gufo::trellis2;

namespace {
int failures = 0;
void Expect(bool c, const std::string& what) {
    if (!c) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

// Same toy as multiview_ref.py: v[i,c] = 0.5 in[i,c] - 0.25 in[i,(c+1)%W] + cond[(i*C+c)%L] * (1 + t/1000).
tr::FlowForward Toy(int n, int channels, int width, int cond_len, int* calls) {
    return [=](const float* in, const float* cond, float t_model, std::vector<float>* v, std::string*) {
        ++*calls;
        v->assign(static_cast<std::size_t>(n) * channels, 0.f);
        for (int i = 0; i < n; ++i)
            for (int c = 0; c < channels; ++c)
                (*v)[static_cast<std::size_t>(i * channels + c)] =
                    0.5f * in[i * width + c] - 0.25f * in[i * width + (c + 1) % width] + cond[(i * channels + c) % cond_len] * (1.f + t_model / 1000.f);
        return true;
    };
}

bool Close(const std::vector<float>& a, const std::vector<float>& b, float tol) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (!(std::fabs(a[i] - b[i]) <= tol * (1.f + std::fabs(b[i])))) {
            std::cerr << "  [" << i << "] " << a[i] << " vs " << b[i] << '\n';
            return false;
        }
    return true;
}
}  // namespace

int main() {
    const std::vector<float> x0 = {0.3f, -1.2f, 0.8f, 0.1f, -0.5f, 0.7f};
    const std::vector<float> extra = {0.2f, -0.4f, 0.9f};
    const std::vector<float> c0 = {0.5f, -0.3f, 0.8f, 0.1f}, c1 = {-0.6f, 0.4f, 0.2f, -0.9f}, c2 = {0.05f, 0.7f, -0.2f, 0.3f};
    const std::vector<float> zero(4, 0.f);
    const tr::SamplerParams shape{5, 7.5f, 0.5f, 0.6f, 1.0f, 3.0f};
    const tr::SamplerParams tex{4, 1.0f, 0.0f, 0.6f, 0.9f, 3.0f};
    std::string error;

    // Reference values: python -P -E tools/trellis2/referenz/multiview_ref.py
    const std::vector<float> ref_single = {-1.64825742f, 0.287414494f, -2.54935144f, -0.628604999f, -1.82134263f, 1.11513009f};
    const std::vector<float> ref_stochastic = {-0.255075033f, -1.07485847f, -0.968129525f, 0.524083485f, -0.428056107f, -0.260054608f};
    const std::vector<float> ref_multi = {-0.0749302227f, -1.57125633f, -0.525519853f, 0.636549522f, -0.246924575f, -0.72747252f};
    const std::vector<float> ref_tex = {0.0282285156f, -0.719011719f, -0.168292969f, 0.527628906f, -0.0864521484f, 0.524841797f};

    int calls = 0;
    std::vector<float> x = x0;
    Expect(tr::sample_flow(Toy(3, 2, 2, 4, &calls), x, 3, 2, nullptr, 0, {c0.data()}, zero.data(), shape, 5, tr::MultiMode::kStochastic, {}, &error) &&
               Close(x, ref_single, 2e-5f),
           "single view = reference");
    // rescaled t of 5 steps with r=3: 1, 0.923, 0.818, 0.667, 0.429 -> 4 guided steps (t >= 0.6): 5 + 4 calls
    Expect(calls == 9, "single view: one call per step plus the negative pass in the interval");

    calls = 0;
    x = x0;
    Expect(tr::sample_flow(Toy(3, 2, 2, 4, &calls), x, 3, 2, nullptr, 0, {c0.data(), c1.data(), c2.data()}, zero.data(), shape, 5,
                           tr::MultiMode::kStochastic, {}, &error) && Close(x, ref_stochastic, 2e-5f),
           "stochastic 3 views = reference");
    Expect(calls == 9, "stochastic costs the same as one view");

    calls = 0;
    x = x0;
    Expect(tr::sample_flow(Toy(3, 2, 2, 4, &calls), x, 3, 2, nullptr, 0, {c0.data(), c1.data(), c2.data()}, zero.data(), shape, 5,
                           tr::MultiMode::kMultiDiffusion, {}, &error) && Close(x, ref_multi, 2e-5f),
           "multidiffusion 3 views = reference");
    Expect(calls == 5 * 3 + 4, "multidiffusion: views calls per step plus one negative pass");

    calls = 0;
    x = x0;
    Expect(tr::sample_flow(Toy(3, 2, 3, 4, &calls), x, 3, 2, &extra, 1, {c0.data(), c1.data()}, zero.data(), tex, 4,
                           tr::MultiMode::kMultiDiffusion, {}, &error) && Close(x, ref_tex, 2e-5f),
           "texture-style flow (concatenated extra column, no CFG) with 2 views = reference");
    Expect(calls == 8, "no guidance: no negative pass");

    // one view: both modes are the single-image sampler
    std::vector<float> a = x0, b = x0;
    calls = 0;
    tr::sample_flow(Toy(3, 2, 2, 4, &calls), a, 3, 2, nullptr, 0, {c1.data()}, zero.data(), shape, 5, tr::MultiMode::kStochastic, {}, &error);
    tr::sample_flow(Toy(3, 2, 2, 4, &calls), b, 3, 2, nullptr, 0, {c1.data()}, zero.data(), shape, 5, tr::MultiMode::kMultiDiffusion, {}, &error);
    Expect(a == b, "one view: the mode does not matter");

    // schedule, names, cancel, ticks
    Expect(tr::stochastic_view(0, 4) == 0 && tr::stochastic_view(5, 4) == 1 && tr::stochastic_view(11, 3) == 2 && tr::stochastic_view(7, 1) == 0,
           "stochastic view = step % views");
    tr::MultiMode m{};
    Expect(tr::parse_multi_mode("multidiffusion", &m) && m == tr::MultiMode::kMultiDiffusion && tr::parse_multi_mode("stochastic", &m) &&
               m == tr::MultiMode::kStochastic && !tr::parse_multi_mode("concat", &m) && !tr::parse_multi_mode("", &m),
           "mode names");
    Expect(std::string(tr::multi_mode_name(tr::MultiMode::kMultiDiffusion)) == "multidiffusion", "mode name round trip");
    int ticks = 0;
    tr::FlowLoopHooks hooks;
    hooks.tick = [&](int done, int total) { ticks = done == total ? done : ticks; };
    hooks.cancelled = [&] { return ticks >= 99; };
    x = x0;
    Expect(tr::sample_flow(Toy(3, 2, 2, 4, &calls), x, 3, 2, nullptr, 0, {c0.data()}, zero.data(), shape, 5, tr::MultiMode::kStochastic, hooks, &error) && ticks == 5,
           "tick reaches steps");
    hooks.cancelled = [] { return true; };
    error.clear();
    Expect(!tr::sample_flow(Toy(3, 2, 2, 4, &calls), x, 3, 2, nullptr, 0, {c0.data()}, zero.data(), shape, 5, tr::MultiMode::kStochastic, hooks, &error) &&
               error == "cancelled",
           "cancel stops the loop");
    Expect(!tr::sample_flow(Toy(3, 2, 2, 4, &calls), x, 3, 2, nullptr, 0, {}, zero.data(), shape, 5, tr::MultiMode::kStochastic, {}, &error),
           "no condition is an error");

    if (failures == 0) std::cout << "trellis2 flow_sampler: all tests passed\n";
    return failures == 0 ? 0 : 1;
}
