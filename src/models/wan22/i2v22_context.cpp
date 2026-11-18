// Written with AI assistance (Claude Code).
#include "src/models/wan22/i2v22_context.hpp"

#include <algorithm>

namespace gufo::models::wan22 {

std::vector<float> I2VConditionChannels(std::span<const float> condition,
                                        int latent_frames, int height,
                                        int width) {
  if (latent_frames < 1 || height < 1 || width < 1)
    return {};
  const std::size_t frame = static_cast<std::size_t>(height) * width;
  const std::size_t plane = frame * latent_frames;
  if (condition.size() != 16 * plane)
    return {};
  std::vector<float> out(20 * plane, 0.0F);
  for (int c = 0; c < 4; ++c)
    std::fill_n(out.begin() + c * plane, frame, 1.0F);
  std::copy(condition.begin(), condition.end(), out.begin() + 4 * plane);
  return out;
}

bool I2VModelInput(std::span<const float> latents,
                   std::span<const float> condition_channels,
                   std::vector<float>* input) {
  if (latents.empty() || latents.size() % 16 != 0 ||
      condition_channels.size() != latents.size() / 16 * 20)
    return false;
  input->resize(latents.size() + condition_channels.size());
  std::copy(latents.begin(), latents.end(), input->begin());
  std::copy(condition_channels.begin(), condition_channels.end(),
            input->begin() + static_cast<std::ptrdiff_t>(latents.size()));
  return true;
}

}  // namespace gufo::models::wan22
