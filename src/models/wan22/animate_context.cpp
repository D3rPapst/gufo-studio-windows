// Written with AI assistance (Claude Code).
// ComfyUI's WanAnimateToVideo (comfy_extras/nodes_wan.py) and the parts of
// WAN21.concat_cond / WAN22_Animate.extra_conds (comfy/model_base.py) that
// shape its outputs for the model, on inputs already fitted to the video
// size.
#include "src/models/wan22/animate_context.hpp"

#include <algorithm>
#include <cstddef>

namespace gufo::models::wan22 {
namespace {

bool Covers(std::span<const std::uint8_t> frames, int count,
            std::size_t frame_values) {
  return count >= 0 && frames.size() == frame_values * count;
}

// RGB8 frame `f` of `video` into [-1, 1] at `out` (interleaved, as
// VaeEncoder::EncodeVideo reads it).
void Widen(std::span<const std::uint8_t> video, int f, std::size_t values,
           float* out) {
  const std::uint8_t* in = video.data() + static_cast<std::size_t>(f) * values;
  for (std::size_t i = 0; i < values; ++i)
    out[i] = static_cast<float>(in[i]) / 127.5F - 1.0F;
}

}  // namespace

bool BuildAnimateChunk(const AnimateChunkInputs& in, AnimateChunkPlan* plan) {
  const int n = in.length, h = in.height, w = in.width;
  if (plan == nullptr || n < 1 || (n - 1) % 4 != 0 || h < 16 || w < 16 ||
      h % 16 != 0 || w % 16 != 0 || in.video_frame_offset < 0 ||
      in.continue_motion_max_frames < 1)
    return false;
  const std::size_t pixels = static_cast<std::size_t>(h) * w;
  const std::size_t rgb = pixels * 3;
  const std::size_t face_rgb =
      static_cast<std::size_t>(kAnimateFaceSize) * kAnimateFaceSize * 3;
  if (!Covers(in.pose, in.pose_frames, rgb) ||
      !Covers(in.face, in.face_frames, face_rgb) ||
      !Covers(in.background, in.background_frames, rgb) ||
      !Covers(in.continue_motion, in.continue_frames, rgb) ||
      in.mask_frames < 0 || in.mask.size() != pixels * in.mask_frames)
    return false;
  *plan = {};
  AnimateChunkPlan& p = *plan;
  const int lf = 1 + (n - 1) / 4, lh = h / 8, lw = w / 8;
  p.latent_frames = lf;
  p.trim_latent = 1;
  int offset = in.video_frame_offset;

  // continue_motion: its last max frames (at most length) in front of a grey
  // video; the seek moves back by as many frames.
  p.condition.assign(rgb * n, 0.0F);  // grey 0.5 in [0, 1]
  int ref_motion_latent_length = 0;
  if (in.continue_frames > 0) {
    const int take = std::min({in.continue_frames,
                               in.continue_motion_max_frames, n});
    offset = std::max(0, offset - std::min(in.continue_frames,
                                           in.continue_motion_max_frames));
    const int first = in.continue_frames - take;
    for (int f = 0; f < take; ++f)
      Widen(in.continue_motion, first + f, rgb, p.condition.data() + f * rgb);
    p.continue_frames = take;
    ref_motion_latent_length = (take - 1) / 4 + 1;
  }
  p.offset = offset;
  p.ref_motion_latent_length = ref_motion_latent_length;

  // Pose: from the seek on, cut to length, the last frame repeated.
  if (in.pose_frames > offset) {
    const int available = std::min(in.pose_frames - offset, n);
    p.pose.resize(rgb * n);
    for (int f = 0; f < n; ++f)
      Widen(in.pose, offset + std::min(f, available - 1), rgb,
            p.pose.data() + f * rgb);
  }

  // Face crops: from the seek on, cut to length, frame-major [t][3][s][s].
  if (in.face_frames > offset) {
    const int count = std::min(in.face_frames - offset, n);
    const std::size_t plane =
        static_cast<std::size_t>(kAnimateFaceSize) * kAnimateFaceSize;
    p.face.resize(face_rgb * count);
    for (int f = 0; f < count; ++f) {
      const std::uint8_t* src =
          in.face.data() + static_cast<std::size_t>(offset + f) * face_rgb;
      float* dst = p.face.data() + static_cast<std::size_t>(f) * face_rgb;
      for (std::size_t i = 0; i < plane; ++i)
        for (int c = 0; c < 3; ++c)
          dst[c * plane + i] = static_cast<float>(src[3 * i + c]) / 127.5F -
                               1.0F;
    }
    p.face_count = count;
  }

  // Background behind the continue_motion frames.
  const int ref_images_num = std::max(0, ref_motion_latent_length * 4 - 3);
  if (in.background_frames > offset) {
    const int count = std::min(in.background_frames - offset, n);
    for (int f = ref_images_num; f < count; ++f)
      Widen(in.background, offset + f, rgb, p.condition.data() + f * rgb);
  }

  // mask_refmotion [latent_frames * 4, lh, lw]: 1 = generate; 0 on the
  // continue_motion latent frames; the character mask (nearest-exact to the
  // latent size) from ref_images_num on.
  const std::size_t lplane = static_cast<std::size_t>(lh) * lw;
  const int mask_rows = lf * 4;
  std::vector<float> refmotion(lplane * mask_rows, 1.0F);
  std::fill_n(refmotion.begin(),
              std::min<std::size_t>(lplane * ref_motion_latent_length * 4,
                                    refmotion.size()),
              0.0F);
  if (in.mask_frames > 0 &&
      (in.mask_frames > offset || in.mask_frames == 1)) {
    const bool still = in.mask_frames == 1;
    const int count = still ? n : std::min(in.mask_frames - offset, n);
    for (int f = ref_images_num; f < count; ++f) {
      const float* src =
          in.mask.data() + (still ? 0 : static_cast<std::size_t>(offset + f)) *
                               pixels;
      float* dst = refmotion.data() + static_cast<std::size_t>(f) * lplane;
      // nearest-exact: source index floor((i + 0.5) * 8).
      for (int y = 0; y < lh; ++y)
        for (int x = 0; x < lw; ++x)
          dst[static_cast<std::size_t>(y) * lw + x] =
              src[static_cast<std::size_t>(8 * y + 4) * w + 8 * x + 4];
    }
  }
  // view(T, 4) and transpose: channel c of latent frame t is row 4t + c;
  // the reference frame (mask 0) in front; the model reads 1 - mask.
  p.mask.assign(4 * (1 + lf) * lplane, 1.0F);
  for (int c = 0; c < 4; ++c)
    for (int t = 0; t < lf; ++t) {
      const float* src =
          refmotion.data() + static_cast<std::size_t>(4 * t + c) * lplane;
      float* dst = p.mask.data() +
                   (static_cast<std::size_t>(c) * (1 + lf) + 1 + t) * lplane;
      for (std::size_t i = 0; i < lplane; ++i)
        dst[i] = 1.0F - src[i];
    }

  p.trim_image = ref_images_num;
  p.video_frame_offset = offset + n;
  return true;
}

std::vector<AnimateChunkSlot> PlanAnimateChunks(int total_frames, int length,
                                                int continue_frames) {
  std::vector<AnimateChunkSlot> slots;
  if (total_frames < 1 || length < 5 || (length - 1) % 4 != 0 ||
      continue_frames < 1 || (continue_frames - 1) % 4 != 0 ||
      continue_frames >= length)
    return slots;
  const int keep = ((continue_frames - 1) / 4 + 1) * 4 - 3;
  slots.push_back({0, 0, 0});
  int covered = length;  // frames of the video produced so far
  int next_input = length;  // the last node's video_frame_offset output
  while (covered < total_frames) {
    AnimateChunkSlot slot;
    slot.video_frame_offset = next_input;
    slot.continue_frames = continue_frames;
    slot.keep_from = keep;
    const int offset = next_input - continue_frames;
    next_input = offset + length;
    covered = offset + length;
    slots.push_back(slot);
  }
  return slots;
}

bool AnimateFrameRateIsIdentity(const AnimateFrameRate& rate) {
  const bool same_rate =
      rate.target_num == 0 ||
      (rate.source_num > 0 && rate.source_den > 0 && rate.target_den > 0 &&
       rate.target_num * rate.source_den == rate.source_num * rate.target_den);
  return same_rate && rate.skip_first_frames == 0 &&
         rate.select_every_nth == 1 && rate.frame_load_cap == 0;
}

std::vector<int> AnimateFrameIndices(const AnimateFrameRate& rate,
                                     int max_frames, int source_frames) {
  std::vector<int> indices;
  constexpr std::int64_t kMaxRate = 10'000'000;
  if (rate.source_num <= 0 || rate.source_den <= 0 || rate.target_num < 0 ||
      rate.target_den <= 0 || rate.source_num > kMaxRate ||
      rate.source_den > kMaxRate || rate.target_num > kMaxRate ||
      rate.target_den > kMaxRate || rate.skip_first_frames < 0 ||
      rate.select_every_nth < 1 || rate.frame_load_cap < 0 || max_frames < 0 ||
      source_frames == 0)
    return indices;
  const std::int64_t target_num =
      rate.target_num == 0 ? rate.source_num : rate.target_num;
  const std::int64_t target_den =
      rate.target_num == 0 ? rate.source_den : rate.target_den;
  // Frame times in seconds as doubles, summed in VHS's order (Python
  // floats are doubles), so ties fall the same way: base_frame_time = 1 /
  // fps, target_frame_time = 1 / force_rate.
  const double base = 1.0 / (static_cast<double>(rate.source_num) /
                             static_cast<double>(rate.source_den));
  const double target = 1.0 / (static_cast<double>(target_num) /
                               static_cast<double>(target_den));
  // cv_frame_generator: the first frame is grabbed before the loop and
  // time_offset starts at one target frame time.
  int current = 0;
  double offset = target;
  int count = 0, evaluated = -1, added = 0;
  constexpr int kMaxSourceFrames = 10'000'000;
  while (static_cast<int>(indices.size()) < max_frames) {
    if (offset < target) {
      ++current;
      if ((source_frames > 0 && current >= source_frames) ||
          current > kMaxSourceFrames)
        break;
      offset += base;
    }
    if (offset < target)
      continue;
    offset -= target;
    ++count;
    if (count <= rate.skip_first_frames)
      continue;
    ++evaluated;
    if (evaluated % rate.select_every_nth != 0)
      continue;
    if (rate.frame_load_cap > 0 && added >= rate.frame_load_cap)
      break;
    indices.push_back(current);
    ++added;
  }
  return indices;
}

}  // namespace gufo::models::wan22
