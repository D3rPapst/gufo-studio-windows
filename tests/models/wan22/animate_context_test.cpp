// Written with AI assistance (Claude Code).
// Host-side Wan 2.2 Animate conditioning against ComfyUI's
// WanAnimateToVideo (comfy_extras/nodes_wan.py) and WAN21.concat_cond: the
// node's outputs, the frame seeks, the continue_motion / background / mask
// layout and the chunk chain of long-video workflows.
#include "src/models/wan22/animate_context.hpp"
#include "src/models/wan22/animate_host.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

namespace wan22 = gufo::models::wan22;

namespace {

int failures = 0;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL wan22_animate_context_test: " << message << '\n';
    ++failures;
  }
}

bool Near(float a, float b) {
  return std::fabs(a - b) <= 1e-6F;
}

constexpr int kH = 16, kW = 32;  // latent 2 x 4
constexpr std::size_t kRgb = static_cast<std::size_t>(kH) * kW * 3;
constexpr std::size_t kPlane = 2 * 4;

// `frames` frames, frame f filled with value base + f.
std::vector<std::uint8_t> Video(int frames, int base,
                                std::size_t values = kRgb) {
  std::vector<std::uint8_t> video(values * frames);
  for (int f = 0; f < frames; ++f)
    for (std::size_t i = 0; i < values; ++i)
      video[f * values + i] = static_cast<std::uint8_t>(base + f);
  return video;
}

float Wide(int byte) {
  return static_cast<float>(byte) / 127.5F - 1.0F;
}

// Mask channel c of latent frame t (t = 0 is the reference frame).
float MaskAt(const wan22::AnimateChunkPlan& plan, int c, int t,
             std::size_t i = 0) {
  return plan.mask[(static_cast<std::size_t>(c) * (1 + plan.latent_frames) +
                    t) *
                       kPlane +
                   i];
}

void PlainAnimation() {
  // Pose shorter than length (last frame repeated), face cut to length, no
  // continue_motion: grey condition, every video latent generated.
  const auto pose = Video(7, 10);
  const auto face = Video(11, 50, wan22::kAnimateFaceSize *
                                      wan22::kAnimateFaceSize * 3);
  wan22::AnimateChunkInputs in;
  in.length = 9;
  in.height = kH;
  in.width = kW;
  in.pose = pose;
  in.pose_frames = 7;
  in.face = face;
  in.face_frames = 11;
  wan22::AnimateChunkPlan plan;
  Expect(wan22::BuildAnimateChunk(in, &plan), "plain chunk builds");
  Expect(plan.latent_frames == 3 && plan.trim_latent == 1 &&
             plan.trim_image == 0 && plan.video_frame_offset == 9 &&
             plan.continue_frames == 0,
         "outputs: 3 latent frames, trim_latent 1, trim_image 0, offset 9");
  Expect(plan.condition.size() == kRgb * 9 && Near(plan.condition[0], 0.0F) &&
             Near(plan.condition.back(), 0.0F),
         "condition is grey (0 in [-1, 1])");
  Expect(plan.pose.size() == kRgb * 9 && Near(plan.pose[0], Wide(10)) &&
             Near(plan.pose[6 * kRgb], Wide(16)) &&
             Near(plan.pose[8 * kRgb], Wide(16)),
         "pose repeats its last frame up to length");
  Expect(plan.face_count == 9 &&
             plan.face.size() == static_cast<std::size_t>(9) * 3 *
                                     wan22::kAnimateFaceSize *
                                     wan22::kAnimateFaceSize &&
             Near(plan.face[0], Wide(50)) &&
             Near(plan.face.back(), Wide(58)),
         "face cut to length, channel-major frames");
  bool mask_ok = plan.mask.size() == 4 * 4 * kPlane;
  for (int c = 0; c < 4; ++c) {
    mask_ok = mask_ok && Near(MaskAt(plan, c, 0), 1.0F);
    for (int t = 1; t < 4; ++t)
      mask_ok = mask_ok && Near(MaskAt(plan, c, t), 0.0F);
  }
  Expect(mask_ok, "mask: reference frame known, video generated");
}

void SeekAndDrop() {
  // video_frame_offset beyond the face video drops it (ComfyUI sets it to
  // None); pose from the offset on.
  const auto pose = Video(20, 0);
  const auto face = Video(3, 0, wan22::kAnimateFaceSize *
                                    wan22::kAnimateFaceSize * 3);
  wan22::AnimateChunkInputs in;
  in.length = 5;
  in.height = kH;
  in.width = kW;
  in.video_frame_offset = 12;
  in.pose = pose;
  in.pose_frames = 20;
  in.face = face;
  in.face_frames = 3;
  wan22::AnimateChunkPlan plan;
  Expect(wan22::BuildAnimateChunk(in, &plan), "seek chunk builds");
  Expect(Near(plan.pose[0], Wide(12)) && Near(plan.pose[4 * kRgb], Wide(16)),
         "pose seeks to video_frame_offset");
  Expect(plan.face.empty() && plan.face_count == 0,
         "face shorter than the offset is dropped");
  Expect(plan.video_frame_offset == 17, "next offset = offset + length");
}

void ContinueReplace() {
  // continue_motion 9 frames (max 5 used), background + a still character
  // mask: the seek moves back 5, trim_image 5, two latent frames known.
  const auto previous = Video(9, 100);
  const auto background = Video(30, 200);
  std::vector<float> mask(static_cast<std::size_t>(kH) * kW, 0.0F);
  // Character in the right half of the picture: latent columns 2, 3.
  for (int y = 0; y < kH; ++y)
    for (int x = kW / 2; x < kW; ++x)
      mask[static_cast<std::size_t>(y) * kW + x] = 1.0F;
  wan22::AnimateChunkInputs in;
  in.length = 13;
  in.height = kH;
  in.width = kW;
  in.video_frame_offset = 13;
  in.continue_motion = previous;
  in.continue_frames = 9;
  in.background = background;
  in.background_frames = 30;
  in.mask = mask;
  in.mask_frames = 1;
  wan22::AnimateChunkPlan plan;
  Expect(wan22::BuildAnimateChunk(in, &plan), "continue chunk builds");
  Expect(plan.offset == 8 && plan.continue_frames == 5 &&
             plan.ref_motion_latent_length == 2 && plan.trim_image == 5 &&
             plan.video_frame_offset == 21,
         "continue: offset 13 - 5 = 8, trim_image 5, next offset 21");
  Expect(Near(plan.condition[0], Wide(104)) &&
             Near(plan.condition[4 * kRgb], Wide(108)),
         "condition starts with the last 5 continue frames");
  Expect(Near(plan.condition[5 * kRgb], Wide(213)) &&
             Near(plan.condition[12 * kRgb], Wide(220)),
         "background (seeked) behind the continue frames");
  // Rows 0..7 (two latent frames) are zeroed, then ComfyUI writes the
  // character mask from row ref_images_num = 5 on: rows 5..7 (channels 1..3
  // of the second video latent) follow the mask again.
  bool known = true;
  for (int c = 0; c < 4; ++c)
    for (int t = 0; t <= 1; ++t)
      for (std::size_t i = 0; i < kPlane; ++i)
        known = known && Near(MaskAt(plan, c, t, i), 1.0F);
  for (std::size_t i = 0; i < kPlane; ++i)
    known = known && Near(MaskAt(plan, 0, 2, i), 1.0F);
  Expect(known, "reference, first continue latent and row 4 are known");
  Expect(Near(MaskAt(plan, 1, 2, 0), 1.0F) && Near(MaskAt(plan, 1, 2, 3), 0.0F),
         "rows 5..7 follow the character mask (as ComfyUI writes them)");
  // Latent frame 3 (rows 8..11): background columns known, character
  // columns generated; rows 13..15 lie past length (generated).
  Expect(Near(MaskAt(plan, 0, 3, 0), 1.0F) && Near(MaskAt(plan, 0, 3, 1), 1.0F) &&
             Near(MaskAt(plan, 0, 3, 2), 0.0F) &&
             Near(MaskAt(plan, 0, 4, 4), 1.0F) &&
             Near(MaskAt(plan, 0, 4, 7), 0.0F) &&
             Near(MaskAt(plan, 3, 4, 4), 0.0F),
         "character mask: character generated, background kept");
}

void MaskVideoOffset() {
  // A mask video shorter than the seek is ignored (everything generated).
  std::vector<float> mask(static_cast<std::size_t>(kH) * kW * 2, 0.0F);
  wan22::AnimateChunkInputs in;
  in.length = 5;
  in.height = kH;
  in.width = kW;
  in.video_frame_offset = 4;
  in.mask = mask;
  in.mask_frames = 2;
  wan22::AnimateChunkPlan plan;
  Expect(wan22::BuildAnimateChunk(in, &plan), "mask chunk builds");
  Expect(Near(MaskAt(plan, 2, 1), 0.0F) && Near(MaskAt(plan, 1, 2, 5), 0.0F),
         "mask video before the seek is ignored");
}

void Invalid() {
  wan22::AnimateChunkPlan plan;
  wan22::AnimateChunkInputs in;
  in.length = 8;
  in.height = kH;
  in.width = kW;
  Expect(!wan22::BuildAnimateChunk(in, &plan), "length 8 is rejected");
  in.length = 5;
  in.width = 20;
  Expect(!wan22::BuildAnimateChunk(in, &plan), "width 20 is rejected");
  in.width = kW;
  const auto pose = Video(2, 0);
  in.pose = pose;
  in.pose_frames = 3;
  Expect(!wan22::BuildAnimateChunk(in, &plan), "pose size mismatch rejected");
}

void Chunks() {
  // 77-frame chunks with 5 frames of overlap over 200 frames.
  const auto slots = wan22::PlanAnimateChunks(200, 77, 5);
  Expect(slots.size() == 3, "200 frames take three chunks");
  if (slots.size() == 3) {
    Expect(slots[0].video_frame_offset == 0 && slots[0].continue_frames == 0 &&
               slots[0].keep_from == 0,
           "first chunk starts at 0 without continue");
    Expect(slots[1].video_frame_offset == 77 && slots[1].continue_frames == 5 &&
               slots[1].keep_from == 5,
           "second chunk: node offset 77 (seek 72), drops 5 frames");
    Expect(slots[2].video_frame_offset == 149,
           "third chunk: node offset 72 + 77 = 149");
  }
  // Kept frames: 77 + 72 + 72 >= 200.
  Expect(wan22::PlanAnimateChunks(77, 77, 5).size() == 1,
         "one chunk covers its length");
  Expect(wan22::PlanAnimateChunks(78, 77, 1).size() == 2 &&
             wan22::PlanAnimateChunks(78, 77, 1)[1].keep_from == 1,
         "overlap 1 drops one frame");
  Expect(wan22::PlanAnimateChunks(100, 77, 77).empty() &&
             wan22::PlanAnimateChunks(100, 76, 5).empty() &&
             wan22::PlanAnimateChunks(100, 77, 4).empty(),
         "invalid chunk arguments are rejected");
}

void Directions() {
  // numpy.linalg.qr([[1, 2], [3, 4], [5, 6]]) (LAPACK signs).
  std::vector<double> q;
  wan22::HouseholderQ({1, 2, 3, 4, 5, 6}, 3, 2, &q);
  const double expected[] = {-0.16903085, 0.89708523, -0.50709255,
                             0.27602622,  -0.84515425, -0.34503278};
  bool ok = q.size() == 6;
  for (int i = 0; ok && i < 6; ++i)
    ok = std::fabs(q[i] - expected[i]) < 1e-7;
  Expect(ok, "Householder Q matches LAPACK's signs");
  // Motion head: one linear, directions = the first identity columns (no
  // reflection when the column below the diagonal is zero: Q = I columns).
  wan22::MotionHead head;
  head.width = 3;
  head.motion = 2;
  head.weights = {{1, 0, 0, 0, 1, 0}};
  head.biases = {{0.5F, -1.0F}};
  wan22::SetMotionDirections(std::vector<float>{1, 0, 0, 1, 0, 0}, &head);
  const auto motion = wan22::MotionVector(head, std::vector<float>{2, 3, 4});
  Expect(motion.size() == 3 && Near(motion[0], 2.5F) &&
             Near(motion[1], 2.0F) && std::fabs(motion[2]) < 1e-6F,
         "motion vector = fc(x) @ Q^T");
}

void FaceTokens() {
  // Zero convolutions: every layer norm sees a constant row -> 0, so the
  // tokens are out_proj's bias; frame 0 zero, padding token last, frames
  // past the encoder output zero.
  wan22::FaceEncoderWeights w;
  w.in_dim = 2;
  w.heads = 2;
  w.channels = 2;
  w.width = 3;
  w.conv1_weight.assign(4 * 2 * 3, 0.0F);
  w.conv1_bias = {1, 2, 3, 4};
  w.conv2_weight.assign(2 * 2 * 3, 0.0F);
  w.conv2_bias = {5, 6};
  w.conv3_weight = w.conv2_weight;
  w.conv3_bias = {7, 8};
  w.out_weight.assign(3 * 2, 0.0F);
  w.out_bias = {0.25F, 0.5F, 0.75F};
  w.padding = {9, 10, 11};
  Expect(wan22::FaceEncoderFrames(77) == 20 &&
             wan22::FaceEncoderFrames(5) == 2 &&
             wan22::FaceEncoderFrames(1) == 1,
         "face encoder frames: 77 -> 20, 5 -> 2");
  std::vector<float> motion(5 * 2, 1.0F);
  const auto tokens = wan22::RunFaceEncoder(w, motion, 5, 4);
  Expect(tokens.size() == 4 * 3 * 3, "tokens [4, 3, 3]");
  if (tokens.size() == 36) {
    const auto at = [&](int f, int n, int c) { return tokens[(f * 3 + n) * 3 + c]; };
    Expect(Near(at(0, 0, 0), 0) && Near(at(0, 2, 2), 0), "frame 0 is zero");
    Expect(Near(at(1, 0, 1), 0.5F) && Near(at(2, 1, 2), 0.75F) &&
               Near(at(1, 2, 0), 9) && Near(at(2, 2, 2), 11),
           "encoder frames: head tokens then the padding token");
    Expect(Near(at(3, 0, 0), 0) && Near(at(3, 2, 0), 0),
           "frames past the encoder output are zero");
  }
  Expect(wan22::RunFaceEncoder(w, motion, 4, 4).empty(),
         "size mismatch is rejected");
}

}  // namespace

int main() {
  PlainAnimation();
  SeekAndDrop();
  ContinueReplace();
  MaskVideoOffset();
  Invalid();
  Chunks();
  Directions();
  FaceTokens();
  if (failures == 0)
    std::cout << "wan22_animate_context_test: ok\n";
  return failures == 0 ? 0 : 1;
}
