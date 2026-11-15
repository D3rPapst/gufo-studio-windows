// MiniMax H3 provider requests on gufo 2.0 (gufo-studio-windows): the 1.4.9
// rules for LoRA, keyframes, guide frames, presets and provider options,
// through PrepareRequest / DecodePreparedRequest. CPU only.
// Written with AI assistance (Claude Code).
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "src/models/minimax_h3/request.hpp"

namespace {

int failures = 0;

void Check(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL minimax_h3_request_test: " << message << '\n';
    ++failures;
  }
}

// 1x1 RGB PNG.
constexpr const char* kPng =
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADEl"
    "EQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC";

bool Prepare(const std::string& model, const std::string& body,
             gufo::ext::VideoPreparedRequest* prepared,
             gufo::ext::VideoRequestError* error) {
  return gufo::minimax_h3::PrepareRequest(
      {.model = model, .body_json = body, .files = {}}, prepared, error);
}

}  // namespace

int main() {
  namespace h3 = gufo::minimax_h3;
  const auto dir = std::filesystem::temp_directory_path() / "gufo-h3-request-test";
  std::filesystem::create_directories(dir);
  const auto lora_file = dir / "turbo.safetensors";
  const auto vision_file = dir / "mmproj.gguf";
  std::ofstream(lora_file) << "x";
  std::ofstream(vision_file) << "x";

  h3::ProviderOptions options;
  std::string error;
  Check(h3::ParseProviderOptions(
            {{"lora", "turbo=" + lora_file.string() + ",steps=8"},
             {"vision-projector", vision_file.string()}},
            &options, &error),
        "provider options parse");
  Check(options.loras.size() == 1 && options.loras.at("turbo").steps == 8 &&
            options.vision_projector == vision_file,
        "LoRA steps and vision projector are kept");
  h3::ProviderOptions rejected;
  Check(!h3::ParseProviderOptions({{"controlnet", "x"}}, &rejected, &error),
        "unknown provider options fail");
  Check(!h3::ParseProviderOptions({{"lora", "turbo=" + lora_file.string() +
                                                ",speed=2"}},
                                  &rejected, &error),
        "unknown LoRA keys fail");
  h3::SetActiveProviderOptions(options);

  // FilmStudio-style request: 832x512, 4 s, Turbo LoRA (its 8 steps), a first
  // frame and a guide frame at the end.
  gufo::ext::VideoPreparedRequest prepared;
  gufo::ext::VideoRequestError failure;
  const std::string body =
      std::string(R"({"model":"minimax-h3","prompt":"a shot","size":"832x512",)"
                  R"("seconds":"4","first_frame":")") +
      kPng + R"(","gufo":{"lora":"turbo","seed":3,"keyframes":[{"frame":-1,"image":")" +
      kPng + R"("}]}})";
  Check(Prepare("minimax-h3", body, &prepared, &failure),
        "keyframes + LoRA request is accepted");
  Check(prepared.parameters.preset == "exact-832x512",
        "exact at 832x512 becomes exact-832x512");
  Check(prepared.parameters.evaluations == 8,
        "the LoRA's registered steps replace the preset's");
  Check(prepared.seed == 3 && prepared.attachments.size() == 2,
        "seed and both images reach the job");
  Check(prepared.options_json.find("\"lora\":\"turbo\"") != std::string::npos,
        "the LoRA name is in the job options");

  gufo::ext::VideoRequest request;
  request.parameters = prepared.parameters;
  request.options_json = prepared.options_json;
  request.attachments = prepared.attachments;
  h3::GenerationRequest generation;
  Check(h3::DecodePreparedRequest(request, options, dir, &generation, &error),
        "the prepared job decodes");
  Check(generation.keyframes.size() == 2 &&
            generation.keyframes[0].vision &&
            generation.keyframes[0].anchor == h3::KeyframeAnchor::kFirst &&
            !generation.keyframes[1].vision &&
            generation.keyframes[1].frame == -1 &&
            generation.keyframes[0].width == 1,
        "first frame (vision) and guide frame (no vision) are decoded");
  Check(generation.lora_path == lora_file &&
            generation.vision_projector == vision_file,
        "LoRA file and vision projector reach the generation");

  gufo::ext::VideoPreparedRequest unknown;
  Check(!Prepare("minimax-h3",
                 R"({"model":"minimax-h3","prompt":"x","seconds":"5","gufo":{"lora":"nope"}})",
                 &unknown, &failure) &&
            failure.message.find("unknown LoRA") != std::string::npos,
        "an unregistered LoRA is refused at creation");

  gufo::ext::VideoPreparedRequest fast;
  Check(Prepare("minimax-h3-fast",
                R"({"model":"minimax-h3-fast","prompt":"x","size":"512x512","seconds":"5"})",
                &fast, &failure) &&
            fast.parameters.preset == "fast-384",
        "preset-selecting model names still select their preset");
  gufo::ext::VideoPreparedRequest mismatch;
  Check(!Prepare("minimax-h3-fast",
                 R"({"model":"minimax-h3-fast","prompt":"x","seconds":"5","gufo":{"preset":"dev"}})",
                 &mismatch, &failure) &&
            failure.code == "preset_model_mismatch",
        "model alias and gufo.preset must agree");

  gufo::ext::VideoPreparedRequest twice;
  const std::string same_frame =
      std::string(R"({"model":"minimax-h3","prompt":"x","seconds":"5","first_frame":")") +
      kPng + R"(","gufo":{"keyframes":[{"frame":0,"image":")" + kPng + R"("}]}})";
  Check(!Prepare("minimax-h3", same_frame, &twice, &failure) &&
            failure.code == "invalid_keyframes",
        "two images on one frame fail");

  // MiniMaxH3SigmaShift: gufo.sigma_shift reaches the generation; the
  // trained 12 / 3 stay out of the job options.
  gufo::ext::VideoPreparedRequest shifted;
  Check(Prepare("minimax-h3",
                R"({"model":"minimax-h3","prompt":"x","seconds":"5","gufo":{"sigma_shift":{"audio":6.5}}})",
                &shifted, &failure),
        "gufo.sigma_shift is accepted");
  gufo::ext::VideoRequest shifted_request;
  shifted_request.parameters = shifted.parameters;
  shifted_request.options_json = shifted.options_json;
  h3::GenerationRequest shifted_generation;
  Check(h3::DecodePreparedRequest(shifted_request, options, dir,
                                  &shifted_generation, &error) &&
            shifted_generation.audio_sigma_shift == 6.5F &&
            shifted_generation.video_sigma_shift == 12.0F,
        "the audio shift reaches the generation, the video shift stays 12");
  gufo::ext::VideoPreparedRequest trained;
  Check(Prepare("minimax-h3",
                R"({"model":"minimax-h3","prompt":"x","seconds":"5","gufo":{"sigma_shift":{"video":12,"audio":3}}})",
                &trained, &failure) &&
            trained.options_json.find("sigma_shift") == std::string::npos,
        "the trained shifts add no job option");
  for (const char* bad : {R"({"audio":0})", R"({"video":101})",
                          R"({"audio":"3"})", R"({"text":3})", "3"}) {
    gufo::ext::VideoPreparedRequest refused;
    Check(!Prepare("minimax-h3",
                   std::string(R"({"model":"minimax-h3","prompt":"x","seconds":"5","gufo":{"sigma_shift":)") +
                       bad + "}}",
                   &refused, &failure) &&
              failure.code == "invalid_sigma_shift",
          "an invalid gufo.sigma_shift is refused");
  }

  // Audio noise mask: {"audio", "mask"} reaches the generation as the
  // decoded source and the mask rows.
  const std::string silence =
      "data:audio/wav;base64,UklGRqQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YYAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==";
  gufo::ext::VideoPreparedRequest masked;
  Check(Prepare("minimax-h3",
                std::string(R"({"model":"minimax-h3","prompt":"x","seconds":"5","gufo":{"audio_mask":{"audio":")") +
                    silence + R"(","mask":[0,0,1,0]}}})",
                &masked, &failure),
        "gufo.audio_mask is accepted");
  gufo::ext::VideoRequest masked_request;
  masked_request.parameters = masked.parameters;
  masked_request.options_json = masked.options_json;
  masked_request.attachments = masked.attachments;
  h3::GenerationRequest masked_generation;
  Check(h3::DecodePreparedRequest(masked_request, options, dir,
                                  &masked_generation, &error) &&
            masked_generation.audio_mask_source.has_value() &&
            masked_generation.audio_mask.size() == 4 &&
            masked_generation.audio_mask[2] == 1.0F,
        "the audio mask and its source reach the generation");
  for (const char* bad : {R"([0,0,0])", R"([0,2])", R"([])", R"("0")"}) {
    gufo::ext::VideoPreparedRequest refused;
    Check(!Prepare("minimax-h3",
                   std::string(R"({"model":"minimax-h3","prompt":"x","seconds":"5","gufo":{"audio_mask":{"audio":")") +
                       silence + R"(","mask":)" + bad + "}}}",
                   &refused, &failure) &&
              failure.code == "invalid_audio_mask",
          "an invalid gufo.audio_mask.mask is refused");
  }
  gufo::ext::VideoPreparedRequest no_audio;
  Check(!Prepare("minimax-h3",
                 R"({"model":"minimax-h3","prompt":"x","seconds":"5","gufo":{"audio_mask":{"mask":[0,0]}}})",
                 &no_audio, &failure) &&
            failure.code == "invalid_audio_mask",
        "an audio mask without audio is refused");

  std::error_code ignored;
  std::filesystem::remove_all(dir, ignored);
  if (failures == 0)
    std::cout << "MiniMax H3 request tests passed.\n";
  return failures == 0 ? 0 : 1;
}
