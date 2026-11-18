// gufo.guidance_steps (CFG on the first N steps only) through the provider's
// BuildRequest: accepted values, the default and the HTTP 400 cases
// (std::invalid_argument). No model, no GPU. gufo-studio-windows 2.2.3.

#include <iostream>
#include <stdexcept>
#include <string>

#include "src/models/qwen_image_21/model.hpp"
#include "src/models/qwen_image_21/request.hpp"

namespace {

using namespace gufo::models::qwen_image_21;

int failures = 0;

void Check(bool ok, const std::string& what) {
  if (!ok) {
    std::cerr << "FAIL: " << what << '\n';
    ++failures;
  }
}

Request Build(const std::string& extension_json) {
  gufo::ext::ImageRequest in;
  in.prompt = "a fox";
  in.width = 256;
  in.height = 256;
  in.extension_json = extension_json;
  return BuildRequest(in, ProviderState{});
}

bool Rejected(const std::string& extension_json) {
  try {
    (void)Build(extension_json);
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

}  // namespace

int main() {
  // Missing: every step guided (the 2.2.2 behaviour).
  Check(Build("").guidance_steps == 0, "default without extension");
  Check(Build(R"({"gufo":{"guidance":4}})").guidance_steps == 0,
        "default with guidance");
  const auto set = Build(
      R"({"gufo":{"guidance":4,"negative_prompt":"blur","guidance_steps":6}})");
  Check(set.guidance_steps == 6 && set.guidance == 4.0F &&
            set.negative_prompt == "blur",
        "guidance_steps 6");
  Check(Build(R"({"gufo":{"guidance":4,"guidance_steps":0}})")
                .guidance_steps == 0,
        "explicit 0");
  Check(Build(R"({"gufo":{"guidance":4,"guidance_steps":100}})")
                .guidance_steps == 100,
        "upper bound 100");
  // Without CFG the field is accepted and has no effect (like
  // negative_prompt at guidance 1).
  Check(Build(R"({"gufo":{"guidance_steps":3}})").guidance_steps == 3,
        "accepted at guidance 1");
  for (const char* bad :
       {R"({"gufo":{"guidance_steps":-1}})",
        R"({"gufo":{"guidance_steps":1.5}})",
        R"({"gufo":{"guidance_steps":101}})",
        R"({"gufo":{"guidance_steps":"4"}})",
        R"({"gufo":{"guidance_steps":true}})",
        R"({"gufo":{"guidance_steps":null}})",
        R"({"gufo":{"guidance_step":4}})"})
    Check(Rejected(bad), std::string("rejected: ") + bad);
  if (failures) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "qwen_image_21_guidance_steps_test: all passed\n";
  return 0;
}
