// Small ONNX graph runtime for the PaddleOCR PP-OCRv4 exports of the
// /v1/vision/detect "ocr" task (reference/convert_ocr.py writes the node list
// and the weights). CPU fp32, NCHW, OpenMP; only the ops of the two exports
// (DBNet detector with PP-LCNetV3 backbone, SVTR-LCNet recognizer) are
// implemented, the converter refuses anything else. Not a general ONNX
// runtime: one input, one output, batch 1 in practice (N > 1 works where the
// ops allow it). Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_CCIP_CPU_OCR_NET_HPP_
#define GUFO_MODELS_CCIP_CPU_OCR_NET_HPP_

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace gufo::models::ccip::cpu {

// A dense tensor: float data, or int64 data for shape arithmetic.
struct NetTensor {
  std::vector<std::int64_t> shape;
  std::vector<float> f;
  std::vector<std::int64_t> i;
  bool is_int{false};
  [[nodiscard]] std::int64_t numel() const;
};

class OcrNet {
public:
  virtual ~OcrNet() = default;
  // `dir` holds <stem>.safetensors + <stem>.json from reference/convert_ocr.py.
  [[nodiscard]] static std::unique_ptr<OcrNet> Load(const std::filesystem::path& dir,
                                                    const std::string& stem, std::string* error);
  // Runs the graph on one float input; returns the single output. Throws
  // std::runtime_error on shape errors. Thread-safe (no mutable state).
  [[nodiscard]] virtual NetTensor Run(NetTensor input) const = 0;
};

}  // namespace gufo::models::ccip::cpu

#endif  // GUFO_MODELS_CCIP_CPU_OCR_NET_HPP_
