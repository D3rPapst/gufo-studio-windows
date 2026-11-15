#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>

#include "src/models/qwen3_asr/audio.hpp"
#include "src/models/qwen3_asr/transcription.hpp"
#include "tests/support/model_api_test_support.hpp"

namespace {

namespace qwen3_asr = gufo::models::qwen3_asr;
using gufo::server::AsrService;
using gufo::server::AsrServiceOptions;
using gufo::server::HandleAudioAsrApiRequest;
using gufo::server::HttpRequest;
using gufo::server::HttpResponse;

[[noreturn]] void Fail(const std::string& message) {
  std::cerr << "FAIL qwen3_asr_audio_api_test: " << message << '\n';
  std::exit(1);
}

void Check(bool condition, const std::string& message) {
  if (!condition) {
    Fail(message);
  }
}

class FakeRunner {
public:
  bool Run(const qwen3_asr::TranscriptionRequest& request,
           const qwen3_asr::CancellationCheck&,
           qwen3_asr::TranscriptionResult* result, std::string* error) {
    ++calls;
    wav_bytes = request.wav.size();
    context = request.context;
    language = request.language;
    maximum_tokens = request.max_new_tokens;
    word_timestamps = request.word_timestamps;
    result->language = "English";
    if (request.word_timestamps && words)
      result->words = {{.word = " A", .start = 0.0, .end = 0.32},
                       {.word = " native", .start = 0.32, .end = 0.8},
                       {.word = " transcript.", .start = 0.8, .end = 1.46}};
    result->text = "A native transcript.";
    result->audio_samples = 32000;
    result->audio_tokens = 25;
    result->generated_ids = {1, 2, 3};
    result->timings.audio_encoder_ms = 10.0;
    result->timings.text_decoder_ms = 20.0;
    if (request.on_text && !request.on_text("A native")) {
      if (error != nullptr)
        *error = "cancelled";
      return false;
    }
    return true;
  }

  int calls{0};
  std::size_t wav_bytes{0};
  std::string context;
  std::optional<std::string> language;
  std::size_t maximum_tokens{0};
  bool word_timestamps{false};
  bool words{true};  // the fake model has word timing
};

AsrService MakeService(FakeRunner* runner) {
  return AsrService(AsrServiceOptions{
      .model_root = {},
      .validate_model = false,
      .model_id = "qwen3-asr-1.7b",
      .runner =
          [runner](const qwen3_asr::TranscriptionRequest& request,
                   const qwen3_asr::CancellationCheck& cancelled,
                   qwen3_asr::TranscriptionResult* result, std::string* error) {
            return runner->Run(request, cancelled, result, error);
          },
  });
}

std::string Multipart(std::string response_format = "json",
                      std::string_view extra_field = {},
                      std::string_view stream = {}) {
  constexpr std::string_view boundary = "gufo-test-boundary";
  std::string body;
  const auto append = [&](std::string_view name, std::string_view value,
                          std::string_view filename = {}) {
    body += "--";
    body += boundary;
    body += "\r\nContent-Disposition: form-data; name=\"";
    body += name;
    body += '"';
    if (!filename.empty()) {
      body += "; filename=\"";
      body += filename;
      body += '"';
    }
    body += "\r\n\r\n";
    body.append(value.data(), value.size());
    body += "\r\n";
  };
  append("file", "RIFFfake", "sample.wav");
  append("model", "qwen3-asr");
  append("language", "en");
  append("prompt", "Names: Gufo.");
  append("response_format", response_format);
  append("max_tokens", "42");
  append("temperature", "0");
  if (!stream.empty())
    append("stream", stream);
  if (!extra_field.empty()) {
    append(extra_field, "logprobs");
  }
  body += "--";
  body += boundary;
  body += "--\r\n";
  return body;
}

HttpResponse Send(AsrService& service, std::string body,
                  std::string content_type =
                      "multipart/form-data; boundary=gufo-test-boundary") {
  return HandleAudioAsrApiRequest(
      HttpRequest{
          .method = "POST",
          .path = "/v1/audio/transcriptions",
          .query = {},
          .body = std::move(body),
          .headers = {{"Content-Type", std::move(content_type)}},
          .is_cancelled = {},
      },
      service);
}

bool HasHeader(const HttpResponse& response, std::string_view name,
               std::string_view value) {
  return std::ranges::any_of(response.headers, [&](const auto& header) {
    return header.first == name && header.second == value;
  });
}

void TestJsonRequest() {
  FakeRunner runner;
  AsrService service = MakeService(&runner);
  const HttpResponse response = Send(service, Multipart());
  Check(response.status == 200 &&
            response.body.find("\"text\":\"A native transcript.\"") !=
                std::string::npos,
        "valid multipart request returns OpenAI JSON");
  Check(runner.calls == 1 && runner.wav_bytes == 8U &&
            runner.context == "Names: Gufo." &&
            runner.language == std::optional<std::string>("en") &&
            runner.maximum_tokens == 42U,
        "multipart fields reach the model contract");
  Check(HasHeader(response, "X-Gufo-ASR-Backend", "native-hip"),
        "native backend is exposed");
  Check(response.log_details.find("encoder_ms=") != std::string::npos &&
            response.log_details.find("decoder_ms=") != std::string::npos &&
            response.log_details.find("A native transcript.") ==
                std::string::npos &&
            response.log_details.find("Names: Gufo.") == std::string::npos,
        "transcription diagnostics report work without transcript or context");
}

void TestFormatsAndValidation() {
  FakeRunner runner;
  AsrService service = MakeService(&runner);
  const HttpResponse text = Send(service, Multipart("text"));
  Check(text.status == 200 && text.body == "A native transcript." &&
            HasHeader(text, "Content-Type", "text/plain; charset=utf-8"),
        "text response format is supported");
  const HttpResponse verbose = Send(service, Multipart("verbose_json"));
  Check(verbose.status == 200 &&
            verbose.body.find("\"duration\":2") != std::string::npos &&
            verbose.body.find("\"language\":\"english\"") != std::string::npos,
        "verbose_json exposes duration and language");
  Check(Send(service, Multipart("srt")).status == 400,
        "unsupported response formats fail closed");
  Check(Send(service, Multipart("json", "typo")).status == 200,
        "unknown multipart fields are ignored");
  Check(Send(service, Multipart("json", "include[]")).status == 400,
        "logprobs requests fail closed");
  Check(Send(service, Multipart(), "application/json").status == 400,
        "non-multipart requests fail closed");
  Check(runner.calls == 3, "invalid requests never invoke inference");
}

void TestStreamingAndChunks() {
  FakeRunner runner;
  AsrService service = MakeService(&runner);
  const auto response = Send(service, Multipart("json", {}, "true"));
  Check(response.status == 200 && response.streaming_body && runner.calls == 0,
        "streaming defers inference until the response is consumed");
  std::string events;
  response.streaming_body([&](std::string_view part) {
    events += part;
    return true;
  });
  Check(
      runner.calls == 1 &&
          events.find("\"delta\":\"A native\"") != std::string::npos &&
          events.find("\"delta\":\" transcript.\"") != std::string::npos &&
          events.find("\"type\":\"transcript.text.done\"") != std::string::npos,
      "OpenAI SSE emits incremental deltas and a complete transcript");
  const auto disconnected = Send(service, Multipart("json", {}, "true"));
  int writes = 0;
  disconnected.streaming_body([&](std::string_view) {
    ++writes;
    return false;
  });
  Check(writes == 1 && disconnected.stream_log->error_code == "cancelled",
        "disconnected stream cancels inference without a done event");
  Check(!Send(service, Multipart("json", {}, "false")).streaming_body,
        "stream=false retains the ordinary JSON contract");
  Check(Send(service, Multipart("json", {}, "yes")).status == 400 &&
            Send(service, Multipart("text", {}, "true")).status == 400,
        "invalid streaming configurations fail closed");

  constexpr std::size_t limit = 4 * qwen3_asr::kAudioSampleRate;
  const auto short_chunk =
      qwen3_asr::SplitAudio(std::vector<float>(1600), 1600);
  Check(short_chunk.size() == 1 && short_chunk.front().samples == 1600,
        "a short unsplit input does not require a half-second chunk budget");
  std::vector<float> waveform(limit * 2 + 137, 0.25F);
  std::fill(waveform.begin() + 3 * qwen3_asr::kAudioSampleRate,
            waveform.begin() + 3 * qwen3_asr::kAudioSampleRate + 1600, 0.0F);
  const auto chunks = qwen3_asr::SplitAudio(waveform, limit);
  std::size_t consumed = 0;
  for (const auto& chunk : chunks) {
    Check(
        chunk.offset == consumed && chunk.samples > 0 && chunk.samples <= limit,
        "long audio partitions exactly within the hard capacity");
    consumed += chunk.samples;
  }
  Check(consumed == waveform.size() &&
            chunks.front().samples == 3 * qwen3_asr::kAudioSampleRate,
        "chunking prefers a quiet boundary without dropping samples");
  for (const std::size_t budget : {13U, 104U, 725U}) {
    const auto samples = qwen3_asr::AudioSamplesForTokenBudget(budget);
    Check(qwen3_asr::AudioEmbeddingTokenCount(samples / 160) <= budget &&
              qwen3_asr::AudioEmbeddingTokenCount((samples + 1) / 160) > budget,
          "audio capacity includes convolutional tails");
  }
}

// A verbose_json request with the given repeated granularity fields.
std::string GranularityMultipart(std::initializer_list<std::string_view> values,
                                 std::string_view format = "verbose_json") {
  std::string body;
  const auto append = [&](std::string_view name, std::string_view value,
                          std::string_view filename = {}) {
    body += "--gufo-test-boundary\r\nContent-Disposition: form-data; name=\"";
    body += name;
    body += '"';
    if (!filename.empty()) {
      body += "; filename=\"";
      body += filename;
      body += '"';
    }
    body += "\r\n\r\n";
    body += value;
    body += "\r\n";
  };
  append("file", "RIFFfake", "sample.wav");
  append("response_format", format);
  for (const auto value : values)
    append("timestamp_granularities[]", value);
  body += "--gufo-test-boundary--\r\n";
  return body;
}

void TestTimestampGranularities() {
  FakeRunner runner;
  AsrService service = MakeService(&runner);
  const HttpResponse words = Send(service, GranularityMultipart({"word", "segment"}));
  Check(words.status == 200 && runner.word_timestamps &&
            words.body.find("\"words\":[{\"word\":\"A\",\"start\":0,\"end\":0.32}") !=
                std::string::npos &&
            words.body.find("{\"word\":\"transcript.\",\"start\":0.8,\"end\":1.46}") !=
                std::string::npos &&
            words.body.find("\"segments\":[]") != std::string::npos,
        "word granularity lists trimmed OpenAI words beside segments");
  const HttpResponse segments = Send(service, GranularityMultipart({"segment"}));
  Check(segments.status == 200 && !runner.word_timestamps &&
            segments.body.find("\"words\"") == std::string::npos,
        "segment granularity alone asks for no words");
  runner.words = false;
  const HttpResponse none = Send(service, GranularityMultipart({"word"}));
  Check(none.status == 200 && none.body.find("\"words\":[]") != std::string::npos,
        "a model without word timing answers an empty word list");
  const int calls = runner.calls;
  Check(Send(service, GranularityMultipart({"word"}, "json")).status == 400 &&
            Send(service, GranularityMultipart({"sentence"})).status == 400 &&
            Send(service, GranularityMultipart({"word", "character"})).status == 400,
        "granularities need verbose_json and known values");
  Check(Send(service, Multipart("json", "model")).status == 400,
        "other repeated fields stay duplicates");
  Check(runner.calls == calls, "rejected granularities never invoke inference");
}

}  // namespace

int main() {
  TestJsonRequest();
  TestFormatsAndValidation();
  TestStreamingAndChunks();
  TestTimestampGranularities();
  std::cout << "PASS qwen3_asr_audio_api_test\n";
  return 0;
}
