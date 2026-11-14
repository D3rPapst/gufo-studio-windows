// YuE2-3B songs behind /v1/audio/music (`gufo serve music --yue2-vae DIR`).
// gufo-studio-windows; no upstream capability (see media_service.hpp).
// Weights: CC BY-NC 4.0 plus the creator permission in MODEL_LICENSE
// (docs/models/yue2.md). Written with AI assistance (Claude Code).
#ifndef GUFO_SERVER_MEDIA_YUE2_API_HPP_
#define GUFO_SERVER_MEDIA_YUE2_API_HPP_

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>

#include "src/cli/serve/media/media_service.hpp"

namespace gufo::models::yue2 {
class SongPipeline;
}

namespace gufo::server {

/// YuE2 behind the music API of `gufo serve music`:
///   POST /v1/audio/music {lyrics ("[Verse]" / "[Chorus]" sections),
///        tags | prompt | style (genre, instruments, voice), language
///        (appended to the style when it is not mentioned there),
///        duration (upper bound in seconds, 0 = the model ends the song,
///        <= 360), seed, cot ("full" | "melody" | "off"), abc (own score),
///        guidance_scale, steps (ODE steps, default 32), response_format,
///        bitrate_kbps, async}
///        -> the audio (async: 202 + job JSON with the planned `abc`)
///   GET  /v1/audio/music/{id}          job status
///   GET  /v1/audio/music/{id}/content  the audio
/// One worker generates the jobs in order; results are kept for an hour.
class Yue2MusicService final : public MediaService {
public:
  static constexpr std::size_t kQueueCapacity = 8;
  static constexpr std::size_t kKeptJobs = 16;

  Yue2MusicService(const std::filesystem::path& model_dir,
                   const std::filesystem::path& vae_dir, std::string model_id);
  ~Yue2MusicService();
  Yue2MusicService(const Yue2MusicService&) = delete;
  Yue2MusicService& operator=(const Yue2MusicService&) = delete;

  [[nodiscard]] bool ready() const override { return pipeline_ != nullptr; }
  [[nodiscard]] const std::string& initialization_error() const {
    return init_error_;
  }
  [[nodiscard]] std::string model_id() const override { return model_id_; }
  [[nodiscard]] std::string_view capability() const override {
    return "music";
  }
  [[nodiscard]] std::string_view owned_by() const override {
    return "operator-supplied-yue2";
  }
  [[nodiscard]] bool HandlesPath(std::string_view path) const override;
  [[nodiscard]] std::string_view unavailable_message() const override {
    return "YuE2 music is not ready";
  }
  [[nodiscard]] std::string_view unavailable_code() const override {
    return "music_service_unavailable";
  }
  void Describe(json::Value* entry, bool ready) const override;

  HttpResponse Handle(const HttpRequest& request) override;

private:
  struct Job;
  HttpResponse Submit(const HttpRequest& request);
  HttpResponse Status(const std::string& id, bool content);
  void Worker();
  void ExpireLocked();

  std::string model_id_;
  std::string init_error_;
  std::shared_ptr<models::yue2::SongPipeline> pipeline_;
  std::mutex mutex_;
  std::condition_variable changed_;
  std::deque<std::shared_ptr<Job>> queue_;
  std::map<std::string, std::shared_ptr<Job>> jobs_;
  std::uint64_t next_id_{1};
  bool stop_{false};
  std::thread worker_;
};

/// True when `dir` holds YuE2-3B (qwen.tiktoken next to model.safetensors).
[[nodiscard]] bool IsYue2ModelDir(const std::filesystem::path& dir);

}  // namespace gufo::server

#endif  // GUFO_SERVER_MEDIA_YUE2_API_HPP_
