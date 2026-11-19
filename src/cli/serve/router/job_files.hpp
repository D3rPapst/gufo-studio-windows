#ifndef GUFO_CLI_SERVE_ROUTER_JOB_FILES_HPP_
#define GUFO_CLI_SERVE_ROUTER_JOB_FILES_HPP_

// Finished video and 3D jobs read from the worker's job folder (2.2.13).
//
// `gufo serve video` and `gufo serve objekt` keep every job in
// `<root>/<id>/` (`root` from the preset section, the worker's `--root`):
// `job.json` with the job's state, and for a finished job its result
// (`output.mp4` / `output.ppm`, or `model.glb` + `preview.png`). Once the
// job is finished, nothing in that folder changes any more, so the router
// answers the status and the result reads itself instead of loading the
// worker (19.11.: a `/content` read of a finished Wan 2.2 clip reloaded the
// evicted 54.8 GB model only to hand out a file).
//
// No sockets, no router state: the router decides which model owns the job
// (and checks the account before, in the gate); this part only reads files.

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/router/preset.hpp"

namespace gufo::router::job_files {

enum class Kind { kVideo, kObjekt };

/// One read of a job resource.
struct Target {
  Kind kind{Kind::kVideo};
  std::string id;
  /// "" = status, "content", "preview" (3D only).
  std::string part;
};

/// `video_` + 24 lowercase hex digits (video_jobs.cpp IsVideoId).
[[nodiscard]] bool IsVideoId(std::string_view id);
/// `obj_` + 24 lowercase hex digits (objekt_service.cpp IsObjektId).
[[nodiscard]] bool IsObjektId(std::string_view id);

/// A GET of `/v1/videos/<id>[/content]` or
/// `/v1/3d/generations/<id>[/content|/preview]` with a well-formed id;
/// nullopt for anything else (other methods, other paths, ids that do not
/// match the worker's pattern: those take the normal route).
[[nodiscard]] std::optional<Target> ParseRead(std::string_view method,
                                              std::string_view path);

/// The job folder root of a preset whose worker keeps jobs of `kind` on
/// disk: `serve video` (or a `[video/<id>]` section) for video jobs,
/// `serve objekt` for 3D jobs, each with `--root`. nullopt otherwise (no
/// root given, or another worker such as `serve upscale`).
[[nodiscard]] std::optional<std::filesystem::path> JobRootOf(
    const PresetModel& preset, Kind kind);

/// The answer for a finished job, read from `root` (`now` in unix
/// seconds). nullopt when the folder does not hold a finished, unexpired
/// job of that id: then only the worker can answer (it may still run or
/// queue the job). Never writes anything.
///
/// - status: the worker's job object (video: the shape of video_api.cpp
///   SnapshotValue; 3D: the stored job object).
/// - content / preview of a completed job: the file, with Range support
///   for video (206 / 416 like the worker). A missing or unsafe file
///   (symlink, outside the root) is 404 (`video_not_found` /
///   `objekt_file_missing`).
/// - content of a failed or cancelled job: 409 `video_not_ready` /
///   `objekt_not_ready`, as the worker answers.
///
/// Every answer means the job is finished: the router may drop its
/// open-job record.
[[nodiscard]] std::optional<server::HttpResponse> ReadFinished(
    const std::filesystem::path& root, const Target& target,
    const server::HttpRequest& request, long long now);

/// The 404 the worker answers for an unknown job id of `kind`.
[[nodiscard]] server::HttpResponse NotFound(Kind kind);

}  // namespace gufo::router::job_files

#endif  // GUFO_CLI_SERVE_ROUTER_JOB_FILES_HPP_
