// Shared CPU queue and pool decision of the router's CPU services (see
// cpu_pool.hpp). Written with AI assistance (Claude Code).
#include "src/cli/serve/router/cpu_pool.hpp"

#include <algorithm>
#include <cctype>
#include <string_view>
#include <utility>

#include "src/core/json.hpp"

namespace gufo::router::cpu {
namespace {

bool EqualsIgnoreCase(std::string_view a, std::string_view b) {
  if (a.size() != b.size())
    return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (std::tolower(static_cast<unsigned char>(a[i])) !=
        std::tolower(static_cast<unsigned char>(b[i])))
      return false;
  }
  return true;
}

std::string Lower(std::string text) {
  for (char& c : text)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return text;
}

}  // namespace

CpuQueue::Result CpuQueue::Enter(const std::function<bool()>& cancelled) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (busy_ && line_.size() >= max_waiting_)
    return Result::kFull;
  const std::uint64_t ticket = next_++;
  line_.push_back(ticket);
  while (busy_ || line_.front() != ticket) {
    cv_.wait_for(lock, std::chrono::milliseconds(200));
    if (cancelled && cancelled()) {
      line_.erase(std::find(line_.begin(), line_.end(), ticket));
      cv_.notify_all();
      return Result::kCancelled;
    }
  }
  line_.pop_front();
  busy_ = true;
  return Result::kOk;
}

void CpuQueue::Leave() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    busy_ = false;
  }
  cv_.notify_all();
}

bool CpuQueue::Idle() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return !busy_ && line_.empty();
}

std::size_t CpuQueue::Waiting() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return line_.size();
}

bool CpuQueue::Busy() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return busy_;
}

Route Decide(const server::HttpRequest& request, bool have_partner, bool local_busy,
             bool local_missing) {
  if (!request.header("X-Gufo-Forwarded").empty())
    return request.header(kIfIdleHeader).empty() ? Route::kLocal : Route::kLocalIfIdle;
  const std::string side = Lower(request.header("X-Gufo-Pool"));
  if (side == "local")
    return Route::kLocal;
  if (side == "remote")
    return Route::kPartnerOnly;
  if (have_partner && (local_busy || local_missing))
    return Route::kTryPartner;
  return Route::kLocal;
}

bool PartnerAnswered(const server::HttpResponse& answer) {
  return !(answer.status == 404 || answer.status == 502 || answer.status == 503 ||
           answer.status == 504);
}

server::HttpResponse Busy(const std::string& what) {
  json::Value detail = json::Value::object();
  detail["message"] = what + " queue is full, try again";
  detail["type"] = "server_error";
  detail["code"] = "busy";
  json::Value body = json::Value::object();
  body["error"] = std::move(detail);
  server::HttpResponse response;
  response.status = 503;
  response.reason = "Service Unavailable";
  response.body = body.dump();
  response.headers = {{"Content-Type", "application/json"}, {"Retry-After", "5"}};
  return response;
}

server::HttpRequest RelayRequest(const server::HttpRequest& request, std::string body,
                                 bool force) {
  server::HttpRequest relay;
  relay.method = request.method;
  relay.path = request.path;
  relay.query = request.query;
  relay.body = std::move(body);
  relay.is_cancelled = request.is_cancelled;
  relay.request_id = request.request_id;
  relay.client_id = request.client_id;
  relay.headers.emplace_back("Content-Type", "application/json");
  relay.headers.emplace_back("Accept", "application/json");
  if (!force)
    relay.headers.emplace_back(kIfIdleHeader, "1");
  // The caller's priority travels along; its credentials do not.
  for (const auto& [name, value] : request.headers) {
    if (EqualsIgnoreCase(name, "X-Gufo-Priority"))
      relay.headers.emplace_back(name, value);
  }
  return relay;
}

}  // namespace gufo::router::cpu
