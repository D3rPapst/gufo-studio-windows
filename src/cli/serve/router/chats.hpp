#ifndef GUFO_CLI_SERVE_ROUTER_CHATS_HPP_
#define GUFO_CLI_SERVE_ROUTER_CHATS_HPP_

// Chat history per account on the router (gufo-studio-windows 2.1.8,
// decision e51): the /chat/ page keeps its conversations here when the
// user is signed in, instead of the browser's localStorage only. One JSON
// file per conversation under `<users dir>\chats\<account>\<id>.json`,
// an `index.json` beside them (id, title, model, time, size) for the
// list. At most kMaxChats per account (the oldest by `zeit` go), at most
// kMaxChatBytes per conversation. Nothing of it goes to the pool partner:
// every router keeps its own accounts' chats. Thread-safe.

#include <cstddef>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/json.hpp"

namespace gufo::router::auth {

/// Conversations kept per account.
inline constexpr std::size_t kMaxChats = 100;
/// Bytes per stored conversation (its JSON as sent). The page leaves out
/// image data above kMaxChatImageBytes per image, keeping a placeholder.
inline constexpr std::size_t kMaxChatBytes = 2 * 1024 * 1024;
inline constexpr std::size_t kMaxChatImageBytes = 256 * 1024;

/// Conversation ids: 1..64 of [A-Za-z0-9_-].
[[nodiscard]] bool IsValidChatId(std::string_view id);

class ChatStore {
public:
  /// `root` is the `chats` folder (the users file's folder + "chats").
  explicit ChatStore(std::filesystem::path root) : root_(std::move(root)) {}
  void SetRoot(std::filesystem::path root);

  /// The account's conversations, newest `zeit` first: [{id, titel,
  /// modell, zeit, bytes}].
  [[nodiscard]] json::Value List(const std::string& account);
  /// The stored JSON text of one conversation, nullopt when unknown.
  [[nodiscard]] std::optional<std::string> Get(const std::string& account,
                                               const std::string& id);
  /// Stores `chat` (a JSON object; its `id` is set to `id`) and drops the
  /// oldest beyond kMaxChats; `removed` gets their ids. False with
  /// `error` when it cannot be written.
  [[nodiscard]] bool Put(const std::string& account, const std::string& id,
                         json::Value chat, std::vector<std::string>* removed,
                         std::string* error);
  /// False when there was no such conversation.
  bool Remove(const std::string& account, const std::string& id);
  /// Removes every conversation of the account; returns how many.
  std::size_t RemoveAll(const std::string& account);

private:
  struct Meta {
    std::string id;
    std::string titel;
    std::string modell;
    double zeit{0};
    long long bytes{0};
  };
  [[nodiscard]] std::filesystem::path DirOf(const std::string& account) const;
  [[nodiscard]] std::vector<Meta> IndexLocked(const std::string& account);
  void SaveIndexLocked(const std::string& account,
                       const std::vector<Meta>& index);
  [[nodiscard]] static Meta MetaOf(const std::string& id,
                                   const json::Value& chat, long long bytes);

  std::filesystem::path root_;
  std::mutex mutex_;
};

}  // namespace gufo::router::auth

#endif  // GUFO_CLI_SERVE_ROUTER_CHATS_HPP_
