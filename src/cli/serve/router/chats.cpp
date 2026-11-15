#include "src/cli/serve/router/chats.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iterator>
#include <system_error>
#include <utility>

namespace gufo::router::auth {
namespace {

std::string ReadFile(const std::filesystem::path& file, bool* ok) {
  std::ifstream in(file, std::ios::binary);
  *ok = static_cast<bool>(in);
  if (!*ok)
    return {};
  return std::string(std::istreambuf_iterator<char>(in),
                     std::istreambuf_iterator<char>());
}

/// Temporary file + rename: a reader never sees half a conversation.
bool WriteFileAtomic(const std::filesystem::path& file, std::string_view text,
                     std::string* error) {
  std::error_code ec;
  std::filesystem::create_directories(file.parent_path(), ec);
  const std::filesystem::path tmp = file.string() + ".tmp";
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out) {
      *error = "cannot write " + tmp.string();
      return false;
    }
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!out) {
      *error = "write failed: " + tmp.string();
      return false;
    }
  }
  std::filesystem::rename(tmp, file, ec);
  if (ec) {
    *error = "cannot replace " + file.string() + ": " + ec.message();
    std::filesystem::remove(tmp, ec);
    return false;
  }
  return true;
}

}  // namespace

bool IsValidChatId(std::string_view id) {
  if (id.empty() || id.size() > 64)
    return false;
  return std::all_of(id.begin(), id.end(), [](char c) {
    const auto u = static_cast<unsigned char>(c);
    return std::isalnum(u) || c == '_' || c == '-';
  });
}

void ChatStore::SetRoot(std::filesystem::path root) {
  const std::lock_guard<std::mutex> lock(mutex_);
  root_ = std::move(root);
}

std::filesystem::path ChatStore::DirOf(const std::string& account) const {
  // Account names are [a-z0-9._-] (IsValidName) and never "." or "..".
  return root_ / account;
}

ChatStore::Meta ChatStore::MetaOf(const std::string& id,
                                  const json::Value& chat, long long bytes) {
  Meta meta;
  meta.id = id;
  meta.titel = chat.member_str("titel").substr(0, 200);
  meta.modell = chat.member_str("modell").substr(0, 200);
  meta.zeit = chat.member_double("zeit", 0.0);
  meta.bytes = bytes;
  return meta;
}

std::vector<ChatStore::Meta> ChatStore::IndexLocked(const std::string& account) {
  const std::filesystem::path dir = DirOf(account);
  std::vector<Meta> index;
  bool ok = false;
  const std::string text = ReadFile(dir / "index.json", &ok);
  if (ok) {
    try {
      const json::Value doc = json::parse(text);
      if (const json::Value* list = doc.find("chats");
          list != nullptr && list->is_array()) {
        for (const json::Value& item : list->items()) {
          Meta meta;
          meta.id = item.member_str("id");
          if (!IsValidChatId(meta.id))
            continue;
          meta.titel = item.member_str("titel");
          meta.modell = item.member_str("modell");
          meta.zeit = item.member_double("zeit", 0.0);
          meta.bytes = static_cast<long long>(item.member_double("bytes", 0.0));
          index.push_back(std::move(meta));
        }
        return index;
      }
    } catch (const std::exception&) {
      // Rebuilt from the files below.
    }
  }
  // No or a broken index: read every conversation once and write it anew.
  std::error_code ec;
  if (!std::filesystem::is_directory(dir, ec))
    return index;
  for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
    const std::filesystem::path& path = entry.path();
    if (path.extension() != ".json" || path.filename() == "index.json")
      continue;
    const std::string id = path.stem().string();
    if (!IsValidChatId(id))
      continue;
    bool read = false;
    const std::string body = ReadFile(path, &read);
    if (!read)
      continue;
    try {
      index.push_back(MetaOf(id, json::parse(body),
                             static_cast<long long>(body.size())));
    } catch (const std::exception&) {
      // A broken file stays out of the list.
    }
  }
  SaveIndexLocked(account, index);
  return index;
}

void ChatStore::SaveIndexLocked(const std::string& account,
                                const std::vector<Meta>& index) {
  json::Value list = json::Value::array();
  for (const Meta& meta : index) {
    json::Value item = json::Value::object();
    item["id"] = meta.id;
    item["titel"] = meta.titel;
    item["modell"] = meta.modell;
    item["zeit"] = meta.zeit;
    item["bytes"] = meta.bytes;
    list.push_back(std::move(item));
  }
  json::Value doc = json::Value::object();
  doc["chats"] = std::move(list);
  std::string error;
  (void)WriteFileAtomic(DirOf(account) / "index.json", doc.dump(), &error);
}

json::Value ChatStore::List(const std::string& account) {
  const std::lock_guard<std::mutex> lock(mutex_);
  std::vector<Meta> index = IndexLocked(account);
  std::stable_sort(index.begin(), index.end(),
                   [](const Meta& a, const Meta& b) { return a.zeit > b.zeit; });
  json::Value out = json::Value::array();
  for (const Meta& meta : index) {
    json::Value item = json::Value::object();
    item["id"] = meta.id;
    item["titel"] = meta.titel;
    item["modell"] = meta.modell;
    item["zeit"] = meta.zeit;
    item["bytes"] = meta.bytes;
    out.push_back(std::move(item));
  }
  return out;
}

std::optional<std::string> ChatStore::Get(const std::string& account,
                                          const std::string& id) {
  if (!IsValidChatId(id))
    return std::nullopt;
  const std::lock_guard<std::mutex> lock(mutex_);
  bool ok = false;
  std::string text = ReadFile(DirOf(account) / (id + ".json"), &ok);
  if (!ok)
    return std::nullopt;
  return text;
}

bool ChatStore::Put(const std::string& account, const std::string& id,
                    json::Value chat, std::vector<std::string>* removed,
                    std::string* error) {
  if (!IsValidChatId(id)) {
    *error = "invalid conversation id";
    return false;
  }
  chat["id"] = id;
  const std::string text = chat.dump(json::Value::Numbers::kAsParsed);
  const std::lock_guard<std::mutex> lock(mutex_);
  const std::filesystem::path dir = DirOf(account);
  if (!WriteFileAtomic(dir / (id + ".json"), text, error))
    return false;
  std::vector<Meta> index = IndexLocked(account);
  std::erase_if(index, [&id](const Meta& meta) { return meta.id == id; });
  // The conversation just written stays first; of the others the newest
  // kMaxChats - 1 stay.
  std::stable_sort(index.begin(), index.end(),
                   [](const Meta& a, const Meta& b) { return a.zeit > b.zeit; });
  while (index.size() + 1 > kMaxChats) {
    const std::string gone = index.back().id;
    index.pop_back();
    std::error_code ec;
    std::filesystem::remove(dir / (gone + ".json"), ec);
    removed->push_back(gone);
  }
  index.insert(index.begin(),
               MetaOf(id, chat, static_cast<long long>(text.size())));
  SaveIndexLocked(account, index);
  return true;
}

bool ChatStore::Remove(const std::string& account, const std::string& id) {
  if (!IsValidChatId(id))
    return false;
  const std::lock_guard<std::mutex> lock(mutex_);
  std::vector<Meta> index = IndexLocked(account);
  std::error_code ec;
  const bool existed =
      std::filesystem::remove(DirOf(account) / (id + ".json"), ec);
  const std::size_t before = index.size();
  std::erase_if(index, [&id](const Meta& meta) { return meta.id == id; });
  if (index.size() != before || existed)
    SaveIndexLocked(account, index);
  return existed;
}

std::size_t ChatStore::RemoveAll(const std::string& account) {
  const std::lock_guard<std::mutex> lock(mutex_);
  const std::vector<Meta> index = IndexLocked(account);
  std::size_t count = 0;
  for (const Meta& meta : index) {
    std::error_code ec;
    if (std::filesystem::remove(DirOf(account) / (meta.id + ".json"), ec))
      ++count;
  }
  SaveIndexLocked(account, {});
  return count;
}

}  // namespace gufo::router::auth
