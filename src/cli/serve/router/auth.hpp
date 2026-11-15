#ifndef GUFO_CLI_SERVE_ROUTER_AUTH_HPP_
#define GUFO_CLI_SERVE_ROUTER_AUTH_HPP_

// Users and access control for the router front (gufo-studio-windows
// 2.0.3, concept NUTZERVERWALTUNG-KONZEPT-2026-11-13): the users file
// (PBKDF2-SHA256 password hashes, API tokens, remembered sessions), the
// in-memory sessions, the per-request gate (loopback -> bearer -> cookie ->
// 401/302), the per-area rights, the word filter for image/video prompts,
// the login/users endpoints and the access log. Everything is behind the
// preset's `[router]` section; with `auth = off` (the default) the gate
// only answers /router/me, /login, /logout and the users API from loopback
// and passes every other request through untouched.

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/router/chats.hpp"
#include "src/core/json.hpp"

namespace gufo::router::auth {

/// Router-to-router calls carry the shared pool secret (preset
/// `pool-secret-env`); a partner without it is answered 401 and counts as
/// unreachable (B5 keeps the pool from hanging).
inline constexpr const char* kPoolKeyHeader = "X-Gufo-Pool-Key";
/// Set by the entry router on every relayed or pulled request: the user
/// the request runs for and its rights (Rights::Encode). The partner trusts
/// them when the pool key is valid; a client's own copy is dropped.
inline constexpr const char* kUserHeader = "X-Gufo-User";
inline constexpr const char* kRightsHeader = "X-Gufo-Rights";
/// The router's own calls to its front (jobs runner, figure check, ComfyUI
/// adapter) carry a per-process random key; only loopback may use it.
inline constexpr const char* kInternalHeader = "X-Gufo-Internal";
inline constexpr const char* kSessionCookie = "gufo_sitzung";
inline constexpr const char* kTokenPrefix = "gufo_";
/// The areas a user may be limited to (Rights::bereiche).
inline constexpr std::string_view kAreas[] = {
    "text", "bild", "video", "stimme", "musik", "3d", "ablage", "pool",
    "status"};
inline constexpr const char* kRoles[] = {"admin", "nutzer", "dienst"};
/// PBKDF2-SHA256 rounds for new password hashes.
inline constexpr long long kPasswordRounds = 600000;
inline constexpr std::chrono::seconds kSessionTtl{12 * 3600};
inline constexpr std::chrono::seconds kRememberTtl{30 * 24 * 3600};
/// Failed logins per address and minute before a 60 s block.
inline constexpr int kFailuresPerMinute = 5;
inline constexpr std::chrono::seconds kFailureBlock{60};
/// Access log files older than this are removed at rotation.
inline constexpr int kLogKeepDays = 30;

enum class Mode { kOff, kLan, kAll };
enum class LoopbackPolicy { kAdmin, kLogin };

[[nodiscard]] const char* ModeName(Mode mode);
[[nodiscard]] bool ParseMode(std::string_view text, Mode* out);
[[nodiscard]] bool ParseLoopbackPolicy(std::string_view text,
                                       LoopbackPolicy* out);
[[nodiscard]] bool IsArea(std::string_view area);
[[nodiscard]] bool IsRole(std::string_view role);
/// User and token names: 1..32 of [a-z0-9._-].
[[nodiscard]] bool IsValidName(std::string_view name);
/// "lokal" (the router itself, loopback) and "partner" (the partner
/// router) name no user and cannot be created.
[[nodiscard]] bool IsReservedName(std::string_view name);

/// Per-user rights (concept section 3). Empty lists mean "the role
/// decides" (every area, every model).
struct Rights {
  std::vector<std::string> bereiche;
  std::vector<std::string> modelle;
  /// Word filter on image/video prompts; nullopt = role default (admin
  /// off, everyone else on).
  std::optional<bool> filter;
  /// Media jobs per day, 0 = unlimited (videos count twice). Since 2.1.8
  /// only image, video, music, 3D, speech, upscale and media jobs count;
  /// chat/text and speech recognition do not.
  long long tageslimit{0};
  /// May send work to the pool partner (otherwise X-Gufo-Pool: local).
  bool pool{true};

  [[nodiscard]] bool AllowsArea(std::string_view area) const;
  [[nodiscard]] bool AllowsModel(std::string_view model) const;
  [[nodiscard]] bool FilterOn(std::string_view role) const;
  /// "bereiche=text,bild;modelle=a,b;filter=an;tageslimit=0;pool=an" for
  /// kRightsHeader; Decode reads it back (unknown keys ignored).
  [[nodiscard]] std::string Encode() const;
  [[nodiscard]] static Rights Decode(std::string_view text);
  [[nodiscard]] json::Value ToJson() const;
  [[nodiscard]] static Rights FromJson(const json::Value& value);
};

struct Token {
  std::string name;
  std::string hash;  // SHA-256 hex of the plaintext token
  long long erstellt{0};
  long long ablauf{0};   // unix seconds, 0 = never
  long long zuletzt{0};  // last use, 0 = never
};

/// A "stay signed in" session stored in the users file (survives a router
/// restart); plain sessions live only in Gate's memory.
struct RememberedSession {
  std::string hash;
  long long ablauf{0};
};

struct User {
  std::string name;
  std::string rolle{"nutzer"};
  std::string salt_hex;  // 16 bytes
  std::string hash_hex;  // PBKDF2-SHA256, 32 bytes
  long long runden{0};
  Rights rechte;
  std::vector<Token> token;
  std::vector<RememberedSession> sitzungen;
  long long erstellt{0};
  long long letzte_anmeldung{0};
  bool gesperrt{false};

  [[nodiscard]] bool admin() const { return rolle == "admin"; }
  /// What the users API lists: no hashes, no salts.
  [[nodiscard]] json::Value PublicJson() const;
};

// Crypto helpers (OpenSSL, already linked for the webhook HMAC).
[[nodiscard]] std::string RandomHex(std::size_t bytes);
[[nodiscard]] std::string Sha256Hex(std::string_view text);
/// PBKDF2-HMAC-SHA256 of `password` with the raw `salt`, `rounds` rounds,
/// 32 bytes as hex. Returns "" when OpenSSL refuses.
[[nodiscard]] std::string Pbkdf2Hex(std::string_view password,
                                    std::string_view salt, long long rounds);
[[nodiscard]] bool ConstantTimeEquals(std::string_view a, std::string_view b);
[[nodiscard]] std::string HexOf(std::string_view bytes);
[[nodiscard]] std::string BytesOfHex(std::string_view hex);
[[nodiscard]] long long NowUnix();

/// The users file: loaded once, written whole after every change
/// (temporary file + rename). Thread-safe.
class UserStore {
public:
  explicit UserStore(std::filesystem::path file,
                     long long rounds = kPasswordRounds);

  /// A missing file is an empty store (first setup); a file that does not
  /// parse is an error.
  [[nodiscard]] bool Load(std::string* error);
  [[nodiscard]] bool Save(std::string* error) const;
  /// Points the store at another file and loads it (the setup assistant
  /// of a router started with --einrichtung learns the data folder, and
  /// with it the users file, from the page's step 3).
  [[nodiscard]] bool Relocate(std::filesystem::path file, std::string* error);
  [[nodiscard]] const std::filesystem::path& path() const { return file_; }
  [[nodiscard]] std::filesystem::path dir() const {
    return file_.parent_path();
  }
  /// The file exists on disk.
  [[nodiscard]] bool exists() const;
  [[nodiscard]] bool empty() const;
  [[nodiscard]] bool HasAdmin() const;
  [[nodiscard]] std::vector<User> List() const;
  [[nodiscard]] std::optional<User> Find(std::string_view name) const;

  /// Adds or replaces (by name) and saves.
  [[nodiscard]] bool Put(User user, std::string* error);
  /// Adds a new user with its password in one save: nothing is stored
  /// when the name is taken or the password is refused (the setup
  /// assistant's first admin never exists without a password).
  [[nodiscard]] bool Create(User user, std::string_view password,
                            std::string* error);
  [[nodiscard]] bool Remove(std::string_view name, std::string* error);
  [[nodiscard]] bool SetPassword(std::string_view name,
                                 std::string_view password,
                                 std::string* error);
  /// The user when the password matches and the account is not blocked.
  [[nodiscard]] std::optional<User> Verify(std::string_view name,
                                           std::string_view password) const;

  /// Creates `gufo_<64 hex>` for `name`, stores its hash and returns the
  /// plaintext (shown once). `ablauf` 0 = never.
  [[nodiscard]] std::string CreateToken(std::string_view name,
                                        std::string_view token_name,
                                        long long ablauf, std::string* error);
  [[nodiscard]] bool RevokeToken(std::string_view name,
                                 std::string_view token_name,
                                 std::string* error);
  /// The user owning a live token (not expired, account not blocked);
  /// notes the use.
  [[nodiscard]] std::optional<User> FindByToken(std::string_view token);

  /// Stores the hash of a "stay signed in" session id.
  [[nodiscard]] bool RememberSession(std::string_view name,
                                     std::string_view session_id,
                                     long long ablauf, std::string* error);
  /// `ablauf` (may be null) receives the stored expiry (unix seconds).
  [[nodiscard]] std::optional<User> FindBySession(
      std::string_view session_id, long long* ablauf = nullptr) const;
  void ForgetSession(std::string_view session_id);
  void NoteLogin(std::string_view name);

  [[nodiscard]] json::Value ToJson() const;
  /// Replaces the contents from a users file document (old files may
  /// lack fields: defaults apply).
  [[nodiscard]] bool FromJson(const json::Value& value, std::string* error);

private:
  [[nodiscard]] bool SaveLocked(std::string* error) const;
  [[nodiscard]] User* FindLocked(std::string_view name);
  [[nodiscard]] const User* FindLocked(std::string_view name) const;
  void SetPasswordLocked(User* user, std::string_view password) const;

  std::filesystem::path file_;
  long long rounds_;
  mutable std::mutex mutex_;
  std::vector<User> users_;
};

/// Who a request runs for.
struct Identity {
  enum class Source {
    kNone,      // not authenticated
    kOff,       // auth = off: everyone is "lokal", admin
    kLoopback,  // 127.0.0.1/::1 with auth-loopback = admin
    kInternal,  // the router's own call (kInternalHeader)
    kPool,      // the partner router (kPoolKeyHeader)
    kBearer,
    kCookie
  };
  Source source{Source::kNone};
  std::string name;
  std::string rolle;
  Rights rechte;
  /// The name is a placeholder, not a user: the router itself, loopback
  /// as admin, auth = off, the partner router. Owns nothing.
  bool placeholder{false};
  /// A bearer value without the gufo_ prefix came from loopback and was
  /// ignored (an SDK's placeholder key): noted in the log.
  bool bearer_ignored{false};

  [[nodiscard]] bool authenticated() const { return source != Source::kNone; }
  [[nodiscard]] bool admin() const { return rolle == "admin"; }
  /// The partner and the router's own calls are never checked again (the
  /// entry router did).
  [[nodiscard]] bool trusted() const {
    return source == Source::kOff || source == Source::kLoopback ||
           source == Source::kInternal || source == Source::kPool;
  }
  /// The partner router itself (pool key, no relayed user): trusted for
  /// work, jobs, pull and status, but no admin (users API, model control
  /// and bulk delete stay closed to it).
  [[nodiscard]] bool partner_router() const {
    return source == Source::kPool && rolle == "partner";
  }
  /// A user the partner relays (pool key + X-Gufo-User) who is no admin:
  /// the own-jobs rule applies here as well.
  [[nodiscard]] bool relayed_user() const {
    return source == Source::kPool && rolle != "partner" && !admin();
  }
  [[nodiscard]] const char* SourceName() const;
};

/// The `[router]` preset keys, resolved.
struct Settings {
  Mode mode{Mode::kOff};
  LoopbackPolicy loopback{LoopbackPolicy::kAdmin};
  std::filesystem::path users_file;
  /// Environment variable with the shared pool secret and its value.
  std::string pool_secret_env;
  std::string pool_secret;
  /// Origins whose pages may make browser writes besides this router's
  /// own address, normalised (Gate::NormalizeOrigin): set them through
  /// AllowOrigins so the preset's spelling never matters.
  std::vector<std::string> csrf_allow_origins;
  void AllowOrigins(const std::vector<std::string>& origins);
};

/// The pool secret outgoing partner calls carry (remote.cpp adds it to
/// every RemoteTarget); "" = none.
void SetPoolSecret(std::string secret);
[[nodiscard]] const std::string& PoolSecret();

/// What a request needs (Classify).
struct Route {
  enum class Kind {
    kFree,         // /health, /login/ files, /gemeinsam/ files
    kLogin,        // POST /login
    kLogout,       // POST /logout
    kMe,           // GET /router/me
    kPasswort,     // POST /router/me/passwort (own password)
    kMeChats,      // /router/me/chats[/<id>] (2.1.8): the own chat history
    kFilter,       // /router/filter[/pruefen] (2.1.8): the word list,
                   // loopback or admin; another user 403
    kFilterSync,   // POST /router/pool/filter (2.1.8): the partner's word
                   // list exchange, pool key only, else 404
    kUsers,        // /router/users*, /router/zugriff: loopback or admin
    kAdminPage,    // /verwaltung/, /einrichtung/: loopback or admin, else 404
    kSetup,        // /router/setup/* (2.1 setup assistant): loopback or
                   // admin, else 404; the router answers (einrichtung.cpp)
    kPage,         // GET /<page>/...: login (302), then the page's area
    kAnyUser,      // any authenticated user (model list, status reads)
    kAdminControl, // model load/unload, placement: admin
    kJobs,         // /v1/jobs*: area ablage, own jobs only
    kWork,         // generation endpoints
    kMcp,          // /mcp (2.1): bearer of any role, loopback, auth = off;
                   // a cookie is 403 mcp_token_required, the partner 404
    kOther         // unknown path: authenticated, the router answers
  };
  Kind kind{Kind::kOther};
  /// "" = none; "bild|video" = either.
  std::string area;
  /// POST on a media endpoint: counts against tageslimit (videos twice;
  /// chat/text and speech recognition 0 since 2.1.8).
  int counts{0};
  /// Prompts run through the word filter.
  bool filtered{false};
};

[[nodiscard]] Route Classify(const server::HttpRequest& req);
/// The area a page serves ("bild" for /bild/, "" when none/unknown).
[[nodiscard]] std::string PageArea(std::string_view path);
[[nodiscard]] bool IsLoopback(std::string_view address);

/// Lower case for the filter: ASCII and the Latin-1 letters of UTF-8
/// (Ä Ö Ü É ... -> ä ö ü é; ß stays), so "TÖTEN" meets "töt*".
[[nodiscard]] std::string LowerUtf8(std::string_view text);

/// Word list limits (2.1.8): active words, entries with tombstones, bytes
/// per word, kept backups of filter.json.
inline constexpr std::size_t kFilterMaxWords = 500;
inline constexpr std::size_t kFilterMaxEntries = 5000;
inline constexpr std::size_t kFilterMaxWordBytes = 100;
inline constexpr std::size_t kFilterKeepBackups = 10;

/// Word filter for prompts: case-insensitive whole-word match (`*` at the
/// end = any word form) on the prompt strings of a JSON or multipart body.
///
/// Since 2.1.8 the list lives in `filter.json` next to the users file:
/// per word `{aktiv, geaendert_ms, von}` (a deleted word stays as a
/// tombstone, aktiv = false, so a partner never brings it back). The pool
/// partners keep their lists in step (Gate::SyncFilterOnce): per word the
/// newer `geaendert_ms` wins, on a tie the larger `von` (the router's
/// address), then the tombstone. `filter-woerter.txt` stays as a mirror of
/// the active words (written after every change; an edit of it by hand is
/// taken over as a change of this machine). A first start without
/// filter.json takes the text file over (geaendert_ms 0).
class WordFilter {
public:
  struct Entry {
    bool aktiv{true};
    long long geaendert_ms{0};
    std::string von;
  };
  using Entries = std::map<std::string, Entry>;
  /// Called (with the list's lock held: never call back into the filter)
  /// with the entries a change of this machine set: the list API, a hand
  /// edit of the text file. Never for entries merged from the partner.
  using OnChange = std::function<void(const Entries&)>;

  explicit WordFilter(std::filesystem::path file);
  /// Writes the default list when neither file exists.
  void EnsureDefaults();
  /// The first blocked word found in the request's prompt texts, "" when
  /// clean. `comfy` scans every string of the body (ComfyUI graphs).
  [[nodiscard]] std::string Hit(const server::HttpRequest& req, bool comfy);
  /// The first blocked word in a chat request's user texts (2.0.5):
  /// `prompt`, the last user message of `messages` and the last user item
  /// of `input`, as a string or text parts; "" when clean.
  [[nodiscard]] std::string HitChat(const server::HttpRequest& req);
  /// The first blocked word in the named top-level fields of a JSON body
  /// (2.1.8): a string or an array of strings each; "" when clean. Music
  /// reads tags/prompt/style/caption/lyrics, speech input/instructions.
  [[nodiscard]] std::string HitFields(const server::HttpRequest& req,
                                      std::initializer_list<const char*> keys);
  /// Every string of a parsed body (a ComfyUI graph).
  [[nodiscard]] std::string HitJson(const json::Value& body);
  [[nodiscard]] std::string HitText(std::string_view text);
  /// Active words.
  [[nodiscard]] std::size_t size();

  /// This router's name in `von` (its address; "lokal" by default).
  void SetSelf(std::string name);
  [[nodiscard]] std::string self();
  /// Test hook: the clock for geaendert_ms (unix milliseconds).
  void SetClockMs(std::function<long long()> now_ms);
  void SetOnChange(OnChange on_change);
  [[nodiscard]] Entries All();
  [[nodiscard]] std::vector<std::string> Comments();
  [[nodiscard]] std::filesystem::path json_file() const { return json_; }
  /// Sets the active list to `words` (lines; empty lines and `#`
  /// comments are skipped): new words become active, missing ones
  /// tombstones, both stamped now by this router. Keeps a backup
  /// `filter.json.bak-<time>` (the last kFilterKeepBackups). `changed`
  /// gets what changed. False with `error` for an invalid line or more
  /// than kFilterMaxWords words.
  bool Replace(const std::vector<std::string>& words, Entries* changed,
               std::string* error);
  /// Takes every entry of `incoming` that is newer than the own one
  /// (Newer); `changed` gets them. Invalid words are skipped.
  bool Merge(const Entries& incoming, Entries* changed, std::string* error);
  /// `text` trimmed and lower-cased into `key`; false with `why` for an
  /// empty line, a space, a control character, a comment, a `*` not at
  /// the end or more than kFilterMaxWordBytes bytes.
  [[nodiscard]] static bool NormalizeWord(std::string_view text,
                                          std::string* key, std::string* why);
  /// `a` replaces `b`: newer geaendert_ms, then the larger von, then the
  /// tombstone.
  [[nodiscard]] static bool Newer(const Entry& a, const Entry& b);
  [[nodiscard]] static json::Value EntriesJson(const Entries& entries);
  [[nodiscard]] static bool EntriesFromJson(const json::Value& value,
                                            Entries* out, std::string* error);

private:
  void Reload();
  void ImportTextLocked(bool initial, Entries* changed);
  bool SaveLocked(bool backup, bool export_text, std::string* error);
  void RebuildLocked();
  void CapLocked();
  [[nodiscard]] long long NowMs() const;
  std::filesystem::path file_;
  std::filesystem::path json_;
  std::mutex mutex_;
  std::vector<std::string> words_;
  Entries entries_;
  std::vector<std::string> comments_;
  std::filesystem::file_time_type text_at_{};
  bool text_known_{false};
  bool loaded_{false};
  std::string self_{"lokal"};
  std::function<long long()> now_ms_;
  OnChange on_change_;
};

/// One answer of the partner to POST /router/pool/filter: status 0 = not
/// reached (`error` says why).
struct FilterReply {
  int status{0};
  std::string body;
  std::string error;
};
using FilterCall = std::function<FilterReply(const std::string& body)>;

/// Prompt texts of a body: JSON `prompt`/`negative_prompt` (any depth) or
/// every string with `all_strings`; multipart text parts named so.
[[nodiscard]] std::vector<std::string> PromptTexts(
    const server::HttpRequest& req, bool all_strings);

/// One unit of work a ComfyUI graph (POST /prompt) resolves to: the
/// area and model it runs on, how it counts against the daily limit and
/// the prompt texts the word filter sees.
struct WorkItem {
  std::string area;
  std::string model;
  int counts{1};
  std::vector<std::string> texts;
};

/// The request gate. One per router; thread-safe.
class Gate {
public:
  struct Hooks {
    /// The model a request names (the router's own resolver).
    std::function<std::string(const server::HttpRequest&)> model_of;
    /// The owner (metadata.nutzer) of a local job, nullopt when unknown.
    std::function<std::optional<std::string>(const std::string& job_id)>
        job_owner;
    /// The tasks a ComfyUI /prompt body (parsed once by the gate) resolves
    /// to (the adapter's plan); nullopt when the graph does not build (the
    /// adapter answers that).
    std::function<std::optional<std::vector<WorkItem>>(
        const server::HttpRequest&, const json::Value& body)>
        comfy_plan;
    /// The ComfyUI prompt whose finished outputs list a file (`/view`'s
    /// filename, subfolder, type); nullopt for a file no history entry
    /// names (an upload, a file of a deleted entry): those stay open.
    std::function<std::optional<std::string>(const std::string& filename,
                                             const std::string& subfolder,
                                             const std::string& type)>
        comfy_file_prompt;
  };

  Gate(Settings settings, std::shared_ptr<UserStore> store, Hooks hooks);

  [[nodiscard]] const Settings& settings() const { return settings_; }
  [[nodiscard]] const std::string& internal_key() const {
    return internal_key_;
  }
  [[nodiscard]] UserStore& store() { return *store_; }
  [[nodiscard]] WordFilter& filter() { return filter_; }
  [[nodiscard]] ChatStore& chats() { return chats_; }

  /// Word list exchange with the pool partner (2.1.8). No timer: the
  /// router asks for a full exchange at its start and when the partner is
  /// reachable again (or still owed one), a local change is pushed at
  /// once. `call` sends one POST /router/pool/filter to the partner.
  void SetFilterPartner(std::string partner_name, FilterCall call);
  /// Queues a full exchange (both lists merged on both sides).
  void RequestFilterSync();
  /// A push or exchange did not reach the partner: a full exchange is owed.
  [[nodiscard]] bool FilterSyncOpen();
  /// Runs the queued work once (a full exchange, else the pending push);
  /// true when nothing failed. The loop calls it; tests call it directly.
  bool SyncFilterOnce();
  /// The sync thread: waits for queued work (no timer) until Stop.
  void FilterSyncLoop();
  void StopFilterSync();
  /// {partner, rechner, offen, zuletzt_ok_ms, zuletzt_versuch_ms, fehler}.
  [[nodiscard]] json::Value FilterSyncStatus();

  /// The front hook: answers the request itself (refusals, login, me,
  /// users API) or hands a checked copy to `next` and post-processes the
  /// answer (job lists of a user). With auth = off everything but the
  /// auth endpoints passes through untouched.
  [[nodiscard]] server::HttpResponse Serve(
      const server::HttpRequest& req,
      const std::function<server::HttpResponse(const server::HttpRequest&)>&
          next);

  /// Who the request is (loopback -> internal -> pool -> bearer -> cookie).
  [[nodiscard]] Identity Identify(const server::HttpRequest& req);
  /// The refusal for `req` under `who` (401/302/403/400/404/429) or
  /// nullopt when it may proceed. `route` is Classify(req). `body` (may be
  /// null) is the request's parsed JSON body when the caller has it (a job
  /// submit is parsed once in Serve).
  [[nodiscard]] std::optional<server::HttpResponse> Refuse(
      const server::HttpRequest& req, const Route& route, const Identity& who,
      const json::Value* body = nullptr);
  /// Rewrites a request that proceeds: drops client-supplied gufo
  /// headers, sets kUserHeader/kRightsHeader (the job API stamps the
  /// owner from it), pins the pool side. Never touches the body.
  void Rewrite(server::HttpRequest* req, const Identity& who);
  /// A user's job list shows only its own jobs; its job reads 404 for
  /// others' jobs; the answer to a prompt or a direct video names its
  /// owner.
  void After(const server::HttpRequest& req, const Route& route,
             const Identity& who, server::HttpResponse* response);
  /// `/v1/models` for a user (2.0.5): the worker command lines (paths on
  /// this machine) are left out of every `status`.
  static void HideModelArgs(server::HttpResponse* response);
  /// `/router/status` for a user (2.0.5): the shared queue's id lists and
  /// the remote jobs keep only the user's own jobs.
  void HideOthersJobs(const Identity& who, server::HttpResponse* response);

  /// Login/logout/me/users endpoints; nullopt when `req` is none of them.
  [[nodiscard]] std::optional<server::HttpResponse> AuthEndpoint(
      const server::HttpRequest& req, const Route& route, const Identity& who);

  /// Failed logins per address: blocked for kFailureBlock after
  /// kFailuresPerMinute failures within a minute.
  [[nodiscard]] bool Blocked(std::string_view address, long long* retry_after);
  void NoteFailure(std::string_view address);
  void ClearFailures(std::string_view address);

  /// In-memory sessions (the remembered ones are also in the store).
  [[nodiscard]] std::string NewSession(const User& user, bool remember);
  void EndSession(std::string_view id);
  /// Every session of `name` in memory (a password change or a block),
  /// except `keep` (the session that changed the password).
  void EndSessionsOf(std::string_view name, std::string_view keep = {});
  [[nodiscard]] std::optional<Identity> SessionIdentity(std::string_view id);

  /// Append to zugriff.log (time, user, address, method path, model,
  /// result); never prompts.
  void Log(const server::HttpRequest& req, const Identity& who,
           std::string_view result, std::string_view model = {});
  [[nodiscard]] std::vector<std::string> LogTail(std::size_t lines);
  [[nodiscard]] std::filesystem::path log_file() const;

  /// Test hook: the clock the gate uses.
  void SetClock(std::function<long long()> now) { now_ = std::move(now); }
  /// Daily count of a user (tests).
  [[nodiscard]] long long DailyCount(std::string_view name);
  /// Test hook: how many owner entries are kept before the oldest
  /// confirmed ones go (default 20000).
  void SetOwnerLimit(std::size_t limit) { owner_limit_ = limit; }

  struct OwnerEntry {
    std::string name;
    long long at{0};
    /// Taken for a request in flight, not confirmed by an answer yet.
    bool reserved{false};
    /// Requests of `name` sharing the reservation; it goes at 0 unless
    /// an answer confirmed it.
    int in_flight{0};
    /// Changes whenever the entry is taken afresh (another user, or after
    /// it expired): a release names the generation it took, so a request
    /// of an earlier holding never frees a later one.
    unsigned long long generation{0};
  };
  /// Owners of things the job store does not know: ComfyUI prompts
  /// ("prompt:<id>") and direct video jobs ("video:<id>"), remembered
  /// from the answers for a day (in memory; unknown = not checked). The
  /// first owner wins: false when a live entry names someone else.
  bool RememberOwner(std::string_view key, std::string_view name);
  /// Takes `key` for `name` before the work runs (a client-chosen prompt
  /// id), atomically: false when someone else holds it; the same user may
  /// hold it several times (in flight). RememberOwner confirms it,
  /// ReleaseOwner ends one holding: a reservation nobody holds and no
  /// answer confirmed goes.
  /// Returns the generation taken (0 = another user holds it).
  unsigned long long ReserveOwner(std::string_view key, std::string_view name);
  void ReleaseOwner(std::string_view key, std::string_view name,
                    unsigned long long generation);
  /// The client-chosen prompt id of a /prompt body, once every check
  /// passed: reserved for a real user, only checked for a placeholder;
  /// 409 prompt_id_taken when another user holds it. `key` and
  /// `generation` receive what ReleaseOwner needs ("" / 0 when nothing
  /// was reserved).
  [[nodiscard]] std::optional<server::HttpResponse> ReservePrompt(
      const Identity& who, const json::Value& body, std::string* key,
      unsigned long long* generation);
  /// "HTTPS://Host:443/" -> "https://host": scheme and host lower case,
  /// the default port of the scheme dropped, no trailing slash.
  [[nodiscard]] static std::string NormalizeOrigin(std::string_view origin);
  [[nodiscard]] std::optional<std::string> OwnerOf(std::string_view key);
  /// Browser requests (a cookie session, loopback as admin) that change
  /// something must come from this router's own pages: a cross-site
  /// fetch or an Origin of another site is 403 csrf_rejected. Tokens, the
  /// partner and the router's own calls are no browsers and pass.
  /// Nullopt = fine.
  [[nodiscard]] std::optional<server::HttpResponse> CsrfRefusal(
      const server::HttpRequest& req, const Identity& who) const;
  /// The header rule alone (any identity): Sec-Fetch-Site: cross-site,
  /// or an Origin (normalised, without its scheme) that is not exactly
  /// the Host header (host and port) and not in
  /// Settings::csrf_allow_origins (pages on another port of this machine
  /// need an entry there).
  [[nodiscard]] std::optional<server::HttpResponse> CsrfCheck(
      const server::HttpRequest& req) const;

private:
  struct SessionInfo {
    std::string name;
    long long ablauf{0};
    bool remembered{false};
  };
  [[nodiscard]] bool SetupPending() const;
  [[nodiscard]] json::Value MeJson(const Identity& who) const;
  [[nodiscard]] server::HttpResponse Login(const server::HttpRequest& req);
  [[nodiscard]] server::HttpResponse Logout(const server::HttpRequest& req,
                                            const Identity& who);
  /// POST /router/me/passwort {"alt", "neu"}: a signed-in user (admin or
  /// nutzer, never a placeholder or a service account) sets its own
  /// password; its other sessions end, this one stays.
  [[nodiscard]] server::HttpResponse ChangePassword(
      const server::HttpRequest& req, const Identity& who);
  [[nodiscard]] server::HttpResponse UsersApi(const server::HttpRequest& req,
                                              const Identity& who);
  /// GET/DELETE /router/me/chats, GET/PUT/DELETE /router/me/chats/<id>
  /// (2.1.8): a signed-in account's own conversations (chats.hpp); the
  /// router itself, loopback as admin, auth = off and the partner have
  /// none (403 no_account: the page keeps them in the browser).
  [[nodiscard]] server::HttpResponse ChatsApi(const server::HttpRequest& req,
                                              const Identity& who);
  /// GET/PUT /router/filter, POST /router/filter/pruefen (2.1.8).
  [[nodiscard]] server::HttpResponse FilterApi(const server::HttpRequest& req,
                                               const Identity& who);
  /// POST /router/pool/filter from the partner (2.1.8).
  [[nodiscard]] server::HttpResponse FilterSyncEndpoint(
      const server::HttpRequest& req, const Identity& who);
  [[nodiscard]] json::Value FilterListJson();
  void QueueFilterPush(const WordFilter::Entries& changed);
  [[nodiscard]] bool CountDaily(const Identity& who, int weight,
                                long long limit);
  void RotateLogLocked();
  [[nodiscard]] long long Now() const { return now_ ? now_() : NowUnix(); }

  Settings settings_;
  std::shared_ptr<UserStore> store_;
  Hooks hooks_;
  WordFilter filter_;
  ChatStore chats_;
  struct FilterSync {
    std::string partner;
    FilterCall call;
    bool want_full{false};
    WordFilter::Entries push;
    bool open{false};
    bool stopping{false};
    long long last_try_ms{0};
    long long last_ok_ms{0};
    std::string last_error;
  };
  std::mutex sync_mutex_;
  std::condition_variable sync_cv_;
  FilterSync sync_;
  std::string internal_key_;
  std::function<long long()> now_;
  std::mutex mutex_;
  std::map<std::string, SessionInfo> sessions_;
  std::map<std::string, std::deque<long long>> failures_;
  std::map<std::string, std::pair<std::string, long long>> daily_;
  std::map<std::string, OwnerEntry> owners_;
  std::size_t owner_limit_{20000};
  unsigned long long owner_generation_{0};
  long long owner_purged_at_{0};
  std::mutex log_mutex_;
  std::string log_day_;
};

/// JSON error body in the router's shape.
[[nodiscard]] server::HttpResponse AuthError(int status,
                                             std::string_view reason,
                                             std::string_view message,
                                             std::string_view code);

}  // namespace gufo::router::auth

#endif  // GUFO_CLI_SERVE_ROUTER_AUTH_HPP_
