// The "person_count" task of POST /v1/vision/detect (see vision_persons.hpp).
// Written with AI assistance (Claude Code).
#include "src/cli/serve/router/vision_persons.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <utility>

#include "src/cli/serve/router/vision_chat.hpp"

namespace gufo::router::vision {
namespace {

using server::HttpRequest;
using server::HttpResponse;

int WholeNumber(const json::Value& value, const std::string& what, int low, int high) {
  if (!value.is_number() || value.as_double() != std::floor(value.as_double()) ||
      value.as_double() < low || value.as_double() > high)
    throw std::invalid_argument(what + " must be a whole number from " + std::to_string(low) +
                                " to " + std::to_string(high));
  return static_cast<int>(value.as_double());
}

template <std::size_t N>
bool OneOf(const std::string& name, const std::array<const char*, N>& names) {
  return std::any_of(names.begin(), names.end(),
                     [&name](const char* known) { return name == known; });
}

/// A reference name: 1-64 bytes, no control characters, no quotes (it is
/// put into the question).
bool GoodName(const std::string& name) {
  if (name.empty() || name.size() > 64)
    return false;
  return std::none_of(name.begin(), name.end(), [](char c) {
    const auto u = static_cast<unsigned char>(c);
    return u < 0x20 || u == 0x7f || c == '"' || c == '\\';
  });
}

std::vector<PersonReference> ParseReferences(const json::Value& value, const std::string& what) {
  if (!value.is_array() || value.size() > static_cast<std::size_t>(kMaxPersonReferences))
    throw std::invalid_argument(what + " must list up to " +
                                std::to_string(kMaxPersonReferences) + " references");
  std::vector<PersonReference> out;
  for (std::size_t i = 0; i < value.size(); ++i) {
    const json::Value& item = value.items()[i];
    const std::string at = what + "[" + std::to_string(i) + "]";
    PersonReference reference;
    reference.name = "Figur " + std::to_string(i + 1);
    if (item.is_string()) {
      reference.image = item.str();
    } else if (item.is_object()) {
      for (const auto& [key, member] : item.members()) {
        if (key == "name") {
          if (!member.is_string() || !GoodName(member.str()))
            throw std::invalid_argument(at + ".name must be 1-64 characters without quotes");
          reference.name = member.str();
        } else if (key == "image" || key == "bild") {
          if (!member.is_string())
            throw std::invalid_argument(at + ".image must be a string");
          reference.image = member.str();
        } else {
          throw std::invalid_argument(at + ": use name and image");
        }
      }
    } else {
      throw std::invalid_argument(at + " must be an image or {\"name\", \"image\"}");
    }
    if (reference.image.empty())
      throw std::invalid_argument(at + ": image missing");
    for (const PersonReference& known : out) {
      if (known.name == reference.name)
        throw std::invalid_argument(at + ": name " + reference.name + " given twice");
    }
    out.push_back(std::move(reference));
  }
  return out;
}

std::string PersonsText(int n) {
  return std::to_string(n) + (n == 1 ? " person" : " persons");
}

}  // namespace

PersonCountRequest ParsePersonCount(const json::Value& body, bool requested) {
  PersonCountRequest request;
  const json::Value* options = body.find(kPersonCountOptions);
  const bool given = options != nullptr && !options->is_null();
  if (!requested) {
    if (given)
      throw std::invalid_argument(
          "person_count_options given, but the task person_count is not requested");
    return request;
  }
  request.enabled = true;
  if (!given)
    return request;
  if (!options->is_object())
    throw std::invalid_argument("person_count_options must be an object");
  bool expected_seen = false;
  bool references_seen = false;
  for (const auto& [key, value] : options->members()) {
    const std::string what = std::string(kPersonCountOptions) + "." + key;
    if (key == "model") {
      if (!value.is_string() || value.str().empty() || value.str().size() > 128)
        throw std::invalid_argument(what + " must be a model id");
      request.model = value.str();
    } else if (key == "thinking") {
      if (!value.is_bool())
        throw std::invalid_argument(what + " must be true or false");
      request.thinking = value.as_bool();
    } else if (key == "timeout_seconds") {
      request.timeout_seconds = WholeNumber(value, what, 5, 600);
    } else if (key == "wait_seconds") {
      request.wait_seconds = WholeNumber(value, what, 0, 3600);
    } else if (key == "max_tokens") {
      request.max_tokens = WholeNumber(value, what, 32, 8192);
    } else if (key == "expected" || key == "erwartet") {
      if (expected_seen)
        throw std::invalid_argument(what + ": expected given twice");
      expected_seen = true;
      if (!value.is_null())
        request.expected = WholeNumber(value, what, 0, kMaxExpectedPersons);
    } else if (key == "references" || key == "referenzen") {
      if (references_seen)
        throw std::invalid_argument(what + ": references given twice");
      references_seen = true;
      if (!value.is_null())
        request.references = ParseReferences(value, what);
    } else {
      throw std::invalid_argument(what +
                                  ": use model, thinking, expected, references, "
                                  "timeout_seconds, wait_seconds or max_tokens");
    }
  }
  return request;
}

std::string PersonPrompt(const PersonCountRequest& options) {
  std::string wer;
  std::string references;
  if (options.references.empty()) {
    wer = "a short German description, for example \"Mann im grauen Mantel\"";
  } else {
    std::string names;
    for (std::size_t i = 0; i < options.references.size(); ++i)
      names += (i == 0 ? "" : ", ") + options.references[i].name;
    wer = "the name of the reference character this person is (same face, hair and clothes), "
          "or \"unbekannt\"";
    references = "The pictures after the scene are reference pictures of the film's "
                 "characters, in this order: " +
                 names +
                 ". They are not part of the scene: never list or check them. Two persons in "
                 "the scene who are the same character are duplicate_person. ";
  }
  const std::string count =
      options.expected >= 0
          ? "count_mismatch (the scene should show exactly " + PersonsText(options.expected) +
                "; report it when your list has a different number, stelle says who is "
                "missing or extra)"
          : "count_mismatch (only when a number of persons is expected; here none is, so do "
            "not use it)";
  return "Wie viele Personen sind auf diesem Bild, und ist jede so möglich? The picture "
         "\"Szene\" is a frame of an AI-generated film. " +
         references +
         "First list every real person in the scene in personen, each with stelle (where in "
         "the picture, in German, for example \"links am Tisch\") and wer (" +
         wer +
         "). List a person once, also when only part of the body is visible. Do not list "
         "people on posters, photos or screens, reflections in mirrors or windows, statues, "
         "or faces drawn on a monitor. Do not list a faint, see-through ghost figure as a "
         "person: report it. Then look at the whole picture, not only at the faces, and "
         "report in befunde only what is clearly impossible for real people, each with art, "
         "stelle (where, in German) and sicherheit (hoch = clearly visible, mittel = "
         "probably, niedrig = possible but unclear): extra_limb (an arm, hand, leg or head "
         "that belongs to no person, a third arm on one body, or a faint, see-through body "
         "part or ghost figure through which the wall, a desk or the background shows), "
         "duplicate_person (the same person, with the same face, hair and clothes, appears "
         "twice in the scene; also a see-through ghost double of a person), " +
         count +
         ", sonstiges (another impossible thing about the persons, for example two bodies "
         "merged into one). Hidden body parts are NOT an error: a person partly behind a "
         "desk, a chair, another person or an object, cut off by the picture border, hands "
         "behind the back or in pockets, legs under a table or a face turned away are "
         "normal; report nothing for them. Shadows, dark lighting, drawing style and blur "
         "are no finding. In doubt report nothing: a normal picture has befunde = []. "
         "sicher = false when the picture is dark, blurred or crowded, or you are unsure "
         "about the persons or their number. befund: one short German sentence, for example "
         "\"zwei Personen, alles normal\" or \"dritter Arm an der linken Person\". Answer "
         "only with the JSON object.";
}

json::Value PersonChatBody(const PersonCountRequest& options, const std::string& scene_url,
                           const std::vector<std::string>& reference_urls) {
  json::Value content = json::Value::array();
  content.push_back(TextPart("Szene:"));
  content.push_back(ImagePart(scene_url));
  for (std::size_t i = 0; i < reference_urls.size() && i < options.references.size(); ++i) {
    content.push_back(TextPart("Referenz " + options.references[i].name + ":"));
    content.push_back(ImagePart(reference_urls[i]));
  }
  content.push_back(TextPart(PersonPrompt(options)));

  // The order of the properties is the order the model writes them: the
  // persons first, the findings and the verdict after them.
  json::Value person_properties = json::Value::object();
  person_properties["stelle"] = TypeSchema("string");
  person_properties["wer"] = TypeSchema("string");
  json::Value personen = json::Value::object();
  personen["type"] = "array";
  personen["items"] = ObjectSchema(std::move(person_properties));
  json::Value finding_properties = json::Value::object();
  finding_properties["art"] = EnumSchema(kPersonBefundArten);
  finding_properties["stelle"] = TypeSchema("string");
  finding_properties["sicherheit"] = EnumSchema(kSicherheiten);
  json::Value befunde = json::Value::object();
  befunde["type"] = "array";
  befunde["items"] = ObjectSchema(std::move(finding_properties));
  json::Value properties = json::Value::object();
  properties["personen"] = std::move(personen);
  properties["befunde"] = std::move(befunde);
  properties["sicher"] = TypeSchema("boolean");
  properties["befund"] = TypeSchema("string");
  return VisionChatBody(options.model, std::move(content), kPersonCountTask,
                        ObjectSchema(std::move(properties)), options.max_tokens, options.thinking);
}

PersonVerdict ParsePersonAnswer(std::string_view chat_body) {
  const json::Value verdict = ChatAnswerObject(chat_body);
  const auto text_of = [](const json::Value* value, const std::string& name) {
    if (value == nullptr || !value->is_string())
      throw std::runtime_error(name + " missing or not a string");
    return value->str();
  };
  PersonVerdict out;
  const json::Value* personen = verdict.find("personen");
  if (personen == nullptr || !personen->is_array())
    throw std::runtime_error("personen missing or not a list");
  if (personen->size() > static_cast<std::size_t>(kMaxListedPersons))
    throw std::runtime_error("personen lists more than " + std::to_string(kMaxListedPersons));
  for (const json::Value& item : personen->items()) {
    if (!item.is_object())
      throw std::runtime_error("personen[] is no object");
    PersonSeen person;
    person.stelle = text_of(item.find("stelle"), "personen[].stelle");
    person.wer = text_of(item.find("wer"), "personen[].wer");
    out.personen.push_back(std::move(person));
  }
  const json::Value* befunde = verdict.find("befunde");
  if (befunde == nullptr || !befunde->is_array())
    throw std::runtime_error("befunde missing or not a list");
  for (const json::Value& item : befunde->items()) {
    if (!item.is_object())
      throw std::runtime_error("befunde[] is no object");
    PersonFinding finding;
    finding.art = text_of(item.find("art"), "befunde[].art");
    if (!OneOf(finding.art, kPersonBefundArten))
      throw std::runtime_error("befunde[].art: unknown '" + finding.art.substr(0, 40) + "'");
    finding.stelle = text_of(item.find("stelle"), "befunde[].stelle");
    finding.sicherheit = text_of(item.find("sicherheit"), "befunde[].sicherheit");
    if (!OneOf(finding.sicherheit, kSicherheiten))
      throw std::runtime_error("befunde[].sicherheit: unknown '" +
                               finding.sicherheit.substr(0, 40) + "'");
    out.befunde.push_back(std::move(finding));
  }
  const json::Value* sicher = verdict.find("sicher");
  if (sicher == nullptr || !sicher->is_bool())
    throw std::runtime_error("sicher missing or not true/false");
  out.sicher = sicher->as_bool();
  out.befund = text_of(verdict.find("befund"), "befund");
  return out;
}

void CheckPersonCount(const PersonCountRequest& options, PersonVerdict& verdict) {
  const auto is_mismatch = [](const PersonFinding& f) { return f.art == "count_mismatch"; };
  if (options.expected < 0) {
    verdict.befunde.erase(
        std::remove_if(verdict.befunde.begin(), verdict.befunde.end(), is_mismatch),
        verdict.befunde.end());
    return;
  }
  const int listed = static_cast<int>(verdict.personen.size());
  if (listed == options.expected ||
      std::any_of(verdict.befunde.begin(), verdict.befunde.end(), is_mismatch))
    return;
  PersonFinding finding;
  finding.art = "count_mismatch";
  finding.stelle = std::to_string(listed) + " gezählt, " + std::to_string(options.expected) +
                   " erwartet";
  finding.sicherheit = verdict.sicher ? "hoch" : "mittel";
  finding.quelle = "gufo";
  verdict.befunde.push_back(std::move(finding));
}

std::string PersonUrteil(const PersonVerdict& verdict) {
  bool any = false;
  for (const PersonFinding& finding : verdict.befunde) {
    if (finding.sicherheit == "hoch")
      return "fehler";
    any = true;
  }
  return any || !verdict.sicher ? "pruefen" : "ok";
}

json::Value CheckPersons(const PersonCountRequest& options, const models::ccip::RgbImage& image,
                         const std::vector<models::ccip::RgbImage>& references,
                         int detector_persons,
                         const std::function<HttpResponse(const HttpRequest&)>& serve,
                         const HttpRequest& outer) {
  using Clock = std::chrono::steady_clock;
  const auto started = Clock::now();
  json::Value entry = json::Value::object();
  if (options.expected >= 0)
    entry["erwartet"] = options.expected;
  entry["detektor_personen"] = detector_persons;
  const auto finish = [&entry, started] {
    entry["seconds"] =
        RoundedSeconds(std::chrono::duration<double>(Clock::now() - started).count());
    return std::move(entry);
  };
  if (outer.is_cancelled && outer.is_cancelled()) {
    entry["fehler"] = "cancelled";
    return finish();
  }
  if (!serve) {
    entry["fehler"] = "no chat on this router";
    return finish();
  }
  double waited = -1;
  try {
    std::vector<std::string> reference_urls;
    for (const auto& reference : references)
      reference_urls.push_back(ImageDataUrl(reference, 0, kReferenceMaxSide));
    const std::string body =
        PersonChatBody(options, ImageDataUrl(image, 0, kPersonMaxSide), reference_urls).dump();
    const ChatLimits limits{options.model, options.wait_seconds, options.timeout_seconds};
    std::string model_gone;
    const std::string answer = AskVisionChat(body, limits, serve, outer, &model_gone, &waited);
    PersonVerdict verdict = ParsePersonAnswer(answer);
    CheckPersonCount(options, verdict);
    entry["personen"] = static_cast<int>(verdict.personen.size());
    json::Value liste = json::Value::array();
    for (const PersonSeen& person : verdict.personen) {
      json::Value item = json::Value::object();
      item["stelle"] = person.stelle;
      item["wer"] = person.wer;
      liste.push_back(std::move(item));
    }
    entry["liste"] = std::move(liste);
    json::Value befunde = json::Value::array();
    for (const PersonFinding& finding : verdict.befunde) {
      json::Value item = json::Value::object();
      item["art"] = finding.art;
      item["stelle"] = finding.stelle;
      item["sicherheit"] = finding.sicherheit;
      item["quelle"] = finding.quelle;
      befunde.push_back(std::move(item));
    }
    entry["befunde"] = std::move(befunde);
    entry["sicher"] = verdict.sicher;
    entry["befund"] = verdict.befund;
    entry["urteil"] = PersonUrteil(verdict);
  } catch (const std::exception& error) {
    entry["fehler"] = error.what();
  }
  if (waited >= 0)
    entry["waited_seconds"] = waited;
  return finish();
}

json::Value PersonCountModelInfo(const PersonCountRequest& options) {
  json::Value info = json::Value::object();
  info["model"] = options.model;
  info["detector"] = "person";
  info["thinking"] = options.thinking;
  info["references"] = static_cast<int>(options.references.size());
  info["timeout_seconds"] = options.timeout_seconds;
  info["wait_seconds"] = options.wait_seconds;
  return info;
}

}  // namespace gufo::router::vision
