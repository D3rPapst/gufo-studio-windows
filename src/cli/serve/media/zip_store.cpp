// Minimal store-only ZIP writer (see zip_store.hpp).
// Written with AI assistance (Claude Code).
#include "src/cli/serve/media/zip_store.hpp"

#include <zlib.h>

#include <algorithm>
#include <limits>
#include <system_error>

namespace gufo::server {
namespace {

constexpr std::uint64_t kMaxZipBytes = std::numeric_limits<std::uint32_t>::max();
constexpr std::size_t kMaxEntries = 65535;
constexpr std::uint16_t kDosDate = (0U << 9U) | (1U << 5U) | 1U;  // 1980-01-01
constexpr std::uint16_t kDosTime = 0;

void Put16(std::string* out, std::uint32_t v) {
  out->push_back(static_cast<char>(v & 0xFFU));
  out->push_back(static_cast<char>((v >> 8U) & 0xFFU));
}

void Put32(std::string* out, std::uint32_t v) {
  Put16(out, v & 0xFFFFU);
  Put16(out, v >> 16U);
}

std::uint32_t Crc32(const std::uint8_t* data, std::size_t size) {
  uLong crc = crc32(0L, Z_NULL, 0);
  while (size > 0) {
    const uInt chunk = static_cast<uInt>(
        std::min<std::size_t>(size, std::numeric_limits<uInt>::max()));
    crc = crc32(crc, data, chunk);
    data += chunk;
    size -= chunk;
  }
  return static_cast<std::uint32_t>(crc);
}

}  // namespace

bool ZipStoreWriter::Open(const std::filesystem::path& path,
                          std::string* error) {
  path_ = path;
  out_.open(path, std::ios::binary | std::ios::trunc);
  if (!out_) {
    *error = "cannot create " + path.string();
    return false;
  }
  written_ = 0;
  entries_.clear();
  return true;
}

bool ZipStoreWriter::Add(std::string_view name, const std::uint8_t* data,
                         std::size_t size, std::string* error) {
  if (!out_.is_open()) {
    *error = "zip is not open";
    return false;
  }
  if (entries_.size() >= kMaxEntries) {
    *error = "zip holds at most 65535 entries";
    return false;
  }
  const std::uint64_t header = 30 + name.size();
  // The entry and the central directory (46 + name per entry, 22 end
  // record) must stay below 4 GiB.
  std::uint64_t directory = 22;
  for (const Entry& e : entries_)
    directory += 46 + e.name.size();
  directory += 46 + name.size();
  if (written_ + header + size + directory > kMaxZipBytes) {
    *error = "zip would exceed 4 GiB";
    return false;
  }
  Entry entry{std::string(name), Crc32(data, size),
              static_cast<std::uint32_t>(size),
              static_cast<std::uint32_t>(written_)};
  std::string local;
  Put32(&local, 0x04034b50U);
  Put16(&local, 10);  // version needed: 1.0 (stored)
  Put16(&local, 0);   // flags
  Put16(&local, 0);   // method: store
  Put16(&local, kDosTime);
  Put16(&local, kDosDate);
  Put32(&local, entry.crc);
  Put32(&local, entry.size);  // compressed
  Put32(&local, entry.size);  // uncompressed
  Put16(&local, static_cast<std::uint32_t>(name.size()));
  Put16(&local, 0);  // extra
  local.append(name);
  out_.write(local.data(), static_cast<std::streamsize>(local.size()));
  if (size > 0)
    out_.write(reinterpret_cast<const char*>(data),
               static_cast<std::streamsize>(size));
  if (!out_) {
    *error = "cannot write " + path_.string();
    return false;
  }
  written_ += header + size;
  entries_.push_back(std::move(entry));
  return true;
}

bool ZipStoreWriter::Finish(std::string* error) {
  if (!out_.is_open()) {
    *error = "zip is not open";
    return false;
  }
  std::string directory;
  for (const Entry& e : entries_) {
    Put32(&directory, 0x02014b50U);
    Put16(&directory, 20);  // made by: 2.0, MS-DOS
    Put16(&directory, 10);  // needed
    Put16(&directory, 0);
    Put16(&directory, 0);
    Put16(&directory, kDosTime);
    Put16(&directory, kDosDate);
    Put32(&directory, e.crc);
    Put32(&directory, e.size);
    Put32(&directory, e.size);
    Put16(&directory, static_cast<std::uint32_t>(e.name.size()));
    Put16(&directory, 0);  // extra
    Put16(&directory, 0);  // comment
    Put16(&directory, 0);  // disk
    Put16(&directory, 0);  // internal attributes
    Put32(&directory, 0);  // external attributes
    Put32(&directory, e.offset);
    directory.append(e.name);
  }
  const auto count = static_cast<std::uint32_t>(entries_.size());
  const auto directory_size = static_cast<std::uint32_t>(directory.size());
  Put32(&directory, 0x06054b50U);
  Put16(&directory, 0);
  Put16(&directory, 0);
  Put16(&directory, count);
  Put16(&directory, count);
  Put32(&directory, directory_size);
  Put32(&directory, static_cast<std::uint32_t>(written_));
  Put16(&directory, 0);  // comment
  out_.write(directory.data(), static_cast<std::streamsize>(directory.size()));
  out_.close();
  if (out_.fail()) {
    *error = "cannot write " + path_.string();
    return false;
  }
  return true;
}

void ZipStoreWriter::Abort() {
  if (out_.is_open())
    out_.close();
  std::error_code ec;
  std::filesystem::remove(path_, ec);
}

}  // namespace gufo::server
