// Minimal ZIP writer without compression (method 0, "store"): local headers,
// central directory, end record; no ZIP64. Used for the lossless PNG frames
// of /v1/videos/upscale ("output_format": "png_zip").
// Written with AI assistance (Claude Code).
#ifndef GUFO_CLI_SERVE_MEDIA_ZIP_STORE_HPP_
#define GUFO_CLI_SERVE_MEDIA_ZIP_STORE_HPP_

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::server {

/// Streams entries into a ZIP file on disk. Every entry is stored as given
/// (CRC-32 from zlib), dated 1980-01-01 00:00. Limits of the plain format:
/// at most 65535 entries and 4 GiB of archive; Add() refuses past them.
class ZipStoreWriter {
public:
  bool Open(const std::filesystem::path& path, std::string* error);
  bool Add(std::string_view name, const std::uint8_t* data, std::size_t size,
           std::string* error);
  /// Writes the central directory and closes the file.
  bool Finish(std::string* error);
  /// Closes and deletes the file.
  void Abort();
  [[nodiscard]] std::size_t entries() const noexcept { return entries_.size(); }

private:
  struct Entry {
    std::string name;
    std::uint32_t crc{0};
    std::uint32_t size{0};
    std::uint32_t offset{0};
  };
  std::filesystem::path path_;
  std::ofstream out_;
  std::uint64_t written_{0};
  std::vector<Entry> entries_;
};

}  // namespace gufo::server

#endif  // GUFO_CLI_SERVE_MEDIA_ZIP_STORE_HPP_
