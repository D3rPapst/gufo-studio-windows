// One-pass parallel PNG encoder for large RGB frames (Real-ESRGAN results).
// Written with AI assistance (Claude Code).
#include "src/cli/serve/media/png_fast.hpp"

#include <zlib.h>

#include <algorithm>
#include <cstring>
#include <thread>

namespace gufo::server {
namespace {

constexpr int kMinStripRows = 64;
constexpr int kMaxThreads = 8;

void PutU32(std::vector<std::uint8_t>* out, std::uint32_t v) {
  out->push_back(static_cast<std::uint8_t>(v >> 24U));
  out->push_back(static_cast<std::uint8_t>(v >> 16U));
  out->push_back(static_cast<std::uint8_t>(v >> 8U));
  out->push_back(static_cast<std::uint8_t>(v));
}

// Appends a chunk: length, type, data, CRC over type and data.
void PutChunk(std::vector<std::uint8_t>* out, const char* type,
              const std::uint8_t* data, std::size_t size) {
  PutU32(out, static_cast<std::uint32_t>(size));
  const std::size_t start = out->size();
  out->insert(out->end(), type, type + 4);
  if (size > 0)
    out->insert(out->end(), data, data + size);
  uLong crc = crc32(0L, Z_NULL, 0);
  // crc32 takes uInt lengths; feed large chunks in pieces.
  const std::uint8_t* p = out->data() + start;
  std::size_t left = size + 4;
  while (left > 0) {
    const uInt n = static_cast<uInt>(std::min<std::size_t>(left, 1U << 30U));
    crc = crc32(crc, p, n);
    p += n;
    left -= n;
  }
  PutU32(out, static_cast<std::uint32_t>(crc));
}

struct Strip {
  int row0{0}, row1{0};
  bool last{false};
  std::vector<std::uint8_t> raw;  // filtered rows
  std::vector<std::uint8_t> deflated;
  uLong adler{1};
  bool ok{false};
};

void RunStrip(const std::uint8_t* rgb, int w, Strip* s) {
  const std::size_t stride = static_cast<std::size_t>(w) * 3;
  const int rows = s->row1 - s->row0;
  s->raw.resize(static_cast<std::size_t>(rows) * (stride + 1));
  for (int y = 0; y < rows; ++y) {
    const std::uint8_t* in =
        rgb + static_cast<std::size_t>(s->row0 + y) * stride;
    std::uint8_t* o =
        s->raw.data() + static_cast<std::size_t>(y) * (stride + 1);
    o[0] = 1;  // Sub: byte minus the same channel of the pixel to the left
    std::memcpy(o + 1, in, std::min<std::size_t>(3, stride));
    for (std::size_t i = 3; i < stride; ++i)
      o[1 + i] = static_cast<std::uint8_t>(in[i] - in[i - 3]);
  }
  s->adler = adler32_z(1L, s->raw.data(), s->raw.size());
  z_stream z{};
  if (deflateInit2(&z, 1, Z_DEFLATED, -15, 8, Z_DEFAULT_STRATEGY) != Z_OK)
    return;
  s->deflated.resize(deflateBound(&z, static_cast<uLong>(s->raw.size())) + 16);
  z.next_in = s->raw.data();
  z.avail_in = static_cast<uInt>(s->raw.size());
  z.next_out = s->deflated.data();
  z.avail_out = static_cast<uInt>(s->deflated.size());
  const int status = deflate(&z, s->last ? Z_FINISH : Z_FULL_FLUSH);
  const bool done = s->last ? status == Z_STREAM_END
                            : status == Z_OK && z.avail_in == 0;
  s->deflated.resize(z.total_out);
  deflateEnd(&z);
  std::vector<std::uint8_t>().swap(s->raw);
  s->ok = done;
}

}  // namespace

std::vector<std::uint8_t> EncodePngRgbFast(const std::uint8_t* rgb24, int w,
                                           int h, int threads) {
  if (rgb24 == nullptr || w <= 0 || h <= 0 || w > (1 << 20) || h > (1 << 20))
    return {};
  // A strip's raw bytes must fit deflate's 32-bit avail_in.
  const std::size_t row_bytes = static_cast<std::size_t>(w) * 3 + 1;
  if (threads <= 0)
    threads = static_cast<int>(
        std::clamp(std::thread::hardware_concurrency() / 2U, 1U,
                   static_cast<unsigned>(kMaxThreads)));
  int count = std::clamp(h / kMinStripRows, 1, std::max(1, threads));
  while (static_cast<std::size_t>((h + count - 1) / count) * row_bytes >
         (1U << 30U))
    ++count;
  std::vector<Strip> strips(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    strips[i].row0 = static_cast<int>(static_cast<long long>(h) * i / count);
    strips[i].row1 =
        static_cast<int>(static_cast<long long>(h) * (i + 1) / count);
    strips[i].last = i + 1 == count;
  }
  if (count == 1) {
    RunStrip(rgb24, w, &strips[0]);
  } else {
    std::vector<std::thread> pool;
    pool.reserve(strips.size());
    for (Strip& s : strips)
      pool.emplace_back(RunStrip, rgb24, w, &s);
    for (std::thread& t : pool)
      t.join();
  }
  std::size_t deflated = 0;
  uLong adler = 1;
  for (const Strip& s : strips) {
    if (!s.ok)
      return {};
    deflated += s.deflated.size();
    adler = adler32_combine(
        adler, s.adler,
        static_cast<z_off_t>(static_cast<std::size_t>(s.row1 - s.row0) *
                             row_bytes));
  }
  std::vector<std::uint8_t> idat;
  idat.reserve(deflated + 6);
  idat.push_back(0x78);  // deflate, 32 KiB window
  idat.push_back(0x01);  // fastest compression, FCHECK makes 0x7801 % 31 == 0
  for (const Strip& s : strips)
    idat.insert(idat.end(), s.deflated.begin(), s.deflated.end());
  PutU32(&idat, static_cast<std::uint32_t>(adler));
  strips.clear();

  std::vector<std::uint8_t> png;
  png.reserve(idat.size() + 64);
  static constexpr std::uint8_t kSignature[8] = {0x89, 'P',  'N',  'G',
                                                 '\r', '\n', 0x1A, '\n'};
  png.insert(png.end(), kSignature, kSignature + 8);
  std::vector<std::uint8_t> ihdr;
  PutU32(&ihdr, static_cast<std::uint32_t>(w));
  PutU32(&ihdr, static_cast<std::uint32_t>(h));
  // 8 bit, RGB, deflate, adaptive filtering, no interlace.
  ihdr.insert(ihdr.end(), {8, 2, 0, 0, 0});
  PutChunk(&png, "IHDR", ihdr.data(), ihdr.size());
  // IDAT chunks of at most 1 GiB (PNG allows 2^31 - 1).
  std::size_t offset = 0;
  do {
    const std::size_t n =
        std::min<std::size_t>(idat.size() - offset, 1U << 30U);
    PutChunk(&png, "IDAT", idat.data() + offset, n);
    offset += n;
  } while (offset < idat.size());
  PutChunk(&png, "IEND", nullptr, 0);
  return png;
}

}  // namespace gufo::server
