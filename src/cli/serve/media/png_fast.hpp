#ifndef GUFO_CLI_SERVE_MEDIA_PNG_FAST_HPP_
#define GUFO_CLI_SERVE_MEDIA_PNG_FAST_HPP_

#include <cstdint>
#include <vector>

namespace gufo::server {

/// 8-bit RGB PNG of rgb24 [h, w, 3], written in one pass: every row uses the
/// Sub filter, horizontal strips of rows are deflated (zlib level 1) on up to
/// `threads` threads and joined into one zlib stream (each strip but the last
/// ends with a full flush, the Adler-32 sums are combined). Lossless; a 3328x
/// 2048 frame takes tens of milliseconds instead of seconds. threads <= 0
/// picks min(8, hardware threads / 2). Returns an empty vector on failure.
[[nodiscard]] std::vector<std::uint8_t> EncodePngRgbFast(
    const std::uint8_t* rgb24, int w, int h, int threads = 0);

}  // namespace gufo::server

#endif  // GUFO_CLI_SERVE_MEDIA_PNG_FAST_HPP_
