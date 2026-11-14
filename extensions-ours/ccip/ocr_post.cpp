// imgutils.ocr.detect._boxes_from_bitmap natively. The geometry steps follow
// their reference implementations line by line so that the boxes match
// imgutils exactly on the same heat map:
//   FindContoursList  OpenCV 4.11 imgproc/src/contours_new.cpp (Suzuki border
//                     following, RETR_LIST, CHAIN_APPROX_SIMPLE)
//   ConvexHull        OpenCV 4.11 imgproc/src/convhull.cpp (Sklansky, int path)
//   MinAreaRect       OpenCV 4.11 imgproc/src/rotcalipers.cpp (float math)
//   BoxPoints         OpenCV 4.11 core/src/types.cpp RotatedRect::points
//   FillPoly          OpenCV 4.11 imgproc/src/drawing.cpp (CollectPolyEdges,
//                     FillEdgeCollection, 8-connected Line, clipLine)
//   OffsetRound       Clipper 6.4.2 ClipperOffset (pyclipper 1.4), JT_ROUND,
//                     ET_CLOSEDPOLYGON; the final union is not needed because
//                     only the convex hull of the result is used
//   polygon area/length as shapely (GEOS) computes them.
// OpenCV and Clipper are BSD/Apache-2.0 and Boost licensed; this is a port of
// the algorithms, not a copy of either library. Float expressions keep the
// operand order and precision of the originals (this file must not be built
// with -ffast-math or FMA contraction).
// Written with AI assistance (Claude Code).
#include <algorithm>
#include <array>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include "ccip/ocr.hpp"

namespace gufo::models::ccip {
namespace {

struct Pt {
  int x;
  int y;
};
struct Pt2f {
  float x;
  float y;
};
struct RotRect {
  Pt2f center{0, 0};
  Pt2f size{0, 0};  // width, height
  float angle{0};
};

// ------------------------------------------------------------- findContours
constexpr std::int8_t kMaskRight = static_cast<std::int8_t>(0x80);
constexpr std::int8_t kMaskNew = 2;
constexpr std::int8_t kMaskFlags = static_cast<std::int8_t>(0xFE);
constexpr std::int8_t kMaskBlack = 1;
constexpr int kMaxSize = 16;
constexpr Pt kChainDeltas[8] = {{1, 0}, {1, -1}, {0, -1}, {-1, -1},
                                {-1, 0}, {-1, 1}, {0, 1},  {1, 1}};

inline std::ptrdiff_t Delta(int s, std::ptrdiff_t step) {
  const Pt d = kChainDeltas[s % 8];
  return d.x + d.y * step;
}

// icvFetchContourEx<schar> for RETR_LIST (nbd = MASK8_NEW), points only at
// direction changes (CHAIN_APPROX_SIMPLE).
std::vector<Pt> FetchContour(std::int8_t* image, std::ptrdiff_t step, Pt start, bool is_hole,
                             Pt origin) {
  std::vector<Pt> pts;
  const std::int8_t nbd = kMaskNew;
  std::int8_t* i0 = image + start.y * step + start.x;
  std::int8_t* i1 = nullptr;
  std::int8_t* i3 = nullptr;
  std::int8_t* i4 = nullptr;
  Pt pt = origin;
  int s_end = is_hole ? 0 : 4;
  int s = s_end;
  do {
    s = (s - 1) & 7;
    i1 = i0 + Delta(s, step);
  } while (*i1 == 0 && s != s_end);

  if (s == s_end) {
    *i0 = static_cast<std::int8_t>(nbd | kMaskRight);
    pts.push_back(pt);
    return pts;
  }
  i3 = i0;
  int prev_s = s ^ 4;
  for (;;) {
    s_end = s;
    s = std::min(s, kMaxSize - 1);
    while (s < kMaxSize - 1) {
      ++s;
      i4 = i3 + Delta(s, step);
      if (*i4 != 0) break;
    }
    s &= 7;
    if (static_cast<unsigned>(s - 1) < static_cast<unsigned>(s_end)) {
      *i3 = static_cast<std::int8_t>(nbd | kMaskRight);
    } else if (*i3 == kMaskBlack) {
      *i3 = nbd;
    }
    if (s != prev_s) pts.push_back(pt);
    prev_s = s;
    pt.x += kChainDeltas[s].x;
    pt.y += kChainDeltas[s].y;
    if (i4 == i0 && i3 == i1) break;
    i3 = i4;
    s = (s + 4) & 7;
  }
  return pts;
}

// cv::findContours(mask, RETR_LIST, CHAIN_APPROX_SIMPLE): `mask` is width x
// height, nonzero = foreground. Result in OpenCV's order (last found first).
std::vector<std::vector<Pt>> FindContoursList(const std::uint8_t* mask, int width, int height) {
  const int pw = width + 2;
  const int ph = height + 2;
  std::vector<std::int8_t> img(static_cast<std::size_t>(pw) * ph, 0);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      img[static_cast<std::size_t>(y + 1) * pw + x + 1] =
          mask[static_cast<std::size_t>(y) * width + x] != 0 ? 1 : 0;
  std::int8_t* data = img.data();
  const std::ptrdiff_t step = pw;
  auto at = [&](int y, int x) -> int { return data[y * step + x]; };

  std::vector<std::vector<Pt>> found;
  Pt scan_pt{1, 1};
  Pt lnbd{0, 1};
  const int w1 = pw - 1;
  const int h1 = ph - 1;
  for (;;) {
    // ContourScanner_::findNext
    int x = scan_pt.x;
    int y = scan_pt.y;
    Pt last_pos = lnbd;
    int prev = at(y, x - 1);
    bool got = false;
    for (; y < h1 && !got; y++) {
      int p = 0;
      for (; x < w1; x++) {
        for (; x < w1 && (p = at(y, x)) == prev; x++) {
        }
        if (x >= w1) break;
        // contourScan
        bool is_hole = false;
        bool start = true;
        if (!(prev == 0 && p == 1)) {
          if (p != 0 || prev < 1) {
            start = false;
          } else {
            if (prev & kMaskFlags) last_pos.x = x - 1;
            is_hole = true;
          }
        }
        if (start) {
          last_pos.x = x - (is_hole ? 1 : 0);
          const Pt start_pt{x - (is_hole ? 1 : 0), y};
          found.push_back(FetchContour(data, step, start_pt, is_hole,
                                       Pt{start_pt.x - 1, start_pt.y - 1}));
          scan_pt = Pt{x + 1, y};
          lnbd = last_pos;
          got = true;
          break;
        }
        prev = p;
        if (prev & kMaskFlags) last_pos.x = x;
      }
      if (got) break;
      last_pos = Pt{0, y + 1};
      x = 1;
      prev = 0;
    }
    if (!got) break;
  }
  std::reverse(found.begin(), found.end());
  return found;
}

// ------------------------------------------------------------- convexHull
inline int Sign(long long v) { return (v > 0) - (v < 0); }

int Sklansky(const Pt* const* array, int start, int end, int* stack, int nsign, int sign2) {
  const int incr = end > start ? 1 : -1;
  int pprev = start, pcur = pprev + incr, pnext = pcur + incr;
  int stacksize = 3;
  if (start == end || (array[start]->x == array[end]->x && array[start]->y == array[end]->y)) {
    stack[0] = start;
    return 1;
  }
  stack[0] = pprev;
  stack[1] = pcur;
  stack[2] = pnext;
  end += incr;
  while (pnext != end) {
    const int cury = array[pcur]->y;
    const int nexty = array[pnext]->y;
    const int by = nexty - cury;
    if (Sign(by) != nsign) {
      const int ax = array[pcur]->x - array[pprev]->x;
      const int bx = array[pnext]->x - array[pcur]->x;
      const int ay = cury - array[pprev]->y;
      const long long convexity =
          static_cast<long long>(ay) * bx - static_cast<long long>(ax) * by;
      if (Sign(convexity) == sign2 && (ax != 0 || ay != 0)) {
        pprev = pcur;
        pcur = pnext;
        pnext += incr;
        stack[stacksize] = pnext;
        stacksize++;
      } else if (pprev == start) {
        pcur = pnext;
        stack[1] = pcur;
        pnext += incr;
        stack[2] = pnext;
      } else {
        stack[stacksize - 2] = pnext;
        pcur = pprev;
        pprev = stack[stacksize - 4];
        stacksize--;
      }
    } else {
      pnext += incr;
      stack[stacksize - 1] = pnext;
    }
  }
  return --stacksize;
}

// cv::convexHull(points, hull, clockwise=false, returnPoints=true), CV_32S.
std::vector<Pt> ConvexHull(const std::vector<Pt>& points) {
  const int total = static_cast<int>(points.size());
  if (total == 0) return {};
  const Pt* data0 = points.data();
  std::vector<const Pt*> pointer(static_cast<std::size_t>(total));
  for (int i = 0; i < total; i++) pointer[static_cast<std::size_t>(i)] = &data0[i];
  std::vector<int> stack_buf(static_cast<std::size_t>(total) + 2);
  std::vector<int> hullbuf(static_cast<std::size_t>(total));
  int* stack = stack_buf.data();
  int nout = 0;
  int miny_ind = 0, maxy_ind = 0;
  std::sort(pointer.begin(), pointer.end(), [](const Pt* p1, const Pt* p2) {
    if (p1->x != p2->x) return p1->x < p2->x;
    if (p1->y != p2->y) return p1->y < p2->y;
    return p1 < p2;
  });
  for (int i = 1; i < total; i++) {
    const int y = pointer[static_cast<std::size_t>(i)]->y;
    if (pointer[static_cast<std::size_t>(miny_ind)]->y > y) miny_ind = i;
    if (pointer[static_cast<std::size_t>(maxy_ind)]->y < y) maxy_ind = i;
  }
  const Pt* const* arr = pointer.data();
  if (pointer[0]->x == pointer[static_cast<std::size_t>(total - 1)]->x &&
      pointer[0]->y == pointer[static_cast<std::size_t>(total - 1)]->y) {
    hullbuf[static_cast<std::size_t>(nout++)] = 0;
  } else {
    int* tl_stack = stack;
    int tl_count = Sklansky(arr, 0, maxy_ind, tl_stack, -1, 1);
    int* tr_stack = stack + tl_count;
    int tr_count = Sklansky(arr, total - 1, maxy_ind, tr_stack, -1, -1);
    // clockwise == false
    std::swap(tl_stack, tr_stack);
    std::swap(tl_count, tr_count);
    for (int i = 0; i < tl_count - 1; i++)
      hullbuf[static_cast<std::size_t>(nout++)] = static_cast<int>(arr[tl_stack[i]] - data0);
    for (int i = tr_count - 1; i > 0; i--)
      hullbuf[static_cast<std::size_t>(nout++)] = static_cast<int>(arr[tr_stack[i]] - data0);
    const int stop_idx =
        tr_count > 2 ? tr_stack[1] : tl_count > 2 ? tl_stack[tl_count - 2] : -1;

    int* bl_stack = stack;
    int bl_count = Sklansky(arr, 0, miny_ind, bl_stack, 1, -1);
    int* br_stack = stack + bl_count;
    int br_count = Sklansky(arr, total - 1, miny_ind, br_stack, 1, 1);
    if (stop_idx >= 0) {
      const int check_idx = bl_count > 2               ? bl_stack[1]
                            : bl_count + br_count > 2 ? br_stack[2 - bl_count]
                                                      : -1;
      if (check_idx == stop_idx ||
          (check_idx >= 0 && arr[check_idx]->x == arr[stop_idx]->x &&
           arr[check_idx]->y == arr[stop_idx]->y)) {
        bl_count = std::min(bl_count, 2);
        br_count = std::min(br_count, 2);
      }
    }
    for (int i = 0; i < bl_count - 1; i++)
      hullbuf[static_cast<std::size_t>(nout++)] = static_cast<int>(arr[bl_stack[i]] - data0);
    for (int i = br_count - 1; i > 0; i--)
      hullbuf[static_cast<std::size_t>(nout++)] = static_cast<int>(arr[br_stack[i]] - data0);

    if (nout >= 3) {
      int min_idx = 0, max_idx = 0, lt = 0;
      for (int i = 1; i < nout; i++) {
        const int idx = hullbuf[static_cast<std::size_t>(i)];
        lt += hullbuf[static_cast<std::size_t>(i - 1)] < idx;
        if (lt > 1 && lt <= i - 2) break;
        if (idx < hullbuf[static_cast<std::size_t>(min_idx)]) min_idx = i;
        if (idx > hullbuf[static_cast<std::size_t>(max_idx)]) max_idx = i;
      }
      const int mmdist = std::abs(max_idx - min_idx);
      if ((mmdist == 1 || mmdist == nout - 1) && (lt <= 1 || lt >= nout - 2)) {
        const int ascending = (max_idx + 1) % nout == min_idx;
        const int i0 = ascending ? min_idx : max_idx;
        int j = i0;
        if (i0 > 0) {
          int i = 0;
          for (; i < nout; i++) {
            const int curr_idx = stack[i] = hullbuf[static_cast<std::size_t>(j)];
            const int next_j = j + 1 < nout ? j + 1 : 0;
            const int next_idx = hullbuf[static_cast<std::size_t>(next_j)];
            if (i < nout - 1 && (ascending != (curr_idx < next_idx))) break;
            j = next_j;
          }
          if (i == nout) std::memcpy(hullbuf.data(), stack, sizeof(int) * static_cast<std::size_t>(nout));
        }
      }
    }
  }
  std::vector<Pt> hull(static_cast<std::size_t>(nout));
  for (int i = 0; i < nout; i++) hull[static_cast<std::size_t>(i)] = data0[hullbuf[static_cast<std::size_t>(i)]];
  return hull;
}

// ------------------------------------------------------------- minAreaRect
void Rotate90CCW(const Pt2f& in, Pt2f& out) {
  out.x = -in.y;
  out.y = in.x;
}
void Rotate90CW(const Pt2f& in, Pt2f& out) {
  out.x = in.y;
  out.y = -in.x;
}
void Rotate180(const Pt2f& in, Pt2f& out) {
  out.x = -in.x;
  out.y = -in.y;
}
bool FirstVecIsRight(const Pt2f& vec1, const Pt2f& vec2) {
  Pt2f tmp;
  Rotate90CW(vec1, tmp);
  return tmp.x * vec2.x + tmp.y * vec2.y < 0;
}

// rotatingCalipers(points, n, CALIPERS_MINAREARECT, out)
void RotatingCalipers(const Pt2f* points, int n, float* out) {
  float minarea = FLT_MAX;
  union Buf {
    float f[8];
    int i[8];
  } buffer{};
  std::vector<float> inv_vect_length(static_cast<std::size_t>(n));
  std::vector<Pt2f> vect(static_cast<std::size_t>(n));
  int left = 0, bottom = 0, right = 0, top = 0;
  int seq[4] = {-1, -1, -1, -1};
  Pt2f rot_vect[4];
  float orientation = 0;
  float base_a;
  float base_b = 0;
  float left_x, right_x, top_y, bottom_y;
  Pt2f pt0 = points[0];
  left_x = right_x = pt0.x;
  top_y = bottom_y = pt0.y;
  for (int i = 0; i < n; i++) {
    double dx, dy;
    if (pt0.x < left_x) left_x = pt0.x, left = i;
    if (pt0.x > right_x) right_x = pt0.x, right = i;
    if (pt0.y > top_y) top_y = pt0.y, top = i;
    if (pt0.y < bottom_y) bottom_y = pt0.y, bottom = i;
    const Pt2f pt = points[(i + 1) & (i + 1 < n ? -1 : 0)];
    dx = pt.x - pt0.x;
    dy = pt.y - pt0.y;
    vect[static_cast<std::size_t>(i)].x = static_cast<float>(dx);
    vect[static_cast<std::size_t>(i)].y = static_cast<float>(dy);
    inv_vect_length[static_cast<std::size_t>(i)] = static_cast<float>(1. / std::sqrt(dx * dx + dy * dy));
    pt0 = pt;
  }
  {
    double ax = vect[static_cast<std::size_t>(n - 1)].x;
    double ay = vect[static_cast<std::size_t>(n - 1)].y;
    for (int i = 0; i < n; i++) {
      const double bx = vect[static_cast<std::size_t>(i)].x;
      const double by = vect[static_cast<std::size_t>(i)].y;
      const double convexity = ax * by - ay * bx;
      if (convexity != 0) {
        orientation = (convexity > 0) ? 1.f : (-1.f);
        break;
      }
      ax = bx;
      ay = by;
    }
  }
  base_a = orientation;
  seq[0] = bottom;
  seq[1] = right;
  seq[2] = top;
  seq[3] = left;
  for (int k = 0; k < n; k++) {
    int main_element = 0;
    rot_vect[0] = vect[static_cast<std::size_t>(seq[0])];
    Rotate90CW(vect[static_cast<std::size_t>(seq[1])], rot_vect[1]);
    Rotate180(vect[static_cast<std::size_t>(seq[2])], rot_vect[2]);
    Rotate90CCW(vect[static_cast<std::size_t>(seq[3])], rot_vect[3]);
    for (int i = 1; i < 4; i++) {
      if (FirstVecIsRight(rot_vect[i], rot_vect[main_element])) main_element = i;
    }
    {
      const int pindex = seq[main_element];
      const float lead_x = vect[static_cast<std::size_t>(pindex)].x * inv_vect_length[static_cast<std::size_t>(pindex)];
      const float lead_y = vect[static_cast<std::size_t>(pindex)].y * inv_vect_length[static_cast<std::size_t>(pindex)];
      switch (main_element) {
        case 0:
          base_a = lead_x;
          base_b = lead_y;
          break;
        case 1:
          base_a = lead_y;
          base_b = -lead_x;
          break;
        case 2:
          base_a = -lead_x;
          base_b = -lead_y;
          break;
        default:
          base_a = -lead_y;
          base_b = lead_x;
          break;
      }
    }
    seq[main_element] += 1;
    seq[main_element] = (seq[main_element] == n) ? 0 : seq[main_element];
    {
      float dx = points[seq[1]].x - points[seq[3]].x;
      float dy = points[seq[1]].y - points[seq[3]].y;
      const float width = dx * base_a + dy * base_b;
      dx = points[seq[2]].x - points[seq[0]].x;
      dy = points[seq[2]].y - points[seq[0]].y;
      const float height = -dx * base_b + dy * base_a;
      const float area = width * height;
      if (area <= minarea) {
        minarea = area;
        buffer.i[0] = seq[3];
        buffer.f[1] = base_a;
        buffer.f[2] = width;
        buffer.f[3] = base_b;
        buffer.f[4] = height;
        buffer.i[5] = seq[0];
        buffer.f[6] = area;
      }
    }
  }
  const float A1 = buffer.f[1];
  const float B1 = buffer.f[3];
  const float A2 = -buffer.f[3];
  const float B2 = buffer.f[1];
  const float C1 = A1 * points[buffer.i[0]].x + points[buffer.i[0]].y * B1;
  const float C2 = A2 * points[buffer.i[5]].x + points[buffer.i[5]].y * B2;
  const float idet = 1.f / (A1 * B2 - A2 * B1);
  const float px = (C1 * B2 - C2 * B1) * idet;
  const float py = (A1 * C2 - A2 * C1) * idet;
  out[0] = px;
  out[1] = py;
  out[2] = A1 * buffer.f[2];
  out[3] = B1 * buffer.f[2];
  out[4] = A2 * buffer.f[4];
  out[5] = B2 * buffer.f[4];
}

RotRect MinAreaRect(const std::vector<Pt>& points) {
  const std::vector<Pt> hull_i = ConvexHull(points);
  std::vector<Pt2f> hpoints(hull_i.size());
  for (std::size_t i = 0; i < hull_i.size(); ++i)
    hpoints[i] = Pt2f{static_cast<float>(hull_i[i].x), static_cast<float>(hull_i[i].y)};
  const int n = static_cast<int>(hpoints.size());
  RotRect box;
  if (n > 2) {
    Pt2f out[3];
    RotatingCalipers(hpoints.data(), n, reinterpret_cast<float*>(out));
    box.center.x = out[0].x + (out[1].x + out[2].x) * 0.5f;
    box.center.y = out[0].y + (out[1].y + out[2].y) * 0.5f;
    box.size.x = static_cast<float>(std::sqrt(static_cast<double>(out[1].x) * out[1].x +
                                              static_cast<double>(out[1].y) * out[1].y));
    box.size.y = static_cast<float>(std::sqrt(static_cast<double>(out[2].x) * out[2].x +
                                              static_cast<double>(out[2].y) * out[2].y));
    box.angle = static_cast<float>(std::atan2(static_cast<double>(out[1].y), static_cast<double>(out[1].x)));
  } else if (n == 2) {
    box.center.x = (hpoints[0].x + hpoints[1].x) * 0.5f;
    box.center.y = (hpoints[0].y + hpoints[1].y) * 0.5f;
    const double dx = hpoints[1].x - hpoints[0].x;
    const double dy = hpoints[1].y - hpoints[0].y;
    box.size.x = static_cast<float>(std::sqrt(dx * dx + dy * dy));
    box.size.y = 0;
    box.angle = static_cast<float>(std::atan2(dy, dx));
  } else if (n == 1) {
    box.center = hpoints[0];
  }
  // (float)(box.angle*180/CV_PI): float * int stays float, then / double.
  const float scaled = box.angle * 180;
  box.angle = static_cast<float>(scaled / 3.14159265358979323846);
  return box;
}

std::array<Pt2f, 4> BoxPoints(const RotRect& r) {
  const double angle = r.angle * 3.14159265358979323846 / 180.;
  const float b = static_cast<float>(std::cos(angle)) * 0.5f;
  const float a = static_cast<float>(std::sin(angle)) * 0.5f;
  std::array<Pt2f, 4> pt{};
  pt[0].x = r.center.x - a * r.size.y - b * r.size.x;
  pt[0].y = r.center.y + b * r.size.y - a * r.size.x;
  pt[1].x = r.center.x + a * r.size.y - b * r.size.x;
  pt[1].y = r.center.y - b * r.size.y - a * r.size.x;
  pt[2].x = 2 * r.center.x - pt[0].x;
  pt[2].y = 2 * r.center.y - pt[0].y;
  pt[3].x = 2 * r.center.x - pt[1].x;
  pt[3].y = 2 * r.center.y - pt[1].y;
  return pt;
}

// imgutils _get_mini_boxes: corners in the order top-left, top-right,
// bottom-right, bottom-left (by the sort on x, then y), and min(w, h).
std::array<Pt2f, 4> MiniBox(const std::vector<Pt>& contour, float* sside) {
  const RotRect rect = MinAreaRect(contour);
  std::array<Pt2f, 4> p = BoxPoints(rect);
  std::stable_sort(p.begin(), p.end(), [](const Pt2f& a, const Pt2f& b) { return a.x < b.x; });
  int i1, i2, i3, i4;
  if (p[1].y > p[0].y) {
    i1 = 0;
    i4 = 1;
  } else {
    i1 = 1;
    i4 = 0;
  }
  if (p[3].y > p[2].y) {
    i2 = 2;
    i3 = 3;
  } else {
    i2 = 3;
    i3 = 2;
  }
  *sside = std::min(rect.size.x, rect.size.y);
  return {p[static_cast<std::size_t>(i1)], p[static_cast<std::size_t>(i2)],
          p[static_cast<std::size_t>(i3)], p[static_cast<std::size_t>(i4)]};
}

// ------------------------------------------------------------- fillPoly
constexpr int kXYShift = 16;
constexpr std::int64_t kXYOne = std::int64_t{1} << kXYShift;

struct PolyEdge {
  int y0{0}, y1{0};
  std::int64_t x{0}, dx{0};
  PolyEdge* next{nullptr};
};

struct Mask {
  std::vector<std::uint8_t> data;
  int width{0};
  int height{0};
  void Set(int x, int y) { data[static_cast<std::size_t>(y) * width + x] = 1; }
};

bool ClipLine(std::int64_t width, std::int64_t height, std::int64_t& x1, std::int64_t& y1,
              std::int64_t& x2, std::int64_t& y2) {
  const std::int64_t right = width - 1, bottom = height - 1;
  if (width <= 0 || height <= 0) return false;
  int c1 = (x1 < 0) + (x1 > right) * 2 + (y1 < 0) * 4 + (y1 > bottom) * 8;
  int c2 = (x2 < 0) + (x2 > right) * 2 + (y2 < 0) * 4 + (y2 > bottom) * 8;
  if ((c1 & c2) == 0 && (c1 | c2) != 0) {
    std::int64_t a;
    if (c1 & 12) {
      a = c1 < 8 ? 0 : bottom;
      x1 += static_cast<std::int64_t>(static_cast<double>(a - y1) * (x2 - x1) / (y2 - y1));
      y1 = a;
      c1 = (x1 < 0) + (x1 > right) * 2;
    }
    if (c2 & 12) {
      a = c2 < 8 ? 0 : bottom;
      x2 += static_cast<std::int64_t>(static_cast<double>(a - y2) * (x2 - x1) / (y2 - y1));
      y2 = a;
      c2 = (x2 < 0) + (x2 > right) * 2;
    }
    if ((c1 & c2) == 0 && (c1 | c2) != 0) {
      if (c1) {
        a = c1 == 1 ? 0 : right;
        y1 += static_cast<std::int64_t>(static_cast<double>(a - x1) * (y2 - y1) / (x2 - x1));
        x1 = a;
        c1 = 0;
      }
      if (c2) {
        a = c2 == 1 ? 0 : right;
        y2 += static_cast<std::int64_t>(static_cast<double>(a - x2) * (y2 - y1) / (x2 - x1));
        x2 = a;
        c2 = 0;
      }
    }
  }
  return (c1 | c2) == 0;
}

// Line(img, pt1, pt2, color, 8) via LineIterator(connectivity 8, leftToRight).
void Line8(Mask& img, Pt pt1, Pt pt2) {
  std::int64_t x1 = pt1.x, y1 = pt1.y, x2 = pt2.x, y2 = pt2.y;
  if (static_cast<unsigned>(pt1.x) >= static_cast<unsigned>(img.width) ||
      static_cast<unsigned>(pt2.x) >= static_cast<unsigned>(img.width) ||
      static_cast<unsigned>(pt1.y) >= static_cast<unsigned>(img.height) ||
      static_cast<unsigned>(pt2.y) >= static_cast<unsigned>(img.height)) {
    if (!ClipLine(img.width, img.height, x1, y1, x2, y2)) return;
  }
  Pt p1{static_cast<int>(x1), static_cast<int>(y1)};
  const Pt p2{static_cast<int>(x2), static_cast<int>(y2)};
  int delta_x = 1, delta_y = 1;
  int dx = p2.x - p1.x;
  int dy = p2.y - p1.y;
  if (dx < 0) {  // leftToRight
    dx = -dx;
    dy = -dy;
    p1 = p2;
  }
  if (dy < 0) {
    dy = -dy;
    delta_y = -1;
  }
  const bool vert = dy > dx;
  if (vert) {
    std::swap(dx, dy);
    std::swap(delta_x, delta_y);
  }
  int err = dx - (dy + dy);
  const int plus_delta = dx + dx;
  const int minus_delta = -(dy + dy);
  int minus_shift = delta_x;
  int plus_shift = 0;
  int minus_step = 0;
  int plus_step = delta_y;
  const int count = dx + 1;
  if (vert) {
    std::swap(plus_step, plus_shift);
    std::swap(minus_step, minus_shift);
  }
  Pt p = p1;
  for (int i = 0; i < count; i++) {
    img.Set(p.x, p.y);
    const int mask = err < 0 ? -1 : 0;
    err += minus_delta + (plus_delta & mask);
    p.x += minus_shift + (plus_shift & mask);
    p.y += minus_step + (plus_step & mask);
  }
}

void FillEdgeCollection(Mask& img, std::vector<PolyEdge>& edges) {
  PolyEdge tmp;
  const int total = static_cast<int>(edges.size());
  int y_max = INT_MIN, y_min = INT_MAX;
  std::int64_t x_max = -1, x_min = INT64_MAX;  // 0xFFFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF
  const std::int64_t delta = kXYOne - 1;
  if (total < 2) return;
  for (int i = 0; i < total; i++) {
    PolyEdge& e1 = edges[static_cast<std::size_t>(i)];
    const std::int64_t x1 = e1.x + (e1.y1 - e1.y0) * e1.dx;
    y_min = std::min(y_min, e1.y0);
    y_max = std::max(y_max, e1.y1);
    x_min = std::min(x_min, e1.x);
    x_max = std::max(x_max, e1.x);
    x_min = std::min(x_min, x1);
    x_max = std::max(x_max, x1);
  }
  if (y_max < 0 || y_min >= img.height || x_max < 0 ||
      x_min >= (static_cast<std::int64_t>(img.width) << kXYShift))
    return;
  std::sort(edges.begin(), edges.end(), [](const PolyEdge& e1, const PolyEdge& e2) {
    return e1.y0 - e2.y0 ? e1.y0 < e2.y0 : e1.x - e2.x ? e1.x < e2.x : e1.dx < e2.dx;
  });
  tmp.y0 = INT_MAX;
  edges.push_back(tmp);
  int i = 0;
  tmp.next = nullptr;
  PolyEdge* e = &edges[static_cast<std::size_t>(i)];
  y_max = std::min(y_max, img.height);
  for (int y = e->y0; y < y_max; y++) {
    PolyEdge* last;
    PolyEdge* prelast;
    PolyEdge* keep_prelast;
    int draw = 0;
    const int clipline = y < 0;
    prelast = &tmp;
    last = tmp.next;
    while (last || e->y0 == y) {
      if (last && last->y1 == y) {
        prelast->next = last->next;
        last = last->next;
        continue;
      }
      keep_prelast = prelast;
      if (last && (e->y0 > y || last->x < e->x)) {
        prelast = last;
        last = last->next;
      } else if (i < total) {
        prelast->next = e;
        e->next = last;
        prelast = e;
        e = &edges[static_cast<std::size_t>(++i)];
      } else {
        break;
      }
      if (draw) {
        if (!clipline) {
          int x1, x2;
          if (keep_prelast->x > prelast->x) {
            x1 = static_cast<int>((prelast->x + delta) >> kXYShift);
            x2 = static_cast<int>(keep_prelast->x >> kXYShift);
          } else {
            x1 = static_cast<int>((keep_prelast->x + delta) >> kXYShift);
            x2 = static_cast<int>(prelast->x >> kXYShift);
          }
          if (x1 < img.width && x2 >= 0) {
            if (x1 < 0) x1 = 0;
            if (x2 >= img.width) x2 = img.width - 1;
            for (int xx = x1; xx <= x2; ++xx) img.Set(xx, y);
          }
        }
        keep_prelast->x += keep_prelast->dx;
        prelast->x += prelast->dx;
      }
      draw ^= 1;
    }
    keep_prelast = nullptr;
    do {
      prelast = &tmp;
      last = tmp.next;
      PolyEdge* last_exchange = nullptr;
      while (last != keep_prelast && last->next != nullptr) {
        PolyEdge* te = last->next;
        if (last->x > te->x) {
          prelast->next = te;
          last->next = te->next;
          te->next = last;
          prelast = te;
          last_exchange = prelast;
        } else {
          prelast = last;
          last = te;
        }
      }
      if (last_exchange == nullptr) break;
      keep_prelast = last_exchange;
    } while (keep_prelast != tmp.next && keep_prelast != &tmp);
  }
}

// cv::fillPoly(img, [v], 1) with LINE_8, shift 0, no offset.
void FillPoly(Mask& img, const std::array<Pt, 4>& v) {
  std::vector<PolyEdge> edges;
  const int count = 4;
  edges.reserve(count + 1);
  struct P64 {
    std::int64_t x, y;
  };
  P64 pt0{static_cast<std::int64_t>(v[count - 1].x) << kXYShift, v[count - 1].y};
  for (int i = 0; i < count; i++) {
    P64 pt1{static_cast<std::int64_t>(v[static_cast<std::size_t>(i)].x) << kXYShift,
            v[static_cast<std::size_t>(i)].y};
    P64 pt0c = pt0, pt1c = pt1;
    std::int64_t t0x = (pt0.x + (kXYOne >> 1)) >> kXYShift;
    std::int64_t t1x = (pt1.x + (kXYOne >> 1)) >> kXYShift;
    std::int64_t t0y = pt0.y, t1y = pt1.y;
    Line8(img, Pt{static_cast<int>(t0x), static_cast<int>(t0y)},
          Pt{static_cast<int>(t1x), static_cast<int>(t1y)});
    if (static_cast<std::uint64_t>(t0x) >= static_cast<std::uint64_t>(img.width) ||
        static_cast<std::uint64_t>(t1x) >= static_cast<std::uint64_t>(img.width) ||
        static_cast<std::uint64_t>(t0y) >= static_cast<std::uint64_t>(img.height) ||
        static_cast<std::uint64_t>(t1y) >= static_cast<std::uint64_t>(img.height)) {
      ClipLine(img.width, img.height, t0x, t0y, t1x, t1y);
      if (t0y != t1y) {
        pt0c.y = t0y;
        pt1c.y = t1y;
      }
    }
    pt0c.x = t0x << kXYShift;
    pt1c.x = t1x << kXYShift;
    if (pt0.y != pt1.y) {
      PolyEdge edge;
      edge.dx = (pt1c.x - pt0c.x) / (pt1c.y - pt0c.y);
      if (pt0.y < pt1.y) {
        edge.y0 = static_cast<int>(pt0.y);
        edge.y1 = static_cast<int>(pt1.y);
        edge.x = pt0c.x + (pt0.y - pt0c.y) * edge.dx;
      } else {
        edge.y0 = static_cast<int>(pt1.y);
        edge.y1 = static_cast<int>(pt0.y);
        edge.x = pt1c.x + (pt1.y - pt1c.y) * edge.dx;
      }
      edges.push_back(edge);
    }
    pt0 = pt1;
  }
  FillEdgeCollection(img, edges);
}

// imgutils _box_score_fast.
double BoxScoreFast(const float* heat, int w, int h, const std::array<Pt2f, 4>& box) {
  float min_x = box[0].x, max_x = box[0].x, min_y = box[0].y, max_y = box[0].y;
  for (const Pt2f& p : box) {
    min_x = std::min(min_x, p.x);
    max_x = std::max(max_x, p.x);
    min_y = std::min(min_y, p.y);
    max_y = std::max(max_y, p.y);
  }
  const int xmin = std::clamp(static_cast<int>(std::floor(min_x)), 0, w - 1);
  const int xmax = std::clamp(static_cast<int>(std::ceil(max_x)), 0, w - 1);
  const int ymin = std::clamp(static_cast<int>(std::floor(min_y)), 0, h - 1);
  const int ymax = std::clamp(static_cast<int>(std::ceil(max_y)), 0, h - 1);
  Mask mask;
  mask.width = xmax - xmin + 1;
  mask.height = ymax - ymin + 1;
  mask.data.assign(static_cast<std::size_t>(mask.width) * mask.height, 0);
  std::array<Pt, 4> poly{};
  for (std::size_t k = 0; k < 4; ++k) {
    // float32 - int32 stays float32 (exact here), then astype(int32) truncates.
    poly[k].x = static_cast<int>(box[k].x - static_cast<float>(xmin));
    poly[k].y = static_cast<int>(box[k].y - static_cast<float>(ymin));
  }
  FillPoly(mask, poly);
  double sum = 0.0;
  std::size_t count = 0;
  for (int y = 0; y < mask.height; ++y) {
    const float* row = heat + static_cast<std::size_t>(ymin + y) * w + xmin;
    const std::uint8_t* m = mask.data.data() + static_cast<std::size_t>(y) * mask.width;
    for (int x = 0; x < mask.width; ++x) {
      if (m[x]) {
        sum += row[x];
        ++count;
      }
    }
  }
  return count != 0 ? sum * (1. / static_cast<double>(count)) : 0.0;
}

// ------------------------------------------------------------- Clipper offset
struct IntPoint {
  std::int64_t x, y;
  bool operator==(const IntPoint& o) const { return x == o.x && y == o.y; }
  bool operator!=(const IntPoint& o) const { return !(*this == o); }
};
struct DoublePoint {
  double x, y;
};

inline std::int64_t ClipperRound(double val) {
  return (val < 0) ? static_cast<std::int64_t>(val - 0.5) : static_cast<std::int64_t>(val + 0.5);
}

DoublePoint UnitNormal(const IntPoint& pt1, const IntPoint& pt2) {
  if (pt2.x == pt1.x && pt2.y == pt1.y) return DoublePoint{0, 0};
  double dx = static_cast<double>(pt2.x - pt1.x);
  double dy = static_cast<double>(pt2.y - pt1.y);
  const double f = 1 * 1.0 / std::sqrt(dx * dx + dy * dy);
  dx *= f;
  dy *= f;
  return DoublePoint{dy, -dx};
}

double ClipperArea(const std::vector<IntPoint>& poly) {
  const int size = static_cast<int>(poly.size());
  if (size < 3) return 0;
  double a = 0;
  for (int i = 0, j = size - 1; i < size; ++i) {
    a += (static_cast<double>(poly[static_cast<std::size_t>(j)].x) + poly[static_cast<std::size_t>(i)].x) *
         (static_cast<double>(poly[static_cast<std::size_t>(j)].y) - poly[static_cast<std::size_t>(i)].y);
    j = i;
  }
  return -a * 0.5;
}

// ClipperOffset: AddPath(path, jtRound, etClosedPolygon); Execute(delta)
// without the final union (convex input, positive delta: one convex ring).
std::vector<IntPoint> OffsetRound(const std::vector<IntPoint>& path, double delta) {
  constexpr double kPi = 3.141592653589793238;
  constexpr double kTwoPi = kPi * 2;
  constexpr double kDefArcTolerance = 0.25;
  const double arc_tolerance = 0.25;  // pyclipper default
  // AddPath: strip duplicates.
  int high = static_cast<int>(path.size()) - 1;
  if (high < 0) return {};
  while (high > 0 && path[0] == path[static_cast<std::size_t>(high)]) high--;
  std::vector<IntPoint> contour;
  contour.push_back(path[0]);
  int j = 0;
  for (int i = 1; i <= high; i++) {
    if (contour[static_cast<std::size_t>(j)] != path[static_cast<std::size_t>(i)]) {
      j++;
      contour.push_back(path[static_cast<std::size_t>(i)]);
    }
  }
  if (j < 2) return {};
  // FixOrientations: the only path holds the lowest point.
  if (!(ClipperArea(contour) >= 0)) std::reverse(contour.begin(), contour.end());
  if (std::fabs(delta) < 1.0E-20) return contour;
  double y;
  if (arc_tolerance <= 0.0) y = kDefArcTolerance;
  else if (arc_tolerance > std::fabs(delta) * kDefArcTolerance) y = std::fabs(delta) * kDefArcTolerance;
  else y = arc_tolerance;
  double steps = kPi / std::acos(1 - y / std::fabs(delta));
  if (steps > std::fabs(delta) * kPi) steps = std::fabs(delta) * kPi;
  double m_sin = std::sin(kTwoPi / steps);
  const double m_cos = std::cos(kTwoPi / steps);
  const double steps_per_rad = steps / kTwoPi;
  if (delta < 0.0) m_sin = -m_sin;
  const int len = static_cast<int>(contour.size());
  if (len == 0 || (delta <= 0 && len < 3)) return {};
  std::vector<DoublePoint> normals;
  normals.reserve(static_cast<std::size_t>(len));
  for (int k = 0; k < len - 1; ++k)
    normals.push_back(UnitNormal(contour[static_cast<std::size_t>(k)], contour[static_cast<std::size_t>(k) + 1]));
  normals.push_back(UnitNormal(contour[static_cast<std::size_t>(len - 1)], contour[0]));
  std::vector<IntPoint> dest;
  int k = len - 1;
  for (int jj = 0; jj < len; ++jj) {
    const DoublePoint& nk = normals[static_cast<std::size_t>(k)];
    const DoublePoint& nj = normals[static_cast<std::size_t>(jj)];
    const IntPoint& src = contour[static_cast<std::size_t>(jj)];
    // OffsetPoint(j, k, jtRound)
    double sin_a = (nk.x * nj.y - nj.x * nk.y);
    bool done = false;
    if (std::fabs(sin_a * delta) < 1.0) {
      const double cos_a = (nk.x * nj.x + nj.y * nk.y);
      if (cos_a > 0) {
        dest.push_back(IntPoint{ClipperRound(src.x + nk.x * delta), ClipperRound(src.y + nk.y * delta)});
        done = true;
      }
    } else if (sin_a > 1.0) {
      sin_a = 1.0;
    } else if (sin_a < -1.0) {
      sin_a = -1.0;
    }
    if (!done) {
      if (sin_a * delta < 0) {
        dest.push_back(IntPoint{ClipperRound(src.x + nk.x * delta), ClipperRound(src.y + nk.y * delta)});
        dest.push_back(src);
        dest.push_back(IntPoint{ClipperRound(src.x + nj.x * delta), ClipperRound(src.y + nj.y * delta)});
      } else {
        // DoRound(j, k)
        const double a = std::atan2(sin_a, (nk.x * nj.x + nk.y * nj.y));
        const int steps_n = std::max(static_cast<int>(ClipperRound(steps_per_rad * std::fabs(a))), 1);
        double X = nk.x, Y = nk.y, X2;
        for (int i = 0; i < steps_n; ++i) {
          dest.push_back(IntPoint{ClipperRound(src.x + X * delta), ClipperRound(src.y + Y * delta)});
          X2 = X;
          X = X * m_cos - m_sin * Y;
          Y = X2 * m_sin + Y * m_cos;
        }
        dest.push_back(IntPoint{ClipperRound(src.x + nj.x * delta), ClipperRound(src.y + nj.y * delta)});
      }
    }
    k = jj;
  }
  return dest;
}

// shapely Polygon(box).area * ratio / Polygon(box).length (GEOS formulas).
double UnclipDistance(const std::array<Pt2f, 4>& box, double ratio) {
  double xs[5], ys[5];
  for (int i = 0; i < 4; ++i) {
    xs[i] = box[static_cast<std::size_t>(i)].x;
    ys[i] = box[static_cast<std::size_t>(i)].y;
  }
  xs[4] = xs[0];
  ys[4] = ys[0];
  double sum = 0.0;
  const double x0 = xs[0];
  for (int i = 1; i < 4; i++) {
    const double x = xs[i] - x0;
    const double y1 = ys[i + 1];
    const double y2 = ys[i - 1];
    sum += x * (y2 - y1);
  }
  const double area = std::fabs(sum / 2.0);
  double length = 0.0;
  for (int i = 1; i < 5; ++i) {
    const double dx = xs[i] - xs[i - 1];
    const double dy = ys[i] - ys[i - 1];
    length += std::sqrt(dx * dx + dy * dy);
  }
  return area * ratio / length;
}

}  // namespace

std::vector<TextBox> TextBoxesFromHeatmap(const float* heat, int width, int height,
                                          const OcrOptions& options) {
  std::vector<std::uint8_t> bitmap(static_cast<std::size_t>(width) * height);
  for (std::size_t i = 0; i < bitmap.size(); ++i) bitmap[i] = heat[i] >= static_cast<float>(options.heat_threshold);
  const auto contours = FindContoursList(bitmap.data(), width, height);
  const std::size_t num = std::min(contours.size(), static_cast<std::size_t>(std::max(0, options.max_candidates)));
  std::vector<TextBox> boxes;
  constexpr float kMinSize = 3;
  for (std::size_t index = 0; index < num; ++index) {
    float sside = 0;
    const std::array<Pt2f, 4> points = MiniBox(contours[index], &sside);
    if (sside < kMinSize) continue;
    const double score = BoxScoreFast(heat, width, height, points);
    if (options.box_threshold > score) continue;
    std::vector<IntPoint> path;
    for (const Pt2f& p : points)
      path.push_back(IntPoint{static_cast<std::int64_t>(p.x), static_cast<std::int64_t>(p.y)});
    const std::vector<IntPoint> grown =
        OffsetRound(path, UnclipDistance(points, options.unclip_ratio));
    if (grown.empty()) continue;
    std::vector<Pt> grown_pts;
    for (const IntPoint& p : grown) grown_pts.push_back(Pt{static_cast<int>(p.x), static_cast<int>(p.y)});
    float sside2 = 0;
    const std::array<Pt2f, 4> box = MiniBox(grown_pts, &sside2);
    if (sside2 < kMinSize + 2) continue;
    TextBox tb;
    for (std::size_t k = 0; k < 4; ++k) {
      // np.round(x / width * dest_width) in float32, dest = width here.
      const float fx = box[k].x / static_cast<float>(width) * static_cast<float>(width);
      const float fy = box[k].y / static_cast<float>(height) * static_cast<float>(height);
      tb.points[k][0] = static_cast<int>(std::clamp(std::nearbyint(fx), 0.0f, static_cast<float>(width)));
      tb.points[k][1] = static_cast<int>(std::clamp(std::nearbyint(fy), 0.0f, static_cast<float>(height)));
    }
    tb.score = score;
    boxes.push_back(tb);
  }
  return boxes;
}

}  // namespace gufo::models::ccip
