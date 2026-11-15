#ifndef GUFO_CLI_SERVE_ROUTER_COMFY_NODE_INFO_HPP_
#define GUFO_CLI_SERVE_ROUTER_COMFY_NODE_INFO_HPP_

// Node signatures for the ComfyUI adapter's /object_info: inputs (type,
// default, range, tooltip), outputs and category as ComfyUI reports them,
// for the classes where clients build graphs from /object_info (the
// MiniMax H3 nodes). Ranges are gufo's where gufo accepts less than
// ComfyUI (e.g. H3 width/height 256..1344); every other class keeps the
// bare "served by gufo" entry.

#include <optional>
#include <string_view>

#include "src/core/json.hpp"

namespace gufo::router::comfy {

/// The /object_info entry of `class_type`, or nullopt when gufo has no
/// signature for it.
std::optional<json::Value> NodeInfo(std::string_view class_type);

}  // namespace gufo::router::comfy

#endif  // GUFO_CLI_SERVE_ROUTER_COMFY_NODE_INFO_HPP_
