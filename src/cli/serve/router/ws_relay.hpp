#ifndef GUFO_CLI_SERVE_ROUTER_WS_RELAY_HPP_
#define GUFO_CLI_SERVE_ROUTER_WS_RELAY_HPP_

#include <string>
#include <string_view>

#include "src/cli/serve/router/proxy.hpp"
#include "src/cli/serve/websocket.hpp"

namespace gufo::router {

/// Perform the RFC 6455 client handshake against the worker (fresh
/// Sec-WebSocket-Key, require 101 + correct Sec-WebSocket-Accept computed via
/// SHA-1), then pump raw bytes in both directions until either side closes or
/// `front.cancelled()`. Frames are never parsed or re-masked: client frames
/// are already masked and the worker's replies are not, so both directions
/// pass through untouched. The front frame reader is stopped for the duration
/// and the descriptor stays owned by the front server. Any handshake failure
/// returns false without throwing.
bool RelayWebSocket(const UpstreamTarget& target,
                    gufo::server::WebSocket& front, std::string* error);

/// Same relay, but the worker handshake carries the client's request target
/// (path plus query, e.g. "/v1/realtime?model=x") and its
/// Sec-WebSocket-Protocol offer: workers that route WebSockets by path and
/// require a subprotocol (`gufo serve asr` realtime sessions) otherwise answer
/// the root-path upgrade with a plain 200.
bool RelayWebSocket(const UpstreamTarget& target,
                    gufo::server::WebSocket& front,
                    std::string_view request_target,
                    std::string_view protocol, std::string* error);

}  // namespace gufo::router

#endif  // GUFO_CLI_SERVE_ROUTER_WS_RELAY_HPP_
