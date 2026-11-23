#pragma once

// gufo-studio-windows: eager stand-in for upstream's state memory
// (gufo/ext/experimental/state_memory.hpp: mem::SessionState, mem::Region,
// mem::StateAccount, mem::DemandMapped). This fork has no demand-mapped state
// (upstream #1251/#1308/#1333), so every declared buffer is allocated whole and
// zeroed when the session is created, and the region calls inkling2 makes before
// writes and restores have nothing to do. Consequence: a session takes its full
// context of K/V at once (32 KiB per token), so the preset must keep a bounded
// `context` (the 1M native context of upstream needs the mem layer).
// KI-unterstuetzt (Claude Code).

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdio>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/inkling2/compat/state.hpp"

namespace gufo::mem {

/// Never demand-mapped in this fork.
[[nodiscard]] inline bool DemandMapped() noexcept { return false; }

/// Upstream charges demand-mapped pages to a session's account; with whole
/// allocations there is nothing to charge, so sessions have none (null).
class StateAccount {};

class SharedPage;

/// A buffer allocated whole: every row is backed from the start, nothing is
/// shared with checkpoints, so writes and restores need no preparation.
class Region {
public:
  [[nodiscard]] bool Ensure(std::size_t, std::string* = nullptr) {
    return true;
  }
  [[nodiscard]] bool EnsureAhead(std::size_t) { return true; }
  [[nodiscard]] bool PrepareWrite(std::size_t, std::size_t, std::size_t,
                                  std::string* = nullptr) {
    return true;
  }
  [[nodiscard]] std::vector<bool> Restore(
      std::span<const std::shared_ptr<SharedPage>>, std::size_t) {
    return {};
  }
  [[nodiscard]] bool SettleRemaps(std::string* = nullptr) { return true; }
  [[nodiscard]] std::size_t shared_pages() const { return 0; }
};

class SessionState {
public:
  [[nodiscard]] static std::unique_ptr<SessionState> Create(
      const StateDeclaration& declaration, std::string* error,
      bool /*charged*/ = true) {
    std::unique_ptr<SessionState> s(new SessionState());
    const auto& buffers = declaration.buffers();
    s->data_.assign(buffers.size(), nullptr);
    s->regions_.assign(buffers.size(), nullptr);
    for (StateBufferId id = 0; id < buffers.size(); ++id) {
      const std::size_t bytes = declaration.Bytes(id, StateBytes::kSize);
      void* p = nullptr;
      hipError_t e = hipMalloc(&p, bytes);
      if (e == hipErrorOutOfMemory) {
        (void)hipGetLastError();
        std::fprintf(stderr,
                     "inkling2: session state of %zu bytes ran out of device "
                     "memory; falling back to managed memory\n",
                     bytes);
        e = hipMallocManaged(&p, bytes, hipMemAttachGlobal);
        if (e == hipSuccess)
          (void)hipMemAdvise(p, bytes, hipMemAdviseSetCoarseGrain, 0);
      }
      if (e != hipSuccess || p == nullptr) {
        (void)hipGetLastError();
        if (error)
          *error = "cannot allocate " + std::to_string(bytes) +
                   " bytes of session state: " + hipGetErrorString(e);
        return nullptr;
      }
      s->data_[id] = p;
      s->allocated_bytes_ += bytes;
      if (hipMemset(p, 0, bytes) != hipSuccess) {
        (void)hipGetLastError();
        if (error)
          *error = "cannot clear session state";
        return nullptr;
      }
      if (buffers[id].kind != StateKind::kFixed)
        s->regions_[id] = std::make_shared<Region>();
    }
    if (hipDeviceSynchronize() != hipSuccess) {
      (void)hipGetLastError();
      if (error)
        *error = "cannot clear session state";
      return nullptr;
    }
    return s;
  }
  ~SessionState() {
    for (void* p : data_)
      if (p)
        (void)hipFree(p);
  }
  SessionState(const SessionState&) = delete;
  SessionState& operator=(const SessionState&) = delete;

  [[nodiscard]] void* data(StateBufferId id) const { return data_.at(id); }
  template<typename T>
  [[nodiscard]] T* data(StateBufferId id) const {
    return static_cast<T*>(data(id));
  }
  [[nodiscard]] const std::shared_ptr<Region>& region(StateBufferId id) const {
    return regions_.at(id);
  }
  [[nodiscard]] const std::shared_ptr<StateAccount>& account() const noexcept {
    return account_;
  }
  [[nodiscard]] std::size_t allocated_bytes() const noexcept {
    return allocated_bytes_;
  }

private:
  SessionState() = default;

  std::shared_ptr<StateAccount> account_;
  std::vector<void*> data_;
  std::vector<std::shared_ptr<Region>> regions_;
  std::size_t allocated_bytes_{0};
};

}  // namespace gufo::mem
