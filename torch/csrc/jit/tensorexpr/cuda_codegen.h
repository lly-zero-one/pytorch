#pragma once

#include <unordered_map>
#include <unordered_set>

#include "ATen/ATen.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/nvrtc_stub/ATenNVRTC.h"
#include "c10/cuda/CUDACachingAllocator.h"
#include "c10/cuda/CUDAGuard.h"
#include "torch/csrc/jit/resource_guard.h"
#include "torch/csrc/jit/tensorexpr/codegen.h"
#include "torch/csrc/jit/tensorexpr/ir.h"
#include "torch/csrc/jit/tensorexpr/ir_printer.h"
#include "torch/csrc/jit/tensorexpr/ir_visitor.h"
#include "torch/csrc/jit/tensorexpr/unique_name_manager.h"

namespace torch {
namespace jit {
namespace tensorexpr {

// A class that overrides the underlying IRPrinter to produce Cuda C.
class CudaPrinter : public IRPrinter {
 public:
  CudaPrinter(std::ostream* os, UniqueNameManager* name_manager)
      : IRPrinter(*os), os_(os), name_manager_(name_manager) {}

  void visit(const Variable* v) override {
    os() << name_manager_->get_unique_name(v);
  }

  void visit(const For* v);

  std::ostream& os() {
    return *os_;
  }

  const std::vector<Expr>& gpu_block_extents() const {
    return gpu_block_extents_;
  }

  const std::vector<Expr>& gpu_thread_extents() const {
    return gpu_thread_extents_;
  }

 private:
  std::ostream* os_ = nullptr;
  UniqueNameManager* name_manager_ = nullptr;
  std::vector<Expr> gpu_block_extents_;
  std::vector<Expr> gpu_thread_extents_;
};

// Construct Cuda C from the buffer and tensor input, and invoke the kernel
// when real arguments are provided.
class CudaCodeGen : public CodeGen {
 public:
  template <typename... Ts>
  CudaCodeGen(const Stmt& stmt, Ts... ts)
      : CodeGen(stmt, std::forward<Ts>(ts)...) {
    Initialize();
  }

  ~CudaCodeGen() override {}

  template <typename... Ts>
  void operator()(const Ts&... ts) {
    call(std::vector<CallArg>({CallArg(ts)...}));
  }

 private:
  TORCH_API void Initialize();

  TORCH_API void call(const std::vector<CallArg>& args);

  void CompileToNVRTC(const std::string& code);

  UniqueNameManager name_manager_;
  std::ostringstream oss_;
  std::unique_ptr<CudaPrinter> printer_;
  CUfunction function_;
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
