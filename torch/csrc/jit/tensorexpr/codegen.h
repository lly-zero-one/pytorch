#pragma once

#include "torch/csrc/jit/tensorexpr/buffer.h"
#include "torch/csrc/jit/tensorexpr/ir.h"
#include "torch/csrc/jit/tensorexpr/tensor.h"

namespace torch {
namespace jit {
namespace tensorexpr {

template <typename T>
class PaddedBuffer;

class CodeGen {
 public:
  class BufferArg;
  class CallArg;

  template <typename... Ts>
  CodeGen(const Stmt& stmt, Ts... ts)
      : ir_node_(stmt.node()), buffer_args_({BufferArg(ts)...}) {}

  template <typename... Ts>
  CodeGen(const Expr& expr, Ts... ts)
      : ir_node_(expr.node()), buffer_args_({BufferArg(ts)...}) {}

  CodeGen(const IRNode* node) : ir_node_(node) {}

  virtual ~CodeGen() {}

  RefHandle<IRNode>& ir_node() {
    return ir_node_;
  }

  const RefHandle<IRNode>& ir_node() const {
    return ir_node_;
  }

  std::vector<BufferArg>& buffer_args() {
    return buffer_args_;
  }

  const std::vector<BufferArg>& buffer_args() const {
    return buffer_args_;
  }

  virtual void bind(const BufferArg& buf, const CallArg& data) {
    LOG(FATAL) << "Unimplemented interface";
  }

  virtual void run() {
    LOG(FATAL) << "Unimplemented interface";
  }

 private:
  RefHandle<IRNode> ir_node_;
  std::vector<BufferArg> buffer_args_;
};

class CodeGen::BufferArg {
 public:
  BufferArg(const Buffer& buffer)
      : var_(buffer.data()), dtype_(buffer.dtype()) {}
  BufferArg(const Tensor& tensor)
      : var_(tensor.function().func_var()),
        dtype_(tensor.function().body().dtype()) {}
  BufferArg(const Function& func)
      : var_(func.func_var()), dtype_(func.body().dtype()) {}
  BufferArg(const Var& var) : var_(var), dtype_(var.dtype()), isVar_(true) {}

  const Var& var() const {
    return var_;
  }
  Var& var() {
    return var_;
  }
  Dtype dtype() const {
    return dtype_;
  }

  bool isVar() const {
    return isVar_;
  }

 private:
  Var var_;
  Dtype dtype_;
  bool isVar_{false};
};

class CodeGen::CallArg {
 public:
  template <typename T>
  CallArg(const PaddedBuffer<T>& buffer);

  template <typename T>
  CallArg(const std::vector<T>& buffer) : ptr_(const_cast<T*>(buffer.data())) {}

  CallArg(void* ptr) : ptr_(ptr) {}

  CallArg(int32_t i) : ival_(i) {}

  CallArg(float f) : fval_(f) {}

  void* data() const {
    return ptr_;
  }

  int32_t intData() const {
    return ival_;
  }

  float floatData() const {
    return fval_;
  }

 private:
  union {
    void* ptr_;
    float fval_;
    int32_t ival_;
  };
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
