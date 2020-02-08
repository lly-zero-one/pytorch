#pragma once

#include <cmath>
#include <unordered_map>
#include <vector>

#include <c10/util/Logging.h>
#include "torch/csrc/jit/tensorexpr/buffer.h"
#include "torch/csrc/jit/tensorexpr/function.h"
#include "torch/csrc/jit/tensorexpr/ir.h"
#include "torch/csrc/jit/tensorexpr/ir_printer.h"
#include "torch/csrc/jit/tensorexpr/tensor.h"
#include "torch/csrc/jit/tensorexpr/types.h"

namespace torch {
namespace jit {
namespace compiler {

class Value {
 public:
  Value() : dtype_(kInt32) {
    i32_values.push_back(0);
  }
  Value(int v) : dtype_(kInt32) {
    i32_values.push_back(v);
  }
  Value(float v) : dtype_(kFloat32) {
    f32_values.push_back(v);
  }
  Value(const std::vector<int>& v)
      : dtype_(Dtype(kInt32, v.size())), i32_values(v) {}
  Value(const std::vector<float>& v)
      : dtype_(Dtype(kFloat32, v.size())), f32_values(v) {}

  template <typename T>
  T as() const;

  template <typename T>
  const std::vector<T>& as_vec() const;

  Dtype dtype() const {
    return dtype_;
  }

 private:
  Dtype dtype_;
  std::vector<int32> i32_values;
  std::vector<float> f32_values;
  void* ptr;
};

template <>
inline int Value::as<int>() const {
  CHECK_EQ(dtype_, kInt32) << "invalid dtype";
  return i32_values[0];
}

template <>
inline float Value::as<float>() const {
  CHECK_EQ(dtype_, kFloat32) << "invalid dtype";
  return f32_values[0];
}

template <>
inline const std::vector<float>& Value::as_vec<float>() const {
  CHECK_EQ(dtype_.scalar_type(), kFloat32) << "invalid dtype";
  return f32_values;
}

template <>
inline const std::vector<int>& Value::as_vec<int>() const {
  CHECK_EQ(dtype_.scalar_type(), kInt32) << "invalid dtype";
  return i32_values;
}

template <typename T>
class PaddedBuffer;

class SimpleIREvaluator : public IRVisitor {
 public:
  class BufferArg {
   public:
    BufferArg(const Buffer& buffer) : var_(buffer.data()) {}
    BufferArg(const Tensor& tensor) : var_(tensor.function().func_var()) {}
    BufferArg(const Function& func) : var_(func.func_var()) {}
    const Var& var() const {
      return var_;
    }
    Var& var() {
      return var_;
    }

   private:
    Var var_;
  };

  class CallArg {
   public:
    template <typename T>
    CallArg(const PaddedBuffer<T>& buffer);

    template <typename T>
    CallArg(const std::vector<T>& buffer)
        : ptr_(const_cast<T*>(buffer.data())) {}

    void* data() {
      return ptr_;
    }

   private:
    void* ptr_ = nullptr;
  };

  SimpleIREvaluator() {}

  template <typename... Ts>
  SimpleIREvaluator(const Stmt& stmt, Ts... ts)
      : ir_node_(stmt.node()), buffer_args_({BufferArg(ts)...}) {}

  template <typename... Ts>
  SimpleIREvaluator(const Expr& expr, Ts... ts)
      : ir_node_(expr.node()), buffer_args_({BufferArg(ts)...}) {}

  template <typename Buf>
  void bindBuffer(Buf b, void *d) {
    buffer_mapping_[BufferArg(b).var().node()] = d;
  }

  void eval() {
    ir_node_.node()->accept(this);
  }

  template <typename... Ts>
  void operator()(const Ts&... ts) {
    std::vector<CallArg> args({CallArg(ts)...});
    CHECK_EQ(args.size(), buffer_args_.size());
    BufferMapping buffer_mapping;
    for (int i = 0; i < args.size(); i++) {
      buffer_mapping[buffer_args_[i].var().node()] = args[i].data();
    }
    this->SetBufferMapping(buffer_mapping);
    ir_node_.node()->accept(this);
  }

  void visit(const Add* v) override {
    visit_binary_op(v);
  }
  void visit(const Sub* v) override {
    visit_binary_op(v);
  }
  void visit(const Mul* v) override {
    visit_binary_op(v);
  }
  void visit(const Div* v) override {
    visit_binary_op(v);
  }
  void visit(const Max* v) override {
    visit_binary_op(v, v->propagate_nans());
  }
  void visit(const Min* v) override {
    visit_binary_op(v, v->propagate_nans());
  }

  void visit(const CompareSelect* v) override {
    visit_compare_select_op(v, v->compare_select_op());
  }

  template <typename T>
  Value binary_op(
      const Value& lhs,
      const Value& rhs,
      IRNodeType op_type,
      bool option = false) {
    std::vector<T> lhs_v = lhs.as_vec<T>();
    std::vector<T> rhs_v = rhs.as_vec<T>();
    std::vector<T> result_v(lhs_v.size());
    for (int i = 0; i < lhs_v.size(); i++) {
      switch (op_type) {
        case IRNodeType::kAdd:
          result_v[i] = lhs_v[i] + rhs_v[i];
          break;
        case IRNodeType::kSub:
          result_v[i] = lhs_v[i] - rhs_v[i];
          break;
        case IRNodeType::kMul:
          result_v[i] = lhs_v[i] * rhs_v[i];
          break;
        case IRNodeType::kDiv:
          result_v[i] = lhs_v[i] / rhs_v[i];
          break;
        case IRNodeType::kMax:
          result_v[i] = std::fmax(lhs_v[i], rhs_v[i]);
          if (option) {
            // Propagate NaNs
            if (std::isnan(lhs_v[i])) {
              result_v[i] = lhs_v[i];
            } else if (std::isnan(rhs_v[i])) {
              result_v[i] = rhs_v[i];
            }
          }
          break;
        case IRNodeType::kMin:
          result_v[i] = std::fmin(lhs_v[i], rhs_v[i]);
          if (option) {
            // Propagate NaNs
            if (std::isnan(lhs_v[i])) {
              result_v[i] = lhs_v[i];
            } else if (std::isnan(rhs_v[i])) {
              result_v[i] = rhs_v[i];
            }
          }
          break;
        default:
          // TODO: change to a proper error report
          throw std::runtime_error("invalid operator type");
      }
    }
    return Value(result_v);
  }

  template <typename T>
  Value compare_select_op(
      const Value& lhs,
      const Value& rhs,
      CompareSelectOperation cmp_op) {
    std::vector<T> lhs_v = lhs.as_vec<T>();
    std::vector<T> rhs_v = rhs.as_vec<T>();
    std::vector<int> result_v(lhs_v.size());
    for (int i = 0; i < lhs_v.size(); i++) {
      switch (cmp_op) {
        case CompareSelectOperation::kEQ:
          result_v[i] = (lhs_v[i] == rhs_v[i]) ? 1 : 0;
          break;
        case CompareSelectOperation::kGT:
          result_v[i] = (lhs_v[i] > rhs_v[i]) ? 1 : 0;
          break;
        case CompareSelectOperation::kGE:
          result_v[i] = (lhs_v[i] >= rhs_v[i]) ? 1 : 0;
          break;
        case CompareSelectOperation::kLT:
          result_v[i] = (lhs_v[i] < rhs_v[i]) ? 1 : 0;
          break;
        case CompareSelectOperation::kLE:
          result_v[i] = (lhs_v[i] <= rhs_v[i]) ? 1 : 0;
          break;
        default:
          // TODO: change to a proper error report
          throw std::runtime_error("invalid operator type");
      }
    }
    return Value(result_v);
  }

  template <typename Op>
  void visit_binary_op(const BinaryOpNode<Op>* v, bool option = false) {
    v->lhs().accept(this);
    Value lhs_v = value_;
    v->rhs().accept(this);
    Value rhs_v = value_;
    CHECK_EQ(lhs_v.dtype(), rhs_v.dtype());
    IRNodeType expr_type = v->expr_type();
    if (lhs_v.dtype().scalar_type() == kFloat32) {
      value_ = binary_op<float>(lhs_v, rhs_v, expr_type);
    } else if (lhs_v.dtype().scalar_type() == kInt32) {
      value_ = binary_op<int>(lhs_v, rhs_v, expr_type);
    } else {
      LOG(FATAL) << "invalid dtype: " << lhs_v.dtype();
    }
  }

  template <typename Op>
  void visit_compare_select_op(
      const BinaryOpNode<Op>* v,
      CompareSelectOperation cmp_op = CompareSelectOperation::kEQ) {
    v->lhs().accept(this);
    Value lhs_v = value_;
    v->rhs().accept(this);
    Value rhs_v = value_;
    CHECK_EQ(lhs_v.dtype(), rhs_v.dtype());
    if (lhs_v.dtype().scalar_type() == kFloat32) {
      value_ = compare_select_op<float>(lhs_v, rhs_v, cmp_op);
    } else if (lhs_v.dtype().scalar_type() == kInt32) {
      value_ = compare_select_op<int>(lhs_v, rhs_v, cmp_op);
    } else {
      LOG(FATAL) << "invalid dtype: " << lhs_v.dtype();
    }
  }

  void visit(const IntImm* v) override {
    value_ = Value(v->value());
  }
  void visit(const FloatImm* v) override {
    value_ = Value(v->value());
  }

  void visit(const Let* v) override {
    const Variable* var = v->var().AsNode<Variable>();
    CHECK(var != nullptr);
    v->value().accept(this);
    Value value = value_;
    auto iter = eval_context_.find(var);
    // TODO: make the same value settable multiple times.
    CHECK(iter == eval_context_.end())
        << "var must not exist in the context before";
    eval_context_[var] = value_;

    v->body().accept(this);

    eval_context_.erase(var);
  }

  void visit(const Variable* v) override {
    auto iter = eval_context_.find(v);
    CHECK(iter != eval_context_.end())
        << "var must be defined in the context before";
    value_ = iter->second;
  }

  void visit(const Cast* v) override {
    const Expr& src_value = v->src_value();
    src_value.accept(this);
    Dtype dst_dtype = v->dtype();
    Dtype src_dtype = src_value.dtype();
    CHECK_EQ(src_dtype.lanes(), dst_dtype.lanes());
    if (src_dtype != dst_dtype) {
      if (src_dtype == kFloat32 && dst_dtype == kInt32) {
        const std::vector<float>& src_values = value_.as_vec<float>();
        std::vector<int> dst_values(src_values.size());
        for (int i = 0; i < src_dtype.lanes(); ++i) {
          dst_values[i] = static_cast<int>(src_values[i]);
        }
        this->value_ = Value(dst_values);
      } else if (src_dtype == kInt32 && dst_dtype == kFloat32) {
        const std::vector<int>& src_values = value_.as_vec<int>();
        std::vector<float> dst_values(src_values.size());
        for (int i = 0; i < src_dtype.lanes(); ++i) {
          dst_values[i] = static_cast<float>(src_values[i]);
        }
        this->value_ = Value(dst_values);
      }
    }
  }

  void visit(const For* v) override {
    const BaseExprNode* var_node = v->var().node();
    v->start().accept(this);
    int start = value_.as<int>();
    v->stop().accept(this);
    int stop = value_.as<int>();
    auto iter = eval_context_.find(var_node);
    CHECK(iter == eval_context_.end())
        << "var in For must not exist in eval context";
    for (int i = start; i < stop; i++) {
      eval_context_[var_node] = Value(i);
      v->body().accept(this);
    }
    eval_context_.erase(var_node);
  }

  void visit(const Ramp* v) override {
    v->base().accept(this);
    int base = value().as<int>();
    v->stride().accept(this);
    int stride = value().as<int>();
    int lanes = v->lanes();

    std::vector<int> values(lanes);
    for (int i = 0; i < lanes; i++) {
      values[i] = base + i * stride;
    }

    value_ = Value(values);
  }

  void visit(const Broadcast* v) override {
    v->value().accept(this);
    Value value = this->value();
    int lanes = v->lanes();
    if (value.dtype() == kInt32) {
      std::vector<int> v(lanes, value.as<int>());
      value_ = Value(v);
    } else if (value.dtype() == kFloat32) {
      std::vector<float> v(lanes, value.as<float>());
      value_ = Value(v);
    } else {
      LOG(FATAL) << "invalid dtype: " << value.dtype();
    }
  }

  void visit(const Load* v) override {
    const Variable* base_node = v->base_handle().node();
    auto iter = buffer_mapping_.find(base_node);
    CHECK(iter != buffer_mapping_.end());
    void* ptr = iter->second;

    v->index().accept(this);
    std::vector<int> index = value().as_vec<int>();
    v->mask().accept(this);
    std::vector<int> mask = value().as_vec<int>();
    Dtype v_sdtype = v->dtype().scalar_type();
    if (v_sdtype == kFloat32) {
      float* ptr_f = static_cast<float*>(ptr);
      std::vector<float> v(index.size());
      for (int i = 0; i < index.size(); i++) {
        if (mask[i]) {
          v[i] = ptr_f[index[i]];
        }
      }
      value_ = Value(v);
    } else if (v_sdtype == kInt32) {
      int* ptr_i = static_cast<int*>(ptr);
      std::vector<int> v(index.size());
      for (int i = 0; i < index.size(); i++) {
        if (mask[i]) {
          v[i] = ptr_i[index[i]];
        }
      }
      value_ = Value(v);
    } else {
      LOG(FATAL) << "Invalid dtype: " << v_sdtype;
    }
  }

  void visit(const Store* v) override {
    const Variable* base_node = v->base_handle().node();
    auto iter = buffer_mapping_.find(base_node);
    CHECK(iter != buffer_mapping_.end());
    void* ptr = iter->second;

    v->index().accept(this);
    std::vector<int> index = value().as_vec<int>();
    v->mask().accept(this);
    std::vector<int> mask = value().as_vec<int>();
    CHECK_EQ(index.size(), mask.size());
    Dtype v_sdtype = v->value().dtype().scalar_type();
    if (v_sdtype == kFloat32) {
      v->value().accept(this);
      std::vector<float> value = this->value().as_vec<float>();
      CHECK_EQ(index.size(), value.size());
      float* ptr_f = static_cast<float*>(ptr);
      for (int i = 0; i < index.size(); i++) {
        if (mask[i]) {
          ptr_f[index[i]] = value[i];
        }
      }
    } else if (v_sdtype == kInt32) {
      v->value().accept(this);
      std::vector<int> value = this->value().as_vec<int>();
      CHECK_EQ(index.size(), value.size());
      int* ptr_i = static_cast<int*>(ptr);
      for (int i = 0; i < index.size(); i++) {
        if (mask[i]) {
          ptr_i[index[i]] = value[i];
        }
      }
    } else {
      LOG(FATAL) << "Invalid dtype: " << v_sdtype;
    }
  }

  void visit(const BaseCallNode* v) override {
    LOG(FATAL) << "unsupported";
  }

  void visit(const Intrinsics* v) override {
    std::vector<Value> values(v->nparams());
    for (int i = 0; i < v->nparams(); i++) {
      v->param(i).accept(this);
      values[i] = this->value();
    }
    std::vector<float> v1;
    if (values.size() >= 1) {
      v1 = values[0].as_vec<float>();
    }
    std::vector<float> v2;
    if (values.size() >= 2) {
      v2 = values[1].as_vec<float>();
      CHECK_EQ(v1.size(), v2.size()) << "mismatch vectorize sizes";
    }
    CHECK_LE(values.size(), 2)
        << "no support for intrinsics for more than two operand yet";
    std::vector<float> result(v1.size(), -1);
    if (values.size() == 1) {
      for (int i = 0; i < v1.size(); i++) {
        result[i] = compute_intrinsics(v->op_type(), v1[i]);
      }
    } else {
      for (int i = 0; i < v1.size(); i++) {
        result[i] = compute_intrinsics(v->op_type(), v1[i], v2[i]);
      }
    }
    value_ = Value(result);
  }

  void visit(const Allocate* v) override {
    const Variable* buffer_var = v->buffer_var().AsNode<Variable>();
    std::vector<Expr> dims = v->dims();
    int total_byte_size = v->dtype().byte_size();
    for (int i = 0; i < dims.size(); i++) {
      dims[i].accept(this);
      total_byte_size *= value_.as<int>();
    }
    int int_count = (total_byte_size + sizeof(int) - 1) / sizeof(int);
    std::unique_ptr<std::vector<int>> buffer(new std::vector<int>(int_count));
    auto iter = buffer_mapping_.find(buffer_var);
    if (iter != buffer_mapping_.end() && iter->second != nullptr) {
      throw std::runtime_error(
          "Allocate a buffer that has already been allocated: " +
          buffer_var->name_hint());
    }
    buffer_mapping_[buffer_var] = buffer->data();
    internal_buffers_.insert(std::make_pair(buffer_var, std::move(buffer)));
  }

  void visit(const Free* v) override {
    const Variable* buffer_var = v->buffer_var().AsNode<Variable>();
    int count = internal_buffers_.erase(buffer_var);
    if (count == 0) {
      throw std::runtime_error(
          "Free a buffer that is not currently bound: " +
          buffer_var->name_hint());
    }
  }

  Value value() const {
    return value_;
  }

 private:
  static float compute_intrinsics(IntrinsicsOp op_type, float v) {
    switch (op_type) {
      case kSin:
        return std::sin(v);
      case kCos:
        return std::cos(v);
      case kTan:
        return std::tan(v);
      case kAsin:
        return std::asin(v);
      case kAcos:
        return std::acos(v);
      case kAtan:
        return std::atan(v);
      case kSinh:
        return std::sinh(v);
      case kCosh:
        return std::cosh(v);
      case kTanh:
        return std::tanh(v);
      case kExp:
        return std::exp(v);
      case kFabs:
        return std::fabs(v);
      case kLog:
        return std::log(v);
      case kLog2:
        return std::log2(v);
      case kLog10:
        return std::log10(v);
      case kErf:
        return std::erf(v);
      case kSqrt:
        return std::sqrt(v);
      case kRsqrt:
        return 1.0f / std::sqrt(v);
      case kCeil:
        return std::ceil(v);
      case kFloor:
        return std::floor(v);
      case kRound:
        return std::round(v);
      case kTrunc:
        return std::trunc(v);
      default:
        throw std::runtime_error("invalid op_type: " + std::to_string(op_type));
    }
  }

  static float compute_intrinsics(IntrinsicsOp op_type, float v1, float v2) {
    switch (op_type) {
      case kPow:
        return std::pow(v1, v2);
      case kFmod:
        return std::fmod(v1, v2);
      default:
        throw std::runtime_error("nvalid op_type: " + std::to_string(op_type));
    }
  }

  using BufferMapping = std::unordered_map<const BaseExprNode*, void*>;
  void SetBufferMapping(const BufferMapping& buffer_mapping) {
    buffer_mapping_ = buffer_mapping;
  }
  void SetBufferMapping(const std::vector<std::pair<Var, void*>>& entries) {
    for (const std::pair<Var, void*>& entry : entries) {
      buffer_mapping_[entry.first.node()] = entry.second;
    }
  }

  RefHandle<IRNode> ir_node_;
  std::vector<BufferArg> buffer_args_;

  Value value_;
  std::unordered_map<const BaseExprNode*, Value> eval_context_;
  BufferMapping buffer_mapping_;
  std::unordered_map<const Variable*, std::unique_ptr<std::vector<int>>>
      internal_buffers_;
};

using VarMapping = std::vector<std::pair<Expr, Expr>>;

class VarSubMutator : public IRMutator {
 public:
  VarSubMutator(const VarMapping& var_mapping) {
    for (const auto& entry : var_mapping) {
      const Expr& key = entry.first;
      const Expr& value = entry.second;
      const Variable* key_var = key.AsNode<Variable>();
      CHECK(key_var != nullptr);
      var_mapping_[key_var] = value;
    }
  }

  Expr mutate(const Variable* var) override {
    auto iter = var_mapping_.find(var);
    if (iter == var_mapping_.end()) {
      return Expr(const_cast<Variable*>(var));
    }
    return iter->second;
  }

 private:
  std::unordered_map<const Variable*, Expr> var_mapping_;
};

inline Expr Substitute(Expr* expr, const VarMapping& var_mapping) {
  VarSubMutator var_sub(var_mapping);
  return expr->accept_mutator(&var_sub);
}

inline Stmt Substitute(Stmt* stmt, const VarMapping& var_mapping) {
  VarSubMutator var_sub(var_mapping);
  return stmt->accept_mutator(&var_sub);
}

} // namespace compiler
} // namespace jit
} // namespace torch
