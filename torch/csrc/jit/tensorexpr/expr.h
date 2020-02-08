#pragma once

#include "torch/csrc/jit/tensorexpr/ir_mutator.h"
#include "torch/csrc/jit/tensorexpr/ir_visitor.h"
#include "torch/csrc/jit/tensorexpr/refcount.h"
#include "torch/csrc/jit/tensorexpr/types.h"

namespace torch {
namespace jit {
namespace tensorexpr {

// The commomn class between all IR nodes.
class IRNode : public RefCounted {
 public:
  virtual void accept(IRVisitor* visitor) const = 0;
  virtual ~IRNode() {}
};

// The common base between all expression node.
class Expr;
class BaseExprNode : public IRNode {
 public:
  explicit BaseExprNode(Dtype dtype) : dtype_(dtype) {}
  Dtype dtype() const {
    return dtype_;
  }
  virtual Expr accept_mutator(IRMutator* mutator) = 0;

 private:
  Dtype dtype_;
};

// The common base between all statement node.
class BaseStmtNode : public IRNode {
 public:
  BaseStmtNode() {}
  virtual Stmt accept_mutator(IRMutator* mutator) = 0;
};

// A CRTP pattern to accept visitors for children class,
// and dispatch back to the children.
template <class Op, class Base = BaseExprNode>
class ExprNode : public Base {
 public:
  using ExprNodeBase = ExprNode<Op>;
  void accept(IRVisitor* visitor) const override {
    visitor->visit(static_cast<const Op*>(this));
  }
  Expr accept_mutator(IRMutator* mutator) override;
  // pass the constructor to the base class
  using Base::Base;
};

template <class Op>
class StmtNode : public BaseStmtNode {
 public:
  using StmtNodeBase = StmtNode<Op>;
  void accept(IRVisitor* visitor) const override {
    visitor->visit(static_cast<const Op*>(this));
  }
  Stmt accept_mutator(IRMutator* mutator) override;
  StmtNode() {}
};

// A refcounted pointer to the underlying ExprNode.
// Also serves the primary way to build and operate on other expressions.
class TORCH_API Expr : public RefHandle<BaseExprNode> {
 public:
  using BaseHandle = RefHandle<BaseExprNode>;
  explicit Expr() : BaseHandle(nullptr) {}
  explicit Expr(const BaseExprNode* node) : BaseHandle(node) {}

  void accept(IRVisitor* visitor) const {
    // TODO: Consider implement this without using recursion. Otherwise,
    // if the expression tree is degenerate and too long, it could cause a
    // stack overflow.
    if (node() == nullptr) {
      return;
    }
    node()->accept(visitor);
  }

  Expr accept_mutator(IRMutator* mutator) {
    if (node() == nullptr) {
      return Expr();
    }
    return node()->accept_mutator(mutator);
  }

  Expr(int v);
  Expr(float v);

  template <class Op>
  Op* AsNode() {
    return dynamic_cast<Op*>(this->node());
  }

  template <class Op>
  const Op* AsNode() const {
    return const_cast<Expr*>(this)->AsNode<Op>();
  }

  Dtype dtype() const {
    return node()->dtype();
  }

  // Handling the math operators.
  Expr operator+(const Expr& other) const;
  Expr operator-(const Expr& other) const;
  Expr operator*(const Expr& other) const;
  Expr operator/(const Expr& other) const;
  Expr operator==(const Expr& other) const;
  Expr operator!=(const Expr& other) const;
  Expr operator>(const Expr& other) const;
  Expr operator>=(const Expr& other) const;
  Expr operator<(const Expr& other) const;
  Expr operator<=(const Expr& other) const;
};

class Stmt : public RefHandle<BaseStmtNode> {
 public:
  using BaseHandle = RefHandle<BaseStmtNode>;
  Stmt() {}
  explicit Stmt(const BaseStmtNode* node) : BaseHandle(node) {}

  void accept(IRVisitor* visitor) const {
    if (node() == nullptr) {
      return;
    }
    node()->accept(visitor);
  }

  Stmt accept_mutator(IRMutator* mutator) {
    if (node() == nullptr) {
      return Stmt();
    }
    return node()->accept_mutator(mutator);
  }

  bool empty() const {
    return node() == nullptr;
  }

  template <class Op>
  const Op* AsNode() const {
    return dynamic_cast<const Op*>(this->node());
  }
};

template <class Op, class Base>
Expr ExprNode<Op, Base>::accept_mutator(IRMutator* mutator) {
  ExprNode* this_mutable = const_cast<ExprNode*>(this);
  return mutator->mutate(static_cast<Op*>(this_mutable));
}

template <class Op>
Stmt StmtNode<Op>::accept_mutator(IRMutator* mutator) {
  StmtNode* this_mutable = const_cast<StmtNode*>(this);
  return mutator->mutate(static_cast<Op*>(this_mutable));
}

inline bool same_node(const Expr& expr1, const Expr& expr2) {
  return expr1.AsNode<BaseExprNode>() == expr2.AsNode<BaseExprNode>();
}

inline bool same_node(const Stmt& stmt1, const Stmt& stmt2) {
  return stmt1.AsNode<BaseStmtNode>() == stmt2.AsNode<BaseStmtNode>();
}

TORCH_API Expr sin(const Expr& v);
TORCH_API Expr cos(const Expr& v);
TORCH_API Expr tan(const Expr& v);
TORCH_API Expr asin(const Expr& v);
TORCH_API Expr acos(const Expr& v);
TORCH_API Expr atan(const Expr& v);
TORCH_API Expr sinh(const Expr& v);
TORCH_API Expr cosh(const Expr& v);
TORCH_API Expr tanh(const Expr& v);
TORCH_API Expr exp(const Expr& v);
TORCH_API Expr fabs(const Expr& v);
TORCH_API Expr log(const Expr& v);
TORCH_API Expr log2(const Expr& v);
TORCH_API Expr log10(const Expr& v);
TORCH_API Expr erf(const Expr& v);
TORCH_API Expr sqrt(const Expr& v);
TORCH_API Expr rsqrt(const Expr& v);
TORCH_API Expr ceil(const Expr& v);
TORCH_API Expr floor(const Expr& v);
TORCH_API Expr round(const Expr& v);
TORCH_API Expr trunc(const Expr& v);
TORCH_API Expr pow(const Expr& v1, const Expr& v2);
TORCH_API Expr fmod(const Expr& v1, const Expr& v2);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
