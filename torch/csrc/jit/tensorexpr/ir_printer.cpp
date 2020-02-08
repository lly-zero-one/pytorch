#include "torch/csrc/jit/tensorexpr/ir_printer.h"

namespace torch {
namespace jit {
namespace compiler {

IRPrinter::IRPrinter(std::ostream& os) : os(os) {}

void IRPrinter::print(Expr expr) {
  expr.accept(this);
}

void IRPrinter::print(Stmt stmt) {
  stmt.accept(this);
}

// TODO: change whether to include the parenthesis to the parent expression,
// we need to look at the operator precedence to make the output simpler.
#define BINARY_ACCEPT(os, v, op_str) \
  os << "(";                         \
  v->lhs().accept(this);             \
  os << " " << op_str << " ";        \
  v->rhs().accept(this);             \
  os << ")";

void IRPrinter::visit(const Add* v) {
  BINARY_ACCEPT(os, v, "+");
}

void IRPrinter::visit(const Sub* v) {
  BINARY_ACCEPT(os, v, "-");
}

void IRPrinter::visit(const Mul* v) {
  BINARY_ACCEPT(os, v, "*");
}

void IRPrinter::visit(const Div* v) {
  BINARY_ACCEPT(os, v, "/");
}

void IRPrinter::visit(const Max* v) {
  os << "Max(";
  v->lhs().accept(this);
  os << ", ";
  v->rhs().accept(this);
  os << ", " << (unsigned int)v->propagate_nans() << ")";
}

void IRPrinter::visit(const Min* v) {
  os << "Min(";
  v->lhs().accept(this);
  os << ", ";
  v->rhs().accept(this);
  os << ", " << (unsigned int)v->propagate_nans() << ")";
}

void IRPrinter::visit(const CompareSelect* v) {
  CompareSelectOperation cmp_op = v->compare_select_op();
  os << "CompareSelect(";
  v->lhs().accept(this);
  os << ", ";
  v->rhs().accept(this);
  os << ", ";
  switch (cmp_op) {
    case CompareSelectOperation::kEQ:
      os << "EQ";
      break;
    case CompareSelectOperation::kGT:
      os << "GT";
      break;
    case CompareSelectOperation::kGE:
      os << "GE";
      break;
    case CompareSelectOperation::kLT:
      os << "LT";
      break;
    case CompareSelectOperation::kLE:
      os << "LE";
      break;
    default:
      throw std::runtime_error("invalid compare select operator");
  }
  os << ")";
}

void IRPrinter::visit(const IntImm* v) {
  os << v->value();
}

void IRPrinter::visit(const FloatImm* v) {
  os << v->value();
}

void IRPrinter::visit(const Cast* v) {
  auto dtype = v->dtype();
  os << dtype << "(";
  v->src_value().accept(this);
  os << ")";
}

void IRPrinter::visit(const Variable* v) {
  os << v->name_hint();
}

void IRPrinter::visit(const Let* v) {
  os << "(let ";
  v->var().accept(this);
  os << " = ";
  v->value().accept(this);
  os << " in ";
  v->body().accept(this);
  os << ")";
}

void IRPrinter::visit(const Ramp* v) {
  os << "Ramp(" << v->base() << ", " << v->stride() << ", " << v->lanes()
     << ")";
}

void IRPrinter::visit(const Load* v) {
  // TODO: support the mask case
  os << v->base_handle() << "[" << v->index() << "]";
}

void IRPrinter::visit(const For* v) {
  std::string var_name = v->var().name_hint();
  os << "for (" << var_name << " = " << v->start() << "; " << var_name << " < "
     << v->stop() << "; " << var_name << "++) {" << std::endl;
  os << v->body() << std::endl;
  os << "}";
}

void IRPrinter::visit(const Block* v) {
  for (int i = 0; i < v->nstmts(); ++i) {
    os << v->stmt(i) << std::endl;
  }
}

void IRPrinter::visit(const Store* v) {
  // TODO: handle the mask
  os << v->base_handle() << "[" << v->index() << "] = " << v->value();
}

void IRPrinter::visit(const Broadcast* v) {
  os << "Broadcast(" << v->value() << ", " << v->lanes() << ")";
}

void IRPrinter::visit(const BaseCallNode* v) {
  os << v->func_name() << "(";
  for (int i = 0; i < v->nparams(); i++) {
    if (i > 0) {
      os << ", ";
    }
    os << v->param(i);
  }
  os << ")";
}

void IRPrinter::visit(const Allocate* v) {
  os << "Allocate(" << v->buffer_var() << ", " << v->dtype();
  os << ", {";
  const std::vector<Expr>& dims = v->dims();
  for (int i = 0; i < dims.size(); i++) {
    if (i != 0) {
      os << ", ";
    }
    os << dims[i];
  }
  os << "})";
}

void IRPrinter::visit(const Free* v) {
  os << "Free(" << v->buffer_var() << ")";
}

std::ostream& operator<<(std::ostream& stream, const Expr& expr) {
  IRPrinter p(stream);
  p.print(expr);
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const Stmt& stmt) {
  IRPrinter p(stream);
  p.print(stmt);
  return stream;
}

} // namespace compiler
} // namespace jit
} // namespace torch
