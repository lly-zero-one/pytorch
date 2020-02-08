#include "torch/csrc/jit/tensorexpr/expr.h"

#include "torch/csrc/jit/tensorexpr/ir.h"

namespace torch {
namespace jit {
namespace compiler {

Expr Expr::operator+(const Expr& other) const {
  return Add::make(*this, other);
}

Expr Expr::operator-(const Expr& other) const {
  return Sub::make(*this, other);
}

Expr Expr::operator*(const Expr& other) const {
  return Mul::make(*this, other);
}

Expr Expr::operator/(const Expr& other) const {
  return Div::make(*this, other);
}

Expr::Expr(int v) : Expr(IntImm::make(v)) {}

Expr::Expr(float v) : Expr(FloatImm::make(v)) {}

Expr sin(const Expr& v) {
  return Intrinsics::make(kSin, v);
}

Expr cos(const Expr& v) {
  return Intrinsics::make(kCos, v);
}

Expr tan(const Expr& v) {
  return Intrinsics::make(kTan, v);
}

Expr asin(const Expr& v) {
  return Intrinsics::make(kAsin, v);
}

Expr acos(const Expr& v) {
  return Intrinsics::make(kAcos, v);
}

Expr atan(const Expr& v) {
  return Intrinsics::make(kAtan, v);
}

Expr sinh(const Expr& v) {
  return Intrinsics::make(kSinh, v);
}

Expr cosh(const Expr& v) {
  return Intrinsics::make(kCosh, v);
}

Expr tanh(const Expr& v) {
  return Intrinsics::make(kTanh, v);
}

Expr exp(const Expr& v) {
  return Intrinsics::make(kExp, v);
}

Expr fabs(const Expr& v) {
  return Intrinsics::make(kFabs, v);
}

Expr log(const Expr& v) {
  return Intrinsics::make(kLog, v);
}

Expr log2(const Expr& v) {
  return Intrinsics::make(kLog2, v);
}

Expr log10(const Expr& v) {
  return Intrinsics::make(kLog10, v);
}

Expr erf(const Expr& v) {
  return Intrinsics::make(kErf, v);
}

Expr sqrt(const Expr& v) {
  return Intrinsics::make(kSqrt, v);
}

Expr rsqrt(const Expr& v) {
  return Intrinsics::make(kRsqrt, v);
}

Expr ceil(const Expr& v) {
  return Intrinsics::make(kCeil, v);
}

Expr floor(const Expr& v) {
  return Intrinsics::make(kFloor, v);
}

Expr round(const Expr& v) {
  return Intrinsics::make(kRound, v);
}

Expr trunc(const Expr& v) {
  return Intrinsics::make(kTrunc, v);
}

Expr pow(const Expr& v1, const Expr& v2) {
  return Intrinsics::make(kPow, v1, v2);
}

Expr fmod(const Expr& v1, const Expr& v2) {
  return Intrinsics::make(kFmod, v1, v2);
}

} // namespace compiler
} // namespace jit
} // namespace torch
