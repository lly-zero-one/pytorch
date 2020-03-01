#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include "test/cpp/tensorexpr/test_base.h"

#include "test/cpp/tensorexpr/padded_buffer.h"
#include "torch/csrc/jit/tensorexpr/buffer.h"
#include "torch/csrc/jit/tensorexpr/eval.h"
#include "torch/csrc/jit/tensorexpr/function.h"
#include "torch/csrc/jit/tensorexpr/ir.h"
#include "torch/csrc/jit/tensorexpr/ir_printer.h"
#include "torch/csrc/jit/tensorexpr/schedule.h"
#include "torch/csrc/jit/tensorexpr/tensor.h"

namespace torch {
namespace jit {

using namespace torch::jit::tensorexpr;
using namespace torch::jit::tensorexpr::schedule;

void testExprSimple01() {
  KernelScope kernel_scope;
  Tensor tensor =
      Compute("f", {{16, "X"}, {5, "y"}}, [](const Var& x, const Var& y) {
        return Expr(1.0f) + cast<float>(x) * x + cast<float>(y) * y;
      });
  Var x = tensor.function().arg(0);
  Var y = tensor.function().arg(1);
  Schedule sch = Schedule::make({tensor});
  Var x_outer;
  Var x_inner;
  Var x_tail;
  TensorOperation tail_op;
  tensor.SplitWithTail(x, 2, true, &x_outer, &x_inner, &x_tail, &tail_op);

  Var x_2;
  Var x_1;
  Var x_tail_2;
  TensorOperation tail_op_2;
  tensor.SplitWithTail(x_outer, 2, true, &x_2, &x_1, &x_tail_2, &tail_op_2);
}

void testExprLower01() {
  KernelScope kernel_scope;
  Tensor tensor =
      Compute("f", {{16, "x"}, {5, "y"}}, [](const Var& x, const Var& y) {
        return Expr(1.0f) + cast<float>(x) * x + cast<float>(y) * y;
      });
  Var x = tensor.function().arg(0);
  Var y = tensor.function().arg(1);
  Schedule sch = Schedule::make({tensor});
  Stmt stmt = sch.Lower();
  std::ostringstream oss;
  oss << stmt;
  ASSERT_GT(oss.str().size(), 20);
  ASSERT_LT(oss.str().size(), 200);
}

void testExprSimple02() {
  KernelScope kernel_scope;
  auto func = [](const Expr& x, const Expr& y) {
    return Expr(1.0f) + cast<float>(x) * x + cast<float>(y) * y;
  };
  Tensor tensor = Compute("f", {{26, "x"}, {5, "y"}}, func);
  Var x = tensor.function().arg(0);
  Var y = tensor.function().arg(1);
  Schedule sch = Schedule::make({tensor});
  Var x_outer;
  Var x_inner;
  Var x_tail;
  TensorOperation tail_op;
  tensor.SplitWithTail(x, 4, true, &x_outer, &x_inner, &x_tail, &tail_op);

  Stmt stmt = sch.Lower();
  std::ostringstream oss;
  oss << stmt;
  ASSERT_GT(oss.str().size(), 200);
  ASSERT_LT(oss.str().size(), 600);

  {
    // Compare to a reference loop structure structure.
    Var x_outer("x.outer", kInt32);
    Var x_inner("x.inner", kInt32);
    Var y("y", kInt32);
    Var x_tail("x.tail", kInt32);
    Var f("f", kHandle);
    Expr x_1 = x_outer * 4 + x_inner;
    Stmt stmt1 = For::make(
        x_outer,
        0,
        6,
        For::make(
            x_inner,
            0,
            4,
            For::make(
                y, 0, 5, Store::make(f, x_1 * 5 + y * 1, func(x_1, y), 1))));
    Expr x_2 = x_tail + Expr(6) * 4;
    Stmt stmt2 = For::make(
        x_tail,
        0,
        2,
        For::make(y, 0, 5, Store::make(f, x_2 * 5 + y * 1, func(x_2, y), 1)));
    Stmt stmt = Block::make({stmt1, stmt2});

    std::ostringstream oss_ref;
    oss_ref << stmt;
    ASSERT_EQ(oss.str(), oss_ref.str());
  }

  {
    PaddedBuffer<float> f_v(26, 5, "f_v");
    PaddedBuffer<float> f_ref(26, 5, "f_res");

    SimpleIREvaluator ir_eval(stmt, tensor);
    ir_eval(f_v);

    for (int x = 0; x < 26; x++) {
      for (int y = 0; y < 5; y++) {
        f_ref(x, y) = 1 + x * x + y * y;
      }
    }

    ExpectAllNear(f_v, f_ref, 1e-5);
  }
}

void testExprSplitWithMask01() {
  KernelScope kernel_scope;
  const int M = 26;
  const int N = 5;
  Buffer a_buf("a", kFloat32, {M, N});
  Buffer b_buf("b", kFloat32, {M, N});
  Tensor tensor =
      Compute("f", {{M, "m"}, {N, "n"}}, [&](const Expr& m, const Expr& n) {
        return a_buf(m, n) + b_buf(m, n) + 1.0f;
      });
  Var m = tensor.function().arg(0);
  Var n = tensor.function().arg(1);
  Var n_outer;
  Var n_inner;

  Schedule sch({tensor});
  tensor.SplitWithMask(n, 4, true, &n_outer, &n_inner);

  Stmt stmt = sch.Lower();

  PaddedBuffer<float> a_v(M, N, "a");
  PaddedBuffer<float> b_v(M, N, "b");
  PaddedBuffer<float> c_v(M, N, "c");
  PaddedBuffer<float> c_ref(M, N, "c_ref");
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      a_v(m, n) = 2 * m;
      b_v(m, n) = 3 * n;
      c_ref(m, n) = a_v(m, n) + b_v(m, n) + 1.0f;
    }
  }

  SimpleIREvaluator(stmt, a_buf, b_buf, tensor)(a_v, b_v, c_v);

  ExpectAllNear(c_v, c_ref, 1e-5);
}

void testScheduleBroadcastAddBuffer() {
  KernelScope kernel_scope;
  const int M = 4;
  const int N = 5;
  const int K = 6;
  Buffer a_buf("a", kFloat32, {M, N});
  Buffer b_buf("b", kFloat32, {N, K});
  Tensor c = Compute(
      "broadcast_add",
      {{M, "m"}, {N, "n"}, {K, "k"}},
      [&](const Var& m, const Var& n, const Var& k) {
        return a_buf(m, n) + b_buf(n, k);
      });
  Schedule sch({c});
  Stmt stmt = sch.Lower();

  PaddedBuffer<float> a_v(M, N, "a_v");
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      a_v(m, n) = 7 * m * n;
    }
  }
  a_v.Backup();

  PaddedBuffer<float> b_v(N, K, "b_v");
  for (int n = 0; n < N; n++) {
    for (int k = 0; k < K; k++) {
      b_v(n, k) = 11 * n * k;
    }
  }
  b_v.Backup();

  PaddedBuffer<float> c_v(M, N, K, "c_buf");
  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c);
  ir_eval(a_v, b_v, c_v);

  a_v.CheckBackup();
  b_v.CheckBackup();
  PaddedBuffer<float> c_ref(M, N, K, "c_ref");
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      for (int k = 0; k < K; k++) {
        c_ref(m, n, k) = 7 * m * n + 11 * n * k;
      }
    }
  }
  ExpectAllNear(c_v, c_ref, 1e-5);
}

void testScheduleFunctionCall01() {
  KernelScope kernel_scope;
  const int M = 4;
  const int N = 5;
  const int K = 6;
  Buffer a_buf("a", kFloat32, {M, N});
  Buffer b_buf("b", kFloat32, {N, K});
  Tensor c = Compute(
      "broadcast_add",
      {{M, "m"}, {N, "n"}, {K, "k"}},
      [&](const Var& m, const Var& n, const Var& k) {
        return a_buf(m, n) + b_buf(n, k);
      });
  Tensor d = Compute(
      "d",
      {{M, "m"}, {N, "n"}, {K, "k"}},
      [&](const Var& m, const Var& n, const Var& k) { return c(m, n, k) + 1; });

  Schedule sch({d});
  Stmt stmt = sch.Lower();
  std::ostringstream oss;
  oss << stmt;
  ASSERT_GT(oss.str().size(), 100);

  PaddedBuffer<float> a_v(M, N);
  PaddedBuffer<float> b_v(N, K);
  PaddedBuffer<float> c_v(M, N, K);
  PaddedBuffer<float> d_v(M, N, K);
  PaddedBuffer<float> d_ref(M, N, K);

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      a_v(i, j) = i * i;
    }
  }
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < K; j++) {
      b_v(i, j) = j * j;
    }
  }
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < K; k++) {
        d_ref(i, j, k) = a_v(i, j) + b_v(j, k) + 1;
      }
    }
  }

  SimpleIREvaluator eval(stmt, a_buf, b_buf, d);
  eval(a_v, b_v, d_v);

  ExpectAllNear(d_v, d_ref, 1e-5);
}

static std::string remove_space(const std::string& str) {
  std::string str_new = str;
  str_new.erase(
      remove_if(str_new.begin(), str_new.end(), isspace), str_new.end());
  return str_new;
}

void InlineFunc01Helper(const std::vector<std::string>& inline_order) {
  KernelScope kernel_scope;
  const int M = 4;
  const int N = 5;
  const int K = 6;
  Buffer a_buf("a", kFloat32, {M, N});
  Buffer b_buf("b", kFloat32, {N, K});
  Buffer c_buf("c", kFloat32, {M, N});
  Buffer d_buf("d", kFloat32, {M, K});

  Tensor x = Compute(
      "x",
      {{M, "m1"}, {N, "n1"}, {K, "k1"}},
      [&](const Var& m, const Var& n, const Var& k) {
        return a_buf(m, n) * b_buf(n, k);
      });
  Tensor y = Compute(
      "y",
      {{M, "m2"}, {N, "n2"}, {K, "k2"}},
      [&](const Var& m, const Var& n, const Var& k) {
        return c_buf(m, n) * d_buf(m, k) + x(m, n, k);
      });
  Tensor z = Compute(
      "z",
      {{M, "m3"}, {N, "n3"}, {K, "k3"}},
      [&](const Var& m, const Var& n, const Var& k) {
        return x(m, n, k) + y(m, n, k);
      });

  Schedule sch({z});
  for (const std::string& order : inline_order) {
    if (order == "x") {
      x.ComputeInline();
    } else if (order == "y") {
      y.ComputeInline();
    } else {
      throw std::runtime_error("Invalid order: " + order);
    }
  }
  Stmt stmt = sch.Lower();

  std::ostringstream oss;
  oss << stmt;
  std::string str1 = remove_space(oss.str());

  {
    PaddedBuffer<float> a_v(M, N);
    PaddedBuffer<float> b_v(N, K);
    PaddedBuffer<float> c_v(M, N);
    PaddedBuffer<float> d_v(M, K);

    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        a_v(i, j) = i * i;
      }
    }
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < K; j++) {
        a_v(i, j) = j * j;
      }
    }
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        c_v(i, j) = i + j;
      }
    }
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < K; j++) {
        d_v(i, j) = i * j;
      }
    }

    PaddedBuffer<float> z_v(M, N, K);
    PaddedBuffer<float> z_ref(M, N, K);
    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
          z_ref(m, n, k) = a_v(m, n) * b_v(n, k) * 2 + c_v(m, n) * d_v(m, k);
        }
      }
    }

    SimpleIREvaluator eval(stmt, a_buf, b_buf, c_buf, d_buf, z);
    eval(a_v, b_v, c_v, d_v, z_v);
    ExpectAllNear(z_v, z_ref, 1e-5);
  }

  if (inline_order.size() == 2) {
    Tensor z2 = Compute(
        "z",
        {{M, "m3"}, {N, "n3"}, {K, "k3"}},
        [&](const Var& m, const Var& n, const Var& k) {
          return a_buf(m, n) * b_buf(n, k) +
              (c_buf(m, n) * d_buf(m, k) + a_buf(m, n) * b_buf(n, k));
        });
    Schedule sch2({z2});
    Stmt stmt2 = sch2.Lower();

    std::ostringstream oss2;
    oss2 << stmt2;
    std::string str2 = remove_space(oss2.str());

    ASSERT_EQ(str1, str2);
    ASSERT_GT(str1.size(), 100);
  }
}

void testScheduleInlineFunc01() {
  InlineFunc01Helper({"x", "y"});
  InlineFunc01Helper({"y", "x"});
  InlineFunc01Helper({"x"});
  InlineFunc01Helper({"y"});
  InlineFunc01Helper({});
}

void testScheduleFuserStyle() {
  KernelScope kernel_scope;
  const int kVectorSize = 8;
  const int kVectorCount = 128;
  const int kTotalSize = kVectorSize * kVectorCount;

  Buffer a_buf(Var("A", kHandle), kFloat32, {Expr(kTotalSize)});
  Var a = a_buf.data();

  Tensor b =
      Compute("f", {{kTotalSize, "i"}}, [&](const std::vector<Var>& axes) {
        return a_buf(axes[0]) + 11.0f;
      });

  Tensor c =
      Compute("g", {{kTotalSize, "i"}}, [&](const std::vector<Var>& axes) {
        return b(axes[0]) + 1.0f;
      });

  Schedule sch({b, c});
  Stmt s = sch.Lower();

  std::vector<float> a_data(kTotalSize, 7.0f);
  std::vector<float> b_data(kTotalSize, 0.0f);
  std::vector<float> c_data(kTotalSize, 0.0f);
  SimpleIREvaluator(s, a_buf, b, c)(a_data, b_data, c_data);

  for (int i = 0; i < kTotalSize; i++) {
    ASSERT_EQ(b_data[i], 18.0f);
    ASSERT_EQ(c_data[i], 19.0f);
  }
}

void testScheduleFuserThreeArg() {
  KernelScope kernel_scope;
  const int kVectorSize = 8;
  const int kVectorCount = 128;
  const int kTotalSize = kVectorSize * kVectorCount;

  Buffer a(Var("A", kHandle), kFloat32, {Expr(kTotalSize)});
  Buffer b(Var("B", kHandle), kFloat32, {Expr(kTotalSize)});
  Buffer c(Var("C", kHandle), kFloat32, {Expr(kTotalSize)});
  Buffer d(Var("D", kHandle), kFloat32, {Expr(kTotalSize)});

  Tensor e = Compute(
      "e", {{kTotalSize, "i"}}, [&](const Var& i) { return a(i) + b(i); });
  Tensor f = Compute(
      "f", {{kTotalSize, "i"}}, [&](const Var& i) { return e(i) + c(i); });
  Tensor g = Compute(
      "g", {{kTotalSize, "i"}}, [&](const Var& i) { return f(i) + d(i); });

  Schedule sch({g});
  e.ComputeInline();
  f.ComputeInline();
  Stmt s = sch.Lower();

  std::vector<float> a_data(kTotalSize, 1.0f);
  std::vector<float> b_data(kTotalSize, 2.0f);
  std::vector<float> c_data(kTotalSize, 3.0f);
  std::vector<float> d_data(kTotalSize, 4.0f);
  std::vector<float> g_data(kTotalSize, 0.0f);
  SimpleIREvaluator(s, a, b, c, d, g)(a_data, b_data, c_data, d_data, g_data);

  for (int i = 0; i < kTotalSize; i++) {
    ASSERT_EQ(g_data[i], 10.0f);
  }
}

void testScheduleDynamicShape2D() {
  KernelScope kernel_scope;
  auto testWithSize = [](int32_t M, int32_t N) {
    Var m("m", kInt32);
    Var n("n", kInt32);
    Buffer a(Var("a", kHandle), kFloat32, {m, n});
    Buffer b(Var("b", kHandle), kFloat32, {m, n});
    Tensor c =
        Compute("c", {{m, "m"}, {n, "n"}}, [&](const Var& i, const Var& j) {
          return a(i, j) + b(i, j);
        });
    auto sch = Schedule::make({c});
    Stmt s = sch.Lower();
    SimpleIREvaluator cg(s, {a, b, c, m, n});
    std::vector<float> aData(M * N, 1.0f);
    std::vector<float> bData(M * N, 2.0f);
    std::vector<float> cData(M * N, 0.0f);
    cg.call({aData, bData, cData, M, N});
    ExpectAllNear(cData, std::vector<float>(M * N, 3.0f), 1e-7);
  };
  testWithSize(1, 8);
  testWithSize(16, 32);
  testWithSize(37, 11);
}

} // namespace jit
} // namespace torch
