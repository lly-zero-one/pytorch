#ifndef NNC_TESTS_TEST_UTILS_H_INCLUDED__
#define NNC_TESTS_TEST_UTILS_H_INCLUDED__

#include <gtest/gtest.h>
#include <unordered_map>
#include <vector>

#include "torch/csrc/jit/tensorexpr/buffer.h"
#include "torch/csrc/jit/tensorexpr/eval.h"
#include "torch/csrc/jit/tensorexpr/function.h"
#include "torch/csrc/jit/tensorexpr/ir.h"
#include "torch/csrc/jit/tensorexpr/tensor.h"
#include "torch/csrc/jit/tensorexpr/tests/padded_buffer.h"

namespace torch {
namespace jit {
namespace compiler {

template <class T>
class SimpleTensorEvaluator {
 public:
  void evaluate(const Tensor& t, std::vector<T>* output) {
    int ndim = t.ndim();
    std::vector<int> dims;
    int size = 1;
    for (int i = 0; i < ndim; i++) {
      SimpleIREvaluator expr_eval(t.dim(i));
      expr_eval();
      int dim = expr_eval.value().template as<int>();
      dims.push_back(dim);
      size *= dim;
    }
    const Function& func = t.function();
    const Expr& body = func.body();
    eval_func(dims, func, 0, output, body);
  }

 private:
  void eval_func(
      const std::vector<int>& dims,
      const Function& func,
      int level,
      std::vector<T>* output,
      const Expr& body) {
    if (level >= dims.size()) {
      SimpleIREvaluator expr_eval(body);
      expr_eval();
      output->push_back(expr_eval.value().template as<T>());
      return;
    }
    for (int i = 0; i < dims[level]; i++) {
      Expr wrapped_body = Let::make(func.arg(level), Expr(i), body);
      eval_func(dims, func, level + 1, output, wrapped_body);
    }
  }
};

template <typename U, typename V>
void ExpectAllNear(
    const std::vector<U>& v1,
    const std::vector<U>& v2,
    V threshold,
    const std::string& name = "") {
  ASSERT_EQ(v1.size(), v2.size());
  for (int i = 0; i < v1.size(); i++) {
    EXPECT_NEAR(v1[i], v2[i], threshold)
        << "element index: " << i << ", name: " << name;
  }
}

template <typename T>
static void assertAllEqual(const std::vector<T>& vec, const T& val) {
  for (auto const& elt : vec) {
    ASSERT_EQ(elt, val);
  }
}
} // namespace compiler
} // namespace jit
} // namespace torch

#endif // NNC_TESTS_TEST_UTILS_H_INCLUDED__
