#include "test/cpp/tensorexpr/test_base.h"
#include <sstream>
#include <stdexcept>

#include <cmath>

#include "torch/csrc/jit/tensorexpr/buffer.h"
#include "torch/csrc/jit/tensorexpr/cuda_codegen.h"
#include "torch/csrc/jit/tensorexpr/schedule.h"
#include "torch/csrc/jit/tensorexpr/tensor.h"
#include "test/cpp/tensorexpr/padded_buffer.h"

namespace torch {
namespace jit {
using namespace torch::jit::compiler;
using namespace torch::jit::compiler::schedule;

void testCudaTestVectorAdd01() {
  const int N = 1024;
  Buffer a_buf("a", kFloat32, {N});
  Buffer b_buf("b", kFloat32, {N});
  Tensor c = Compute(
      "c", {{N, "n"}}, [&](const Var& n) { return a_buf(n) + b_buf(n); });
  Schedule sch({c});
  Stmt stmt = sch.Lower();
  CudaCodeGen cuda_cg(stmt, c, a_buf, b_buf);
  PaddedBuffer<float> a_v(N);
  PaddedBuffer<float> b_v(N);
  PaddedBuffer<float> c_v(N);
  PaddedBuffer<float> c_ref(N);
  for (int i = 0; i < N; i++) {
    a_v(i) = i;
    b_v(i) = i * i;
    c_ref(i) = a_v(i) + b_v(i);
  }

  cuda_cg(c_v, a_v, b_v);

#if 0
  ExpectAllNear(c_v, c_ref, 1e-5);
#endif
}
} // namespace jit
} // namespace torch
