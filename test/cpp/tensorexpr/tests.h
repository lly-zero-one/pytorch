#pragma once

/**
 * See README.md for instructions on how to add a new test.
 */
#include <c10/macros/Export.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace jit {
#define TH_FORALL_TESTS(_)      \
  _(ExprBasicValueTest)         \
  _(ExprBasicValueTest02)       \
  _(ExprLetTest01)              \
  _(ExprLetTest02)              \
  _(ExprVectorAdd01)            \
  _(ExprCompareSelectEQ)        \
  _(ExprSubstitute01)           \
  _(ExprMath01)                 \
  _(ExprUnaryMath01)            \
  _(ExprBinaryMath01)           \
  _(ExprDynamicShapeAdd)        \
  _(IRPrinterBasicValueTest)    \
  _(IRPrinterBasicValueTest02)  \
  _(IRPrinterLetTest01)         \
  _(IRPrinterLetTest02)         \
  _(IRPrinterCastTest)          \
  _(ExprSimple01)               \
  _(ExprLower01)                \
  _(ExprSimple02)               \
  _(ScheduleBroadcastAddBuffer) \
  _(ScheduleFunctionCall01)     \
  _(ScheduleInlineFunc01)       \
  _(ScheduleFuserStyle)         \
  _(ScheduleFuserThreeArg)      \
  _(TypeTest01)                 \
  _(AsmjitIntImmTest)           \
  _(AsmjitIntAddTest)           \
  _(AsmjitIntSubTest)           \
  _(AsmjitIntMulTest)           \
  _(AsmjitIntDivTest)           \
  _(LLVMIntImmTest)             \
  _(LLVMFloatImmTest)           \
  _(LLVMIntAddTest)             \
  _(LLVMIntSubTest)             \
  _(LLVMIntMulTest)             \
  _(LLVMIntDivTest)             \
  _(LLVMIntToFloatCastTest)     \
  _(LLVMFloatToIntCastTest)     \
  _(LLVMLetTest01)              \
  _(LLVMLetTest02)              \
  _(LLVMBufferTest)             \
  _(LLVMBlockTest)              \
  _(LLVMLoadStoreTest)          \
  _(LLVMVecLoadStoreTest)       \
  _(LLVMMemcpyTest)             \
  _(LLVMBzeroTest)              \
  _(LLVMElemwiseAdd)            \
  _(LLVMElemwiseAddFloat)       \
  _(LLVMElemwiseLog10Float)     \
  _(LLVMElemwiseMaxInt)         \
  _(LLVMElemwiseMinInt)         \
  _(LLVMElemwiseMaxNumFloat)    \
  _(LLVMElemwiseMaxNumNaNFloat) \
  _(LLVMElemwiseMinNumFloat)    \
  _(LLVMElemwiseMinNumNaNFloat) \
  _(LLVMCompareSelectIntEQ)     \
  _(LLVMCompareSelectFloatEQ)   \
  _(LLVMStoreFloat)             \
  _(LLVMSimpleMath01)           \
  _(LLVMComputeMul)             \
  _(LLVMBroadcastAdd)           \
  _(LLVMDynamicShapeAdd)        \
  _(CudaTestVectorAdd01)        \
  _(ATen_cast_Float)            \
  _(ATennegInt)                 \
  _(ATennegFloat)               \
  _(ATenaddInt)                 \
  _(ATenaddFloat)               \
  _(ATensubInt)                 \
  _(ATensubFloat)               \
  _(ATenlerp)                   \
  _(ATenaddcmulInt)             \
  _(ATenaddcmulFloat)           \
  _(ATenmulInt)                 \
  _(ATenmulFloat)               \
  _(ATendivInt)                 \
  _(ATendivFloat)               \
  _(ATenmaxInt)                 \
  _(ATenmaxFloat)               \
  _(ATenminInt)                 \
  _(ATenminFloat)               \
  _(ATen_sigmoid_backward)      \
  _(ATen_tanh_backward)         \
  _(ATenreciprocal)             \
  _(ATenreluInt)                \
  _(ATenreluFloat)              \
  _(ATenlogFloat)               \
  _(ATenlog10Float)             \
  _(ATenlog2Float)              \
  _(ATenexpFloat)               \
  _(ATenerfFloat)               \
  _(ATencosFloat)               \
  _(ATeneqInt)                  \
  _(ATengeInt)                  \
  _(ATengtInt)                  \
  _(ATenleInt)                  \
  _(ATenltInt)

#define TH_FORALL_TESTS_CUDA(_)

#define DECLARE_TENSOREXPR_TEST(name) void test##name();
TH_FORALL_TESTS(DECLARE_TENSOREXPR_TEST)
TH_FORALL_TESTS_CUDA(DECLARE_TENSOREXPR_TEST)
#undef DECLARE_TENSOREXPR_TEST

} // namespace jit
} // namespace torch
