#include "caffe2/operators/tt_sls_op.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(TTSparseLengthsSum, TTSparseLengthsSumOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(TTSparseLengthsSumGradient, TTSparseLengthsSumGradientOp<float, CPUContext>);

// The TT-layer serves as a low-rank decomposition of a fully connected layer.
// The inputs are the same as to an FC layer, but the number of the parameters
// are greatly reduced.
OPERATOR_SCHEMA(TTSparseLengthsSum)
    .NumInputs(5)
    .NumOutputs(5)
    .SetDoc(R"DOC(
This operator introduce a new, parameter efficient embedding layer, termed TT–embedding, which
can be plugged in into any model and trained end-to-end. The benefits of our compressed TT–layer
are twofold. Firstly, instead of storing huge embedding matrix, it stores a sequence of much smaller
2-dimensional and 3-dimensional tensors, necessary for reconstructing the required embeddings,
which allows compressing the model significantly at the cost of a negligible performance drop.
Secondly, the overall number of parameters can be relatively small (and constant) during the whole
training stage, which allows to use larger batches or train efficiently in a case of limited resources.
)DOC")
    .Arg("factor_i", "vector<int>: factorization of voc size")
    .Arg("factor_j", "vector<int>: factorization of emb size")
    .Arg("ranks", "int[] Ranks of cores")
    .Arg("emb_size", "int: the size of each embedding entry")
    .Input(0, "core0", "tensor core 0")
    .Input(1, "core1", "tensor core 1")
    .Input(2, "core2", "tensor core 2")
    .Input(3, "indices", "index for embedding")
    .Input(4, "lengths", "segment lengths")
    .Output(0, "OUTPUT", "Aggregated tensor")
    .Output(1, "core0_output", "intermediate mm result from core0 for backward path")
    .Output(2, "core1_output", "intermediate mm result from core1 for backward path")
    .Output(3, "core2_output", "intermediate mm result from core2 for backward path")
    .Output(4, "indices", "the index for each core");

OPERATOR_SCHEMA(TTSparseLengthsSumGradient)
    .NumInputs(10)
    .NumOutputs(3);

class GetTTSparseLengthsGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
  // set up the input and output
  return SingleGradientDef(
    "TTSparseLengthsSumGradient",
    "",
    // CORE0, CORE1, CORE2, INDICES, LENGTHS, CORE0_output, CORE1_output, CORE2_output, indices, dY
    vector<string>{I(0), I(1), I(2), I(3), I(4), O(1), O(2), O(3), O(4), GO(0)},
    // dCore0, dCore1, dCore2
    vector<string>{GI(0), GI(1), GI(2)});
  };
};

REGISTER_GRADIENT(TTSparseLengthsSum, GetTTSparseLengthsGradient)

} // namespace
} // namespace caffe2
