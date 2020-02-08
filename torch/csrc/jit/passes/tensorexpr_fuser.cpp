#include <torch/csrc/autograd/record_function.h>
#include <torch/csrc/jit/custom_operator.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/operator_options.h>
#include <torch/csrc/jit/pass_manager.h>
#include <torch/csrc/jit/passes/alias_analysis.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/tensorexpr/buffer.h>
#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/schedule.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

using namespace torch::jit;
using namespace torch::jit::compiler;

namespace {

const Symbol& getTensorExprSymbol() {
  static Symbol s = Symbol::fromQualString("tensorexpr::Group");
  return s;
}

value_list sortReverseTopological(
    ArrayRef<torch::jit::Value*> inputs,
    torch::jit::Block* block) {
  value_list result;
  for (auto i : inputs) {
    if (i->node()->owningBlock() == block) {
      result.push_back(i);
    }
  }
  // Sort in reverse topological order
  std::sort(
      result.begin(),
      result.end(),
      [&](torch::jit::Value* a, torch::jit::Value* b) {
        return a->node()->isAfter(b->node());
      });
  return result;
}

bool isSupported(Node* node) {
  // TODO:
  return node->kind() == Symbol::fromQualString("aten::add");
}

bool canHandle(Node* node, AliasDb& aliasDb) {
  if (node->kind() == prim::Constant) {
    return true;
  }
  if (node->kind() == prim::Loop) {
    return false; // TODO
  }
  return isSupported(node);
}

#define REQ(cond)                           \
  if (!(cond)) {                            \
    GRAPH_DEBUG("Failed cond " #cond "\n"); \
    return c10::nullopt;                    \
  }

c10::optional<Node*> tryMerge(
    Node* consumer,
    Node* producer,
    AliasDb& aliasDb) {
  GRAPH_DEBUG(
      "Trying producer ",
      producer->kind().toQualString(),
      " and consumer ",
      consumer->kind().toQualString(),
      ":\n");

  // Symbolic checks
  REQ(canHandle(producer, aliasDb));
  REQ(
      (canHandle(consumer, aliasDb) ||
       consumer->kind() == getTensorExprSymbol()));

  // Alias checks
  // Requirement:
  // - moveAfterTopologicallyValid(consumer, producer)
  // - One of:
  //   1) Both are in-place ops
  //   2) Consumer is in-place, producer !hasInputWriters
  //   3) Producer is in-place, consumer !hasOutputWriters
  REQ(aliasDb.moveAfterTopologicallyValid(consumer, producer));

  // 1)
  if (!(aliasDb.isMutable(consumer) && aliasDb.isMutable(producer))) {
    // 2)
    if (aliasDb.isMutable(consumer)) {
      REQ(!aliasDb.hasInputWriters(producer));
      // 3)
    } else if (aliasDb.isMutable(producer)) {
      REQ(!aliasDb.hasOutputWriters(consumer));
    }
  }

  if (!consumer->hasAttribute(attr::Subgraph) &&
      consumer->kind() != getTensorExprSymbol()) {
    consumer =
        SubgraphUtils::createSingletonSubgraph(consumer, getTensorExprSymbol());
  }
  if (producer->kind() == prim::Constant) {
    auto& subgraph = consumer->g(attr::Subgraph);
    Node* in_const = subgraph->createClone(
        producer, [](torch::jit::Value*) -> torch::jit::Value* {
          throw std::runtime_error("unexpected input");
        });
    subgraph->insertNode(in_const);
  } else {
    SubgraphUtils::mergeNodeIntoSubgraph(producer, consumer);
  }
  return consumer;
}
#undef REQ

std::pair<graph_node_list::iterator, bool> scanNode(
    Node* consumer,
    AliasDb& aliasDb,
    torch::jit::Block* block) {
  auto inputs = sortReverseTopological(consumer->inputs(), block);
  for (auto input : inputs) {
    if (auto group = tryMerge(consumer, input->node(), aliasDb)) {
      // we successfully merged, so the new group's `inputs` may have
      // changed. So rescan the new group for more merging opportunities.
      return {group.value()->reverseIterator(), true};
    }
  }
  return {++consumer->reverseIterator(), false};
}

void fuseTensorExprs(std::shared_ptr<Graph>& graph) {
#if TX_DEBUG
  std::cout << "Entering TExprFuser\n";
  std::cout << *graph;
#endif

  AliasDb aliasDb(graph);
  auto block = graph->block();

  bool any_changed = true;
  while (any_changed) {
    any_changed = false;
    for (auto it = block->nodes().rbegin(); it != block->nodes().rend();) {
      bool changed;
      std::tie(it, changed) = scanNode(*it, aliasDb, block);
      any_changed |= changed;
    }
  }

  EliminateCommonSubexpression(graph);
  EliminateDeadCode(graph);

#if TX_DEBUG
  std::cout << "Finishing TExprFuser\n";
  std::cout << *graph;
#endif
}

Dtype texprType(const c10::optional<at::ScalarType>& st) {
  switch (*st) {
    case at::ScalarType::Int:
      return kInt32;
    case at::ScalarType::Float:
      return kFloat32;
    default:
      LOG(FATAL) << "Unhandled datatype";
      return kUninitialized;
  }
}

std::vector<Expr> texprSizes(const c10::VaryingShape& shape) {
  std::vector<Expr> dims;
  for (auto i = 0; i < *shape.size(); i++) {
    dims.push_back(IntImm::make(*shape[i]));
  }
  return dims;
}

std::vector<DimArg> texprDims(torch::jit::Value* v) {
  auto tt = v->type()->cast<TensorType>();
  auto exprDims = texprSizes(tt->sizes());
  return std::vector<DimArg>(exprDims.begin(), exprDims.end());
}

Buffer texprBuffer(const torch::jit::Value* v) {
  auto tt = v->type()->cast<TensorType>();
  return Buffer(
      v->debugName(), texprType(tt->scalarType()), texprSizes(tt->sizes()));
}

template <typename T>
size_t bufferSize(T t) {
  size_t size = 1;
  for (int i = 0; i < t.ndim(); i++) {
    size *= t.dim(i).template AsNode<IntImm>()->value();
  }
  return size;
}

struct TensorExprKernel {
  std::vector<Buffer> buffer_args;
  Tensor* tensor_output;
  std::unordered_map<int64_t, Tensor> tensors;
  Stmt stmt;

  explicit TensorExprKernel(const Node* node) {
    auto subgraph = node->g(attr::Subgraph);

    // Bind inputs to buffers.
    auto inputs = subgraph->inputs();
    for (auto const& input : subgraph->inputs()) {
      Buffer in_buffer = texprBuffer(input);
      tensors.emplace(
          input->unique(),
          Compute(
              "input",
              texprDims(input),
              [in_buffer](const std::vector<Var>& axes) {
                return in_buffer(axes[0]);
              }));
      buffer_args.push_back(std::move(in_buffer));
    }

    // Bind nodes to tensor compute expressions.
    std::unordered_map<int64_t, Expr> constants;
    for (auto const& n : subgraph->nodes()) {
      if (n->kind() == prim::Constant) {
        const auto val = toIValue(n->output()).value();
        if (val.isDouble()) {
          constants[n->output()->unique()] = FloatImm::make(val.toDouble());
        } else if (val.isInt()) {
          constants[n->output()->unique()] = IntImm::make(val.toInt());
        } else {
          LOG(FATAL) << "Unhandled constant datatype";
        }
        continue;
      }

      if (n->kind() == aten::add) {
        auto const& lhs = tensors.at(n->inputs()[0]->unique());
        auto const& rhs = tensors.at(n->inputs()[1]->unique());
        tensors.emplace(
            n->output()->unique(),
            Compute(
                "aten_add",
                texprDims(n->output()),
                [&lhs, &rhs](const std::vector<Var>& axes) {
                  return lhs(axes[0]) + rhs(axes[0]);
                }));
        continue;
      }

      LOG(FATAL) << "Unhandled node kind";
    }

    CHECK(subgraph->outputs().size() == 1)
        << "Only handle single output subgraphs";
    auto const& output = subgraph->outputs()[0];
    CHECK(tensors.count(output->unique())) << "Output must be a tensor";
    tensor_output = &tensors.at(output->unique());
    torch::jit::compiler::schedule::Schedule sch({*tensor_output});
    stmt = sch.Lower();
  }

  void run(Stack& stack) {
    SimpleIREvaluator eval(stmt);
    std::vector<std::vector<float>> backing;

    auto inputs = last(stack, buffer_args.size());
    for (int i = 0; i < buffer_args.size(); i++) {
      eval.bindBuffer(buffer_args[i], inputs[i].toTensor().data_ptr());
    }

    at::Tensor output =
        at::empty(bufferSize(*tensor_output), at::ScalarType::Float);
    eval.bindBuffer(*tensor_output, output.data_ptr());

    eval.eval();
    drop(stack, buffer_args.size());
    stack.insert(stack.end(), std::move(output));
  }
};

Operation createTensorExprOp(const Node* node) {
  return [node](Stack& stack) {
    RECORD_FUNCTION("TensorExpr", std::vector<c10::IValue>());
    auto kernel = std::make_shared<TensorExprKernel>(node);
    kernel->run(stack);
    return 0;
  };
}

c10::OperatorOptions getAliasAnalysisOption(AliasAnalysisKind k) {
  auto options = c10::OperatorOptions();
  options.setAliasAnalysis(k);
  return options;
}

RegisterOperators TensorExprOps({
    torch::jit::Operator(
        getTensorExprSymbol(),
        createTensorExprOp,
        getAliasAnalysisOption(AliasAnalysisKind::PURE_FUNCTION)),
});

RegisterPass pass(fuseTensorExprs);

} // namespace
