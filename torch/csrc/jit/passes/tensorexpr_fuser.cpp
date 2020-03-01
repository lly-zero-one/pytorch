#include <torch/csrc/autograd/record_function.h>
#include <torch/csrc/jit/custom_operator.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/operator_options.h>
#include <torch/csrc/jit/pass_manager.h>
#include <torch/csrc/jit/passes/alias_analysis.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>

using namespace torch::jit;
using namespace torch::jit::tensorexpr;

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
  switch (node->kind()) {
    case aten::add:
    case aten::sub:
    case aten::mul:
    case aten::div:
    case aten::eq:
    case aten::ne:
    case aten::ge:
    case aten::gt:
    case aten::le:
    case aten::lt:
    case aten::min:
    case aten::max:
    case aten::clamp:
    case aten::log10:
    case aten::log:
    case aten::log2:
    case aten::exp:
    case aten::erf:
    case aten::erfc:
    case aten::cos:
    case aten::sin:
    case aten::tan:
    case aten::acos:
    case aten::asin:
    case aten::atan:
    case aten::cosh:
    case aten::sinh:
    case aten::tanh:
    case aten::sqrt:
    case aten::rsqrt:
    case aten::abs:
    case aten::floor:
    case aten::ceil:
    case aten::round:
    case aten::trunc:
    case aten::remainder:
    case prim::ConstantChunk:
    case aten::cat:
    case prim::ListConstruct:
    case aten::sigmoid:
    case aten::relu:
    case aten::addcmul:
    case aten::neg:
    case aten::reciprocal:
    case aten::expm1:
    case aten::lgamma:
#ifndef ENABLE_LLVM
    case aten::frac:
#endif
      return true;
    default:
      return false;
  }
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

  // Only handle complete tensor types
  for (torch::jit::Value* output : consumer->outputs()) {
    REQ(output->isCompleteTensor());
  }

  // Only fuse within a block
  REQ(consumer->owningBlock() == producer->owningBlock());

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
    // Don't initiate a fusion group from prim::ListConstruct
    REQ(consumer->kind() != prim::ListConstruct);

    // Don't initiate a fusion group just for a constant operand
    REQ(producer->kind() != prim::Constant);

    consumer =
        SubgraphUtils::createSingletonSubgraph(consumer, getTensorExprSymbol());
  }

  if (producer->kind() == aten::cat) {
    REQ(producer->inputs()[0]->node()->kind() == prim::ListConstruct);
    REQ(producer->inputs()[0]->uses().size() == 1);
    REQ(producer->inputs()[1]->node()->kind() == prim::Constant);
    Node* listconstruct = producer->inputs()[0]->node();
    Node* constant = producer->inputs()[1]->node();
    SubgraphUtils::mergeNodeIntoSubgraph(producer, consumer);
    auto& subgraph = consumer->g(attr::Subgraph);
    Node* new_const = subgraph->createClone(constant, [](Value*) -> Value* { return nullptr; } );
    subgraph->insertNode(new_const);
    SubgraphUtils::mergeNodeIntoSubgraph(listconstruct, consumer);
  } else {
    if (consumer->kind() == aten::cat) {
      REQ(consumer->inputs()[0]->node()->kind() == prim::ListConstruct);
      REQ(consumer->inputs()[0]->uses().size() == 1);
      REQ(consumer->inputs()[1]->node()->kind() == prim::Constant);
    }
    SubgraphUtils::mergeNodeIntoSubgraph(producer, consumer);
  }

  return consumer;
}
#undef REQ

std::pair<graph_node_list::iterator, bool> scanNode(
    Node* consumer,
    AliasDb& aliasDb) {
  auto inputs =
      sortReverseTopological(consumer->inputs(), consumer->owningBlock());
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
  GRAPH_DUMP("Before TExprFuser: ", graph);

  AliasDb aliasDb(graph);
  auto block = graph->block();

  std::vector<std::pair<graph_node_list_iterator, graph_node_list_iterator>>
      worklist;
  std::unordered_set<torch::jit::Block*> visited_blocks;

  bool any_changed = true;
  while (any_changed) {
    any_changed = false;
    worklist.push_back({block->nodes().rbegin(), block->nodes().rend()});

    while (worklist.size()) {
      auto& it = worklist.back().first;
      auto end = worklist.back().second;

      if (it->blocks().size()) {
        Node* n = *it;
        ++it;
        for (auto b : n->blocks()) {
          if (!visited_blocks.count(b)) {
            worklist.push_back({b->nodes().rbegin(), b->nodes().rend()});
            visited_blocks.insert(b);
          }
        }
      } else {
        bool changed;
        std::tie(it, changed) = scanNode(*it, aliasDb);
        any_changed |= changed;
      }

      if (it == end) {
        worklist.pop_back();
      }
    }
  }

  EliminateCommonSubexpression(graph);
  EliminateDeadCode(graph);

  GRAPH_DUMP("After TExprFuser: ", graph);
}

Operation createTensorExprOp(const Node* node) {
  auto kernel = std::make_shared<TensorExprKernel>(node);
  return [kernel](Stack& stack) {
    RECORD_FUNCTION("TensorExpr", std::vector<c10::IValue>());
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
