#ifndef TCG_GRAPH_H_
#define TCG_GRAPH_H_

#include <cstdint>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>
#include "mlir/IR/Value.h"


template<class T>
using up = std::unique_ptr<T>;

namespace tcg {

struct GraphNode {
  enum OpCode {
    ADD,
    LOAD,
    STORE,
    ONES,
  };
  OpCode op;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  int64_t offset = 0;
  mlir::Value val;

  virtual ~GraphNode() = default;
  OpCode getKind() const {return op;}

  GraphNode(OpCode op, std::vector<int64_t> shape)
      : op(op), shape(std::move(shape)) {
    strides.resize(this->shape.size());
    strides.back() = 1;
    for (int i = shape.size() - 2; i >= 0; i--) {
      strides[i] = this->shape[i + 1] * strides[i + 1];
    }
  };

  static up<GraphNode> ones(const std::vector<int64_t>& shape);
  static up<GraphNode> add(up<GraphNode> lhs, up<GraphNode> rhs);
  static up<GraphNode> load(void *addr, const std::vector<int64_t>& shape);
  static up<GraphNode> store(up<GraphNode> node, void *addr);
};

struct ConstantOpNode : public GraphNode {
  using GraphNode::GraphNode;
  static bool classof(const GraphNode *c) {return c->getKind() == ONES;}
};


struct BinaryOpNode : public GraphNode {
  up<GraphNode> lhs;
  up<GraphNode> rhs;

  BinaryOpNode(OpCode op, const std::vector<int64_t> &shape, up<GraphNode> lhs,
               up<GraphNode> rhs)
      : GraphNode(op, std::move(shape)), lhs(std::move(lhs)),
        rhs(std::move(rhs)){};
  static bool classof(const GraphNode *c) {return c->getKind() == ADD;}
};

struct LoadOpNode : public GraphNode {
  void* src;
  LoadOpNode(void* src, const std::vector<int64_t>& shape): GraphNode(GraphNode::LOAD, shape), src(src) {};
  static bool classof(const GraphNode *c) {return c->getKind() == LOAD;}
};

struct StoreOpNode : public GraphNode {
  void* dst;
  up<GraphNode> src;
  StoreOpNode(void* dst, up<GraphNode> src): GraphNode(GraphNode::STORE, src->shape), dst(dst), src(std::move(src)) {};
  static bool classof(const GraphNode *c) {return c->getKind() == STORE;}
};

void dump(GraphNode &);

void topo_sort(GraphNode* root, std::vector<GraphNode*>& sorted, std::unordered_set<GraphNode*>& visited);

}

#endif
