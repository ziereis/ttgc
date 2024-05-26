//===- AST.cpp - Helper for printing out the Toy AST ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AST dump for the Toy language.
//
//===----------------------------------------------------------------------===//

#include "Graph.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <memory>


namespace tcg {



struct Indent {
  Indent(int &level) : level(level) { ++level; }
  ~Indent() { --level; }
  int &level;
};

class GraphDumper {
public:
  void dump(GraphNode *node);

private:
  void dump(BinaryOpNode *node);
  void dump(LoadOpNode *node);
  void dump(StoreOpNode *node);
  void dump(ConstantOpNode *node);

  void indent() {
    for (int i = 0; i < curIndent; i++)
      llvm::errs() << "  ";
  }
  int curIndent = 0;
};


// the current indentations
#define INDENT()                                                               \
  Indent level_(curIndent);                                                    \
  indent();

/// Dispatch to a generic expressions to the appropriate subclass using RTTI
void GraphDumper::dump(GraphNode *expr) {
  llvm::TypeSwitch<GraphNode *>(expr)
      .Case<LoadOpNode, StoreOpNode, BinaryOpNode, ConstantOpNode>(
          [&](auto *node) { this->dump(node); })
      .Default([&](GraphNode *) {
        // No match, fallback to a generic message
        INDENT();
        llvm::errs() << "<unknown Expr, kind " << expr->getKind() << ">\n";
      });
}

void GraphDumper::dump(ConstantOpNode *node) {
  INDENT();
  llvm::errs() << "CONST: ONES" << "\n";
}

void GraphDumper::dump(LoadOpNode *node) {
  INDENT();
  llvm::errs() << "LOAD: addr -> " << node->src << "\n";
}

void GraphDumper::dump(StoreOpNode *node) {
  INDENT();
  llvm::errs() << "Store: addr -> " << node->dst << "\n";
  dump(node->src.get());
}
/// Print a binary operation, first the operator, then recurse into LHS and RHS.
void GraphDumper::dump(BinaryOpNode *node) {
  INDENT();
  llvm::errs() << "BinOp: " << "ADD" << " " << "\n";
  dump(node->lhs.get());
  dump(node->rhs.get());
}


void dump(GraphNode &node) { GraphDumper().dump(&node); }

up<GraphNode> GraphNode::ones(const std::vector<int64_t>& shape) {
  return std::make_unique<GraphNode>(GraphNode::ONES, shape);
}

up<GraphNode> GraphNode::add(up<GraphNode> lhs, up<GraphNode> rhs) {
  return std::make_unique<BinaryOpNode>(GraphNode::ADD, lhs->shape, std::move(lhs), std::move(rhs));
}

up<GraphNode> GraphNode::load(void *addr, const std::vector<int64_t>& shape) {
  return std::make_unique<LoadOpNode>(addr, shape);
}

up<GraphNode> GraphNode::store(up<GraphNode> paren, void *addr) {
  return std::make_unique<StoreOpNode>(addr,std::move(paren));
}

void topo_sort(GraphNode* root, std::vector<GraphNode*>& sorted, std::unordered_set<GraphNode*>& visited) {
  if (visited.find(root) != visited.end()) {
    return;
  }
  visited.insert(root);
  llvm::TypeSwitch<GraphNode *>(root)
      .Case<LoadOpNode, ConstantOpNode>(
          [&](GraphNode *node) { sorted.push_back(node); })
      .Case<StoreOpNode>([&](StoreOpNode *node) {
        topo_sort(node->src.get(), sorted, visited);
        sorted.push_back(node);
      })
      .Case<BinaryOpNode>([&](BinaryOpNode * node) {
        topo_sort(node->lhs.get(), sorted, visited);
        topo_sort(node->rhs.get(), sorted, visited);
        sorted.push_back(node);
      });
}


} // namespace
