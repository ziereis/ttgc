#ifndef TCG_CONTEXT_H_
#define TCG_CONTEXT_H_

#include "lib/Graph/Graph.h"


namespace tcg {
struct tcgContext {
  mlir::DialectRegistry registry;
  mlir::MLIRContext context;
  std::vector<LoadOpNode *> inputs;
  StoreOpNode *output;
  tcgContext();
};

}

#endif // TCG_CONTEXT_H
