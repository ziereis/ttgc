#ifndef TCG_MLIRGEN_H_
#define TCG_MLIRGEN_H_

#include "lib/Graph/Graph.h"
#include "mlir/IR/BuiltinOps.h"

namespace tcg {
struct tcgContext;
mlir::ModuleOp genMLIR(tcgContext &ctx, GraphNode *root);
mlir::ModuleOp genMLIR(tcgContext &ctx);
}

#endif // TCG_MLIRGEN_H_
