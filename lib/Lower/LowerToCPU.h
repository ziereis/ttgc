#ifndef TCG_LOWERTOCPU_H_
#define TCG_LOWERTOCPU_H_

#include "memory"

namespace mlir {
class ModuleOp;
class Pass;
class MLIRContext;
}

namespace tcg {
std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

void lowerToCPU(mlir::ModuleOp& module, mlir::MLIRContext& context);
} // namespace tcg

#endif // TCG_LOWERTOCPU_H_
