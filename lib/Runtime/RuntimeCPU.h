#ifndef TCG_RUNTIME_CPU_H_
#define TCG_RUNTIME_CPU_H_
#include "llvm/Support/Error.h"

namespace mlir {
class ModuleOp;
}

namespace tcg {
  struct tcgContext;
  llvm::Error runCPU(const tcgContext& ctx, mlir::ModuleOp& module, unsigned int optLevel);

} // namespace tcg

#endif // TCG_RUNTIME_CPU_H_
