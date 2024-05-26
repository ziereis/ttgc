#include "lib/MLIRGen/Context.h"

#include "mlir/InitAllPasses.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"


namespace tcg {

tcgContext::tcgContext() {
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();
  mlir::func::registerAllExtensions(registry);
  mlir::registerAllDialects(registry);
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();
}
}
