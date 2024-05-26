#include "lib/Graph/Graph.h"
#include "lib/MLIRGen/MLIRGen.h"
#include "lib/MLIRGen/Context.h"
#include "lib/Runtime/RuntimeCPU.h"
#include "lib/MLIRGen/Context.h"
#include "lib/Lower/LowerToCPU.h"
#include <llvm/Support/Error.h>

int main(int argc, char **argv) {

  std::vector<float> lhs(256 * 256, 1.0);
  std::vector<float> rhs(256 * 256, 1.0);
  std::vector<float> out(256 * 256, 0.0);

  auto a = tcg::GraphNode::load(lhs.data(), {256, 256});
  auto b = tcg::GraphNode::load(rhs.data(), {256, 256});
  auto c = tcg::GraphNode::add(std::move(a), std::move(b));
  auto d = tcg::GraphNode::store(std::move(c), out.data());

  tcg::tcgContext ctx;

  llvm::errs() << "Generating MLIR\n";

  auto module = tcg::genMLIR(ctx, d.get());

  tcg::lowerToCPU(module, ctx.context);

  module.dump();

  auto res  = runCPU(ctx, module, 0);

  if (res) {
    llvm::errs() << "Error: " << res << "\n";
  }

  for (int i = 0; i < 10; i++) {
    llvm::errs() << out[i] << ",";
  }
  llvm::errs() << "\n";

  return 0;
}
