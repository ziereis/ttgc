#include "lib/Runtime/RuntimeCPU.h"
#include "lib/Graph/Graph.h"
#include "lib/MLIRGen/Context.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/TargetSelect.h"
#include <vector>


namespace tcg {


template <typename T>
static void addShapedMemrefArg(std::vector<void*>& args,
                               T *inputNode) {
  if constexpr(std::is_same<T, LoadOpNode>::value) {
    args.push_back(&inputNode->src); // basePtr
    args.push_back(&inputNode->src); // dataPtr
  } else if constexpr(std::is_same<T, StoreOpNode>::value) {
    args.push_back(&inputNode->dst); // basePtr
    args.push_back(&inputNode->dst); // dataPtr
  } else {
    static_assert(false, "Unsupported node type");
  }
  args.push_back(&inputNode->offset); // offset
  for(auto& dim : inputNode->shape) {
    args.push_back(&dim); // shape
  }
  for(auto& stride : inputNode->strides) {
    args.push_back(&stride); // strides
  }
}

llvm::Error runCPU(const tcgContext& ctx, mlir::ModuleOp& module, unsigned int optLevel) {
  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Register the translation from MLIR to LLVM IR, which must happen before we
  // can JIT-compile.
  mlir::registerBuiltinDialectTranslation(*module.getContext());
  mlir::registerLLVMDialectTranslation(*module.getContext());

  // An optimization pipeline to use within the execution engine.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
  // the module.
  mlir::ExecutionEngineOptions engineOptions;
  engineOptions.transformer = optPipeline;
  auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
  assert(maybeEngine && "failed to construct an execution engine");
  auto engine = std::move(maybeEngine.get());

  std::vector<void*> args;

  for (auto inputNode : ctx.inputs) {
    addShapedMemrefArg(args, static_cast<LoadOpNode*>(inputNode));
  }
  addShapedMemrefArg(args, ctx.output);

  llvm::errs() << "args.size() = " << args.size() << "\n";

  llvm::errs() << "Invoking tensorFn\n";

  llvm::Error res = engine->invokePacked("tensorFn", args);

  return res;

}
}
