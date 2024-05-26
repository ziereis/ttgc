
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"


namespace tcg {
struct CPULoweringPass
    : public mlir::PassWrapper<CPULoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect>();
  }

  void runOnOperation() final;
};

void CPULoweringPass::runOnOperation() {
  mlir::LLVMConversionTarget target(getContext());
  target.addLegalOp<mlir::ModuleOp>();

  mlir::LLVMTypeConverter typeConverter(&getContext());

  mlir::RewritePatternSet patterns(&getContext());
  mlir::populateAffineToStdConversionPatterns(patterns);
  mlir::populateSCFToControlFlowConversionPatterns(patterns);
  mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
  mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                        patterns);
  mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);

  auto module = getOperation();
  if (failed(mlir::applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> createLowerToLLVMPass() {
  return std::make_unique<CPULoweringPass>();
}


mlir::LogicalResult lowerToCPU(mlir::ModuleOp &module, mlir::MLIRContext &context) {
  mlir::PassManager pm(&context);
  mlir::bufferization::OneShotBufferizationOptions options;
  options.bufferizeFunctionBoundaries = true;
  pm.addPass(mlir::bufferization::createOneShotBufferizePass(options));
  mlir::bufferization::BufferResultsToOutParamsOptions bufferizationOptions;
  options.bufferizeFunctionBoundaries = true;
  pm.addPass(mlir::bufferization::createBufferResultsToOutParamsPass(bufferizationOptions));
  pm.addPass(mlir::createConvertLinalgToLoopsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(createLowerToLLVMPass());
  pm.addPass(mlir::createCanonicalizerPass());
  // Run the pass pipeline
  return pm.run(module);
}

} // namespace
