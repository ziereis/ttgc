
#include "lib/Graph/Graph.h"
#include "lib/MLIRGen/MLIRGen.h"
#include "lib/MLIRGen/Context.h"
#include <unordered_set>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/TypeSwitch.h"


using namespace mlir;

namespace tcg {

mlir::ModuleOp genMLIR(tcgContext &ctx, GraphNode *root) {
  ctx.inputs.clear();
  ctx.output = nullptr;

  std::vector<GraphNode *> sorted;
  std::unordered_set<GraphNode *> visisted;
  tcg::topo_sort(root, sorted, visisted);

  mlir::OpBuilder builder(&ctx.context);
  auto module = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());

  std::vector<mlir::Type> inputTypes;
  std::vector<GraphNode *> body;

  for (auto node : sorted) {
    llvm::TypeSwitch<GraphNode *>(node)
        .Case<LoadOpNode>([&](LoadOpNode *node) {
          ctx.inputs.push_back(node);
          inputTypes.push_back(mlir::RankedTensorType::get(node->shape, builder.getF32Type()));
        })
        .Case<StoreOpNode>([&](StoreOpNode *node) {
          ctx.output = node;
        })
        .Default([&](GraphNode *node) {
          body.push_back(node);
        });
  }

  auto functionType = builder.getFunctionType(inputTypes, {});
  auto func = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(),
                                                 "tensorFn", functionType);

  auto &entryBlock = *func.addEntryBlock();

  builder.setInsertionPointToStart(&entryBlock);

  for (int i = 0; i < ctx.inputs.size(); i++) {
    ctx.inputs[i]->val = entryBlock.getArgument(i);
  }

  Value lastRes;

  for (auto node : body) {
    llvm::TypeSwitch<GraphNode *>(node).Case<BinaryOpNode>(
        [&](BinaryOpNode *node) {
          llvm::errs() << "BinaryOpNode\n";
          node->lhs->val.dump();
          node->rhs->val.dump();
          node->val.dump();
          Value outputTensor = builder.create<mlir::tensor::EmptyOp>(
            builder.getUnknownLoc(), node->shape,
            builder.getF32Type());
          lastRes = builder.create<mlir::linalg::AddOp>(builder.getUnknownLoc(),
                                              ValueRange{node->lhs->val, node->rhs->val}, outputTensor)->getResult(0);
          node->val = lastRes;
        })
        .Default([](GraphNode* node) {
          llvm::errs() << "Unhandled node\n";
        });
  }
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), lastRes);

  module.push_back(func);

  return module;
}

mlir::ModuleOp genMLIR(tcgContext &ctx) {

  mlir::OpBuilder builder(&ctx.context);

  auto module = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());

  // // Define the function type (memref<4x4xf32>, memref<4x4xf32>) ->
  // memref<4x4xf32>.
  auto tensorType = mlir::RankedTensorType::get({4, 4}, builder.getF32Type());
  auto functionType =
      builder.getFunctionType({tensorType, tensorType}, {tensorType});

  // // Create a function.
  auto func = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(),
                                                 "add_tensors", functionType);
  auto &entryBlock = *func.addEntryBlock();

  builder.setInsertionPointToStart(&entryBlock);

  // // Get function arguments.
  mlir::Value lhs = entryBlock.getArgument(0);
  mlir::Value rhs = entryBlock.getArgument(1);
  auto resultType = mlir::RankedTensorType::get({4, 4}, builder.getF32Type());
  Value outputTensor = builder.create<mlir::tensor::EmptyOp>(
      builder.getUnknownLoc(), tensorType.getShape(),
      tensorType.getElementType());
  lhs.dump();
  rhs.dump();
  outputTensor.dump();

  auto loc = builder.getUnknownLoc();
  auto inputs = mlir::ValueRange{lhs, rhs};
  llvm::SmallVector<mlir::AffineMap, 2> indexingMaps = {
      builder.getMultiDimIdentityMap(tensorType.getRank()),
      builder.getMultiDimIdentityMap(tensorType.getRank()),
      builder.getMultiDimIdentityMap(tensorType.getRank()),
  };
  llvm::SmallVector<mlir::utils::IteratorType, 2> iteratorTypes{
      mlir::utils::IteratorType::parallel, mlir::utils::IteratorType::parallel};
  llvm::errs() << "Creating generic op\n";
  // // Create a linalg.generic op to perform element-wise addition.
  // auto resultTensor =
  //     builder
  //         .create<mlir::linalg::GenericOp>(
  //             loc, resultType, inputs, outputTensor, indexingMaps,
  //             iteratorTypes,
  //             [](mlir::OpBuilder &b, mlir::Location loc,
  //                mlir::ValueRange args) {
  //               for (auto arg : args) {
  //                 arg.dump();
  //               }
  //               mlir::Value lhs = args[0];
  //               mlir::Value rhs = args[1];
  //               mlir::Value result =
  //                   b.create<mlir::arith::AddFOp>(loc, lhs, rhs);
  //               b.create<mlir::linalg::YieldOp>(loc, result);
  //             })
  //         .getResult(0);

  auto resultTensor = builder.create<mlir::linalg::AddOp>(loc, ValueRange{lhs, rhs}, outputTensor).getResult(0);
  // // Return the result tensor.
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), resultTensor);

  // // Add the function to the module.
  module.push_back(func);

  return module;
}
} // namespace tcg
