package(default_visibility = ["//visibility:public"])

cc_library(
    name = "mlirgen",
    hdrs = [
      "MLIRGen.h",
    ],
    srcs = [
    "MLIRGen.cpp",
    ],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "//lib/Graph:graph",
        "//lib/MLIRGen:context",
    ],
)

cc_library(
    name = "context",
    hdrs = [
      "Context.h",
    ],
    srcs = [
    "Context.cpp",
    ],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:AllExtensions",
        "@llvm-project//mlir:IR",
        "//lib/Graph:graph",
    ],
)
