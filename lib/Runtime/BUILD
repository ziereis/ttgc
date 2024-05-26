package(default_visibility = ["//visibility:public"])

cc_library(
    name = "runtimecpu",
    hdrs = [
      "RuntimeCPU.h",
    ],
    srcs = [
    "RuntimeCPU.cpp",
    ],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:ExecutionEngine",
        "//lib/Graph:graph",
        "//lib/MLIRGen:mlirgen",
        "//lib/MLIRGen:context"
    ],
)
