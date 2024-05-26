package(default_visibility = ["//visibility:public"])

cc_library(
    name = "lowercpu",
    hdrs = [
      "LowerToCPU.h",
    ],
    srcs = [
    "LowerToCPU.cpp",
    ],
    deps = [
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:AllExtensions",
    ],
)
