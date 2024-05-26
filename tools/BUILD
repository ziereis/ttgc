package(
    default_visibility = ["//visibility:public"],
)


cc_binary(
    name = "tcg",
    srcs = [
        "tcg.cpp",
    ],
    deps = [
        "@llvm-project//llvm:Support",
        "//lib/MLIRGen:mlirgen",
        "//lib/MLIRGen:context",
        "//lib/Graph:graph",
        "//lib/Runtime:runtimecpu",
        "//lib/Lower:lowercpu",
    ],
)
