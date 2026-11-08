set(LLVM_TARGET_DEFINITIONS TransformPasses.td)

mlir_tablegen(TransformPasses.h.inc -gen-pass-decls -name MQSSTransforms)

add_public_tablegen_target(MQSSTransformsPassIncGen)
