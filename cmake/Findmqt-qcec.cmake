include(FetchContent)

FetchContent_Declare(
  mqt-qcec
  GIT_REPOSITORY https://github.com/munich-quantum-toolkit/qcec
  GIT_TAG v3.8.0)

FetchContent_MakeAvailable(mqt-qcec)

FetchContent_GetProperties(mqt-qcec)

set(MQT_QCEC_INCLUDE_DIR "${mqt-qcec_SOURCE_DIR}/include")
