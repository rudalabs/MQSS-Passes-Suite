include(FetchContent)

set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

FetchContent_Declare(
  mqt-core
  GIT_REPOSITORY https://github.com/munich-quantum-toolkit/core
  GIT_TAG v3.8.0)

FetchContent_MakeAvailable(mqt-core)

FetchContent_GetProperties(mqt-core)

set(MQT_CORE_INCLUDE_DIR "${mqt-core_SOURCE_DIR}/include")
