include(FetchContent)

set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

FetchContent_Declare(
  mqt-qmap
  GIT_REPOSITORY https://github.com/munich-quantum-toolkit/qmap
  GIT_TAG v3.9.0) # 186b9dcb5bc3395f865e734bc69dc6887071045f

FetchContent_MakeAvailable(mqt-qmap)

FetchContent_GetProperties(mqt-qmap)

set(MQT_QMAP_INCLUDE_DIR "${mqt-qmap_SOURCE_DIR}/include")
