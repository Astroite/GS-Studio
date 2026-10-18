cmake_minimum_required(VERSION 3.24)
project(gsstudio_media_helper LANGUAGES CXX)

if(NOT DEFINED ENV{INSTA360_MEDIA_SDK_ROOT})
  message(FATAL_ERROR "INSTA360_MEDIA_SDK_ROOT must point to the approved Desktop MediaSDK")
endif()

set(INSTA360_MEDIA_SDK_ROOT "$ENV{INSTA360_MEDIA_SDK_ROOT}" CACHE PATH "Desktop MediaSDK root")
add_executable(gsstudio-media-helper src/main.cpp src/insta360_bridge.cpp)
target_compile_features(gsstudio-media-helper PRIVATE cxx_std_20)
target_include_directories(gsstudio-media-helper PRIVATE "${INSTA360_MEDIA_SDK_ROOT}/include")
target_compile_definitions(gsstudio-media-helper PRIVATE UNICODE _UNICODE)

# Library names and the vendor header/API surface are finalized only against the
# approved SDK package. No vendor binary, header, model or licence is vendored here.
find_library(INSTA360_MEDIA_LIBRARY NAMES MediaSDK media_sdk
  PATHS "${INSTA360_MEDIA_SDK_ROOT}/lib" "${INSTA360_MEDIA_SDK_ROOT}/lib/x64"
  NO_DEFAULT_PATH)
if(NOT INSTA360_MEDIA_LIBRARY)
  message(FATAL_ERROR "Could not locate the approved Insta360 MediaSDK import library")
endif()

# probe's camera_model field comes from InsMetaDataSDK, not MediaSDK --
# GetMediaFileInfo() carries no camera-identity field of its own.
find_library(INSTA360_METADATA_LIBRARY NAMES InsMetaDataSDK
  PATHS "${INSTA360_MEDIA_SDK_ROOT}/lib" "${INSTA360_MEDIA_SDK_ROOT}/lib/x64"
  NO_DEFAULT_PATH)
if(NOT INSTA360_METADATA_LIBRARY)
  message(FATAL_ERROR "Could not locate the approved Insta360 InsMetaDataSDK import library")
endif()

target_link_libraries(gsstudio-media-helper PRIVATE
  "${INSTA360_MEDIA_LIBRARY}"
  "${INSTA360_METADATA_LIBRARY}")
