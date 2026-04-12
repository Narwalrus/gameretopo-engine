cmake_minimum_required(VERSION 3.10)
# Allow TBB 2017's old cmake_minimum_required on newer CMake versions
if(POLICY CMP0048)
  cmake_policy(SET CMP0048 OLD)
endif()
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)

project(InstantMeshesBatch CXX C)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific flags to silence TBB 2017 warnings
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -Wno-int-in-bool-context -Wno-uninitialized -Wno-unused-parameter -Wno-sign-compare")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-parameter -Wno-uninitialized")
  # GCC-specific
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-changes-meaning -Wno-maybe-uninitialized")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-maybe-uninitialized")
  endif()
elseif(MSVC)
  # MSVC: suppress the warnings that make TBB 2017 noisy
  add_definitions(/D_CRT_SECURE_NO_WARNINGS /D__TBB_NO_IMPLICIT_LINKAGE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W0 /MP /wd4996")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W0")
endif()

# TBB build (static, no malloc proxy)
set(TBB_BUILD_STATIC ON CACHE BOOL " " FORCE)
set(TBB_BUILD_SHARED OFF CACHE BOOL " " FORCE)
set(TBB_BUILD_TBBMALLOC OFF CACHE BOOL " " FORCE)
set(TBB_BUILD_TBBMALLOC_PROXY OFF CACHE BOOL " " FORCE)
set(TBB_BUILD_TESTS OFF CACHE BOOL " " FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../ext/tbb ext_build/tbb)

# Eigen is header-only, comes from NanoGUI's ext directory
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/../ext/nanogui/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../ext/nanogui/ext/eigen
  ${CMAKE_CURRENT_SOURCE_DIR}/../ext/tbb/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../ext/dset
  ${CMAKE_CURRENT_SOURCE_DIR}/../ext/pss
  ${CMAKE_CURRENT_SOURCE_DIR}/../ext/pcg32
  ${CMAKE_CURRENT_SOURCE_DIR}/../ext/rply
  ${CMAKE_CURRENT_SOURCE_DIR}/../ext/half
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

# Batch-only executable: no GUI files, no viewer, no widgets, no glutil
add_executable(InstantMeshesBatch
  ../src/main_batch.cpp
  ../src/batch.h ../src/batch.cpp
  ../src/common.h
  ../src/meshio.h ../src/meshio.cpp
  ../src/normal.h ../src/normal.cpp
  ../src/adjacency.h ../src/adjacency.cpp
  ../src/meshstats.h ../src/meshstats.cpp
  ../src/hierarchy.h ../src/hierarchy.cpp
  ../src/extract.h ../src/extract.cpp
  ../src/field.h ../src/field.cpp
  ../src/bvh.h ../src/bvh.cpp ../src/aabb.h
  ../src/subdivide.h ../src/subdivide.cpp
  ../src/reorder.h ../src/reorder.cpp
  ../src/serializer.h ../src/serializer.cpp
  ../src/smoothcurve.h ../src/smoothcurve.cpp
  ../src/cleanup.h ../src/cleanup.cpp
  ../src/dedge.h ../src/dedge.cpp
  ../ext/rply/rply.c
)

target_link_libraries(InstantMeshesBatch tbb_static)

if(WIN32 AND NOT MSVC)
  # MinGW cross-compile: static-link runtime to avoid DLL dependencies
  target_link_options(InstantMeshesBatch PRIVATE
    -static-libgcc
    -static-libstdc++
    -Wl,-Bstatic,--whole-archive
    -lwinpthread
    -Wl,--no-whole-archive,-Bdynamic
  )
endif()
# MSVC produces a static exe by default; nothing extra needed.

if(UNIX AND NOT WIN32)
  target_link_libraries(InstantMeshesBatch pthread)
endif()

set_target_properties(InstantMeshesBatch PROPERTIES OUTPUT_NAME "instant_meshes_batch")
