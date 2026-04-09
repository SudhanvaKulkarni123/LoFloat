cmake_minimum_required(VERSION 3.16)
project(lo_float VERSION 0.1 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Header‑only library target
# ---------------------------------------------------------------------------
add_library(lo_float INTERFACE)

# Require C++20 everywhere that links lo_float
target_compile_features(lo_float INTERFACE cxx_std_20)

target_include_directories(lo_float
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>          # path after install
)

target_include_directories(lo_float
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/xsimd/include>
)


if(MSVC)
    target_compile_options(lo_float INTERFACE /W4 /permissive-)
else()
    target_compile_options(lo_float INTERFACE -Wall -Wextra -pedantic)
endif()

option(LOFLOAT_BUILD_TESTS "Build LoFloat unit tests" ON)
if(LOFLOAT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()


include(GNUInstallDirs)

install(TARGETS lo_float
        EXPORT lo_floatConfig)

install(DIRECTORY src/          # copy headers verbatim
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/lo_float)

# CMake package export (config mode)
install(EXPORT lo_floatConfig
        NAMESPACE lo_float::          # target will be lo_float::lo_float
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lo_float)
