cmake_minimum_required(VERSION 3.15)
project(Graduino VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Compilar el motor (Engine) como libreria
add_library(engine INTERFACE)
target_include_directories(engine INTERFACE src)

# 2. Descargar e integrar GoogleTest con FetchContent
# (GoogleTest 1.12.1 es la última versión con soporte oficial para C++11)
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
)
# Evitar conflictos de runtime en Windows/MSVC
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# 3. Habilitar testing
enable_testing()

# 4. Crear ejecutable de tests y linkear con GTest
add_executable(run_tests tests/test_engine.cpp tests/test_nn.cpp)
target_link_libraries(run_tests PRIVATE engine GTest::gtest_main)

# 5. Integrar los tests de GTest directamente en CTest
include(GoogleTest)
gtest_discover_tests(run_tests)
