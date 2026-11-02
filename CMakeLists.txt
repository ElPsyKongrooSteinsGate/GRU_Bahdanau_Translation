cmake_minimum_required(VERSION 3.20)
project(GRU_Bahdanau_Translation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED CMAKE_PREFIX_PATH)
    message(FATAL_ERROR "Set CMAKE_PREFIX_PATH to your LibTorch directory, e.g. C:/Library/libtorch")
endif()

find_package(Torch REQUIRED)
find_package(unofficial-sqlite3 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

add_executable(translate
    src/main.cpp
    src/dataset.cpp
    src/vocab.cpp
    src/model.cpp
    src/trainer.cpp
)

target_include_directories(translate PRIVATE include)
target_link_libraries(translate PRIVATE
    "${TORCH_LIBRARIES}"
    unofficial::sqlite3::sqlite3
    nlohmann_json::nlohmann_json
)

# Required by some LibTorch Windows configurations.
if(MSVC)
    target_compile_options(translate PRIVATE /bigobj)
endif()

# Ensure the executable uses the same C++ runtime setting as LibTorch.
set_property(TARGET translate PROPERTY
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
