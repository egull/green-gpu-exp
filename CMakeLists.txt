cmake_minimum_required(VERSION 3.5)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
project(MPI_READ_LOCK)

if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_BINARY_DIR})
    message(FATAL_ERROR "In-source builds are not allowed")
endif ()

set(CMAKE_CXX_STANDARD 11)
SET( CMAKE_CXX_FLAGS "-O3 -g -fno-omit-frame-pointer -fsanitize=address")

execute_process(
        COMMAND git log -1 --format=%h
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        OUTPUT_VARIABLE TMP_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(GIT_HASH "${PROJECT_NAME}:${TMP_HASH}" CACHE INTERNAL "")


list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

find_package(MPI REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_subdirectory(src)

add_executable(mem_manager.exe mem_manager.cpp)
target_link_libraries(mem_manager.exe PUBLIC mem_manager)
target_link_libraries(mem_manager.exe PUBLIC MPI::MPI_CXX)

target_compile_definitions(mem_manager.exe PRIVATE GIT_HASHES="${GIT_HASH}")

option(Build_Tests "Build tests" ON)
if (Build_Tests)
    enable_testing()
    add_subdirectory(test)
endif ()
install(TARGETS mem_manager.exe DESTINATION bin)
