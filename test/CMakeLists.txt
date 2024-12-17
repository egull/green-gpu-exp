project(_test)

add_executable(cu_solver_test cu_solver_test.cpp)
target_compile_definitions(cu_solver_test PRIVATE TEST_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data"
        GRID_PATH="${grids_SOURCE_DIR}/data")
target_link_libraries(cu_solver_test
        PUBLIC
        Catch2::Catch2
        GREEN::GPU)

include(CTest)
include(Catch)
catch_discover_tests(cu_solver_test)
