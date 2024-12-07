add_library(mem_manager
mem_manager.cpp
)
target_link_libraries(mem_manager PUBLIC MPI::MPI_CXX)


