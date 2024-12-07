add_library(mem_manager
manager.cpp
)
target_link_libraries(mem_manager PUBLIC MPI::MPI_CXX)


