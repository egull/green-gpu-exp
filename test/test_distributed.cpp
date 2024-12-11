#include <catch2/catch_test_macros.hpp>
#include <mpi.h>
#include "mem_manager.hpp"
#include <sstream>

TEST_CASE("mem manager on multiple ranks", "[DIST_MEM_MANAGER]") {
  mem_manager mgr;
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if(size==1) return;

  if(rank==0) mgr.register_memory("polarization 1", 1234);
  if(rank==1) mgr.register_memory("polarization 2", 1234);

  //poll total memory on machine
  MPI_Barrier(MPI_COMM_WORLD);
  REQUIRE(mgr.registered_memory()==2468); 
  MPI_Barrier(MPI_COMM_WORLD);
}
TEST_CASE("register with rank", "[DIST_MEM_MANAGER]") {
  mem_manager mgr;
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if(size==1) return;

  mgr.register_memory("polarization", rank, 1234);

  //poll total memory on machine
  MPI_Barrier(MPI_COMM_WORLD);
  REQUIRE(mgr.registered_memory()==size*1234); 
  MPI_Barrier(MPI_COMM_WORLD);
  mgr.deregister_memory("polarization", rank);
  MPI_Barrier(MPI_COMM_WORLD);
  REQUIRE(mgr.registered_memory()==0); 
}
TEST_CASE("print memory listing", "[DIST_MEM_MANAGER]") {
  mem_manager mgr;
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if(size==1) return;

  mgr.register_memory("polarization", rank, 1234);

  //poll total memory on machine
  MPI_Barrier(MPI_COMM_WORLD);
  REQUIRE(mgr.registered_memory()==size*1234);
  MPI_Barrier(MPI_COMM_WORLD);
  if(!rank) std::cout<<mgr<<std::endl;
  MPI_Barrier(MPI_COMM_WORLD);
}

