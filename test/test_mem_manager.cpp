#include <catch2/catch_test_macros.hpp>
#include <mpi.h>
#include "mem_manager.hpp"

TEST_CASE("construct a mem manager", "[MEM_MANAGER]") {
  mem_manager mgr;

  //poll total memory on machine
  std::size_t total_mem=mgr.total_memory();
  std::cout<<"total memory on this machine is: "<<total_mem/1024./1024./1024.<<" GB"<<std::endl;
  REQUIRE(total_mem>1024); 
  REQUIRE(mgr.registered_memory()==0); 
}
TEST_CASE("register and register twice", "[MEM_MANAGER]") {
  mem_manager mgr;

  //register some memory for a 'polarization'
  mgr.register_memory("polarization", 1234);
  //duplicate registration
  REQUIRE_THROWS(mgr.register_memory("polarization", 1234));
  REQUIRE(mgr.registered_memory()==1234);
  REQUIRE(mgr.registered_memory("polarization")==1234);
}
TEST_CASE("register and deregister", "[MEM_MANAGER]") {
  mem_manager mgr;

  //register some memory for a 'polarization'
  mgr.register_memory("polarization", 1234);
  REQUIRE(mgr.registered_memory()==1234);
  //deregister 'polarization'
  mgr.deregister_memory("polarization");
  REQUIRE(mgr.registered_memory()==0);
}
