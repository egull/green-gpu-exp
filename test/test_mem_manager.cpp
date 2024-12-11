#include <catch2/catch_test_macros.hpp>
#include <mpi.h>
#include "mem_manager.hpp"
#include <sstream>

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
TEST_CASE("print memory listing", "[MEM_MANAGER]") {
  mem_manager mgr;

  //register some memory for a 'polarization'
  mgr.register_memory("polarization", 1234);
  mgr.register_memory("greens function", std::size_t(1024)*1024*1024*3);
  mgr.register_memory("integrals", 1024*2345);

  std::cout<<mgr<<std::endl;
  mgr.poll_mem_usage();
  int Q=200000;
  std::vector<int> LV(Q);
  std::cout<<"adding: "<<sizeof(int)*Q/1024./1024.<<" MB"<<std::endl;
  mgr.poll_mem_usage();
  LV[0]=1;
  LV[Q-1]=1; //so it will not be optimized away
  std::cout<<LV[0]<<" "<<LV[Q-1]<<std::endl; 
}
TEST_CASE("register and deregister random order", "[MEM_MANAGER]") {
  mem_manager mgr;

  //register some memory for a 'polarization'
  mgr.register_memory("polarization 1", 1234);
  mgr.register_memory("polarization 2", 2345);
  mgr.register_memory("polarization 3", 3456);
  REQUIRE(mgr.registered_memory()==7035);
  mgr.deregister_memory("polarization 1");
  REQUIRE(mgr.registered_memory()==5801);
  mgr.register_memory("polarization 1", 1234);
  mgr.register_memory("polarization 4", 4567);
  REQUIRE(mgr.registered_memory()==11602);
  mgr.deregister_memory("polarization 3");
  mgr.deregister_memory("polarization 2");
  mgr.deregister_memory("polarization 4");
  mgr.deregister_memory("polarization 1");
  REQUIRE(mgr.registered_memory()==0);
}
TEST_CASE("too many entries", "[MEM_MANAGER]") {
  mem_manager mgr;
  int limit=100;
  for(int s=0;s<limit-1;++s){
    std::stringstream name; name<<"entry_"<<s;
    mgr.register_memory(name.str(), 7);
  }
  REQUIRE_THROWS(mgr.register_memory("one_too_many", 7));
}
TEST_CASE("too long entry", "[MEM_MANAGER]") {
  mem_manager mgr;
  int char_limit=100;
  std::stringstream name; 
  for(int i=0;i<char_limit;++i) name<<"a";
  mgr.register_memory(name.str(), 7);
  name<<"a";
  REQUIRE_THROWS(mgr.register_memory(name.str(), 7));
}
