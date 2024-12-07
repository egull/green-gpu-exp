#pragma once
#include<cstddef>
#include<string>
#include<unordered_map>

//memory manager. all units are in bytes
class mem_manager{
public:
  mem_manager():
   registered_memory_(0.){
    compute_total_memory();
  }
  //obtain totally available memory on this machine
  const std::size_t &total_memory() const{return total_memory_;}
  //memory that has been registered as used 
  const std::size_t &registered_memory() const{return registered_memory_;}
  //register memory as used
  void register_memory(const std::string &name, const std::size_t &size);
private:
  void compute_total_memory();
  //total memory in hardware. Detected by polling system
  std::size_t total_memory_;  
  
  //memory registered as used 
  std::size_t registered_memory_;  

  std::unordered_map<std::string, std::size_t> entries_;

  //constant for gigabytes
  constexpr static std::size_t GB=1024*1024*1024;
};
