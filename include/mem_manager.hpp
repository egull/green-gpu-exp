#pragma once
#include<cstddef>
#include<string>
#include<unordered_map>
#include<iostream>

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
  const std::size_t &registered_memory(const std::string &s) const{return entries_.at(s);}
  //register memory as used
  void register_memory(const std::string &name, const std::size_t &size);
  //deregister memory and mark memory as free
  void deregister_memory(const std::string &name);
  //getter for mem entries
  const std::unordered_map<std::string, std::size_t> &entries() const{return entries_;}
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
inline std::ostream &operator<<(std::ostream &os, const mem_manager &mgr){
  double GB=1024*1024*1024;
  double MB=1024*1024;
  double kB=1024;
  os<<"#######################Memory Manager#####################";
  os<<"# total memory available: "<<mgr.total_memory()/GB<<" GB"<<std::endl;
  os<<"# "<<std::endl;
  for(auto it=mgr.entries().cbegin(); it!=mgr.entries().cend();++it){
    if(it->second <MB)
      os<<"# "<<it->first<<" :\t"<<it->second/kB<<" kB"<<std::endl;
    else if(it->second <GB)
      os<<"# "<<it->first<<" :\t"<<it->second/MB<<" MB"<<std::endl;
    else 
      os<<"# "<<it->first<<" :\t"<<it->second/GB<<" GB"<<std::endl;
  }
  os<<"# "<<std::endl;
  os<<"##########################################################";
  return os;
}
