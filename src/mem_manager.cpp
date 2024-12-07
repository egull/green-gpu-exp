#include"mem_manager.hpp"
#include <unistd.h>
#include <iostream>
#include <sys/time.h>
#include <sys/resource.h>

void mem_manager::compute_total_memory(){
  //figure out how much memory is available on the machine
  std::size_t pages = sysconf(_SC_PHYS_PAGES);
  std::size_t page_size = sysconf(_SC_PAGE_SIZE);
  total_memory_=pages*page_size;
}
void mem_manager::register_memory(const std::string &name, const std::size_t &size){
  //make sure we have unique entries
  if(entries_.count(name)>0) throw std::runtime_error("Mem manager: entry: "+name+"is already registered");
 
  entries_[name]=size;
  registered_memory_+=size;

  if(registered_memory_>total_memory_){
    //consider throwing to save the user some pain.
    std::cout<<"WARNING: allocated memory> total memory: "<<registered_memory_/GB<<" "<<total_memory_/GB<<" GB"<<std::endl;
  }
}
void mem_manager::deregister_memory(const std::string &name){
  //make sure we have unique entries
  if(entries_.count(name)==0) throw std::runtime_error("Mem manager: entry: "+name+" is not registered");

  std::size_t size=entries_.at(name); 
  entries_.erase(name);
  registered_memory_-=size;

  if(registered_memory_<0){
    throw std::logic_error("registered memory is below zero");
  }
}
std::size_t mem_manager::poll_mem_usage() const{
  //this posix command has a whole bunch of additional info, we're only polling maxrss
  //rss is resident set size
  rusage r;
  getrusage(RUSAGE_SELF, &r);
  return r.ru_maxrss;
}
