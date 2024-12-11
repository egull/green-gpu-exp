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
  if(count(name)>0) throw std::runtime_error("Mem manager: entry: "+name+"is already registered");
  if(num_entries_==MMGR_MAX_REG_ENTRIES-1) throw std::runtime_error("too many registered mem entries, increase max size");
  if(strlen(name.c_str())>MMGR_MAX_STRING_LENGTH) throw std::runtime_error("string is too long");
  entry_t buffer; strcpy(buffer.first, name.c_str()); buffer.second=size;
  memcpy(entries_+num_entries_++,&buffer, sizeof(entry_t));
  registered_memory_+=size;

  if(registered_memory_>total_memory_){
    //consider throwing to save the user some pain.
    std::cout<<"WARNING: allocated memory> total memory: "<<registered_memory_/GB<<" "<<total_memory_/GB<<" GB"<<std::endl;
  }
}
void mem_manager::deregister_memory(const std::string &name){
  //make sure we have unique entries
  if(count(name)==0) throw std::runtime_error("Mem manager: entry: "+name+" is not registered");

  int i=0;
  while(entries_[i].first!=name) i++;
  std::size_t size=entries_[i].second;
  registered_memory_-=size;
  //for(int j=i+1;j<num_entries_;++j) entries_[j-1]=entries_[j]; //copy one back
  memmove(&(entries_[i]), &(entries_[i+1]), sizeof(entry_t)*(num_entries_-i-1));
  num_entries_--;

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
const std::size_t &mem_manager::registered_memory(const std::string &s) const{
  for(int i=0;i<num_entries_;++i){
    if(entries_[i].first==s)
      return entries_[i].second;
  }
  throw std::runtime_error("entry "+s+" not found");
}
std::size_t mem_manager::count(const std::string &s) const{
  std::size_t c=0;
  for(int i=0;i<num_entries_;++i){
    if(entries_[i].first==s) c++;
  }
  return c;
}
