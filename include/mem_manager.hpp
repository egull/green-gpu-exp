#pragma once
#include<cstddef>
#include<string>
#include<iostream>
#include<sstream>
#include<vector>
#include<mpi.h>

#define MMGR_MAX_STRING_LENGTH 100
#define MMGR_MAX_REG_ENTRIES 100

//memory manager. all units are in bytes
class mem_manager{

public:
  typedef std::pair<char[MMGR_MAX_STRING_LENGTH], std::size_t> entry_t;
  typedef entry_t* entries_t;
  mem_manager()
   {
    allocate_memory();
    shmem_ptr_->num_entries_=0;
    shmem_ptr_->registered_memory_=0;
    compute_total_memory();
  }
  ~mem_manager(){
    MPI_Win_free(&shmem_win_);
  }
  //obtain totally available memory on this machine
  const std::size_t &total_memory() const{return total_memory_;}
  //memory that has been registered as used. return total amount 
  const std::size_t &registered_memory() const{return shmem_ptr_->registered_memory_;}
  //memory that has been registered as used. return amoutn belonging to string s
  const std::size_t &registered_memory(const std::string &s) const;
  //register memory as used
  void register_memory(const std::string &name, const std::size_t &size);
  //register memory as used. pass in rank to generate a name
  void register_memory(const std::string &name, int rank, const std::size_t &size);
  //deregister memory and mark memory as free
  void deregister_memory(const std::string &name);
  //deregister memory and mark memory as free, passing in also a rank
  void deregister_memory(const std::string &name, int rank);
  //getter for mem entries
  const entry_t *entries() const{return shmem_ptr_->entries_;}
  //getter for # of entries
  const std::size_t&num_entries() const{return shmem_ptr_->num_entries_;}
  //poll system to get actual process mem usage 
  std::size_t poll_mem_usage()const;

private:
  //shared memory allocation of shared buffer
  void allocate_memory();
  void compute_total_memory();
  std::size_t count(const std::string &s) const;
  //total memory in hardware. Detected by polling system
  std::size_t total_memory_;  
 
  struct shmem{
    //memory registered as used 
    std::size_t registered_memory_;  //size of memory registered
    entry_t entries_[MMGR_MAX_REG_ENTRIES];       //actual entries
    std::size_t num_entries_; //# of entries
  } *shmem_ptr_;  //use this for access

  shmem *shmem_alloc_; //this is only valid on shmem rank 0
  MPI_Win   shmem_win_;

  //MPI shmem communicators,, rank, and size
  MPI_Comm shmem_comm_;
  int shmem_size_;
  int shmem_rank_;

  //constant for gigabytes
  constexpr static std::size_t GB=1024*1024*1024;
};
inline std::ostream &operator<<(std::ostream &os, const mem_manager &mgr){
  double GB=1024*1024*1024;
  double MB=1024*1024;
  double kB=1024;
  os<<"#######################Memory Manager#####################"<<std::endl;
  os<<"# total memory available: "<<mgr.total_memory()/GB<<" GB"<<std::endl;
  os<<"# "<<std::endl;
  for(int i=0;i<mgr.num_entries();++i){
    if(mgr.entries()[i].second <MB)
      os<<"# "<<mgr.entries()[i].first<<" :\t"<<mgr.entries()[i].second/kB<<" kB"<<std::endl;
    else if(mgr.entries()[i].second <GB)
      os<<"# "<<mgr.entries()[i].first<<" :\t"<<mgr.entries()[i].second/MB<<" MB"<<std::endl;
    else 
      os<<"# "<<mgr.entries()[i].first<<" :\t"<<mgr.entries()[i].second/GB<<" GB"<<std::endl;
  }
  os<<"# "<<std::endl;
  std::size_t actual_mem=mgr.poll_mem_usage();
  if(actual_mem<MB)
    os<<"# actual mem used: \t"<<mgr.poll_mem_usage()/kB<<" kB"<<std::endl;
  else if(actual_mem<GB)
    os<<"# actual mem used: \t"<<mgr.poll_mem_usage()/MB<<" MB"<<std::endl;
  else
    os<<"# actual mem used: \t"<<mgr.poll_mem_usage()/GB<<" GB"<<std::endl;
  //if this comes close to 1 we are using too much memory
  os<<"# actual mem div hardware mem  : "<<(double)actual_mem/mgr.total_memory()<<std::endl;
  //if this is far from 1 we substantial memory we did not register
  os<<"# registered mem div actual mem: "<<(double)mgr.registered_memory()/(double)actual_mem<<std::endl;
  os<<"##########################################################";
  return os;
}
