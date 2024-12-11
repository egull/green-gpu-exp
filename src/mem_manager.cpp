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
  if(shmem_ptr_->num_entries_==MMGR_MAX_REG_ENTRIES-1) throw std::runtime_error("too many registered mem entries, increase max size");
  if(strlen(name.c_str())>MMGR_MAX_STRING_LENGTH) throw std::runtime_error("string is too long");
  entry_t buffer; strcpy(buffer.first, name.c_str()); buffer.second=size;
  memcpy(shmem_ptr_->entries_+shmem_ptr_->num_entries_++,&buffer, sizeof(entry_t));
  shmem_ptr_->registered_memory_+=size;

  if(shmem_ptr_->registered_memory_>total_memory_){
    //consider throwing to save the user some pain.
    std::cout<<"WARNING: allocated memory> total memory: "<<shmem_ptr_->registered_memory_/GB<<" "<<total_memory_/GB<<" GB"<<std::endl;
  }
}
void mem_manager::deregister_memory(const std::string &name){
  //make sure we have unique entries
  if(count(name)==0) throw std::runtime_error("Mem manager: entry: "+name+" is not registered");

  int i=0;
  while(shmem_ptr_->entries_[i].first!=name) i++;
  std::size_t size=shmem_ptr_->entries_[i].second;
  shmem_ptr_->registered_memory_-=size;
  memmove(&(shmem_ptr_->entries_[i]), &(shmem_ptr_->entries_[i+1]), sizeof(entry_t)*(shmem_ptr_->num_entries_-i-1));
  shmem_ptr_->num_entries_--;

  if(shmem_ptr_->registered_memory_<0){
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
  for(int i=0;i<shmem_ptr_->num_entries_;++i){
    if(shmem_ptr_->entries_[i].first==s)
      return shmem_ptr_->entries_[i].second;
  }
  throw std::runtime_error("entry "+s+" not found");
}
std::size_t mem_manager::count(const std::string &s) const{
  std::size_t c=0;
  for(int i=0;i<shmem_ptr_->num_entries_;++i){
    if(shmem_ptr_->entries_[i].first==s) c++;
  }
  return c;
}
void mem_manager::allocate_memory(){
  int global_rank; MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
  MPI_Info info; MPI_Info_create(&info);
  MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,global_rank,info,&shmem_comm_);
  MPI_Comm_size(shmem_comm_,&shmem_size_);
  MPI_Comm_rank(shmem_comm_,&shmem_rank_);

  int err=MPI_Win_allocate_shared(shmem_rank_==0?sizeof(shmem):0, sizeof(shmem), MPI_INFO_NULL, shmem_comm_, &shmem_alloc_, &shmem_win_);
  if(err !=MPI_SUCCESS){
    std::cerr<<"memory allocation error on shmem rank: "<<shmem_rank_<<std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  //get a local pointer to shared memory buffer
  MPI_Aint rss2;
  int soT2;
  MPI_Win_shared_query(shmem_win_, 0, &rss2, &soT2, &shmem_ptr_);

  //entries_=new entry_t[MMGR_MAX_REG_ENTRIES];
 
}
