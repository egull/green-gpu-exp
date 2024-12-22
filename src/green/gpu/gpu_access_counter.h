#pragma once
#include"green/integrals/buffered_reader/shared_memory_region.hpp"
class gpu_access_counter{
public:
  gpu_access_counter(const MPI_Comm &comm, std::size_t max):
    gpu_comm_(comm),
    max_(max){
      ctr_.setup_shmem_region(gpu_comm_, 1);
      int rank; MPI_Comm_rank(gpu_comm_, &rank);
      if(rank==0) ctr_[0]=0; //initialize
      MPI_Barrier(gpu_comm_); //sync
  }
  unsigned long long operator()()const{ return ctr_[0];}
  gpu_access_counter & operator++(int){
    ctr_.acquire_exclusive_lock();
    ctr_[0]++;
    ctr_.release_exclusive_lock();
    return *this;
  }
  void increment_to_max_or_wait(){
    ctr_.acquire_exclusive_lock();
    while(ctr_[0]>=max_){
      ctr_.release_exclusive_lock();
      std::this_thread::sleep_for(std::chrono::milliseconds(1)); //go to sleep for one millisecond, then check again
      ctr_.acquire_exclusive_lock();
    }
    ctr_[0]++;
    ctr_.release_exclusive_lock();
  }
  gpu_access_counter & operator--(int){
    ctr_.acquire_exclusive_lock();
    ctr_[0]--;
    ctr_.release_exclusive_lock();
    return *this;
  }
private:
  shared_memory_region<unsigned long long> ctr_;
  const MPI_Comm gpu_comm_;
  std::size_t max_;
};

