#pragma once
#include<iostream>

//-> code this up tomorrow
struct task_t{
  int node;
  int core;
  int cycle;
  int q;
  int k;
  int taskidx;
  int global_rank;
  int active_core_on_node;
  int gpu;
  bool idle;

static std::vector<std::vector<task_t> > define_tasks(int inQ, int nK, int nNodes, int nCores_per_node, int nGPUs, int verbose=0){
  std::cout<<"resource estimation for inQ: "<<inQ<<" nK: "<<nK<<" nNodes: "<<nNodes<<" nCores_per_node: "<<nCores_per_node<<" on "<<nGPUs<<" GPUs"<<std::endl;
  int ntotal=0;
  int nidle=0;

  int cycle=0;
  int parallel_q_njobs=0; //number of parallel q points
  int parallel_q_nbatch=0;//number of k points per q

  std::vector<std::vector<task_t> > task_info;

  for(int startq=0;startq<inQ;startq+=nNodes){
    if(inQ-startq>nNodes){
      parallel_q_nbatch=nCores_per_node;
      parallel_q_njobs=nNodes;
    }else{
      parallel_q_nbatch=nNodes*nCores_per_node/(inQ-startq);
      parallel_q_njobs=(inQ-startq);
    }
    for(int taskidx=0;taskidx<2;++taskidx){ //first and second tasks: compute P0 from g and compute sigma from P
    for(int startk=0; startk<nK;startk+=parallel_q_nbatch){
      int kstep=std::max(parallel_q_nbatch/(nK-startk), 1);
      if(verbose>=4)
        std::cout<<"cycle: "<<cycle<<" parallel_q_nbatch: "<<parallel_q_nbatch<<" parallel_q_njobs: "<<parallel_q_njobs<<" startk: "<<startk<<" kstep: "<<kstep<<std::endl;
      task_info.push_back(std::vector<task_t>(nNodes*nCores_per_node));
      for(int n=0; n<nNodes;++n){
        int active_core_on_node=0;
        for(int c=0;c<nCores_per_node;++c){
          int idx=n*nCores_per_node+c;
          int q_this_core=idx/parallel_q_nbatch+startq;
          int k_this_core=idx%kstep==0?(idx%parallel_q_nbatch)/kstep+startk:nK;
          { 
            task_info[cycle][idx].node=n;
            task_info[cycle][idx].core=c;
            task_info[cycle][idx].global_rank=n*nCores_per_node+c;
            task_info[cycle][idx].cycle=cycle;
            task_info[cycle][idx].q=q_this_core;
            task_info[cycle][idx].k=k_this_core;
            task_info[cycle][idx].active_core_on_node=active_core_on_node;
            task_info[cycle][idx].gpu=active_core_on_node%nGPUs;
            task_info[cycle][idx].taskidx=taskidx;
          }
          if(k_this_core>=nK){ task_info[cycle][idx].idle=true;  nidle++; continue;}
          if(q_this_core>=inQ){task_info[cycle][idx].idle=true; nidle++; continue;}
          if(verbose >=5)
            std::cout<<"cycle: "<<cycle<<" node: "<<n<<" core: "<<c<<" q this core: "<<q_this_core<<" k: "<<k_this_core<<" on GPU: "<<active_core_on_node%nGPUs<<std::endl;
          task_info[cycle][idx].idle=false;
          ntotal++;
          active_core_on_node++;
        }
      }
      cycle++;
    }
    }
  }
  if(ntotal!=inQ*nK*2) throw std::logic_error("problem with task assignment (total).");
  if(ntotal+nidle!=cycle*nNodes*nCores_per_node) throw std::logic_error("problem with task assignment (total+idle).");
  if(verbose>=4){
    std::cout<<"cycles: "<<cycle<<" total: "<<ntotal<<" idle: "<<nidle<<std::endl;
  }
  return task_info;
}
};
// -> promote this to a test
/*int main(int argc, char ** argv){
  if(argc !=6){ throw std::invalid_argument("invoke command as ./a.out inQ nK nNodes nCores_per_node nGPUs");} 
  int inQ=atoi(argv[1]);
  int nK=atoi(argv[2]);
  int nNodes=atoi(argv[3]);
  int nCores_per_node=atoi(argv[4]);
  int nGPUs=atoi(argv[5]);
  define_tasks(inQ, nK, nNodes, nCores_per_node, nGPUs);
}*/
