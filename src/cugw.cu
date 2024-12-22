/*
 * Copyright (c) 2023 University of Michigan
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this
 * software and associated documentation files (the “Software”), to deal in the Software
 * without restriction, including without limitation the rights to use, copy, modify,
 * merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
 * FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <green/gpu/cugw.h>

namespace green::gpu {
  template <typename prec>
    cugw<prec>::cugw(int nts, int nt_batch, int nw_b, int ns, int nk, int ink, int nqkpt, int NQ, int nao, const task_t &this_task, mem_manager *mem_mgr):
      _nts(nts),
      _nt_batch(nt_batch),
      _nw_b(nw_b),
      _ns(ns),
      _nk(nk),
      _ink(ink), 
      _nqkpt(nqkpt), 
      _NQ(NQ), 
      _nao(nao),
      V_Qpm(nullptr, _NQ, _nao, _nao),
      V_Qim(nullptr, _NQ, _nao, _nao),
      Gk1_stij(nullptr, _ns, _nts, _nao, _nao), 
      Gk_smtij(nullptr, _ns, _nts, _nao, _nao),
      PQ_stab(nullptr, _ns, _nts, _NQ, _NQ),
      mem_mgr_(mem_mgr)
      {
    //set the proper GPU and initialize cublas solver
    if (cudaSetDevice(this_task.gpu) != cudaSuccess) throw std::runtime_error("Error in cudaSetDevice2");
    if (cublasCreate(&_handle) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("Node " + std::to_string(this_task.node)+"Core" + std::to_string(this_task.core) + ": error initializing cublas");

    // initialize and transfer device green's function, self-energy and IR matrices
    _X2C = (_ns == 4?true:false);

    //allocate host memory for GPU transfer 
    if (cudaMallocHost(&V_Qpm_hostptr, _NQ* _nao*_nao* sizeof(cuda_complex),cudaHostAllocWriteCombined) != cudaSuccess) throw std::runtime_error("failure to allocate V_Qpm_ptr");
    if (cudaMallocHost(&V_Qim_hostptr, _NQ* _nao*_nao* sizeof(cuda_complex),cudaHostAllocWriteCombined) != cudaSuccess) throw std::runtime_error("failure to allocate V_Qim_ptr");
    if (cudaMallocHost(&Gk1_stij_hostptr, _ns*_nts*_nao*_nao* sizeof(cuda_complex),cudaHostAllocWriteCombined) != cudaSuccess) throw std::runtime_error("failure to allocate Gk1_stij_ptr");
    if (cudaMallocHost(&Gk_smtij_hostptr, _ns*_nts*_nao*_nao* sizeof(cuda_complex), cudaHostAllocWriteCombined) != cudaSuccess) throw std::runtime_error("failure to allocate Gk_smtij_ptr");
    if (cudaMallocHost(&PQ_stab_hostptr, _ns*_nts*_NQ*_NQ* sizeof(cuda_complex), cudaHostAllocWriteCombined) != cudaSuccess) throw std::runtime_error("failure to allocate PQ_stab_ptr");

 
    //allocate host matrix wrappers around host pointers
    V_Qpm.set_ref(V_Qpm_hostptr);
    V_Qim.set_ref(V_Qim_hostptr);
    Gk1_stij.set_ref(Gk1_stij_hostptr);
    Gk_smtij.set_ref(Gk_smtij_hostptr);
    mem_mgr_->register_memory("V_Qpm and V_Qim",this_task.global_rank,2*V_Qpm.size()*sizeof(std::complex<prec>));
    mem_mgr_->register_memory("Gk1_stij, Sigma_stij, and Gk_smtij",this_task.global_rank,2*Gk1_stij.size()*sizeof(std::complex<prec>));
    mem_mgr_->register_memory("P and P0 Q",this_task.global_rank,PQ_stab.size()*sizeof(std::complex<prec>));

    sigma_kstij_device = nullptr;
    g_kstij_device     = nullptr;
    g_ksmtij_device    = nullptr;

  }

  template <typename prec>
  void cugw<prec>::solve_g_to_P0(int _nts, int _ns, int _nk, int _ink, int _nao, const std::vector<size_t>& reduced_to_full,
                               const std::vector<size_t>& full_to_reduced, std::complex<double>* Vk1k2_Qij,
                               ztensor<5>& Sigma_tskij_host, int _devices_rank, int _devices_size,
                               int verbose, irre_pos_callback& irre_pos, mom_cons_callback& momentum_conservation,
                               gw_reader1_callback<prec>& r1, gw_reader2_callback<prec>& r2) {
/*    // this is the main GW loop
    if (!_devices_rank && verbose > 0) std::cout << "GW main loop" << std::endl;
    qpt.verbose() = verbose;

    for (size_t q_reduced_id = _devices_rank; q_reduced_id < _ink; q_reduced_id += _devices_size) {
      if (verbose > 2) std::cout << "q = " << q_reduced_id << std::endl;
      size_t q = reduced_to_full[q_reduced_id];
      qpt.reset_Pqk0();
      for (size_t k = 0; k < _nk; ++k) {
        std::array<size_t, 4> k_vector      = momentum_conservation({
            {k, 0, q}
        });
        size_t                k1            = k_vector[3];
        size_t                k_reduced_id  = full_to_reduced[k];   // irre_pos(index[k]);
        size_t                k1_reduced_id = full_to_reduced[k1];  // irre_pos(index[k1]);
        bool                  need_minus_k  = reduced_to_full[k_reduced_id] != k;
        bool                  need_minus_k1 = reduced_to_full[k1_reduced_id] != k1;

        r1(k, k1, k_reduced_id, k1_reduced_id, k_vector, V_Qpm, Vk1k2_Qij, Gk_smtij, Gk1_stij, need_minus_k, need_minus_k1);

        gw_qkpt<prec>* qkpt = obtain_idle_qkpt(qkpts);
        if (_low_device_memory) {
          if (!_X2C) {
            qkpt->set_up_qkpt_first(Gk1_stij.data(), Gk_smtij.data(), V_Qpm.data(), k_reduced_id, need_minus_k, k1_reduced_id,
                                    need_minus_k1);
          } else {
            // In 2cGW, G(-k) = G*(k) has already been addressed in r1()
            qkpt->set_up_qkpt_first(Gk1_stij.data(), Gk_smtij.data(), V_Qpm.data(), k_reduced_id, false, k1_reduced_id, false);
          }
        } else {
          qkpt->set_up_qkpt_first(nullptr, nullptr, V_Qpm.data(), k_reduced_id, need_minus_k, k1_reduced_id, need_minus_k1);
        }
        qkpt->compute_first_tau_contraction(qpt.Pqk0_tQP(qkpt->all_done_event()), qpt.Pqk0_tQP_lock());
      }
      qpt.wait_for_kpts();
      qpt.scale_Pq0_tQP(1. / _nk);
      qpt.transform_tw();
      qpt.compute_Pq();
      qpt.transform_wt();
      // Write to Sigma(k), k belongs to _ink
      for (size_t k_reduced_id = 0; k_reduced_id < _ink; ++k_reduced_id) {
        size_t k = reduced_to_full[k_reduced_id];
        for (size_t q_or_qinv = 0; q_or_qinv < _nk; ++q_or_qinv) {
          if (full_to_reduced[q_or_qinv] == q_reduced_id) {  // only q and q_inv proceed
            std::array<size_t, 4> k_vector      = momentum_conservation({
                {k, q_or_qinv, 0}
            });
            size_t                k1            = k_vector[3];
            size_t                k1_reduced_id = full_to_reduced[k1];  // irre_pos(index[k1]);
            bool                  need_minus_k1 = reduced_to_full[k1_reduced_id] != k1;
            bool                  need_minus_q  = reduced_to_full[q_reduced_id] != q_or_qinv;

            r2(k, k1, k1_reduced_id, k_vector, V_Qim, Vk1k2_Qij, Gk1_stij, need_minus_k1);

            gw_qkpt<prec>* qkpt = obtain_idle_qkpt(qkpts);
            if (_low_device_memory) {
              if (!_X2C) {
                qkpt->set_up_qkpt_second(Gk1_stij.data(), V_Qim.data(), k_reduced_id, k1_reduced_id, need_minus_k1);
                qkpt->compute_second_tau_contraction(Sigmak_stij.data(),
                                                     qpt.Pqk_tQP(qkpt->all_done_event(), qkpt->stream(), need_minus_q));
                copy_Sigma(Sigma_tskij_host, Sigmak_stij, k_reduced_id, _nts, _ns);
              } else {
                // In 2cGW, G(-k) = G*(k) has already been addressed in r2()
                qkpt->set_up_qkpt_second(Gk1_stij.data(), V_Qim.data(), k_reduced_id, k1_reduced_id, false);
                qkpt->compute_second_tau_contraction_2C(Sigmak_stij.data(),
                                                        qpt.Pqk_tQP(qkpt->all_done_event(), qkpt->stream(), need_minus_q));
                copy_Sigma_2c(Sigma_tskij_host, Sigmak_stij, k_reduced_id, _nts);
              }
            } else {
              qkpt->set_up_qkpt_second(nullptr, V_Qim.data(), k_reduced_id, k1_reduced_id, need_minus_k1);
              qkpt->compute_second_tau_contraction(nullptr, qpt.Pqk_tQP(qkpt->all_done_event(), qkpt->stream(), need_minus_q));
            }
          }
        }
      }
    }
    cudaDeviceSynchronize();
    if (!_low_device_memory and !_X2C) {
      copy_Sigma_from_device_to_host(sigma_kstij_device, Sigma_tskij_host.data(), _ink, _nao, _nts, _ns);
    }*/
  }

  template <typename prec>
  void cugw<prec>::copy_Sigma(ztensor<5>& Sigma_tskij_host, tensor<std::complex<prec>, 4>& Sigmak_stij, int k, int nts,
                                    int ns) {
    /*for (size_t t = 0; t < nts; ++t) {
      for (size_t s = 0; s < ns; ++s) {
        matrix(Sigma_tskij_host(t, s, k)) += matrix(Sigmak_stij(s, t)).template cast<typename std::complex<double>>();
      }
    }*/
  }
  template <typename prec>
  void cugw<prec>::copy_Sigma_2c(ztensor<5>& Sigma_tskij_host, tensor<std::complex<prec>, 4>& Sigmak_4tij, int k, int nts) {
    /*size_t    nao = Sigmak_4tij.shape()[3];
    size_t    nso = 2 * nao;
    MatrixXcf Sigma_ij(nso, nso);
    for (size_t ss = 0; ss < 3; ++ss) {
      size_t a       = (ss % 2 == 0) ? 0 : 1;
      size_t b       = ((ss + 1) / 2 != 1) ? 0 : 1;
      size_t i_shift = a * nao;
      size_t j_shift = b * nao;
      for (size_t t = 0; t < nts; ++t) {
        matrix(Sigma_tskij_host(t, 0, k)).block(i_shift, j_shift, nao, nao) +=
            matrix(Sigmak_4tij(ss, t)).template cast<typename std::complex<double>>();
        if (ss == 2) {
          matrix(Sigma_tskij_host(t, 0, k)).block(j_shift, i_shift, nao, nao) +=
              matrix(Sigmak_4tij(ss, t)).conjugate().transpose().template cast<typename std::complex<double>>();
        }
      }
    }*/
  }

  template <typename prec>
  cugw<prec>::~cugw() {
    if (cublasDestroy(_handle) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublas error destroying handle");
    cudaFreeHost(V_Qpm_hostptr);
    cudaFreeHost(V_Qim_hostptr);
    cudaFreeHost(Gk1_stij_hostptr);
    cudaFreeHost(Gk_smtij_hostptr);
  }

  template class cugw<float>;
  template class cugw<double>;

}  // namespace green::gpu
