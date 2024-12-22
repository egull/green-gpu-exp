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

#ifndef GREEN_GPU_CUGW
#define GREEN_GPU_CUGW
#include <green/gpu/common_defs.h>
#include <green/integrals/common_defs_e.h>

#include <cstring>

#include "cublas_routines_prec.h"
#include "cuda_common.h"
#include "cugw_qpt.h"
#include "task_manager.h"
#include "mem_manager.h"

namespace green::gpu {
  template <typename prec>
  using gw_reader1_callback =
      std::function<void(int, int, int, int, const std::array<size_t, 4>&, tensor<std::complex<prec>, 3>&, std::complex<double>*,
                         tensor<std::complex<prec>, 4>&, tensor<std::complex<prec>, 4>&, bool, bool)>;
  template <typename prec>
  using gw_reader2_callback = std::function<void(int, int, int, const std::array<size_t, 4>&, tensor<std::complex<prec>, 3>&,
                                                 std::complex<double>*, tensor<std::complex<prec>, 4>&, bool)>;

  using irre_pos_callback   = std::function<size_t(size_t)>;
  using mom_cons_callback   = std::function<const std::array<size_t, 4>(const std::array<size_t, 3>&)>;

  class cugw {
    using scalar_t     = typename cu_type_map<std::complex<prec>>::cxx_base_type;
    using cxx_complex  = typename cu_type_map<std::complex<prec>>::cxx_type;
    using cuda_complex = typename cu_type_map<std::complex<prec>>::cuda_type;

    using ptensor3 = green::ndarray::ndarray<std::complex<prec>, 3>;
    using ptensor4 = green::ndarray::ndarray<std::complex<prec>, 4>;

  public:
    cugw(int nts, int nt_batch, int nw_b, int ns, int nk, int ink, int nqkpt, int NQ, int nao, const task_t &this_task, mem_manager *mem_mgr_);
    ~cugw();

    void solve_g_to_P0(int _nts, int _ns, int _nk, int _ink, int _nao, const std::vector<size_t>& reduced_to_full,
               const std::vector<size_t>& full_to_reduced, std::complex<double>* Vk1k2_Qij, ztensor<5>& Sigma_tskij_host,
               int _devices_rank, int _devices_size, int verbose, irre_pos_callback& irre_pos,
               mom_cons_callback& momentum_conservation, gw_reader1_callback<prec>& r1, gw_reader2_callback<prec>& r2);
    void solve_P_to_sigma(int _nts, int _ns, int _nk, int _ink, int _nao, const std::vector<size_t>& reduced_to_full,
               const std::vector<size_t>& full_to_reduced, std::complex<double>* Vk1k2_Qij, ztensor<5>& Sigma_tskij_host,
               int _devices_rank, int _devices_size, int verbose, irre_pos_callback& irre_pos,
               mom_cons_callback& momentum_conservation, gw_reader1_callback<prec>& r1, gw_reader2_callback<prec>& r2);

  private:
    void copy_Sigma(ztensor<5>& Sigma_tskij_host, tensor<std::complex<prec>, 4>& Sigmak_stij, int k, int nts, int ns);
    void copy_Sigma_2c(ztensor<5>& Sigma_tskij_host, tensor<std::complex<prec>, 4>& Sigmak_4tij, int k, int nts);

    const int _nts;
    const int _nt_batch;
    const int _nw_b;
    const int _ns;
    const int _nk;
    const int _ink;
    const int _nqkpt;
    const int _NQ;
    const int _nao;

    bool                           _X2C;
    cublasHandle_t                 _handle;
    cusolverDnHandle_t             _solver_handle;

    //memory of these tensors will be cuda pinned for GPU transfer
    ptensor3 V_Qpm;
    ptensor3 V_Qim;
    ptensor4 Gk1_stij;
    ptensor4 Gk_smtij;
    ptensor4 Sigmak_stij; // = Gk_smtij;
    ptensor4 PQ_stab; // = Gk_smtij;

    mem_manager *mem_mgr_; //pointer to node-local memory manager to keep track of mem used

    //pinned memory pointer
    std::complex<prec> *V_Qpm_hostptr;
    std::complex<prec> *V_Qim_hostptr;
    std::complex<prec> *Gk1_stij_hostptr;
    std::complex<prec> *Gk_smtij_hostptr;
    std::complex<prec> *Sigmak_stij_hostptr; //will use memory of Gk1
    std::complex<prec> *PQ_stab_hostptr;
    std::complex<prec> *P0Q_stab_hostptr;  //will use memory of PQ

    cuda_complex*                  g_kstij_device;
    cuda_complex*                  g_ksmtij_device;
    cuda_complex*                  sigma_kstij_device;

    int*                           sigma_k_locks;
  };
}  // namespace green::gpu

#endif  // GREEN_GPU_CU_ROUTINES_H
