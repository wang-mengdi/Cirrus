#pragma once

#include "PoissonGrid.h"


//launch on block size 128
void NegativeLaplacianSameLevel128(HADeviceGrid<Tile>& grid, thrust::device_vector<HATileInfo<Tile>>& tiles, int launch_tile_num, int subtree_level, uint8_t launch_tile_types, int x_channel, int Ax_channel, bool calc_diag = false);

//Ax=-lap(x)
//will execute on all ghost and leafs
//at the end ghost values are gathered to parent leafs
//the results are calculated on interior cells only
void FullNegativeLaplacian(HADeviceGrid<Tile>& grid, const int x_channel, const int Ax_channel, bool calc_diag = false);

void ProlongateAndUpdate128(HADeviceGrid<Tile>& grid, int coarse_x_channel, int fine_x_channel, int fine_level, T prolong_coeff);

class GMGSolver {
public:
    T one_over_alpha = 1.0;//somehow only 1.0 works for multiple-level
    T prolong_coeff = 1; // 2 is better but does not work for cross-levels

    GMGSolver(T _one_over_alpha = 1, T _prolong_coeff = 1) : d_tmp(7), one_over_alpha(_one_over_alpha), prolong_coeff(_prolong_coeff) {
        gamma_d = thrust::raw_pointer_cast(d_tmp.data());
        beta_d = gamma_d + 1;
        alpha_d = gamma_d + 2;
        gamma_old_d = gamma_d + 3;
        fp_d = gamma_d + 4;
		mean_d = gamma_d + 5;
		count_d = gamma_d + 6;
    }

    void VCycle(HADeviceGrid<Tile>& grid, int x_channel, int f_channel, const int D_channel, int level_iters, int coarsest_iters);

    //solve -lap(x_channel)=b_channel
    //b channel will be modified
    //return: (num_iters, relative_residual)
    //if sync_stride is 0 or -1, we will not perform synced dot and just do max_iters
    //in this case, the returned value will be [max_iters, -1] 
    std::tuple<int, double> solve(HADeviceGrid<Tile>& grid, bool verbose, int max_iters, double relative_tolerance, int level_iters, int coarsest_iters, int sync_stride, bool is_pure_neumann);

    thrust::device_vector<double> d_tmp;
    double* gamma_d;
    double* beta_d;
    double* alpha_d;
    double* gamma_old_d;
    double* fp_d;
    double* mean_d;
	double* count_d;
};
