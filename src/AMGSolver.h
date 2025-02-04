#pragma once

#include "PoissonGrid.h"

void NegativeLaplacianAMG128(HADeviceGrid<Tile>& grid, thrust::device_vector<HATileInfo<Tile>>& tiles, int launch_tile_num, uint8_t launch_tile_types, int x_channel, int coeff_channel, int Ax_channel);

void CalculateAMGCoefficients(HADeviceGrid<Tile>& grid, const int coeff_channel, const uint8_t launch_tile_types);

//void FullNegativeLaplacianAMG(HADeviceGrid<Tile>& grid, const int x_channel, const int coeff_channel, const int Ax_channel);

void GaussSeidelAMG(int iters, int order, HADeviceGrid<Tile>& grid, const int level, const int x_channel, const int coeff_channel, const int rhs_channel);

void VCycleAMG(HADeviceGrid<Tile>& grid, const int x_channel, const int f_channel, const int tmp_channel, const int rhs_channel, const int coeff_channel, int level_iters, int coarsest_iters, const T one_over_alpha, const T prolong_coeff);

class AMGSolver {
public:
    T one_over_alpha = 1.0;//somehow only 1.0 works for multiple-level
    //T one_over_alpha = 1.0 / 8 * 8;
    //T one_over_alpha = 1.0 / 2;
    T prolong_coeff = 1; // 2 is better but does not work for cross-levels
    //T prolong_coeff = 2; // 2 is better but 

    AMGSolver(int _coeff_channel, T _one_over_alpha = 1, T _prolong_coeff = 1) : d_tmp(7), coeff_channel(_coeff_channel),
        one_over_alpha(_one_over_alpha), prolong_coeff(_prolong_coeff)
    {
        gamma_d = thrust::raw_pointer_cast(d_tmp.data());
        beta_d = gamma_d + 1;
        alpha_d = gamma_d + 2;
        gamma_old_d = gamma_d + 3;
        fp_d = gamma_d + 4;
		mean_d = gamma_d + 5;
		count_d = gamma_d + 6;
    }

	void prepareTypesAndCoeffs(HADeviceGrid<Tile>& grid);

    //solve -lap(x_channel)=b_channel
    //b channel will be modified
    //return: (num_iters, relative_residual)
    //if sync_stride is 0 or -1, we will not perform synced dot and just do max_iters
    //in this case, the returned value will be [max_iters, -1] 
    std::tuple<int, double> solve(HADeviceGrid<Tile>& grid, bool verbose, int max_iters, double relative_tolerance, int level_iters, int coarsest_iters, int sync_stride, bool is_pure_neumann);

    int coeff_channel;
    thrust::device_vector<double> d_tmp;
    double* gamma_d;
    double* beta_d;
    double* alpha_d;
    double* gamma_old_d;
    double* fp_d;
	double* mean_d;
    double* count_d;
};
