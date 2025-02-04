#pragma once

#include "PoissonGrid.h"
#include <Eigen/Dense>

//coarsening for Multigrid
//calculate NONLEAF and GHOST cell types from 
void CalcCellTypesFromLeafs(HADeviceGrid<Tile>& grid);


//it calculates divergence on all INTERIOR cells using all velocity nodes including INTERIOR, DIRICHLET and NEUMANN
//NOT propagate u_channel at the beginning (at some times ghost u values are valid)
//will propagate div_channel at the end (leaf-finer fluxes are gathered from ghost)
//exec on LEAF and GHOST tiles
//exec on INTERIOR cells only
void VelocityVolumeDivergenceOnLeafs(HADeviceGrid<Tile>& grid, const int u_channel, const int div_channel);

//it will write to all velocity nodes including INTERIOR, DIRICHLET and NEUMANN
//propagate p_channel at the beginning
//accumulate u_channel at the end (because leaf-finer faces must be gathered from ghost)
//exec on LEAF and GHOST tiles
//exec on all cells
void AddGradientToFaceCenters(HADeviceGrid<Tile>& grid, const int p_channel, const int u_channel);



//take alpha=2 to balance laplacian operator, *2 for updating (alpha /=2), resulting alpha=1
void Restrict(HADeviceGrid<Tile>& grid, const uint8_t fine_channel, const uint8_t coarse_channel, const int coarse_level, const uint8_t launch_types, const T one_over_alpha = 1);
void Prolongate(HADeviceGrid<Tile>& grid, const uint8_t coarse_channel, const uint8_t fine_channel, const int fine_level, const uint8_t launch_types);

void ReCenterLeafVoxels(HADeviceGrid<Tile>& grid, const int channel, double* mean_d, double* count_d);