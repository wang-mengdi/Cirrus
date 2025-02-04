#pragma once

#include "PoissonGrid.h"
#include <Eigen/Dense>


__hostdev__ T QuadraticKernelFast(const T x);
__hostdev__ T QuadraticKernel(const T x);
__hostdev__ T QuadraticKernelDerivativeFast(const T x);
__hostdev__ T QuadraticKernelDerivative(T x);

//__device__ void VelocityJacobian(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk, const Vec& pos, const int node_u_channel, const T h, Eigen::Matrix3<T>& jacobian);
__hostdev__ Vec MatrixTimesVec(const Eigen::Matrix3<T>& A, const Vec& b);

//calculate velocities on leaf nodes and interpolate velocities at face centers on GHOST and NONLEAF tiles
void InterpolateVelocitiesAtAllTiles(HADeviceGrid<Tile>& grid, const int u_channel, const int tmp_u_node_channel);

__device__ void KernelScatterVelocityComponentMAC2(const HATileAccessor<Tile>& acc, const int level, const int axis, const int u_channel, const int uw_channel, const Vec& pos, const Vec& vel, const Eigen::Matrix3<T>& gradu);
__device__ bool KernelIntpVelocityAndJacobianMAC2AtGivenLevel(const HATileAccessor<Tile>& acc, const int level, const Vec& pos, const int u_channel, Vec& vel, Eigen::Matrix3<T>& jacobian);
__device__ bool KernelIntpVelocityAndJacobianMAC2(const HATileAccessor<Tile>& acc, const int fine_level, const int coarse_level, const Vec& pos, const int u_channel, Vec& vel, Eigen::Matrix3<T>& jacobian);

__device__ void VelocityAndJacobian(const HATileAccessor<Tile>& acc, const Vec& pos, const int node_u_channel, Vec& vel, Eigen::Matrix3<T>& jacobian);
__device__ void RK2ForwardPositionAndF(const HATileAccessor<Tile>& acc, const int fine_level, const int coarse_level, const T dt, const int u_channel, const int node_u_channel, Vec& pos, Eigen::Matrix3<T>& F);
__device__ void RK4ForwardPositionAndF(const HATileAccessor<Tile>& acc, const int fine_level, const int coarse_level, const T dt, const int u_channel, const int node_u_channel, Vec& phi, Eigen::Matrix3<T>& F);
//__device__ bool RK4ForwardPositionAndFAtGivenLevel(const HATileAccessor<Tile>& acc, const int level, const T dt, const int u_channel, const int node_u_channel, Vec& phi, Eigen::Matrix3<T>& F, const T eps);
__device__ void RK2ForwardPositionAndT(const HATileAccessor<Tile>& acc, const int fine_level, const int coarse_level, const T dt, const int u_channel, const int node_u_channel, Vec& pos, Eigen::Matrix3<T>& matT);
__device__ void RK4ForwardPositionAndT(const HATileAccessor<Tile>& acc, const int fine_level, const int coarse_level, const T dt, const int u_channel, const int node_u_channel, Vec& phi, Eigen::Matrix3<T>& matT);
__device__ bool RK4ForwardPositionAndTAtGivenLevel(const HATileAccessor<Tile>& acc, const int level, const T dt, const int u_channel, const int node_u_channel, Vec& phi, Eigen::Matrix3<T>& matT);

//void AdvectGridImpulseRK4Forward(HADeviceGrid<Tile>& last_grid, const int last_u_channel, const int last_node_u_channel, const T dt, HADeviceGrid<Tile>& nxt_grid, const int nxt_u_channel);
__device__ void NFMBackMarchPsiAndT(const HATileAccessor<Tile>* accs_d_ptr, const int fine_level, const int coarse_level, const double* time_steps_d_ptr, const int u_channel, const int node_u_channel, const int start_step, const int end_step, Vec& psi, Eigen::Matrix3<T>& matT);
__device__ void NFMBackQueryImpulseAndT(const HATileAccessor<Tile>* accs_d_ptr, const int fine_level, const int coarse_level, const double* time_steps_d_ptr, const int u_channel, const int node_u_channel, const int start_step, const int end_step, Vec& psi, Vec& impulse, Eigen::Matrix3<T>& matT);

void CalculateVorticityMagnitudeOnLeafs(HADeviceGrid<Tile>& grid, const int fine_level, const int coarse_level, const int u_channel, const int tmp_u_node_channel, const int vor_channel);
void CalculateVelocityAndVorticityMagnitudeOnLeafFaceCenters(HADeviceGrid<Tile>& grid, const int fine_level, const int coarse_level, const int u_channel, const int tmp_u_node_channel, const int cv_channel, const int vor_channel);