#pragma once

#include "FlowMap.h"

//execuate in range [0, N)
template <typename Func>
__global__ void ForEachKernel(Func f, const int N, const int numGroups) {
	int base = blockIdx.x * (blockDim.x * numGroups);

	for (int i = 0; i < numGroups; i++) {
		int idx = base + i * blockDim.x + threadIdx.x;
		if (idx < N) {
			f(idx);
		}
	}

	//int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//if (idx < N) {
	//	f(idx);
	//}
}

template<typename Func>
void LaunchIndexFunc(Func f, const int N, const int blockSize = 512, const int numGroups = 4) {
	if (N == 0) return;
	Assert(blockSize % numGroups == 0);
	int numBlocks = (N + blockSize - 1) / blockSize;
	ForEachKernel << <numBlocks, blockSize / numGroups >> > (f, N, numGroups);
}

class Particle {
public:
	int tile_idx, local_offset, idx_in_voxel;
	uint64_t global_idx;
	double start_time;
	Vec impulse;
	Vec pos;
	Eigen::Matrix3<T> gradm;
	Eigen::Matrix3<T> matT;
};



void GenerateParticlesWithDyeDensity(std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr, const int channel, const T threshold, const int number_particles_per_cell, thrust::device_vector<Particle>& particles_d);
void GenerateParticlesUniformlyWithChannelValueOnLevel(std::shared_ptr<HAHostTileHolder<Tile>> holder_all_ptr, const int level, const int channel, const T threshold, const uint8_t sampled_tile_types, const int scale_ratio, thrust::device_vector<Particle>& particles_d);
//void GenerateParticlesUniformlyOnFinestLevel(std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr, const int scale_ratio, thrust::device_vector<Particle>& particles_d);
void GenerateParticlesUniformlyOnGivenLevel(std::shared_ptr<HAHostTileHolder<Tile>> holder_all_ptr, const int level, const uint8_t sampled_tile_types, const int scale_ratio, thrust::device_vector<Particle>& particles_d);
void GenerateParticlesRandomlyInVoxels(
	std::shared_ptr<HAHostTileHolder<Tile>> holder_all_ptr,
	const int level,
	const uint8_t sampled_tile_types,
	const int number_particles_per_voxel,
	thrust::device_vector<Particle>& particles_d);

void CountParticleNumberInLeafCells(HADeviceGrid<Tile>& grid, const thrust::device_vector<Particle>& particles, const int tmp_channel);
void CalcInterestAreaFlagsWithParticlesOnLeafs(const thrust::device_vector<Particle>& particles, HADeviceGrid<Tile>& grid, int tmp_channel);

void RefineWithParticles(HADeviceGrid<Tile>& grid, const thrust::device_vector<Particle>& particles, const int coarse_levels, const int fine_levels, const int counter_channel, bool verbose);
void CoarsenWithParticles(HADeviceGrid<Tile>& grid, const thrust::device_vector<Particle>& particles, const int coarse_levels, const int fine_levels, const int counter_channel, bool verbose);

double LinfNormOfGradMForbenius(const thrust::host_vector<Particle>& particles);
void EraseInvalidParticles(thrust::device_vector<Particle>& particles);
void MarkParticlesOutsideFluidRegionAsInvalid(thrust::device_vector<Particle>& particles, HADeviceGrid<Tile>& grid);

//need to interpolate velocities to nodes before calling this function 
//void GridToParticleVelocityPIC(HADeviceGrid<Tile>& grid, const int u_channel, const int node_u_channel, thrust::device_vector<Particle>& particles);

void ParticleImpulseToPopulatedGridLeafs(HADeviceGrid<Tile>& grid, const thrust::device_vector<Particle>& particles, const int u_channel, const int tmp_u_channel, const int weight_channel);

__device__ void KernelScatterVelocityMAC2(const HATileAccessor<Tile>& acc, const int u_channel, const int uw_channel, const Vec& pos, const Vec& vec, const Eigen::Matrix3<T>& gradv);

void ParticleImpulseToGridMACIntp(HADeviceGrid<Tile>& grid, const thrust::device_vector<Particle>& particles, const int u_channel, const int uw_channel);


void ResetParticleImpulse(HADeviceGrid<Tile>& grid, const int u_channel, const int node_u_channel, thrust::device_vector<Particle>& particles_d);
//void ResetParticlesGradM(HADeviceGrid<Tile>& grid, const int u_channel, const int node_u_channel, thrust::device_vector<Particle>& particles_d);
//void AdvectParticlesRK4Forward(HADeviceGrid<Tile>& grid, const int u_channel, const int node_u_channel, const double dt, thrust::device_vector<Particle>& particles_d);
//void AdvectParticlesAndSingleStepGradMRK4Forward(HADeviceGrid<Tile>& grid, const int u_channel, const int node_u_channel, const double dt, thrust::device_vector<Particle>& particles_d);
////will remove particles that cannot be interpolated at the given level
void AdvectParticlesAndSingleStepGradMRK4ForwardAtGivenLevel(HADeviceGrid<Tile>& grid, const int level, const int u_channel, const int node_u_channel, const double dt, thrust::device_vector<Particle>& particles_d, const bool erase_invalid = true);

class ParticleRecord {
public:
	//Vec pos;
	Particle* ptr;
};

void HistogramSortParticlesAtGivenLevel(HADeviceGrid<Tile>& grid, const int level, const int counter_channel, thrust::device_vector<Particle>& particles_d, thrust::device_vector<int>& tile_prefix_sum_d, thrust::device_vector<ParticleRecord>& sorted_particles_record_d);

__global__ void OptimizedAdvectParticlesAndSingleStepGradMRK4ForwardAtGivenLevel128Kernel(HATileAccessor<Tile> acc, HATileInfo<Tile>* infos_ptr, const int level, const int u_channel, const int node_u_channel, const double dt, int* tile_prefix_sum_ptr, ParticleRecord* records_ptr, const T eps);
void OptimizedAdvectParticlesAndSingleStepGradMRK4ForwardAtGivenLevel(HADeviceGrid<Tile>& grid, const int level, const int u_channel, const int node_u_channel, 
	const double dt, thrust::device_vector<int>& tile_prefix_sum_d, thrust::device_vector<ParticleRecord>& records_d, const T eps = 1e-4);

void OptimizedP2GTransferAtGivenLevel(HADeviceGrid<Tile>& grid, const int level, const int u_channel, const int uw_channel, thrust::device_vector<int>& tile_prefix_sum_d, thrust::device_vector<ParticleRecord>& records_d);