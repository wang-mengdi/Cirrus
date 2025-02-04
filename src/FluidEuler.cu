#include "FluidEuler.h"
#include "Random.h"

#include <cub/cub.cuh>
#include <cub/block/block_reduce.cuh>
#include <thrust/execution_policy.h>

__device__ Vec NFMErodedAdvectionPoint(const int axis, const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const Coord& l_ijk) {
	auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
	auto ng_ijk = g_ijk; ng_ijk[axis]--;
	HATileInfo<Tile> ninfo; Coord nl_ijk;
	acc.findVoxel(info.mLevel, ng_ijk, ninfo, nl_ijk);

	int effect_level = info.mLevel;
	Coord effect_g_ijk = g_ijk;

	if (!ninfo.empty() && (ninfo.mType & GHOST)) {
		effect_level = info.mLevel - 1;
		effect_g_ijk = acc.parentCoord(g_ijk);
	}
	return acc.faceCenterGlobal(axis, effect_level, effect_g_ijk);
}

size_t SmartResizeParticlesForInsert(thrust::device_vector<Particle>& particles, const size_t insert_num) {
	size_t old_capacity = particles.capacity();
	size_t new_capacity = particles.size() + insert_num;

	size_t diff;
	if (new_capacity > old_capacity) {
		diff = new_capacity - old_capacity;
		size_t alloc_capacity = old_capacity + diff * 2;
		particles.reserve(alloc_capacity);

		//Info("alloc capacity: {} particles capacity: {}", alloc_capacity, particles.capacity());
	}
	else {
		diff = 0;
	}

	return diff;
}

void ClearAllNeumannNeighborFaces(HADeviceGrid<Tile>& grid)
{
	grid.launchVoxelFunc(
		[=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
		for (int axis : {0, 1, 2}) {
			bool to_set = false;
			IterateFaceNeighborCellTypes(acc, info, l_ijk, axis, [&](const uint8_t type0, const uint8_t type1) {
				if ((type0 & NEUMANN) || (type1 & NEUMANN) || ((type0 & DIRICHLET) && (type1 & DIRICHLET))) {
					to_set = true;
				}
				});
			if (to_set) {
				info.tile()(Tile::u_channel + axis, l_ijk) = 0;
			}
		}
	}, -1, LEAF, LAUNCH_SUBTREE
	);
}

void FixIsolatedInteriorCells(HADeviceGrid<Tile>& grid, const int tmp_channel) {
	CalculateNeighborTiles(grid);


	//FullNegativeLaplacian(grid, Tile::x_channel, tmp_channel, true);
	NegativeLaplacianSameLevel128(grid, grid.dAllTiles, grid.dAllTiles.size(), -1, LEAF, Tile::x_channel, tmp_channel, true);
	grid.launchVoxelFuncOnAllTiles(
		[=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
		auto& tile = info.tile();

		//{
		//	auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
		//	if (info.mLevel == 5 && (g_ijk == Coord(119, 123, 137) || g_ijk == Coord(119,123,138))) {
		//		printf("level %d g_ijk %d %d %d tile type %d voxel type %d diag %f\n", info.mLevel, g_ijk[0], g_ijk[1], g_ijk[2], info.mType, tile.type(l_ijk), tile(tmp_channel, l_ijk));
		//	}
		//}

		if (tile(tmp_channel, l_ijk) == 0 && tile.type(l_ijk) == INTERIOR) {
			tile.type(l_ijk) = NEUMANN;
		}
	}, LEAF , 4);
	CalcCellTypesFromLeafs(grid);

}



//must calcualte node velocities first
__device__ Vec RK4ForwardPosition(const HATileAccessor<Tile>& acc, const Vec& pos, const double dt, const int u_channel, const int node_u_channel) {
	double c1 = 1.0 / 6.0 * dt, c2 = 1.0 / 3.0 * dt, c3 = 1.0 / 3.0 * dt, c4 = 1.0 / 6.0 * dt;
	Vec vel1 = InterpolateFaceValue(acc, pos, u_channel, node_u_channel);
	Vec pos1 = pos + vel1 * 0.5 * dt;
	Vec vel2 = InterpolateFaceValue(acc, pos1, u_channel, node_u_channel);
	Vec pos2 = pos + vel2 * 0.5 * dt;
	Vec vel3 = InterpolateFaceValue(acc, pos2, u_channel, node_u_channel);
	Vec pos3 = pos + vel3 * dt;
	Vec vel4 = InterpolateFaceValue(acc, pos3, u_channel, node_u_channel);
	return pos + c1 * vel1 + c2 * vel2 + c3 * vel3 + c4 * vel4;
}

__device__ Vec SemiLagrangianBackwardPosition(const HATileAccessor<Tile>& acc, const Vec& pos, const T dt, const int u_channel, const int node_u_channel) {
	auto v0 = InterpolateFaceValue(acc, pos, u_channel, node_u_channel);
	auto pos1 = pos - 0.5 * dt * v0;
	auto v1 = InterpolateFaceValue(acc, pos1, u_channel, node_u_channel);
	auto pos2 = pos - dt * v1;
	return pos2;
}


void MarkOldParticlesAsInvalid(thrust::device_vector<Particle>& particles, const T current_time, const T particle_life) {
	auto particles_ptr = thrust::raw_pointer_cast(particles.data());
	LaunchIndexFunc([=] __device__(int idx) {
		auto& p = particles_ptr[idx];

		if (current_time - p.start_time > particle_life) {
			p.pos = Vec(NODATA, NODATA, NODATA);
		}

	}, particles.size(), 128);
}

//void AdvectParticlesRK2Forward(HADeviceGrid<Tile>& grid, const int u_channel, const int node_u_channel, const double dt, thrust::device_vector<Particle>& particles_d) {
//	//advect particles
//	auto particles_ptr = thrust::raw_pointer_cast(particles_d.data());
//	auto acc = grid.deviceAccessor();
//	LaunchIndexFunc([=] __device__(int idx) {
//		auto& p = particles_ptr[idx];
//
//		RK2ForwardPositionAndT(acc, dt, u_channel, node_u_channel, p.pos, p.matT);
//
//		//p.pos = phi;
//		//p.pos = RK4ForwardPosition(acc, p.pos, dt, Tile::u_channel, node_u_channel);
//	}, particles_d.size(), 128);
//}
//
//


__global__ void CalculateReseedingNumbersOnLeafTiles128Kernel(const T current_time, HATileAccessor<Tile> acc, HATileInfo<Tile>* tile_infos, FluidParams params, const int tmp_channel, const int num_particles_per_cell, int* reseed_number_per_cell) {
	int bidx = blockIdx.x;
	auto& info = tile_infos[bidx];
	auto& tile = info.tile();

	if (!(info.mType & LEAF)) return;

	//int reseed_threshold = num_particles_per_cell / 2;
	int reseed_threshold = 1;

	for (int i = threadIdx.x; i < Tile::SIZE; i += blockDim.x) {
		auto l_ijk = acc.localOffsetToCoord(i);
		int reseed_num = 0;

		//bool has_neumann_neighbor = false;
		//HATileInfo<Tile> ninfo; Coord nl_ijk;
		//if (tile.type(l_ijk) == INTERIOR) {
		//	acc.iterateSameLevelNeighborVoxels(info, l_ijk, [&]__device__(const HATileInfo<Tile>&_ninfo, const Coord & _nl_ijk, const int axis, const int sgn) {
		//		if (!_ninfo.empty()) {
		//			auto& ntile = _ninfo.tile();
		//			if (ntile.type(_nl_ijk) & NEUMANN) {
		//				has_neumann_neighbor = true;
		//				ninfo = _ninfo;
		//				nl_ijk = _nl_ijk;
		//			}
		//		}
		//	});
		//}

		//bool inside_reseeding_region = false;
		//if (has_neumann_neighbor) {
		//	int boundary_axis, boundary_off;
		//	if (!FluidParams::queryBoundaryDirection(acc, ninfo, nl_ijk, boundary_axis, boundary_off)) {
		//		inside_reseeding_region = true;
		//	}
		//}

		
		if (tile.type(l_ijk) == INTERIOR && params.isInParticleGenerationRegion(current_time, acc, info, l_ijk)) {
		//if (inside_reseeding_region) {
			int num_particles = tile(tmp_channel, l_ijk);
			if (num_particles < reseed_threshold) {
				reseed_num = num_particles_per_cell - num_particles;
				//int total_idx = bidx * Tile::SIZE + i;
			}
		}
		reseed_number_per_cell[bidx * Tile::SIZE + i] = reseed_num;
	}
}

void ReseedParticles(HADeviceGrid<Tile>& grid, const FluidParams& params, const int tmp_channel, const double current_time, const int num_particles_per_cell, thrust::device_vector<Particle>& particles) {
	//CPUTimer timer; timer.start();
	static uint64_t global_particle_counter = 0;

	static RandomGenerator rng;
	CountParticleNumberInLeafCells(grid, particles, tmp_channel);

	//cudaDeviceSynchronize(); timer.stop("count particles"); timer.start();

	//int reseed_threshold = num_particles_per_cell / 2;
	//int reseed_threshold = 

	static thrust::device_vector<int> cell_particle_num_d;
	if (cell_particle_num_d.size() < grid.dAllTiles.size() * Tile::SIZE) {
		cell_particle_num_d.resize(grid.dAllTiles.size() * Tile::SIZE);
	}
	//Assert(cell_particle_num_d.size() == grid.dAllTiles.size() * Tile::SIZE,"");

	thrust::fill(cell_particle_num_d.begin(), cell_particle_num_d.end(), 0);
	auto cell_particle_num_d_ptr = thrust::raw_pointer_cast(cell_particle_num_d.data());
	
	auto info_ptr = thrust::raw_pointer_cast(grid.dAllTiles.data());
	auto acc_d = grid.deviceAccessor();
	CalculateReseedingNumbersOnLeafTiles128Kernel << <grid.dAllTiles.size(), 128 >> > (current_time, acc_d, info_ptr, params, tmp_channel, num_particles_per_cell, cell_particle_num_d_ptr);

	//cudaDeviceSynchronize(); timer.stop("calc reseeding number of each voxel"); timer.start();

	static thrust::host_vector<int> cell_particle_num_h;
	cell_particle_num_h = cell_particle_num_d;
	//Info("cell_particle_num_h: {}", cell_particle_num_h);

	auto acc = grid.hostAccessor();
	thrust::host_vector<Particle> reseed_particles_h;

	for (int idx = 0; idx < cell_particle_num_h.size(); idx++) {
		if (cell_particle_num_h[idx] > 0) {
			int tile_idx = idx / Tile::SIZE;
			int l_idx = idx % Tile::SIZE;
			auto& info = grid.hAllTiles[tile_idx];
			auto l_ijk = acc.localOffsetToCoord(l_idx);

			//sample random particles
			auto bbox = acc.voxelBBox(info, l_ijk);
			auto minPoint = bbox.min();
			auto maxPoint = bbox.max();

			for (int i = 0; i < cell_particle_num_h[idx]; i++) {
				auto x = rng.uniform(minPoint[0], maxPoint[0]);
				auto y = rng.uniform(minPoint[1], maxPoint[1]);
				auto z = rng.uniform(minPoint[2], maxPoint[2]);

				Particle p;
				p.global_idx = global_particle_counter++;
				p.pos = Vec(x, y, z);
				p.impulse = Vec(0., 0., 0.);
				p.matT = Eigen::Matrix3<T>::Identity();
				p.start_time = current_time;
				reseed_particles_h.push_back(p);
			}
		}
	}

	//cudaDeviceSynchronize(); timer.stop(fmt::format("generating host reseeding list {}", reseed_particles_h.size())); timer.start();

	int diff = SmartResizeParticlesForInsert(particles, reseed_particles_h.size());

	//cudaDeviceSynchronize(); timer.stop(fmt::format("resize particles {} more", diff)); timer.start();

	particles.insert(particles.end(), reseed_particles_h.begin(), reseed_particles_h.end());

	//cudaDeviceSynchronize(); timer.stop("insert particles to device"); timer.start();

	// auto holder_ptr = grid.getHostTileHolderForLeafs();
	

	// thrust::host_vector<Particle> reseed_particles_h;

	// auto acc = grid.hostAccessor();
	// holder_ptr->iterateLeafCells([&](const HATileInfo<Tile>& info, const Coord& l_ijk) {
	// 	auto& tile = info.tile();
	// 	Vec pos = acc.cellCenter(info, l_ijk);
	// 	if (tile.type(l_ijk) == INTERIOR && params.isInParticleGenerationRegion(acc, info, l_ijk)) {

	// 		int num_particles = tile(tmp_channel, l_ijk);
	// 		if (num_particles < reseed_threshold) {
	// 			//sample random particles
	// 			auto bbox = acc.voxelBBox(info, l_ijk);
	// 			auto minPoint = bbox.min();
	// 			auto maxPoint = bbox.max();

	// 			for (int i = 0; i < num_particles_per_cell - num_particles; i++) {
	// 				auto x = rng.uniform(minPoint[0], maxPoint[0]);
	// 				auto y = rng.uniform(minPoint[1], maxPoint[1]);
	// 				auto z = rng.uniform(minPoint[2], maxPoint[2]);

	// 				Particle p;
	// 				p.pos = Vec(x, y, z);
	// 				p.impulse = Vec(0., 0., 0.);
	// 				p.matT = Eigen::Matrix3<T>::Identity();
	// 				p.start_time = current_time;
	// 				reseed_particles_h.push_back(p);
	// 			}
	// 		}
	// 	}
	// 	});
	
	//particles.insert(particles.end(), reseed_particles_h.begin(), reseed_particles_h.end());
}


////mark interested area (to refine) is min of a tile is 0 and max of a tile is greater than 0
//__global__ void LockedMarkInterestAreaKernelMinAndMax(HATileAccessor<PoissonTile<T>> acc, HATileInfo<PoissonTile<T>>* infos, const uint8_t tmp_channel, int subtree_level, uint8_t launch_types) {
//	const HATileInfo<PoissonTile<T>>& info = infos[blockIdx.x];
//	Coord l_ijk = Coord(threadIdx.x, threadIdx.y, threadIdx.z);
//
//	if (!(info.subtreeType(subtree_level) & launch_types)) {
//		if (l_ijk == Coord(0, 0, 0)) {
//			auto& tile = info.tile();
//			tile.mIsInterestArea = false;
//		}
//		return;
//	}
//
//	auto& tile = info.tile();
//	T value = tile(tmp_channel, l_ijk);
//
//	typedef cub::BlockReduce<T, Tile::DIM, cub::BLOCK_REDUCE_WARP_REDUCTIONS, Tile::DIM, Tile::DIM> BlockReduce;
//	__shared__ typename BlockReduce::TempStorage temp_storage_min;
//	__shared__ typename BlockReduce::TempStorage temp_storage_max;
//
//	T minValue = BlockReduce(temp_storage_min).Reduce(value, cub::Min());
//	T maxValue = BlockReduce(temp_storage_max).Reduce(value, cub::Max());
//
//	if (l_ijk == Coord(0, 0, 0)) {
//		auto& tile = info.tile();
//		//if (maxValue > 0) {
//		//	tile.mIsInterestArea = true;
//		//}
//		//else {
//		//	tile.mIsInterestArea = false;
//		//}
//		printf("minValue: %f, maxValue: %f\n", minValue, maxValue);
//
//		if (minValue == 0 && maxValue > 0) {
//			tile.mIsInterestArea = true;
//			tile.mIsLockedRefine = true;
//		}
//		else {
//			tile.mIsInterestArea = false;
//			tile.mIsLockedRefine = false;
//		}
//	}
//}

// Kernel to mark the interested area based on min and max values in a tile
__global__ void LockedMarkInterestAreaMinAndMax128Kernel(HATileAccessor<PoissonTile<T>> acc, HATileInfo<PoissonTile<T>>* infos, const uint8_t tmp_channel, int subtree_level, uint8_t launch_types) {
	int bi = blockIdx.x;  // Block index
	int ti = threadIdx.x; // Thread index within the block

	const HATileInfo<PoissonTile<T>>& info = infos[bi];

	// Early exit if the subtree type does not match the launch type
	if (!(info.subtreeType(subtree_level) & launch_types)) {
		if (ti == 0) {
			auto& tile = info.tile();
			tile.mIsInterestArea = false;
			tile.mIsLockedRefine = false;
		}
		return;
	}

	auto& tile = info.tile();
	auto dataAsFloat4 = reinterpret_cast<float4*>(tile.mData[tmp_channel]);
	float4 value = dataAsFloat4[ti];

	// Calculate thread-local min and max
	T thread_min = min(min(value.x, value.y), min(value.z, value.w));
	T thread_max = max(max(value.x, value.y), max(value.z, value.w));

	// Use CUB to perform block-wide reduction
	typedef cub::BlockReduce<T, 128> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage_min;
	__shared__ typename BlockReduce::TempStorage temp_storage_max;

	T block_min = BlockReduce(temp_storage_min).Reduce(thread_min, cub::Min());
	T block_max = BlockReduce(temp_storage_max).Reduce(thread_max, cub::Max());

	// The first thread writes the results to the tile metadata
	if (ti == 0) {
		if (block_min == 0 && block_max > 0) {
			tile.mIsInterestArea = true;
			//tile.mIsLockedRefine = true;
		}
		else {
			tile.mIsInterestArea = false;
			//tile.mIsLockedRefine = false;
		}

		// Debugging output
		//printf("Block %d: minValue = %f, maxValue = %f\n", bi, block_min, block_max);
	}
}

int LockedRefineWithNonBoundaryNeumannCellsOneStep(const T current_time, HADeviceGrid<Tile>& grid, const FluidParams params, const int tmp_channel, bool verbose) {
	int coarse_level = params.mCoarseLevel;
	int fine_level = params.mFineLevel;
	auto levelTarget = [coarse_level, fine_level]__device__(const HATileAccessor<Tile> &acc, const HATileInfo<Tile> &info) ->int {
		auto& tile = info.tile();
		if (tile.mIsInterestArea > 0) return fine_level;
		return coarse_level;
	};
	//mark non-boundary neumann cells
	grid.launchVoxelFuncOnAllTiles(
		[=] __device__(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk) {
		auto& tile = info.tile();
		tile(tmp_channel, l_ijk) = 0;
		if (tile.type(l_ijk) & NEUMANN) {
			int boundary_axis, boundary_off;
			if (params.cellType(current_time, acc, info, l_ijk, boundary_axis, boundary_off) == NEUMANN) {
				if (boundary_axis == -1) {
					tile(tmp_channel, l_ijk) = 1;
				}
			}
			//int boundary_axis, boundary_off;
			//if (!FluidParams::queryBoundaryDirection(acc, info, l_ijk, boundary_axis, boundary_off)) {
			//	tile(tmp_channel, l_ijk) = 1;
			//}
		}
	}, LEAF, 4
	);

	{
		//Info("all {} tiles", grid.dAllTiles.size());
		auto info_ptr = thrust::raw_pointer_cast(grid.dAllTiles.data());
		LockedMarkInterestAreaMinAndMax128Kernel << <grid.dAllTiles.size(), 128 >> > (grid.deviceAccessor(), info_ptr, tmp_channel, -1, LEAF);
	}

	auto refine_cnts = RefineLeafsOneStep(grid, levelTarget, verbose);
	SpawnGhostTiles(grid, verbose);
	if (verbose) Info("Refine {} tiles on each layer", refine_cnts);
	auto cnt = std::accumulate(refine_cnts.begin(), refine_cnts.end(), 0);

	//Info("locked refine with non-boundary neumann cells one step: {}", cnt);

	return cnt;
}