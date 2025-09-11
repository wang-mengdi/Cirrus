#include "FMParticles.h"
#include "Random.h"

#include "PoissonIOFunc.h"

#include <cub/cub.cuh>
#include <thrust/remove.h>
#include "FluidEuler.h"
#include <cub/cub.cuh>
#include <cub/device/device_scan.cuh>


void GenerateParticlesWithDyeDensity(std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr, const int channel, const T threshold, const int number_particles_per_cell, thrust::device_vector<Particle>& particles_d) {
	auto& holder = *holder_ptr;
	thrust::host_vector<Particle> particles_h;

	auto acc = holder.coordAccessor();
	holder.iterateLeafCells(
		[&](HATileInfo<Tile>& info, const Coord& l_ijk) {
			auto& tile = info.tile();
			if (tile(channel, l_ijk) >= threshold) {
				auto bbox = acc.voxelBBox(info, l_ijk);
				auto minPoint = bbox.min();
				auto maxPoint = bbox.max();

				RandomGenerator rng;
				for (int i = 0; i < number_particles_per_cell; i++) {
					auto x = rng.uniform(minPoint[0], maxPoint[0]);
					auto y = rng.uniform(minPoint[1], maxPoint[1]);
					auto z = rng.uniform(minPoint[2], maxPoint[2]);

					Particle p;
					p.pos = Vec(x, y, z);
					p.impulse = Vec(0., 0., 0.);
					p.matT = Eigen::Matrix3<T>::Identity();
					p.start_time = -1;
					particles_h.push_back(p);
				}
			}
		}
	);
	particles_d = particles_h;
}


void GenerateParticlesUniformlyWithChannelValueOnLevel(std::shared_ptr<HAHostTileHolder<Tile>> holder_all_ptr, const int level, const int channel, const T threshold, const uint8_t sampled_tile_types, const int scale_ratio, thrust::device_vector<Particle>& particles_d) {
	auto& holder_all = *holder_all_ptr;
	thrust::host_vector<Particle> particles_h;

	auto acc = holder_all.coordAccessor();
	holder_all.iterateLevelCells(
		level,
		[&](HATileInfo<Tile>& info, const Coord& l_ijk) {
			auto& tile = info.tile();
			if (tile(channel, l_ijk) >= threshold) {
				auto bbox = acc.voxelBBox(info, l_ijk);
				auto p_dx = acc.voxelSize(info) / scale_ratio;
				for (int i = 0; i < scale_ratio; i++) {
					for (int j = 0; j < scale_ratio; j++) {
						for (int k = 0; k < scale_ratio; k++) {
							Vec pos = bbox.min() + Vec(i, j, k) * p_dx;
							Particle p;
							p.pos = pos;
							p.impulse = Vec(0., 0., 0.);
							p.matT = Eigen::Matrix3<T>::Identity();
							particles_h.push_back(p);
						}
					}
				}
			}
		}
	);
	particles_d = particles_h;
}

void GenerateParticlesRandomlyInVoxels(
	std::shared_ptr<HAHostTileHolder<Tile>> holder_all_ptr,
	const int level,
	const uint8_t sampled_tile_types,
	const int number_particles_per_voxel,
	thrust::device_vector<Particle>& particles_d) {

	auto& holder_all = *holder_all_ptr;
	thrust::host_vector<Particle> particles_h;

	auto acc = holder_all.coordAccessor();
	holder_all.iterateLevelCells(
		level,
		[&](HATileInfo<Tile>& info, const Coord& l_ijk) {
			if (info.mType & sampled_tile_types) {

				auto& tile = info.tile();
				auto bbox = acc.voxelBBox(info, l_ijk);
				auto minPoint = bbox.min();
				auto maxPoint = bbox.max();

				RandomGenerator rng; // Assume this is available and provides uniform random values
				for (int i = 0; i < number_particles_per_voxel; i++) {
					// Generate random positions within the voxel
					auto x = rng.uniform(minPoint[0], maxPoint[0]);
					auto y = rng.uniform(minPoint[1], maxPoint[1]);
					auto z = rng.uniform(minPoint[2], maxPoint[2]);

					Particle p;
					p.pos = Vec(x, y, z);
					p.impulse = Vec(0., 0., 0.);
					p.matT = Eigen::Matrix3<T>::Identity();
					particles_h.push_back(p);
				}
			}
		});

	particles_d = particles_h;
}


//void GenerateParticlesUniformlyOnFinestLevel(std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr, const int scale_ratio, thrust::device_vector<Particle>& particles_d) {
//	auto& holder = *holder_ptr;
//	thrust::host_vector<Particle> particles_h;
//
//	auto acc = holder.coordAccessor();
//	holder.iterateLeafCells(
//		[&](HATileInfo<Tile>& info, const Coord& l_ijk) {
//			auto& tile = info.tile();
//			if (info.mLevel == holder.mMaxLevel) {
//				auto bbox = acc.voxelBBox(info, l_ijk);
//				auto p_dx = acc.voxelSize(info) / scale_ratio;
//				for (int i = 0; i < scale_ratio; i++) {
//					for (int j = 0; j < scale_ratio; j++) {
//						for (int k = 0; k < scale_ratio; k++) {
//							Vec pos = bbox.min() + Vec(i, j, k) * p_dx;
//							Particle p;
//							p.pos = pos;
//							p.impulse = Vec(0., 0., 0.);
//							p.matT = Eigen::Matrix3<T>::Identity();
//							particles_h.push_back(p);
//						}
//					}
//				}
//			}
//		}
//	);
//	particles_d = particles_h;
//}

void GenerateParticlesUniformlyOnGivenLevel(std::shared_ptr<HAHostTileHolder<Tile>> holder_all_ptr, const int level, const uint8_t sampled_tile_types, const int scale_ratio, thrust::device_vector<Particle>& particles_d) {
	auto& holder_all = *holder_all_ptr;
	thrust::host_vector<Particle> particles_h;

	auto acc = holder_all.coordAccessor();
	holder_all.iterateLevelCells(
		level,
		[&](HATileInfo<Tile>& info, const Coord& l_ijk) {
			if (info.mType & sampled_tile_types) {

				auto& tile = info.tile();
				auto bbox = acc.voxelBBox(info, l_ijk);
				auto p_dx = acc.voxelSize(info) / scale_ratio;
				for (int i = 0; i < scale_ratio; i++) {
					for (int j = 0; j < scale_ratio; j++) {
						for (int k = 0; k < scale_ratio; k++) {
							Vec pos = bbox.min() + Vec(i, j, k) * p_dx;
							Particle p;
							p.pos = pos;
							p.impulse = Vec(0., 0., 0.);
							p.matT = Eigen::Matrix3<T>::Identity();
							particles_h.push_back(p);
						}
					}
				}
			}
		}
	);
	particles_d = particles_h;
}

__global__ void MarkInterestAreaKernel(HATileAccessor<PoissonTile<T>> acc, HATileInfo<PoissonTile<T>>* infos, const uint8_t tmp_channel, int subtree_level, uint8_t launch_types) {
	const HATileInfo<PoissonTile<T>>& info = infos[blockIdx.x];
	Coord l_ijk = Coord(threadIdx.x, threadIdx.y, threadIdx.z);

	if (!(info.subtreeType(subtree_level) & launch_types)) {
		if (l_ijk == Coord(0, 0, 0)) {
			auto& tile = info.tile();
			tile.mIsInterestArea = false;
		}
		return;
	}

	auto& tile = info.tile();
	T value = tile(tmp_channel, l_ijk);

	typedef cub::BlockReduce<T, Tile::DIM, cub::BLOCK_REDUCE_WARP_REDUCTIONS, Tile::DIM, Tile::DIM> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage_min;
	__shared__ typename BlockReduce::TempStorage temp_storage_max;

	T minValue = BlockReduce(temp_storage_min).Reduce(value, thrust::minimum<T>());
	T maxValue = BlockReduce(temp_storage_max).Reduce(value, thrust::maximum<T>());

	if (l_ijk == Coord(0, 0, 0)) {
		auto& tile = info.tile();
		if (maxValue > 0) {
			tile.mIsInterestArea = true;
		}
		else {
			tile.mIsInterestArea = false;
		}
		//if (minValue == 0 && maxValue > 0) {
		//	tile.mIsInterestArea = true;
		//}
		//else {
		//	tile.mIsInterestArea = false;
		//}
	}
}

void CountParticleNumberInLeafCells(HADeviceGrid<Tile>& grid, const thrust::device_vector<Particle>& particles, const int tmp_channel) {
	grid.launchVoxelFuncOnAllTiles(
		[tmp_channel]__device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
		auto& tile = info.tile();
		tile(tmp_channel, l_ijk) = 0;
	}, LEAF, 4
	);
	//Info("reset done");
	auto particles_ptr = thrust::raw_pointer_cast(particles.data());
	auto acc = grid.deviceAccessor();
	LaunchIndexFunc([=] __device__(int idx) {
		auto& p = particles_ptr[idx];
		HATileInfo<Tile> info; Coord l_ijk; Vec frac;
		acc.findLeafVoxelAndFrac(p.pos, info, l_ijk, frac);
		int local_off = acc.localCoordToOffset(l_ijk);
		if (!info.empty()) {
			auto& tile = info.tile();
			atomicAdd(&tile.mData[tmp_channel][local_off], (T)1);
			//atomicAdd(&tile(tmp_channel, l_ijk), 1);
		}
	}, particles.size());
}

void CalcInterestAreaFlagsWithParticlesOnLeafs(const thrust::device_vector<Particle>& particles, HADeviceGrid<Tile>& grid, int tmp_channel) {
	CountParticleNumberInLeafCells(grid, particles, tmp_channel);

	for (int i = 0; i < grid.mNumLayers; i++) {
		if (grid.hNumTiles[i] == 0) continue;
		auto info_ptr = thrust::raw_pointer_cast(grid.dTileArrays[i].data());
		MarkInterestAreaKernel << <grid.hNumTiles[i], dim3(Tile::DIM, Tile::DIM, Tile::DIM) >> > (grid.deviceAccessor(), info_ptr, tmp_channel, -1, LEAF);
	}
}

void CoarsenWithParticles(HADeviceGrid<Tile>& grid, const thrust::device_vector<Particle>& particles, const int coarse_levels, const int fine_levels, const int counter_channel, bool verbose) {
	auto levelTarget = [fine_levels, coarse_levels]__device__(const HATileAccessor<Tile> &acc, const HATileInfo<Tile> &info) ->int {
		auto& tile = info.tile();
		if (tile.mIsInterestArea || tile.mIsLockedRefine) return fine_levels;
		return coarse_levels;
	};

	while (true) {
		CalcInterestAreaFlagsWithParticlesOnLeafs(particles, grid, counter_channel);
		auto coarsen_cnts = CoarsenStep(grid, levelTarget, verbose);
		//SpawnGhostTiles(grid, verbose);
		if (verbose) Info("Deleted {} tiles on each layer", coarsen_cnts);
		auto cnt = std::accumulate(coarsen_cnts.begin(), coarsen_cnts.end(), 0);
		if (cnt == 0) break;
	}
	SpawnGhostTiles(grid, verbose);
}

void RefineWithParticles(HADeviceGrid<Tile>& grid, const thrust::device_vector<Particle>& particles, const int coarse_levels, const int fine_levels, const int counter_channel, bool verbose) {
	auto levelTarget = [fine_levels, coarse_levels]__device__(const HATileAccessor<Tile> &acc, const HATileInfo<Tile> &info) ->int {
		auto& tile = info.tile();
		if (tile.mIsInterestArea || tile.mIsLockedRefine) return fine_levels;
		return coarse_levels;
	};
	while (true) {
		CalcInterestAreaFlagsWithParticlesOnLeafs(particles, grid, counter_channel);

		//polyscope::init();
		//IOFunc::AddTilesToPolyscopeVolumetricMesh(grid, LEAF, "leafs");
		//IOFunc::AddParticleSystemToPolyscope(particles, "particles");
		//polyscope::show();

		auto refine_cnts = RefineLeafsOneStep(grid, levelTarget, verbose);
		SpawnGhostTiles(grid, verbose);
		if (verbose) Info("Refine {} tiles on each layer", refine_cnts);
		auto cnt = std::accumulate(refine_cnts.begin(), refine_cnts.end(), 0);

		//Info("refine leafs: {}", refine_cnts);
		//polyscope::init();
		//IOFunc::AddTilesToPolyscopeVolumetricMesh(grid, LEAF, "leafs");
		//IOFunc::AddTilesToPolyscopeVolumetricMesh(grid, GHOST, "ghosts");
		//IOFunc::AddParticleSystemToPolyscope(particles, "particles");
		//polyscope::show();

		if (cnt == 0) break;
	}
}

double LinfNormOfGradMForbenius(const thrust::host_vector<Particle>& particles) {
	double ret = 0;
	for(const auto& p : particles) {
		ret = std::max<double>(ret, p.gradm.norm());
	}	
	return ret;
}

//remove if it's (NODATA, NODATA, NODATA)
void EraseInvalidParticles(thrust::device_vector<Particle>& particles) {
	//Warn("not erasing");
	//return;

	//Info("before erasing gradm norm {}", LinfNormOfGradMForbenius(particles));

	particles.erase(
		thrust::remove_if(particles.begin(), particles.end(), []__hostdev__(const Particle & p) { return p.pos == Vec(NODATA, NODATA, NODATA); }),
		particles.end()
	);

	//Info("after erasing gradm norm {}", LinfNormOfGradMForbenius(particles));
}

void MarkParticlesOutsideFluidRegionAsInvalid(thrust::device_vector<Particle>& particles, HADeviceGrid<Tile>& grid) {
	auto particles_ptr = thrust::raw_pointer_cast(particles.data());
	auto acc = grid.deviceAccessor();

	//mark particles in NEUMANN cells as Vec(NODATA, NODATA, NODATA)
	LaunchIndexFunc([=] __device__(int idx) {
		auto& p = particles_ptr[idx];
		bool remove_flag = false;

		HATileInfo<Tile> info; Coord l_ijk; Vec frac;
		acc.findLeafVoxelAndFrac(p.pos, info, l_ijk, frac);
		if (info.empty()) remove_flag = true;
		else {
			auto& tile = info.tile();
			if (!(tile.type(l_ijk)&INTERIOR)) {
				remove_flag = true;
			}
		}
		if (remove_flag) {
			p.pos = Vec(NODATA, NODATA, NODATA);
		}
	}, particles.size());
	//EraseInvalidParticles(particles);

	//thrust::host_vector<Particle> particles_h = particles;

	//thrust::remove_if(particles_h.begin(), particles_h.end(), [](const Particle& p) { return p.pos == Vec(NODATA, NODATA, NODATA); });
	//particles = particles_h;
}



//overwrite if a cell contains particles
//result is saved to u_channel at leaf face centers
//tmp_u_channel is used to host intermediate velocity result (on nodes)
//w_channel is used to host intermediate weight result (also on nodes)
//all the channels may host nodal data
//so it uses 4 temp channels
void ParticleImpulseToPopulatedGridLeafs(HADeviceGrid<Tile>& grid, const thrust::device_vector<Particle>& particles, const int u_channel, const int tmp_u_channel, const int weight_channel) {

	auto particles_ptr = thrust::raw_pointer_cast(particles.data());
	auto acc = grid.deviceAccessor();

	//clear
	grid.launchNodeFunc(
		[=]__device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & r_ijk) {
		auto& tile = info.tile();
		for (int axis : {0, 1, 2}) {
			tile.node(u_channel + axis, r_ijk) = 0;
			tile.node(tmp_u_channel + axis, r_ijk) = 0;
		}
		tile.node(weight_channel, r_ijk) = 0;
	},
		-1, LEAF, LAUNCH_SUBTREE
	);

	auto weight = [=]__device__(const Vec & pos, const Vec & node_pos, const T re) ->T {
		T r = (pos - node_pos).length();
		if (r < re) {
			T a = 1 - r / re;
			return a * a * a;
		}
		else return 0;
	};

	LaunchIndexFunc([=]__device__(int idx) {
		auto& p = particles_ptr[idx];
		Vec pos = p.pos;
		//Vec vel = p.impulse;
		Vec p_m = MatrixTimesVec(p.matT.transpose(), p.impulse);
		HATileInfo<Tile> info; Coord l_ijk; Vec frac;
		acc.findLeafVoxelAndFrac(pos, info, l_ijk, frac);
		if (info.empty()) return;

		//this is an initial implementation of: An Adaptive Sampling Approach to Incompressible Particle-Based Fluid
		T re = acc.voxelSize(info) * sqrt(3.f);
		auto& tile = info.tile();

		//add to its 8 corners
		for (int offi : {0, 1}) {
			for (int offj : {0, 1}) {
				for (int offk : {0, 1}) {
					Coord r_ijk = l_ijk + Coord(offi, offj, offk);
					auto node_pos = acc.cellCorner(info, r_ijk);
					auto w = weight(pos, node_pos, re);
					Vec vel = p_m + MatrixTimesVec(p.gradm, node_pos - pos);

					for (int axis : {0, 1, 2}) {
						atomicAdd(&tile.node(u_channel + axis, r_ijk), w * vel[axis]);
					}
					atomicAdd(&tile.node(weight_channel, r_ijk), w);
				}
			}
		}
	}, particles.size());

	//add across-tile-boundary values
	//result is saved to tmp_u_channel at nodes
	grid.launchNodeFunc(
		[=]__device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & r_ijk) {
		auto& tile = info.tile();
		//iterate over 8 possible neighbors

		Vec vs_sum(tile.node(u_channel, r_ijk), tile.node(u_channel + 1, r_ijk), tile.node(u_channel + 2, r_ijk));
		T w_sum = tile.node(weight_channel, r_ijk);

		Coord nb_ijks[8];
		int n_levels[8];
		int nbs = 0;
		//first one (itself)
		nb_ijks[nbs] = info.mTileCoord;
		n_levels[nbs] = info.mLevel;
		nbs++;

		HATileInfo<Tile> n_info; Coord n_l_ijk;
		for (int offx : {-1, 0}) {
			for (int offy : {-1, 0}) {
				for (int offz : {-1, 0}) {
					Coord off(offx, offy, offz);
					acc.findNodeNeighborLeaf(info, r_ijk, off, n_info, n_l_ijk);

					if (!n_info.empty()) {
						//we should only add the node from one tile once
						bool done = false;
						for (int k = 0; k < nbs; k++) {
							if (nb_ijks[k] == n_info.mTileCoord && n_levels[k] == n_info.mLevel) {
								done = true;
								break;
							}
						}
						if (!done) {
							nb_ijks[nbs] = n_info.mTileCoord;
							n_levels[nbs] = n_info.mLevel;
							nbs++;

							Coord n_r_ijk = n_l_ijk;//node coord of neighbor
							if (offx == -1) n_r_ijk[0]++;
							if (offy == -1) n_r_ijk[1]++;
							if (offz == -1) n_r_ijk[2]++;

							Vec n_vs_sum(n_info.tile().node(u_channel, n_r_ijk), n_info.tile().node(u_channel + 1, n_r_ijk), n_info.tile().node(u_channel + 2, n_r_ijk));
							T n_w_sum = n_info.tile().node(weight_channel, n_r_ijk);

							vs_sum += n_vs_sum;
							w_sum += n_w_sum;

						}

					}

				}
			}
		}



		if (w_sum == 0) vs_sum = Vec(0, 0, 0);
		else vs_sum /= w_sum;

		for (int axis : {0, 1, 2}) {
			tile.node(tmp_u_channel + axis, r_ijk) = vs_sum[axis];
		}
	},
		-1, LEAF, LAUNCH_SUBTREE
	);



	//for each cell center, average the values of its 4 corners
	grid.launchVoxelFunc(
		[=]__device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
		auto& tile = info.tile();
		for (int axis : {0, 1, 2}) {
			T v_sum = 0;
			for (int offj : {0, 1}) {
				for (int offk : {0, 1}) {
					Coord offset = acc.rotateCoord(axis, Coord(0, offj, offk));
					Coord r_ijk = l_ijk + offset;
					v_sum += tile.node(tmp_u_channel + axis, r_ijk);
				}
			}
			tile(u_channel + axis, l_ijk) = v_sum / 4;
		}
	},
		-1, LEAF, LAUNCH_SUBTREE
	);
}




//result is saved to (u_channel+0,1,2) at leaf face centers
//(w_channel+0,1,2) is used to host intermediate weight result (on face centers, 3 axes)
void ParticleImpulseToGridMACIntp(HADeviceGrid<Tile>& grid, const thrust::device_vector<Particle>& particles, const int u_channel, const int uw_channel) {

	auto particles_ptr = thrust::raw_pointer_cast(particles.data());
	auto acc = grid.deviceAccessor();

	//clear
	grid.launchVoxelFuncOnAllTiles(
		[=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
		auto& tile = info.tile();
		for (int axis : {0, 1, 2}) {
			tile(u_channel + axis, l_ijk) = 0;
			tile(uw_channel + axis, l_ijk) = 0;
		}
	}, LEAF | GHOST | NONLEAF, 4
	);
	
	LaunchIndexFunc(
		[=]__device__(int idx) {
		auto& p = particles_ptr[idx];
		Vec pos = p.pos;
		Vec vel = MatrixTimesVec(p.matT.transpose(), p.impulse);

		//if (p.gradm.norm() > 1e5) {
		//	printf("particle %d gradm at pos=%f %f %f: %f %f %f %f %f %f %f %f %f\n", idx, pos[0], pos[1], pos[2], p.gradm(0, 0), p.gradm(0, 1), p.gradm(0, 2), p.gradm(1, 0), p.gradm(1, 1), p.gradm(1, 2), p.gradm(2, 0), p.gradm(2, 1), p.gradm(2, 2));
		//}
		KernelScatterVelocityMAC2(acc, u_channel, uw_channel, pos, vel, p.gradm);


	}, particles.size(), 128);

	grid.launchVoxelFuncOnAllTiles(
		[=] __device__(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk) {
		auto& tile = info.tile();

		for (int axis : {0, 1, 2}) {
			auto w = tile(uw_channel + axis, l_ijk);
			if (w > 0) {
				tile(u_channel + axis, l_ijk) /= w;
			}

			//{
			//	auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
			//	if (axis == 2 && g_ijk == Coord(1, 0, 0)) {
			//		printf("global p2g axis %d g_ijk %d %d %d u %f uw %f\n", axis, g_ijk[0], g_ijk[1], g_ijk[2], tile(u_channel + axis, l_ijk), tile(uw_channel + axis, l_ijk));
			//	}
			//}
			//if (tile(u_channel + axis, l_ijk) > 1e5) {
			//	auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
			//	printf("particle impulse to grid mac intp at %d %d %d %f %f %f uw %f %f %f\n", g_ijk[0], g_ijk[1], g_ijk[2], 
			//		tile(u_channel + 0, l_ijk), tile(u_channel + 1, l_ijk), tile(u_channel + 2, l_ijk), 
			//		tile(uw_channel + 0, l_ijk), tile(uw_channel + 1, l_ijk), tile(uw_channel + 2, l_ijk));
			//}
		}
	}, LEAF, 4
	);

	//for (int axis : {0, 1, 2}) {
	//	AccumulateToParents128(grid, u_channel + axis, u_channel + axis, GHOST, 1., true, INTERIOR | DIRICHLET | NEUMANN);
	//}
	//
}



//must calculate node velocities first
void ResetParticleImpulse(HADeviceGrid<Tile>& grid, const int u_channel, const int node_u_channel, thrust::device_vector<Particle>& particles_d) {
	auto particles_ptr = thrust::raw_pointer_cast(particles_d.data());
	auto acc = grid.deviceAccessor();
	LaunchIndexFunc([=] __device__(int idx) {
		auto& p = particles_ptr[idx];
		p.impulse = InterpolateFaceValue(acc, p.pos, u_channel, node_u_channel);

		p.matT = Eigen::Matrix3<T>::Identity();
	}, particles_d.size());
}


////must calculate node velocities first
//void ResetParticlesGradM(HADeviceGrid<Tile>& grid, const int u_channel, const int node_u_channel, thrust::device_vector<Particle>& particles_d) {
//	auto particles_ptr = thrust::raw_pointer_cast(particles_d.data());
//	auto acc = grid.deviceAccessor();
//	LaunchIndexFunc([=] __device__(int idx) {
//		auto& p = particles_ptr[idx];
//
//		Vec m0; Eigen::Matrix3<T> gradu0;
//		KernelIntpVelocityAndJacobianMAC2(acc, p.pos, u_channel, m0, gradu0);
//
//		//printf("reset particle gradm at pos=%f %f %f m0=%f %f %f\n", p.pos[0], p.pos[1], p.pos[2], m0[0], m0[1], m0[2]);
//
//		p.gradm = gradu0;
//	}, particles_d.size(), 128);
//}
//
//void AdvectParticlesRK4Forward(HADeviceGrid<Tile>& grid, const int u_channel, const int node_u_channel, const double dt, thrust::device_vector<Particle>& particles_d) {
//	//advect particles
//	auto particles_ptr = thrust::raw_pointer_cast(particles_d.data());
//	auto acc = grid.deviceAccessor();
//	LaunchIndexFunc([=] __device__(int idx) {
//		auto& p = particles_ptr[idx];
//		Vec phi = p.pos;
//
//		Eigen::Matrix3<T> T_short = Eigen::Matrix3<T>::Identity();
//
//		//printf("rk4 particle advect phi=%f %f %f\n", phi[0], phi[1], phi[2]);
//
//		//RK4ForwardPositionAndT(acc, dt, u_channel, node_u_channel, p.pos, p.matT);
//		RK4ForwardPositionAndT(acc, dt, u_channel, node_u_channel, p.pos, T_short);
//		p.matT = p.matT * T_short;
//
//		//printf("advected particle pos=%f %f %f\n", p.pos[0], p.pos[1], p.pos[2]);
//
//		//given the assumption that gradm is reinitialized each time step
//		//Eigen::Matrix3<T> T_short = Eigen::Matrix3<T>::Identity();
//		//RK4ForwardPositionAndT(acc, dt, u_channel, node_u_channel, phi, T_short);
//
//		p.gradm = T_short.transpose() * p.gradm * T_short;
//
//
//		//p.pos = phi;
//		//p.pos = RK4ForwardPosition(acc, p.pos, dt, Tile::u_channel, node_u_channel);
//	}, particles_d.size(), 128);
//}
//
//void AdvectParticlesAndSingleStepGradMRK4Forward(HADeviceGrid<Tile>& grid, const int u_channel, const int node_u_channel, const double dt, thrust::device_vector<Particle>& particles_d) {
//	//advect particles
//	auto particles_ptr = thrust::raw_pointer_cast(particles_d.data());
//	auto acc = grid.deviceAccessor();
//	LaunchIndexFunc([=] __device__(int idx) {
//		auto& p = particles_ptr[idx];
//		Vec phi = p.pos;
//
//		//reset gradm one step
//		{
//			Vec m0;
//			KernelIntpVelocityAndJacobianMAC2(acc, p.pos, u_channel, m0, p.gradm);
//		}
//
//		Eigen::Matrix3<T> T_short = Eigen::Matrix3<T>::Identity();
//
//		//printf("rk4 particle advect phi=%f %f %f\n", phi[0], phi[1], phi[2]);
//
//		//RK4ForwardPositionAndT(acc, dt, u_channel, node_u_channel, p.pos, p.matT);
//		RK4ForwardPositionAndT(acc, dt, u_channel, node_u_channel, p.pos, T_short);
//		p.matT = p.matT * T_short;
//
//		//printf("advected particle pos=%f %f %f\n", p.pos[0], p.pos[1], p.pos[2]);
//
//		//given the assumption that gradm is reinitialized each time step
//		//Eigen::Matrix3<T> T_short = Eigen::Matrix3<T>::Identity();
//		//RK4ForwardPositionAndT(acc, dt, u_channel, node_u_channel, phi, T_short);
//
//		p.gradm = T_short.transpose() * p.gradm * T_short;
//
//
//		//p.pos = phi;
//		//p.pos = RK4ForwardPosition(acc, p.pos, dt, Tile::u_channel, node_u_channel);
//	}, particles_d.size(), 512, 4);
//}

void AdvectParticlesAndSingleStepGradMRK4ForwardAtGivenLevel(HADeviceGrid<Tile>& grid, const int level, const int u_channel, const int node_u_channel, const double dt, thrust::device_vector<Particle>& particles_d, const bool erase_invalid) {
	//advect particles
	auto particles_ptr = thrust::raw_pointer_cast(particles_d.data());
	auto acc = grid.deviceAccessor();
	LaunchIndexFunc([=] __device__(int idx) {
		auto& p = particles_ptr[idx];
		Vec phi = p.pos;

		bool valid = true;
		//reset gradm one step
		{
			Vec m0;
			if (!KernelIntpVelocityAndJacobianMAC2AtGivenLevel(acc, level, p.pos, u_channel, m0, p.gradm)) valid = false;
		}
		//return;

		//advect forward T
		{
			Eigen::Matrix3<T> T_short = Eigen::Matrix3<T>::Identity();
			if (RK4ForwardPositionAndTAtGivenLevel(acc, level, dt, u_channel, node_u_channel, p.pos, T_short)) {
				p.matT = p.matT * T_short;
				p.gradm = T_short.transpose() * p.gradm * T_short;
			}
			else {
				valid = false;
			}
		}
		if (!valid) {
			p.pos = Vec(NODATA, NODATA, NODATA);
		}
	}, particles_d.size(), 512, 4);
	Info("AdvectParticlesAndSingleStepGradMRK4ForwardAtGivenLevel after advection {} particles erase_invalid {}", particles_d.size(), erase_invalid);

	if (erase_invalid) EraseInvalidParticles(particles_d);

	Info("AdvectParticlesAndSingleStepGradMRK4ForwardAtGivenLevel after erasing {} particles", particles_d.size());
	//thrust::remove_if(particles_d.begin(), particles_d.end(), []__hostdev__(const Particle& p) { return p.pos == Vec(NODATA, NODATA, NODATA); });
}

//void CountParticleNumberInLeafCellsAndRecordToParticles(HADeviceGrid<Tile>& grid, thrust::device_vector<Particle>& particles, const int tmp_channel) {
//	// Reset the particle count in each cell to zero
//	
//	grid.launchVoxelFuncOnAllTiles(
//		[tmp_channel] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
//		auto& tile = info.tile();
//		tile(tmp_channel, l_ijk) = 0;
//	}, LEAF | NONLEAF | GHOST, 4
//	);
//
//	// Get raw pointers to particle data and grid accessor
//	auto particles_ptr = thrust::raw_pointer_cast(particles.data());
//	auto acc = grid.deviceAccessor();
//
//	// Launch kernel to process each particle
//	LaunchIndexFunc([=] __device__(int idx) {
//		auto& p = particles_ptr[idx];
//		HATileInfo<Tile> info; Coord l_ijk; Vec frac;
//
//		// Find the voxel and fractional position for the particle
//		acc.findLeafVoxelAndFrac(p.pos, info, l_ijk, frac);
//
//		//printf("particle %d at %f %f %f empty %d in cell %d %d %d\n", idx, p.pos[0], p.pos[1], p.pos[2],info.empty(), l_ijk[0], l_ijk[1], l_ijk[2]);
//
//		if (!info.empty()) {
//			auto& tile = info.tile();
//
//
//
//			// Atomically increment the particle count in the cell
//			int local_offset = acc.localCoordToOffset(l_ijk);
//			int count_in_voxel = (int)atomicAdd(&tile.mData[tmp_channel][local_offset], (T)1);
//			//int count_in_voxel = atomicAdd(&tile(tmp_channel, l_ijk), (T)1);
//
//
//			//{
//			//	auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
//			//	printf("particle %d at %f %f %f in cell %d %d %d\n", idx, p.pos[0], p.pos[1], p.pos[2], g_ijk[0], g_ijk[1], g_ijk[2]);
//			//}
//
//			// Record the particle's local index within the cell
//			p.tile_idx = tile.mSerialIdx;
//			p.local_offset = local_offset;
//			p.idx_in_voxel = count_in_voxel;
//
//			//if (info.mTileCoord == Coord(14, 48, 21) && l_ijk == Coord(1, 0, 0)) {
//			//	printf("particle %d at %f %f %f in cell %d %d %d count in voxel %d local offset %d tile serial idx %d\n", idx, p.pos[0], p.pos[1], p.pos[2], l_ijk[0], l_ijk[1], l_ijk[2], p.idx_in_voxel, p.local_offset, p.tile_idx);
//			//}
//		}
//		else {
//			p.tile_idx = -1;
//			p.local_offset = -1;
//			p.idx_in_voxel = -1;
//			p.pos = Vec(NODATA, NODATA, NODATA);
//		}
//	}, particles.size());
//}

void CountParticleNumberAtGivenLevelAndRecordToParticles(HADeviceGrid<Tile>& grid, const int level, const int counter_channel, thrust::device_vector<Particle>& particles) {
	// Reset the particle count in each cell to zero
	grid.launchVoxelFunc(
		[counter_channel] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
		auto& tile = info.tile();
		tile.mData[counter_channel][acc.localCoordToOffset(l_ijk)] = 0;
	}, level, LEAF | GHOST | NONLEAF, LAUNCH_LEVEL);

	// Get raw pointers to particle data and grid accessor
	auto particles_ptr = thrust::raw_pointer_cast(particles.data());
	auto acc = grid.deviceAccessor();

	// Launch kernel to process each particle
	LaunchIndexFunc([=] __device__(int idx) {
		auto& p = particles_ptr[idx];
		Coord g_ijk; Vec frac;
		HATileInfo<Tile> info; Coord l_ijk;;

		// Find the voxel and fractional position for the particle
		acc.worldToVoxelAndFraction(level, p.pos, g_ijk, frac);
		acc.findVoxel(level, g_ijk, info, l_ijk);

		//acc.findLeafVoxelAndFrac(p.pos, info, l_ijk, frac);

		//printf("particle %d at %f %f %f empty %d in cell %d %d %d\n", idx, p.pos[0], p.pos[1], p.pos[2],info.empty(), l_ijk[0], l_ijk[1], l_ijk[2]);

		if (!info.empty()) {
			auto& tile = info.tile();



			// Atomically increment the particle count in the cell
			int local_offset = acc.localCoordToOffset(l_ijk);
			int count_in_voxel = (int)atomicAdd(&tile.mData[counter_channel][local_offset], (T)1);
			//int count_in_voxel = atomicAdd(&tile(tmp_channel, l_ijk), (T)1);


			//{
			//	auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
			//	printf("particle %d at %f %f %f in cell %d %d %d\n", idx, p.pos[0], p.pos[1], p.pos[2], g_ijk[0], g_ijk[1], g_ijk[2]);
			//}

			// Record the particle's local index within the cell
			p.tile_idx = tile.mSerialIdx;
			p.local_offset = local_offset;
			p.idx_in_voxel = count_in_voxel;

			//if (info.mTileCoord == Coord(14, 48, 21) && l_ijk == Coord(1, 0, 0)) {
			//	printf("particle %d at %f %f %f in cell %d %d %d count in voxel %d local offset %d tile serial idx %d\n", idx, p.pos[0], p.pos[1], p.pos[2], l_ijk[0], l_ijk[1], l_ijk[2], p.idx_in_voxel, p.local_offset, p.tile_idx);
			//}
		}
		else {
			p.tile_idx = -1;
			p.local_offset = -1;
			p.idx_in_voxel = -1;
			p.pos = Vec(NODATA, NODATA, NODATA);
		}
	}, particles.size());
}

void RunExclusiveScan(thrust::device_vector<int>& tile_prefix_sum_d) {
	int* d_data = thrust::raw_pointer_cast(tile_prefix_sum_d.data());
	size_t temp_storage_bytes = 0;
	void* d_temp_storage = nullptr;

	// Determine temporary storage size
	cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, d_data, d_data, tile_prefix_sum_d.size());

	// Allocate temporary storage
	cudaMalloc(&d_temp_storage, temp_storage_bytes);

	// Perform exclusive scan
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_data, d_data, tile_prefix_sum_d.size());

	// Free temporary storage
	cudaFree(d_temp_storage);
}


void HistogramSortParticlesAtGivenLevel(HADeviceGrid<Tile>& grid, const int level, const int counter_channel, thrust::device_vector<Particle>& particles_d, thrust::device_vector<int>& tile_prefix_sum_d, thrust::device_vector<ParticleRecord>& sorted_particles_record_d) {
	// set tile indices
	int level_num_tiles = grid.hNumTiles[level];
	auto level_info_ptr = thrust::raw_pointer_cast(grid.dTileArrays[level].data());
	auto acc = grid.deviceAccessor();
	LaunchIndexFunc([=] __device__(int idx) {
		auto& info = level_info_ptr[idx];
		auto& tile = info.tile();
		tile.mSerialIdx = idx;
	}, level_num_tiles, 512, 1);


	//cudaDeviceSynchronize(); CheckCudaError("HistogramSortParticles::set serial idx"); Info("set serial idx");

	//count particles in counter_channel
	//CountParticleNumberInLeafCellsAndRecordToParticles(grid, particles_d, counter_channel);
	CountParticleNumberAtGivenLevelAndRecordToParticles(grid, level, counter_channel, particles_d);

	//cudaDeviceSynchronize(); CheckCudaError("HistogramSortParticles::count particles"); Info("count particles");

	//calculate particle number and prefix sum in a block
	//static thrust::device_vector<int> tile_prefix_sum_d;
	tile_prefix_sum_d.resize(level_num_tiles + 1);

	//cudaDeviceSynchronize(); CheckCudaError("HistogramSortParticles::tile prefix resize"); Info("set prefix size");

	auto tile_prefix_sum_ptr = thrust::raw_pointer_cast(tile_prefix_sum_d.data());
	LaunchIndexFunc([=] __device__(int idx) {
		//printf("idx: %d\n", idx);
		auto& info = level_info_ptr[idx];
		auto& tile = info.tile();
		int total_particles = 0;
		for (int i = 0; i < Tile::SIZE; ++i) {
			//Coord l_ijk = acc.localOffsetToCoord(i);

			//int particles_in_cell = (int)tile(counter_channel, l_ijk);
			//somehow you can't use tile(counter_channel, l_ijk) here
			//otherwise you will get something wrong
			int particles_in_cell = (int)tile.mData[counter_channel][i];
			tile.mData[counter_channel][i] = (T)total_particles;
			total_particles += particles_in_cell;
		}
		tile_prefix_sum_ptr[idx] = total_particles;
		//printf("tile %d has %d particles\n", idx, total_particles);

	}, level_num_tiles, 128, 1);

	//cudaDeviceSynchronize(); CheckCudaError("HistogramSortParticles::calc tile prefix sum"); Info("count prefix tile sum");

	//tile_prefix_sum_d.back() = 0;
	int total_counted_particles = thrust::reduce(tile_prefix_sum_d.begin(), tile_prefix_sum_d.end() - 1);
	//Assert(total_counted_particles == particles_d.size(), "total {} particles however counted {} particles", particles_d.size(), total_counted_particles);

	//cudaDeviceSynchronize(); CheckCudaError("HistogramSortParticles::calc total counted particles"); Info("calc total counted particles");

	//calculate prefix sum of total particles in each tile
	thrust::exclusive_scan(
		tile_prefix_sum_d.begin(),
		tile_prefix_sum_d.end(),
		tile_prefix_sum_d.begin()
	);
	//RunExclusiveScan(tile_prefix_sum_d);
	//Assert(tile_prefix_sum_d.back() == particles_d.size(), "last prefix sum {} should be equal to total particles {}", tile_prefix_sum_d.back(), particles_d.size());

	//Info("tile_prefix_sum_d: {}", tile_prefix_sum_d);
	//Info("total counted {} particles", total_counted_particles);

	//cudaDeviceSynchronize(); CheckCudaError("HistogramSortParticles::exclusive scan"); Info("exclusive scan");

	//put each particle to its position in sorted_particles_record_d
	sorted_particles_record_d.resize(total_counted_particles);
	auto sorted_particles_record_ptr = thrust::raw_pointer_cast(sorted_particles_record_d.data());
	auto particles_ptr = thrust::raw_pointer_cast(particles_d.data());
	LaunchIndexFunc([=] __device__(int p_idx) {
		auto& p = particles_ptr[p_idx];
		int tile_idx = p.tile_idx;
		if (tile_idx == -1) return;

		auto& info = level_info_ptr[tile_idx];
		auto& tile = info.tile();
		//Coord l_ijk = acc.localOffsetToCoord(p.local_offset);

		//somehow you can't use tile(counter_channel,l_ijk) here
		int record_idx = tile_prefix_sum_ptr[p.tile_idx] + (int)tile.mData[counter_channel][p.local_offset] + p.idx_in_voxel;
		if (0 <= record_idx && record_idx < total_counted_particles) {
			auto& record = sorted_particles_record_ptr[record_idx];
			//record.pos = p.pos;
			record.ptr = &particles_ptr[p_idx];

			//printf("particle %d idx in voxel %d write to record %d\n", p_idx, p.idx_in_voxel, record_idx);
			//{
			//	if (tile_idx == 24246 && p.local_offset == 64) {
			//		printf("particle %d idx in voxel %d write to record %d at %d %d %d\n", p_idx, p.idx_in_voxel, record_idx, l_ijk[0], l_ijk[1], l_ijk[2]);
			//	}
			//}
		}
	}, particles_d.size(), 128, 1);

	//cudaDeviceSynchronize(); CheckCudaError("HistogramSortParticles::write back record"); Info("write back record");
}

template<int num_channels, int halo>
class LocalVelocityData {
public:
	//static constexpr int halo = 2;
	//static constexpr int SN = 8 + halo * 2;
	//left: halo
	//right: halo+1
	static constexpr int SN = 8 + halo + halo + 1;
	static constexpr int SIZE = SN * SN * SN;

	T h;
	Vec min_pos;
	T data[num_channels][SN][SN][SN];

	__hostdev__ void setGridSize(const T _h, const Coord& center_b_ijk) {
		h = _h;
		Coord min_ijk(center_b_ijk[0] * 8 - halo, center_b_ijk[1] * 8 - halo, center_b_ijk[2] * 8 - halo);
		min_pos = Vec(min_ijk[0], min_ijk[1], min_ijk[2]) * h;
	}

	__hostdev__ static bool isValidSharedCoord(const Coord& s_ijk) {
		return 0 <= s_ijk[0] && s_ijk[0] < SN && 0 <= s_ijk[1] && s_ijk[1] < SN && 0 <= s_ijk[2] && s_ijk[2] < SN;
	}
	__hostdev__ static int sharedCoordToOffset(const Coord& s_ijk) {
		return s_ijk[0] * SN * SN + s_ijk[1] * SN + s_ijk[2];
	}

	__hostdev__ void worldToVoxelAndFrac(const Vec& pos, Coord& ijk, Vec& frac) const {
		Vec v = (pos - min_pos) / h;
		ijk = Coord(floor(v[0]), floor(v[1]), floor(v[2]));
		frac = v - Vec(ijk[0], ijk[1], ijk[2]);
	}

	__hostdev__ T& operator ()(const int axis, const Coord& s_ijk) {
		return data[axis][s_ijk[0]][s_ijk[1]][s_ijk[2]];
	}
	__hostdev__ const T& operator ()(const int axis, const Coord& s_ijk) const {
		return data[axis][s_ijk[0]][s_ijk[1]][s_ijk[2]];
	}

	__hostdev__ Vec faceCenter(const int axis, const Coord& s_ijk)const {
		Vec center = Vec(s_ijk[0], s_ijk[1], s_ijk[2]) * h + min_pos;
		for (int tt : {0, 1, 2}) {
			if (tt != axis) center[tt] += 0.5 * h;
		}
		return center;
	}
	

	//if b_offset==0,0,0, the range is [halo,halo+8)^3
	//voxel idx in [0,512)
	__hostdev__ void setVoxelData(const HATileAccessor<Tile> &acc, HATileInfo<Tile>& info, const Coord& b_offset, const int u_channel, const int vi) {
		//s for "shared", or this class
		Coord s_base_off = Coord(halo + b_offset[0] * 8, halo + b_offset[1] * 8, halo + b_offset[2] * 8);
		Coord l_ijk = acc.localOffsetToCoord(vi);
		Coord s_ijk = s_base_off + l_ijk;
		if (isValidSharedCoord(s_ijk)) {
			for (int axis : {0, 1, 2}) {
				T val = info.empty() ? NODATA : info.tile()(u_channel + axis, l_ijk);
				//(*this)(axis, s_ijk) = data;
				//the performance is basically the same
				data[axis][s_ijk[0]][s_ijk[1]][s_ijk[2]] = val;
			}
		}
	}

	__hostdev__ void addVoxelData(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& b_offset, const int u_channel, const int uw_channel, const int vi) const {
		if (info.empty()) return;
		auto& tile = info.tile();

		// s for "shared", or this class
		Coord s_base_off = Coord(halo + b_offset[0] * 8, halo + b_offset[1] * 8, halo + b_offset[2] * 8);
		Coord l_ijk = acc.localOffsetToCoord(vi);
		Coord s_ijk = s_base_off + l_ijk;

		if (isValidSharedCoord(s_ijk)) {
			bool is_interior =
				(halo + 1 <= l_ijk[0] && l_ijk[0] < 8 - halo)
				&&
				(halo + 1 <= l_ijk[1] && l_ijk[1] < 8 - halo)
				&&
				(halo + 1 <= l_ijk[2] && l_ijk[2] < 8 - halo);

			for (int axis : {0, 1, 2}) {
				if (is_interior) {
					tile.mData[u_channel + axis][vi] = data[axis][s_ijk[0]][s_ijk[1]][s_ijk[2]];
					tile.mData[uw_channel + axis][vi] = data[axis + 3][s_ijk[0]][s_ijk[1]][s_ijk[2]];
				}
				else {
					atomicAdd(&tile.mData[u_channel + axis][vi], data[axis][s_ijk[0]][s_ijk[1]][s_ijk[2]]);
					atomicAdd(&tile.mData[uw_channel + axis][vi], data[axis + 3][s_ijk[0]][s_ijk[1]][s_ijk[2]]);
				}

				//{
				//	auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
				//	if (axis == 2 && g_ijk == Coord(0, 0, 0)) {
				//		printf("add shared weight %f at s_ijk %d %d %d with b_offset %d %d %d to global %d %d %d\n", data[axis + 3][s_ijk[0]][s_ijk[1]][s_ijk[2]], s_ijk[0], s_ijk[1], s_ijk[2], b_offset[0], b_offset[1], b_offset[2], g_ijk[0], g_ijk[1], g_ijk[2]);
				//	}
				//}

				//atomicAdd(&tile.mData[u_channel + axis][vi], data[axis][s_ijk[0]][s_ijk[1]][s_ijk[2]]);
				//atomicAdd(&tile.mData[uw_channel + axis][vi], data[axis + 3][s_ijk[0]][s_ijk[1]][s_ijk[2]]);
			}
		}
	}


	__hostdev__ bool intpVelocityComponentAndGradientWithKernel(int axis, const Vec& pos, T& u, Vec& grad_u)const {
		u = 0;
		grad_u = Vec(0, 0, 0);

		T one_over_h = 1. / h;

		Vec test_pos = pos - Vec(0.5 * h, 0.5 * h, 0.5 * h);
		for (int ii : {0, 1, 2}) {
			if (ii != axis) test_pos[ii] -= 0.5 * h;
		}
		Coord base_ijk; Vec frac;
		test_pos = (test_pos - min_pos) * one_over_h;
		base_ijk = Coord(floor(test_pos[0]), floor(test_pos[1]), floor(test_pos[2]));
		frac = test_pos - Vec(base_ijk[0], base_ijk[1], base_ijk[2]);

		//for example, if axis==0, then test_pos=pos-Vec(0.5h, h, h)
		//relative to the cell min, facex should be (0,0.5h,0.5h)
		//that means, the relative position of pos to the face center should be frac+(0.5,0.5,0.5)
		frac += Vec(0.5, 0.5, 0.5);

		T wi[3], wj[3], wk[3];
		T dwi[3], dwj[3], dwk[3];
		for (int t = 0; t < 3; t++) {
			wi[t] = QuadraticKernelFast(frac[0] - t);
			wj[t] = QuadraticKernelFast(frac[1] - t);
			wk[t] = QuadraticKernelFast(frac[2] - t);
			dwi[t] = QuadraticKernelDerivativeFast(frac[0] - t) * one_over_h;
			dwj[t] = QuadraticKernelDerivativeFast(frac[1] - t) * one_over_h;
			dwk[t] = QuadraticKernelDerivativeFast(frac[2] - t) * one_over_h;
		}

		bool is_valid = (
			0 <= base_ijk[0] && base_ijk[0] + 2 < SN
			&&
			0 <= base_ijk[1] && base_ijk[1] + 2 < SN
			&&
			0 <= base_ijk[2] && base_ijk[2] + 2 < SN
			);
		if (!is_valid) return false;

		//bool is_valid = true;
		for (int offi = 0; offi < 3; ++offi) {
			for (int offj = 0; offj < 3; ++offj) {
				for (int offk = 0; offk < 3; ++offk) {
					Coord neighbor_ijk = base_ijk + Coord(offi, offj, offk);

					{
						T neighbor_value = (*this)(axis, neighbor_ijk);
						if (neighbor_value == NODATA) is_valid = false;

						T w = wi[offi] * wj[offj] * wk[offk];
						Vec dw = Vec(dwi[offi] * wj[offj] * wk[offk], wi[offi] * dwj[offj] * wk[offk], wi[offi] * wj[offj] * dwk[offk]);

						u += w * neighbor_value;
						grad_u += dw * neighbor_value;

						//printf("shared intp pos %f %f %f axis %d base_ijk %d %d %d off %d %d %d neighbor_ijk %d %d %d w %f dw %f %f %f neighbor_value %f\n", pos[0], pos[1], pos[2],axis, base_ijk[0], base_ijk[1], base_ijk[2], offi, offj, offk, neighbor_ijk[0], neighbor_ijk[1], neighbor_ijk[2], w, dw[0], dw[1], dw[2], neighbor_value);
					}
				}
			}
		}

		return is_valid;
	}

	__hostdev__ bool intpVelocityAndGradientWithKernel(const Vec& pos, Vec& vel, Eigen::Matrix3<T>& jacobian) {
		//vel = Vec(0, 0, 0);
		//jacobian = Eigen::Matrix3<T>::Zero();

		for (int axis : {0, 1, 2}) {
			Vec gradu_i;
			if (!intpVelocityComponentAndGradientWithKernel(axis, pos, vel[axis], gradu_i)) return false;
			for (int t : {0, 1, 2}) {
				jacobian(axis, t) = gradu_i[t];
			}
		}

		//printf("shared intp pos %f %f %f vel %f %f %f\n", pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]);
		return true;
	}

	__hostdev__ bool RK4ForwardPositionAndT(const T dt, const int u_channel, Vec& phi, Eigen::Matrix3<T>& matT) {
		Vec u1; Eigen::Matrix3<T> gradu1;
		if (!intpVelocityAndGradientWithKernel(phi, u1, gradu1)) return false;

		Eigen::Matrix3<T> dTdt1 = -matT * gradu1;
		Vec phi1 = phi + 0.5 * dt * u1;
		Eigen::Matrix3<T> T1 = matT + 0.5 * dt * dTdt1;

		Vec u2; Eigen::Matrix3<T> gradu2;
		if (!intpVelocityAndGradientWithKernel(phi1, u2, gradu2)) return false;

		Eigen::Matrix3<T> dTdt2 = -T1 * gradu2;
		Vec phi2 = phi + 0.5 * dt * u2;
		Eigen::Matrix3<T> T2 = matT + 0.5 * dt * dTdt2;

		Vec u3; Eigen::Matrix3<T> gradu3;
		if (!intpVelocityAndGradientWithKernel(phi2, u3, gradu3)) return false;

		Eigen::Matrix3<T> dTdt3 = -T2 * gradu3;
		Vec phi3 = phi + dt * u3;
		Eigen::Matrix3<T> T3 = matT + dt * dTdt3;

		Vec u4; Eigen::Matrix3<T> gradu4;
		if (!intpVelocityAndGradientWithKernel(phi3, u4, gradu4)) return false;

		Eigen::Matrix3<T> dTdt4 = -T3 * gradu4;
		phi = phi + dt / 6.0 * (u1 + 2 * u2 + 2 * u3 + u4);
		matT = matT + dt / 6.0 * (dTdt1 + 2 * dTdt2 + 2 * dTdt3 + dTdt4);

		return true;
	}

	__device__ void scatterVelocityAndGradientWithKernelDirectAtomicAdd(const Vec& pos, const Vec& vel, const Eigen::Matrix3<T>& jacobian) {
		T one_over_h = 1. / h;

		for (int axis : {0, 1, 2}) {
			// 1. Adjust position for face alignment
			Vec test_pos = pos - Vec(0.5 * h, 0.5 * h, 0.5 * h);
			for (int ii : {0, 1, 2}) {
				if (ii != axis) test_pos[ii] -= 0.5 * h;
			}

			// 2. Compute base voxel index and fractional offset
			Coord base_ijk;
			Vec frac;
			test_pos = (test_pos - min_pos) * one_over_h;
			base_ijk = Coord(floor(test_pos[0]), floor(test_pos[1]), floor(test_pos[2]));
			frac = test_pos - Vec(base_ijk[0], base_ijk[1], base_ijk[2]);

			frac += Vec(0.5, 0.5, 0.5);

			T wi[3], wj[3], wk[3];
			for (int t = 0; t < 3; t++) {
				wi[t] = QuadraticKernelFast(frac[0] - t);
				wj[t] = QuadraticKernelFast(frac[1] - t);
				wk[t] = QuadraticKernelFast(frac[2] - t);
			}

			// 3. Iterate over 3x3x3 neighborhood
			for (int offi = 0; offi < 3; ++offi) {
				for (int offj = 0; offj < 3; ++offj) {
					for (int offk = 0; offk < 3; ++offk) {
						Coord neighbor_ijk = base_ijk + Coord(offi, offj, offk);
						bool is_valid = isValidSharedCoord(neighbor_ijk);

						if (isValidSharedCoord(neighbor_ijk))
						{
							T weight = wi[offi] * wj[offj] * wk[offk];
							Vec fpos = faceCenter(axis, neighbor_ijk);

							//vel + gradu@(fpos-pos)
							T scattered_velocity = vel[axis] + Vec(jacobian(axis, 0), jacobian(axis, 1), jacobian(axis, 2)).dot(fpos - pos);
							// Atomic scatter for velocity and weight
							atomicAdd(&(data[axis][neighbor_ijk[0]][neighbor_ijk[1]][neighbor_ijk[2]]), weight * scattered_velocity);
							atomicAdd(&(data[3 + axis][neighbor_ijk[0]][neighbor_ijk[1]][neighbor_ijk[2]]), weight);
						}

					}
				}
			}
		}
	}

	__device__ void scatterVelocityAndGradientWithKernel(const Vec& pos, const Vec& vel, const Eigen::Matrix3<T>& jacobian) {
		T one_over_h = 1. / h;

		for (int axis : {0, 1, 2}) {
			// 1. Adjust position for face alignment
			Vec test_pos = pos - Vec(0.5 * h, 0.5 * h, 0.5 * h);
			for (int ii : {0, 1, 2}) {
				if (ii != axis) test_pos[ii] -= 0.5 * h;
			}

			// 2. Compute base voxel index and fractional offset
			Coord base_ijk;
			Vec frac;
			test_pos = (test_pos - min_pos) * one_over_h;
			base_ijk = Coord(floor(test_pos[0]), floor(test_pos[1]), floor(test_pos[2]));
			frac = test_pos - Vec(base_ijk[0], base_ijk[1], base_ijk[2]);

			frac += Vec(0.5, 0.5, 0.5);

			T wi[3], wj[3], wk[3];
			for (int t = 0; t < 3; t++) {
				wi[t] = QuadraticKernelFast(frac[0] - t);
				wj[t] = QuadraticKernelFast(frac[1] - t);
				wk[t] = QuadraticKernelFast(frac[2] - t);
			}

			// 计算分界点
			int laneid = threadIdx.x % 32;
			unsigned int full_mask = __activemask();
			int local_offset = sharedCoordToOffset(base_ijk);
			int prev_offset = __shfl_up_sync(full_mask, local_offset, 1);
			bool is_boundary = (laneid == 0) || (prev_offset != local_offset);
			// 构造分界点掩码
			unsigned int mark = __brev(__ballot_sync(full_mask, is_boundary));
			unsigned int chk = (mark << (laneid + 1)) | ((1U << (laneid + 1)) - 1);
			int interval = __clz(chk);  // 从当前线程开始的连续 0 数量

			// calculate maximum value of interval to iter in the warp
			int stride = 1, iter = interval;
			// Warp 内归约操作，逐步合并到最大值
			while (stride < 32) {
				int tmp = __shfl_down_sync(full_mask, iter, stride);
				iter = max(iter, tmp);  // 取当前线程和 stride 距离线程的最大 interval
				stride <<= 1;           // 扩大 stride
			}
			// 广播最大值 iter 到整个 warp 的所有线程
			iter = __shfl_sync(full_mask, iter, 0);

			//if (axis == 2 && blockIdx.x == 0 && threadIdx.x < 32) {
			//	printf("laneid %d local offset %d isboundary %d mark %x interval %d iter %d mark << laneid %x\n", laneid, local_offset, is_boundary, mark, interval, iter, (mark << (laneid)));
			//}

			// 3. Iterate over 3x3x3 neighborhood
			for (int offi = 0; offi < 3; ++offi) {
				for (int offj = 0; offj < 3; ++offj) {
					for (int offk = 0; offk < 3; ++offk) {
						Coord neighbor_ijk = base_ijk + Coord(offi, offj, offk);
						bool is_valid = isValidSharedCoord(neighbor_ijk);

						//if (isValidSharedCoord(neighbor_ijk)) 
						{
							T weight = wi[offi] * wj[offj] * wk[offk];
							Vec fpos = faceCenter(axis, neighbor_ijk);

							//vel + gradu@(fpos-pos)
							T scattered_velocity = vel[axis] + Vec(jacobian(axis, 0), jacobian(axis, 1), jacobian(axis, 2)).dot(fpos - pos);

							{
								T warp_velocity_sum = weight * scattered_velocity;
								T warp_weight_sum = weight;
								int stride = 1;
								while (stride <= iter) {
									T neighbor_velocity = __shfl_down_sync(full_mask, warp_velocity_sum, stride);
									T neighbor_weight = __shfl_down_sync(full_mask, warp_weight_sum, stride);

									if (stride <= interval) {
										warp_velocity_sum += neighbor_velocity;
										warp_weight_sum += neighbor_weight;

										//{
										//	if (axis == 2 && neighbor_ijk == Coord(2, 1, 1)) {
										//		printf("lane %d stride %d gather weight %f\n", laneid, stride, neighbor_weight);
										//	}
										//}

									}

									stride <<= 1;
								}

								if (is_boundary && is_valid) {
									atomicAdd(&(data[axis][neighbor_ijk[0]][neighbor_ijk[1]][neighbor_ijk[2]]), warp_velocity_sum);
									atomicAdd(&(data[3 + axis][neighbor_ijk[0]][neighbor_ijk[1]][neighbor_ijk[2]]), warp_weight_sum);

									//if (axis == 2 && neighbor_ijk == Coord(2, 1, 1)) {
									//	printf("shared scatter axis %d block %d thread %d lane %d pos %f %f %f fpos %f %f %f velocity %f weight %f vel sum %f weight sum %f base_ijk %d %d %d to %d %d %d\n", 
									//		axis,blockIdx.x, threadIdx.x, laneid, 
									//		pos[0], pos[1], pos[2], fpos[0], fpos[1], fpos[2], scattered_velocity, weight, warp_velocity_sum, warp_weight_sum, base_ijk[0], base_ijk[1], base_ijk[2], neighbor_ijk[0], neighbor_ijk[1], neighbor_ijk[2]);
									//}
								}
							}





							// Atomic scatter for velocity and weight
							//atomicAdd(&(data[axis][neighbor_ijk[0]][neighbor_ijk[1]][neighbor_ijk[2]]), weight * scattered_velocity);
							//atomicAdd(&(data[3 + axis][neighbor_ijk[0]][neighbor_ijk[1]][neighbor_ijk[2]]), weight);

							//if (axis == 2 && (fpos - Vec(0.003906, 0.003906, 0.070312)).length() < 1e-5) {
							//if(axis==2&&neighbor_ijk==Coord(1,1,10)){
							//if(axis==2&&(pos-Vec(0.001058, 0.006524, 0.062257)).length()<1e-4){
							//	printf("shared scatter axis %d pos %f %f %f fpos %f %f %f velocity %f weight %f base_ijk %d %d %d to %d %d %d\n", axis, pos[0], pos[1], pos[2], fpos[0], fpos[1], fpos[2], scattered_velocity, weight, base_ijk[0], base_ijk[1], base_ijk[2], neighbor_ijk[0], neighbor_ijk[1], neighbor_ijk[2]);
							//}
						}
					}
				}
			}
		}
	}

	__device__ void scatterVelocityAndGradientWithKernelCube4(const Vec& pos, const Vec& vel, const Eigen::Matrix3<T>& jacobian) {
		T one_over_h = 1. / h;
		// 1. Adjust position for face alignment
		Vec test_pos = pos - Vec(h, h, h);

		// 2. Compute base voxel index and fractional offset
		Coord base_ijk;
		Vec frac0;
		test_pos = (test_pos - min_pos) * one_over_h;
		base_ijk = Coord(floor(test_pos[0]), floor(test_pos[1]), floor(test_pos[2]));
		frac0 = test_pos - Vec(base_ijk[0], base_ijk[1], base_ijk[2]);

		for (int axis : {0, 1, 2}) {
			Vec frac = frac0;
			for (int tt : {0, 1, 2}) {
				if (tt == axis) frac[tt] += 1;
				else frac[tt] += 0.5;
			}

			T wi[4], wj[4], wk[4];
			for (int t = 0; t < 4; t++) {
				wi[t] = QuadraticKernel(frac[0] - t);
				wj[t] = QuadraticKernel(frac[1] - t);
				wk[t] = QuadraticKernel(frac[2] - t);
			}

			// 计算分界点
			int laneid = threadIdx.x % 32;
			unsigned int full_mask = __activemask();
			int local_offset = sharedCoordToOffset(base_ijk);
			int prev_offset = __shfl_up_sync(full_mask, local_offset, 1);
			bool is_boundary = (laneid == 0) || (prev_offset != local_offset);
			// 构造分界点掩码
			unsigned int mark = __brev(__ballot_sync(full_mask, is_boundary));
			unsigned int chk = (mark << (laneid + 1)) | ((1U << (laneid + 1)) - 1);
			int interval = __clz(chk);  // 从当前线程开始的连续 0 数量

			// calculate maximum value of interval to iter in the warp
			int stride = 1, iter = interval;
			// Warp 内归约操作，逐步合并到最大值
			while (stride < 32) {
				int tmp = __shfl_down_sync(full_mask, iter, stride);
				iter = max(iter, tmp);  // 取当前线程和 stride 距离线程的最大 interval
				stride <<= 1;           // 扩大 stride
			}
			// 广播最大值 iter 到整个 warp 的所有线程
			iter = __shfl_sync(full_mask, iter, 0);


			// 3. Iterate over 3x3x3 neighborhood
			for (int offi = 0; offi < 4; ++offi) {
				for (int offj = 0; offj < 4; ++offj) {
					for (int offk = 0; offk < 4; ++offk) {
						Coord neighbor_ijk = base_ijk + Coord(offi, offj, offk);
						bool is_valid = isValidSharedCoord(neighbor_ijk);

						//if (isValidSharedCoord(neighbor_ijk)) 
						{
							T weight = wi[offi] * wj[offj] * wk[offk];
							Vec fpos = faceCenter(axis, neighbor_ijk);

							//vel + gradu@(fpos-pos)
							T scattered_velocity = vel[axis] + Vec(jacobian(axis, 0), jacobian(axis, 1), jacobian(axis, 2)).dot(fpos - pos);

							{
								T warp_velocity_sum = weight * scattered_velocity;
								T warp_weight_sum = weight;
								int stride = 1;
								while (stride <= iter) {
									T neighbor_velocity = __shfl_down_sync(full_mask, warp_velocity_sum, stride);
									T neighbor_weight = __shfl_down_sync(full_mask, warp_weight_sum, stride);

									if (stride <= interval) {
										warp_velocity_sum += neighbor_velocity;
										warp_weight_sum += neighbor_weight;
									}

									stride <<= 1;
								}

								if (is_boundary && is_valid) {
									atomicAdd(&(data[axis][neighbor_ijk[0]][neighbor_ijk[1]][neighbor_ijk[2]]), warp_velocity_sum);
									atomicAdd(&(data[3 + axis][neighbor_ijk[0]][neighbor_ijk[1]][neighbor_ijk[2]]), warp_weight_sum);

									//if (axis == 2 && neighbor_ijk == Coord(2, 1, 1)) 
									//{
									//	printf("shared scatter axis %d block %d thread %d lane %d pos %f %f %f fpos %f %f %f velocity %f weight %f vel sum %f weight sum %f base_ijk %d %d %d to %d %d %d\n", 
									//		axis,blockIdx.x, threadIdx.x, laneid, 
									//		pos[0], pos[1], pos[2], fpos[0], fpos[1], fpos[2], scattered_velocity, weight, warp_velocity_sum, warp_weight_sum, base_ijk[0], base_ijk[1], base_ijk[2], neighbor_ijk[0], neighbor_ijk[1], neighbor_ijk[2]);
									//}
								}
							}
						}
					}
				}
			}
		}
	}
};

//each block processes particles in a tile
//blockdim is 128 so one block will iterate over all particles with 128 stride
__global__ void OptimizedAdvectParticlesAndSingleStepGradMRK4ForwardAtGivenLevel128Kernel(HATileAccessor<Tile> acc, HATileInfo<Tile>* infos_ptr, const int level, const int u_channel, const int node_u_channel, const double dt, int* tile_prefix_sum_ptr, ParticleRecord* records_ptr, const T eps) {
	//__shared__ float shared[6][14][14][14];
	__shared__ LocalVelocityData<3, 2> shared_data;

	int tile_idx = blockIdx.x;
	int begin_idx = tile_prefix_sum_ptr[tile_idx];
	int end_idx = tile_prefix_sum_ptr[tile_idx + 1];
	int num_groups = (end_idx - begin_idx + 128 - 1) / 128;

	auto& info = infos_ptr[tile_idx];
	if (info.mLevel != level) {
		//set particles as invalid
		for (int iter = 0; iter < num_groups; iter++) {
			int p_idx = begin_idx + 128 * iter + threadIdx.x;
			if (p_idx < end_idx) {
				auto& p = *records_ptr[p_idx].ptr;
				p.pos = Vec(NODATA, NODATA, NODATA);
			}
		}
		return;
	}

	//if (tile_idx != 24246) return;

	if (threadIdx.x == 0) {
		shared_data.setGridSize(acc.voxelSize(level), info.mTileCoord);
	//	printf("numgroups: %d begin: %d end: %d tilecoord: %d %d %d\n", num_groups,begin_idx, end_idx,  info.mTileCoord[0], info.mTileCoord[1], info.mTileCoord[2]);
	}

	for (int offbi : {-1, 0, 1}) {
		for (int offbj : {-1, 0, 1}) {
			for (int offbk : {-1, 0, 1}) {
				Coord b_offset(offbi, offbj, offbk);
				Coord nb_ijk = info.mTileCoord + b_offset;
				auto n_info = acc.tileInfo(level, nb_ijk);

				for (int i = 0; i < 4; i++) {
					shared_data.setVoxelData(acc, n_info, b_offset, u_channel, i * 128 + threadIdx.x);
				}
			}
		}
	}
	
	__syncthreads();


	for (int iter = 0; iter < num_groups; iter++) {
		int p_idx = begin_idx + 128 * iter + threadIdx.x;
		if (p_idx < end_idx) {

			auto& rc = records_ptr[p_idx];
			auto& p = *rc.ptr;
			Vec phi = p.pos;


			bool valid = true;
			Eigen::Matrix3<T> gradm;
			//reset gradm one step
			{
				Vec m0;
				if (!shared_data.intpVelocityAndGradientWithKernel(p.pos, m0, gradm)) {
					valid = false;
				}

				//if (!KernelIntpVelocityAndJacobianMAC2AtGivenLevel(acc, level, p.pos, u_channel, m0, p.gradm, eps)) valid = false;
			}
			//continue;

			//continue;
			//printf("blockIdx.x %d threadIdx.x %d p_idx %d pos %f %f %f gradm %f %f %f\n", blockIdx.x, threadIdx.x, p_idx, p.pos[0], p.pos[1], p.pos[2], p.gradm(0, 0), p.gradm(0, 1), p.gradm(0, 2));

			//advect forward T
			{
				Eigen::Matrix3<T> T_short = Eigen::Matrix3<T>::Identity();
				//if (RK4ForwardPositionAndTAtGivenLevel(acc, level, dt, u_channel, node_u_channel, p.pos, T_short, eps)) {
				//	p.matT = p.matT * T_short;
				//	p.gradm = T_short.transpose() * p.gradm * T_short;
				//}
				//else {
				//	valid = false;
				//}

				if (shared_data.RK4ForwardPositionAndT(dt, u_channel, p.pos, T_short)) {
					p.matT = p.matT * T_short;
					p.gradm = T_short.transpose() * gradm * T_short;

					//{
					//	auto pos = p.pos;
					//	printf("particle %d gradm at pos=%f %f %f: %f %f %f %f %f %f %f %f %f\n", p_idx, pos[0], pos[1], pos[2], p.gradm(0, 0), p.gradm(0, 1), p.gradm(0, 2), p.gradm(1, 0), p.gradm(1, 1), p.gradm(1, 2), p.gradm(2, 0), p.gradm(2, 1), p.gradm(2, 2));
					//}
				}
				else {
					valid = false;
				}
			}

			//printf("record idx %d valid %d\n", p_idx, valid);

			if (!valid) {
				p.pos = Vec(NODATA, NODATA, NODATA);
			}
		}
	}
}

void OptimizedAdvectParticlesAndSingleStepGradMRK4ForwardAtGivenLevel(HADeviceGrid<Tile>& grid, const int level, const int u_channel, const int node_u_channel, 
	const double dt, thrust::device_vector<int>& tile_prefix_sum_d, thrust::device_vector<ParticleRecord>& records_d, const T eps) {
	//advect particles
	int level_num_tiles = grid.hNumTiles[level];
	auto level_infos_ptr = thrust::raw_pointer_cast(grid.dTileArrays[level].data());
	auto tile_prefix_sum_ptr = thrust::raw_pointer_cast(tile_prefix_sum_d.data());
	auto records_ptr = thrust::raw_pointer_cast(records_d.data());
	auto acc = grid.deviceAccessor();


	OptimizedAdvectParticlesAndSingleStepGradMRK4ForwardAtGivenLevel128Kernel << <level_num_tiles, 128 >> > (acc, level_infos_ptr, level, u_channel, node_u_channel, dt, tile_prefix_sum_ptr, records_ptr, eps);
}

__global__ void P2GTransferAtGivenLevel128Kernel(HATileAccessor<Tile> acc, HATileInfo<Tile>* infos_ptr, const int level, const int u_channel, const int uw_channel,
	int* tile_prefix_sum_ptr, ParticleRecord* records_ptr) {
	__shared__ LocalVelocityData<6, 1> shared_data;

	int tile_idx = blockIdx.x;
	int begin_idx = tile_prefix_sum_ptr[tile_idx];
	int end_idx = tile_prefix_sum_ptr[tile_idx + 1];
	

	auto& info = infos_ptr[tile_idx];
	if (info.empty() || info.mLevel != level) {
		return;
	}

	if (threadIdx.x == 0) {
		shared_data.setGridSize(acc.voxelSize(level), info.mTileCoord);
	}

	//set shared_data to zero
	{
		int num_shared_groups = (shared_data.SIZE + 128 - 1) / 128;
		for (int iter = 0; iter < num_shared_groups; iter++) {
			int svi = 128 * iter + threadIdx.x;
			if (0 <= svi && svi < shared_data.SIZE) {
				for (int i = 0; i < 6; i++) {
					T* ptr = &shared_data.data[i][0][0][0];
					ptr[svi] = 0;
				}
			}
		}
	}
	__syncthreads();


	int num_groups = (end_idx - begin_idx + 128 - 1) / 128;
	for (int iter = 0; iter < num_groups; iter++) {
		int p_idx = begin_idx + 128 * iter + threadIdx.x;
		if (p_idx < end_idx) {
			auto& rc = records_ptr[p_idx];
			auto& p = *rc.ptr;

			// Skip invalid particles
			if (p.pos[0] == NODATA) {
				continue;
			}

			// Scatter velocity and gradient
			Vec vel = MatrixTimesVec(p.matT.transpose(), p.impulse);
			//shared_data.scatterVelocityAndGradientWithKernelCube4(p.pos, vel, p.gradm);
			shared_data.scatterVelocityAndGradientWithKernel(p.pos, vel, p.gradm);
			//shared_data.scatterVelocityAndGradientWithKernelDirectAtomicAdd(p.pos, vel, p.gradm);
		}
	}

	__syncthreads();


	for (int offbi : {-1, 0, 1}) {
		for (int offbj : {-1, 0, 1}) {
			for (int offbk : {-1, 0, 1}) {
				Coord b_offset(offbi, offbj, offbk);
				Coord nb_ijk = info.mTileCoord + b_offset;
				auto n_info = acc.tileInfo(level, nb_ijk);
				if (!n_info.empty()) {
					//if (nb_ijk == Coord(0, 0, 0)) {
					//	printf("shared add from tile %d %d %d to %d %d %d b_offset %d %d %d\n", info.mTileCoord[0], info.mTileCoord[1], info.mTileCoord[2], nb_ijk[0], nb_ijk[1], nb_ijk[2], b_offset[0], b_offset[1], b_offset[2]);
					//}
					for (int i = 0; i < 4; i++) {
						shared_data.addVoxelData(acc, n_info, b_offset, u_channel, uw_channel, i * 128 + threadIdx.x);
					}
				}
			}
		}
	}

}

void OptimizedP2GTransferAtGivenLevel(HADeviceGrid<Tile>& grid, const int level, const int u_channel, const int uw_channel,
	thrust::device_vector<int>& tile_prefix_sum_d, thrust::device_vector<ParticleRecord>& records_d) {
	int level_num_tiles = grid.hNumTiles[level];
	auto level_infos_ptr = thrust::raw_pointer_cast(grid.dTileArrays[level].data());
	auto tile_prefix_sum_ptr = thrust::raw_pointer_cast(tile_prefix_sum_d.data());
	auto records_ptr = thrust::raw_pointer_cast(records_d.data());
	auto acc = grid.deviceAccessor();

	//clear
	grid.launchVoxelFuncOnAllTiles(
		[=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
		auto& tile = info.tile();
		for (int axis : {0, 1, 2}) {
			tile(u_channel + axis, l_ijk) = 0;
			tile(uw_channel + axis, l_ijk) = 0;
		}
	}, LEAF | GHOST | NONLEAF, 4
	);

	P2GTransferAtGivenLevel128Kernel << <level_num_tiles, 128 >> > (acc, level_infos_ptr, level, u_channel, uw_channel, tile_prefix_sum_ptr, records_ptr);

	grid.launchVoxelFuncOnAllTiles(
		[=] __device__(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk) {
		auto& tile = info.tile();

		for (int axis : {0, 1, 2}) {
			auto w = tile(uw_channel + axis, l_ijk);
			if (w > 0) {
				tile(u_channel + axis, l_ijk) /= w;
			}

			//{
			//	auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
			//	if (axis == 2 && g_ijk == Coord(1, 0, 0)) {
			//		printf("optimized p2g axis %d g_ijk %d %d %d u %f uw %f\n", axis, g_ijk[0], g_ijk[1], g_ijk[2], tile(u_channel + axis, l_ijk), tile(uw_channel + axis, l_ijk));
			//	}
			//}
		}
	}, LEAF, 4
	);
}
