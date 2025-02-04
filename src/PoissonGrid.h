#pragma once

#include "HAGrid.h"
#include "PoissonTile.h"

using Tile = PoissonTile<float>;
using T = Tile::T;
using Coord = typename Tile::Coord;
using Vec = Tile::VecType;
constexpr T NODATA = FLT_MAX;


template <class T, typename Func3>
__global__ void TernaryOnArrayKernel(T* a, T* b, T* c, Func3 f, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        f(a[idx], b[idx], c[idx]);
    }
}

template<class T, class Func3>
void TernaryOnArray(T* d_a, T* d_b, T* d_c, Func3 f, int n = 1, int block_size = 512) {
    if (n == 1) block_size = 1;

    int numBlocks = (n + block_size - 1) / block_size;


    TernaryOnArrayKernel << <numBlocks, block_size >> > (d_a, d_b, d_c, f, n);
}

__host__ void SpawnGhostTiles(HADeviceGrid<Tile>& grid, bool verbose = true);

template<class ABFunc>
__host__ std::vector<int> RefineLeafsOneStep(HADeviceGrid<Tile>& grid, ABFunc level_target, bool verbose) {
    //must be called after ghost tiles are properly spawned
    //then, refine leafs for one step
    //this may need to be called multiple times to reach the target level
    Assert(grid.mDeviceSyncFlag, "Grid must be synced before refine step");

    std::vector<int> level_refine_cnts(grid.mNumLayers, 0);

    for (int i = grid.mMaxLevel; i >= 0; i--) {
        thrust::host_vector<int> refine_host(grid.hNumTiles[i]);
        thrust::device_vector<int> refine_flg_dev = refine_host;
        //Info("level {} numtiles {} refine_flg_dev size {}", i, grid.hNumTiles[i], refine_flg_dev.size());
        Assert(refine_flg_dev.size() == grid.hNumTiles[i], "refine_flg_dev size {} mismatch num tiles {}", refine_flg_dev.size(), grid.hNumTiles[i]);


        auto refine_flg_dev_ptr = thrust::raw_pointer_cast(refine_flg_dev.data());
        //first, launch on all 
        grid.launchTileFunc(
            [level_target, i, refine_flg_dev_ptr]__device__(HATileAccessor<Tile>&acc, const uint32_t tile_idx, HATileInfo<Tile>&info) {
            auto& tile = info.tile();
            //a leaf tile need to be refined if it doesn't reach level target, or there is a neighbor leaf on +2 level
            //that means, it has a ghost child, whose REFINE_FLAG is set
            //the flags of ghost tiles will be set in the next step

            //case 1: not reach target level
            bool to_refine = false;
            //case 1: if the target level is higher than the current level, refine
            if (level_target(acc, info) > info.mLevel) {
                to_refine = true;
            }

            //case 2: check ghost children
            //there are no ghost children of the max level
            if (i < acc.mMaxLevel) {
                acc.iterateChildCoords(info.mTileCoord,
                    [&]__device__(const Coord & child_ijk) {
                    auto& child_info = acc.tileInfo(i + 1, child_ijk);
                    if (child_info.isGhost()) {
                        auto& child_tile = child_info.tile();
                        if (child_tile.mStatus & REFINE_FLAG) {
                            to_refine = true;
                        }
                    }
                });
            }

            //we will not refine if the maximum capacity of levels is reached
            if (i + 1 >= acc.mNumLayers) to_refine = false;
            tile.setMask(REFINE_FLAG, to_refine);
            refine_flg_dev_ptr[tile_idx] = to_refine;
        },
            i, LEAF, LAUNCH_LEVEL
        );
        refine_host = refine_flg_dev;

        grid.launchTileFunc(
            [=]__device__(HATileAccessor<Tile>&acc, const uint32_t tile_idx, HATileInfo<Tile>&info) {
            auto& tile = info.tile();
            bool refine_flag = false;
            acc.iterateNeighborCoords(info.mTileCoord,
                [&](const Coord& n_ijk) {
                    auto& n_info = acc.tileInfo(i, n_ijk);
                    if (n_info.isLeaf()) {
                        auto& n_tile = n_info.tile();
                        if (n_tile.mStatus & REFINE_FLAG) {
                            refine_flag = true;
                        }
                    }
                });
            tile.setMask(REFINE_FLAG, refine_flag);
        },
            i, GHOST, LAUNCH_LEVEL
        );

        using Acc = HACoordAccessor<Tile>;
        //we will refine all required tiles to the next level
        //thrust::host_vector<int> refine_host = refine_flg_dev;
        auto h_acc = grid.hostAccessor();
        for (int j = 0; j < refine_host.size(); j++) {
            if (refine_host[j]) {
                level_refine_cnts[i]++;

                //this seems strange, but the compress() function will write mHostTileArray based on hash table
                //so we must modify the hash table element
                auto& tmp_info = grid.hTileArrays[i][j];
                auto& info = h_acc.tileInfo(i, tmp_info.mTileCoord);
                //auto& info = grid.mHostLayers[i].tileInfo(tmp_info.mTileCoord);

                info.mType = NONLEAF;
                auto tile = info.getTile(DEVICE);

                for (int ci = 0; ci < Acc::NUMCHILDREN; ci++) {
                    Coord offset = Acc::childIndexToOffset(ci);
					Coord c_ijk = Acc::childCoord(info.mTileCoord, offset);

                    //if (info.mLevel == 2 && info.mTileCoord == Coord(0, 1, 3)) {
                    //    Info("refine tile {} to ci {} offset {} c_ijk {}", info.mTileCoord, ci, offset, c_ijk);
                    //}

                    grid.setTileHost(i + 1, c_ijk, tile.childTile(offset), LEAF);
                }

                //Acc::iterateChildCoords(info.mTileCoord,
                //    [&](const Coord& c_ijk) {
                //        //auto& c_info = grid.mHostLayers[i + 1].tileInfo(c_ijk);
                //        auto& c_info = h_acc.tileInfo(i + 1, c_ijk);
                //        if (c_info.empty()) {
                //            //create a new tile
                //            grid.setTileHost(i + 1, c_ijk, tile.childTile(), LEAF);
                //            //grid.mHostLayers[i + 1].setTile(c_ijk, Tile(), i + 1, LEAF);
                //        }
                //        else {
                //            //already created (for example, may be a ghost tile), set it to leaf
                //            c_info.mType = LEAF;
                //        }
                //    });

            }
        }

    }

    grid.compressHost(verbose);
    grid.syncHostAndDevice();

    return level_refine_cnts;
}

//return number of deleted tiles each layer
template<class ABFunc>
std::vector<int> CoarsenStep(HADeviceGrid<Tile>& grid, ABFunc level_target, bool verbose) {
    //level_target function is only available on leaf tiles
    //COARSEN_FLAG means you can delete its children and make the tile a LEAF
    //DELETE_FLAG means you can delete the tile itself

    //upstroke from fine to coarse, calculate COARSEN flags based on DELETE flags
    for (int level = grid.mNumLayers - 1; level >= 0; level--) {
        //mark DELETE flag for leaf tiles and COARSEN flag for non-leaf tiles
        //we're not launching GHOST here
        //if there is a same-level neighbor NONLEAF tile that can't be coarsen, we can't delete a LEAF
        //therefore, COARSEN flags are calculated prior to DELETE

        //1: calculate COARSEN flags for NONLEAF tiles and unset DELETE flags for convenience
        //if all 8 children are deletable LEAF tiles, we can mark COARSEN flag
        grid.launchTileFunc(
            [=] __device__(HATileAccessor<Tile>&acc, uint32_t tile_idx, HATileInfo<Tile>&info) {
            auto& tile = info.tile();
            bool to_coarsen = true;
            acc.iterateChildCoords(info.mTileCoord, [&](const Coord& c_ijk) {
                auto cinfo = acc.tileInfoCopy(info.mLevel + 1, c_ijk);//c for child
                if (cinfo.isLeaf()) {
                    const auto& ctile = cinfo.tile();
                    if (!(ctile.mStatus & DELETE_FLAG)) {
                        to_coarsen = false;
                    }
                }
                else {
                    //we're only launching NONLEAF tiles so the child here must be NONLEAF
                    //and its DELETE flag is false
                    to_coarsen = false;
                }
                });
            tile.setMask(COARSEN_FLAG, to_coarsen);
            tile.setMask(DELETE_FLAG, false);
        },
            level, NONLEAF, LAUNCH_LEVEL
        );

        //2: calculate DELETE flags for LEAF tiles, and unset COARSEN flags for convenience
        grid.launchTileFunc(
            [=] __device__(HATileAccessor<Tile>&acc, uint32_t tile_idx, HATileInfo<Tile>&info) {
            auto& tile = info.tile();
            bool to_delete = level_target(acc, info) < info.mLevel;

            //if there is a same-level NONLEAF neighbor that cannot be coarsen, cannot delete
            acc.iterateNeighborCoords(info.mTileCoord, [&](const Coord& nb_ijk) {//nb for neighbor block
                auto ninfo = acc.tileInfoCopy(info.mLevel, nb_ijk);
                if (ninfo.mType & NONLEAF) {
                    auto& ntile = ninfo.tile();
                    if (!(ntile.mStatus & COARSEN_FLAG)) {
                        to_delete = false;
                    }
                }
                });

            tile.setMask(COARSEN_FLAG, false);
            tile.setMask(DELETE_FLAG, to_delete);
        },
            level, LEAF, LAUNCH_LEVEL
        );
    }

    //downstroke from coarse to fine, propagate DELETE flags from COARSEN flags
    for (int level = 0; level < grid.mNumLayers; level++) {
        //3: propagate DELETE flags for LEAF and GHOST based on COASREN flags
        //in the upstroke pass, COARSEN flag is calculated on all NONLEAF tiles and unset on all LEAF tiles
        //if a tile is marked as COARSEN, we can delete its children and mark it as LEAF
        //if DELETE flag is set for a tile, we also set the COARSEN to further delete its children
        grid.launchTileFunc(
            [=] __device__(HATileAccessor<Tile>&acc, uint32_t tile_idx, HATileInfo<Tile>&info) {
            auto& tile = info.tile();
            bool to_delete = false;

            auto pb_ijk = acc.parentCoord(info.mTileCoord);
            auto pinfo = acc.tileInfoCopy(info.mLevel - 1, pb_ijk);
            if (!pinfo.empty()) {
                auto& ptile = pinfo.tile();
                if (ptile.mStatus & COARSEN_FLAG) {
                    to_delete = true;
                }
            }

            tile.setMask(DELETE_FLAG, to_delete);
            if (to_delete) {
                tile.setMask(COARSEN_FLAG, true);
            }
        },
            level, LEAF | GHOST, LAUNCH_LEVEL
        );

        //4: additionally update DELETE flags for GHOST tiles
        //if a GHOST tile does not have a neighbor LEAF tile that will continue to exist, we mark it as DELETE
        //(if the GHOST tile's parent is going to be deleted, the delete flag is already propagated to it)
        grid.launchTileFunc(
            [=] __device__(HATileAccessor<Tile>&acc, uint32_t tile_idx, HATileInfo<Tile>&info) {
            auto& tile = info.tile();
            bool to_delete = true;
            acc.iterateNeighborCoords(info.mTileCoord, [&](const Coord& nb_ijk) {
                auto ninfo = acc.tileInfoCopy(info.mLevel, nb_ijk);
                if (ninfo.mType & LEAF) {
                    auto& ntile = ninfo.tile();
                    if (!(ntile.mStatus & DELETE_FLAG)) {
                        to_delete = false;
                    }
                }
                });
            if (to_delete) tile.setMask(DELETE_FLAG, true);
        },
            level, GHOST, LAUNCH_LEVEL
        );
    }

    //delete tiles with DELETE flag and mark existing COARSEN tiles as leaf
    std::vector<int> deleted_tiles(grid.mNumLayers, 0);
    for (int level = 0; level < grid.mNumLayers; level++) {
        thrust::host_vector<int> stat_h(grid.hNumTiles[level]);
        thrust::device_vector<int> stat_d = stat_h;
        auto stat_d_ptr = thrust::raw_pointer_cast(stat_d.data());
        grid.launchTileFunc(
            [=] __device__(HATileAccessor<Tile>&acc, uint32_t tile_idx, HATileInfo<Tile>&info) {
            auto& tile = info.tile();
            stat_d_ptr[tile_idx] = tile.mStatus;
        },
            level, LEAF | NONLEAF | GHOST, LAUNCH_LEVEL
        );
        stat_h = stat_d;

        int level_deleted = 0;
        auto h_acc = grid.hostAccessor();
        for (int i = 0; i < stat_h.size(); i++) {
            auto b_ijk = grid.hTileArrays[level][i].mTileCoord;
            if (stat_h[i] & DELETE_FLAG) {
                grid.removeTileHost(level, b_ijk);
                level_deleted++;
            }
            else if (stat_h[i] & COARSEN_FLAG) {
                //note that compressHost will overwrite tile array with hash table
                //so we need to modify the info in the hash table
                auto& info = h_acc.tileInfo(level, b_ijk);
                info.mType = LEAF;
            }
        }
        deleted_tiles[level] = level_deleted;
    }

    grid.compressHost(verbose);
    grid.syncHostAndDevice();
    return deleted_tiles;
}

//do multiple spawnghost and refine steps until all tiles reach target level
template<class ABFunc>
void IterativeRefine(HADeviceGrid<Tile>& grid, ABFunc level_target, bool verbose = true) {
    while (true) {
        auto refine_cnts = RefineLeafsOneStep(grid, level_target, verbose);
        SpawnGhostTiles(grid, verbose);
        if (verbose) Info("Refine {} tiles on each layer", refine_cnts);
        auto cnt = std::accumulate(refine_cnts.begin(), refine_cnts.end(), 0);
        if (cnt == 0) break;
    }
}

//All these operators will only perform on interior cells

//apply c[i]=f(a[i],b[i]) on all interior voxels
template<class UnaryOP>
void UnaryTransform(HADeviceGrid<Tile>& grid, const int in_channel, const int out_channel, UnaryOP f, const int level, const uint8_t launch_types, const LaunchMode mode, const uint8_t cell_types = INTERIOR) {
    grid.launchVoxelFunc(
        [=]__device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
        auto& tile = info.tile();
        if (tile.type(l_ijk) & cell_types) {
            tile(out_channel, l_ijk) = f(tile(in_channel, l_ijk));
            //if (l_ijk == Coord(4, 5, 4)) {
            //    printf("tile[%d,%d,%d]=%f type %d\n", l_ijk[0], l_ijk[1], l_ijk[2], tile(out_channel, l_ijk), tile.type(l_ijk));
            //}
        }
    },
        level, launch_types, mode
    );
}

//apply c[i]=f(a[i],b[i]) on all interior voxels
template<class BinaryOP>
void BinaryTransform(HADeviceGrid<Tile>& grid, const int in1_channel, const int in2_channel, const int out_channel, BinaryOP f, const int level, const uint8_t launch_types, const LaunchMode mode, const uint8_t cell_types = INTERIOR) {
    grid.launchVoxelFunc(
        [=]__device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
        auto& tile = info.tile();
        if (tile.type(l_ijk) & cell_types) {
            tile(out_channel, l_ijk) = f(tile(in1_channel, l_ijk), tile(in2_channel, l_ijk));
        }
    },
        level, launch_types, mode
    );
}

//apply d[i]=f(a[i],b[i],c[i]) on all interior voxels
template<class BinaryOP>
void TernaryTransform(HADeviceGrid<Tile>& grid, const int in1_channel, const int in2_channel, const int in3_channel, const int out_channel, BinaryOP f, const int level, const uint8_t launch_types, const LaunchMode mode, const uint8_t cell_types = INTERIOR) {
    grid.launchVoxelFunc(
        [=]__device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
        auto& tile = info.tile();
        if (tile.type(l_ijk) & cell_types) {
            tile(out_channel, l_ijk) = f(tile(in1_channel, l_ijk), tile(in2_channel, l_ijk), f(in3_channel, l_ijk));
        }
    },
        level, launch_types, mode
    );
}

template<class OP2>
void ApplyElementWiseFunc2(HADeviceGrid<Tile>& grid, const int chn0, const int chn1, OP2 f, const int level, const uint8_t launch_types, const LaunchMode mode, const uint8_t cell_types = INTERIOR) {
    grid.launchVoxelFunc(
        [=]__device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
        auto& tile = info.tile();
        if (tile.type(l_ijk) & cell_types) {
            f(tile(chn0, l_ijk), tile(chn1, l_ijk));
        }
    },
        level, launch_types, mode
    );
}


template<class OP4>
void ApplyElementWiseFunc4(HADeviceGrid<Tile>& grid, const int chn0, const int chn1, const int chn2, const int chn3, OP4 f, const int level, const uint8_t launch_types, const LaunchMode mode, const uint8_t cell_types = INTERIOR) {
    grid.launchVoxelFunc(
        [=]__device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
        auto& tile = info.tile();
        if (tile.type(l_ijk) & cell_types) {
            f(tile(chn0, l_ijk), tile(chn1, l_ijk), tile(chn2, l_ijk), tile(chn3, l_ijk));
        }
    },
        level, launch_types, mode
    );
}

void CalculateNeighborTiles(HADeviceGrid<Tile>& grid);

void Copy(HADeviceGrid<Tile>& grid, const int in_channel, const int out_channel, const int level, const uint8_t launch_types, const LaunchMode mode, const uint8_t cell_types = INTERIOR);
void Fill(HADeviceGrid<Tile>& grid, const int channel, const Tile::T val, const int level, const uint8_t launch_types, const LaunchMode mode, const uint8_t cell_types = INTERIOR);
//running on all leafs (level=-1) or a specific level
//out[i] += alpha * in[i]
void Axpy(HADeviceGrid<Tile>& grid, const Tile::T alpha, const uint8_t in_channel, const uint8_t out_channel, const int level, const uint8_t launch_types, const LaunchMode mode);

void DotAsync(double* d_result, HADeviceGrid<Tile>& grid, const uint8_t in1_channel, const uint8_t in2_channel, const uint8_t launch_tile_types);
double Dot(HADeviceGrid<Tile>& grid, const uint8_t in1_channel, const uint8_t in2_channel, const uint8_t launch_tile_types);
//double VelocityLinf(HADeviceGrid<Tile>& grid, const uint8_t u_channel, int level, const uint8_t launch_types, LaunchMode mode);
double VelocityLinfSync(HADeviceGrid<Tile>& grid, const int u_channel, const uint8_t launch_tile_types);
std::tuple<double, double> VolumeWeightedSumAndVolume(HADeviceGrid<Tile>& grid, const int order, const int in_channel, int level, const uint8_t launch_types, LaunchMode mode);
double VolumeWeightedNorm(HADeviceGrid<Tile>& grid, const int order, const int in_channel, int level = -1, const uint8_t launch_types = LEAF, LaunchMode mode = LAUNCH_SUBTREE);

void MeanAsync(HADeviceGrid<Tile>& grid, const int in_channel, const uint8_t launch_tile_types, double* d_mean, double* d_count);

void PropagateValuesToGhostTiles(HADeviceGrid<Tile>& grid, const int coarse_channel, const int fine_channel);
//In general, cell values can be propagated and accumulated casually
//however, we have to take caution when propagating and accumulating face values
//because 
void PropagateValues(HADeviceGrid<Tile>& grid, const int coarse_channel, const int fine_channel, const int fine_level, const uint8_t propagated_tile_types, const LaunchMode mode);

//copy values from parents for tiles specified by propagate_tile_types
//for example, GHOST will propagate ghost values from parents
//this is actually prolongation with sum kernel
//we call it propagate following the convention in SPGrid paper
void PropagateToChildren(HADeviceGrid<Tile>& grid, const int coarse_channel, const int fine_channel, const int target_subtree_level, const uint8_t target_tile_types, const LaunchMode mode, const uint8_t cell_types);
//accumulate values from children (fine levels) to parents (coarse levels)
//will launch parent tiles respective to target_subtree_level and target_tile_types
//calculate on cell_types only, and if child cell is not in cell_type, just ignore its value (set to 0)
//coeff is: parent+=child*coeff
//if additive, will add to parent value, otherwise will overwrite parent value
void AccumulateToParents(HADeviceGrid<Tile>& grid, const int fine_channel, const int coarse_channel, const int target_subtree_level, const uint8_t target_tile_types, const LaunchMode mode, const uint8_t cell_types, const Tile::T coeff, bool additive);

void AccumulateToParents128(HADeviceGrid<Tile>& grid, const int fine_channel, const int coarse_channel, const uint8_t fine_tile_types, const Tile::T coeff, bool additive, uint8_t cell_types);

//tricky thing: there are two possibilities: (1) ghost tiles do not contain valid data, (2) ghost-leaf faces contain valid data
//for (1), you need to (a) propagate faces before calculating velocity nodes, (b) use findNodeNeighborLeaf in CalcLeafNodeValuesFromFaceCenters
//(c) use findPlusFaceIntpVoxel in InterpolateFaceValue, which basically tries to find the neighbor leaf cell
//for (2), you need to (a) calculate correct ghost values and not propagate, (b) set weight 
void CalcLeafNodeValuesFromFaceCenters(HADeviceGrid<Tile>& grid, const int u_channel, const int node_u_channel);
void CalcLeafNodeValuesFromCellCenters(HADeviceGrid<Tile>& grid, const int cell_channel, const int node_channel);

__device__ Tile::T InterpolateCellValue(const HATileAccessor<Tile>& acc, const Vec& pos, const int cell_channel, const int node_channel);
__device__ Vec InterpolateFaceValue(const HATileAccessor<Tile>& acc, const Vec& pos, const int u_channel, const int node_u_channel);

__device__ thrust::tuple<uint8_t, uint8_t> FaceNeighborCellTypes(const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const Coord& l_ijk, const int axis);

template<class FuncII>
__hostdev__ void IterateFaceNeighborCellTypes(const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const Coord& l_ijk, const int axis, FuncII f) {
    auto& tile = info.tile();
    uint8_t type0 = tile.type(l_ijk);

    auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
    auto ng_ijk = g_ijk; ng_ijk[axis]--;
    HATileInfo<Tile> ninfo; Coord nl_ijk;
    acc.findVoxel(info.mLevel, ng_ijk, ninfo, nl_ijk);

    if (!ninfo.empty()) {
        if (!ninfo.isLeaf()) {
            for (int offj : {0, 1}) {
                for (int offk : {0, 1}) {
                    Coord child_offset = acc.rotateCoord(axis, Coord(1, offj, offk));
                    Coord nc_ijk = acc.childCoord(ng_ijk, child_offset);
					HATileInfo<Tile> nc_info; Coord ncl_ijk;
					acc.findVoxel(info.mLevel + 1, nc_ijk, nc_info, ncl_ijk);
                    if (!nc_info.empty()) {
                        auto& nctile = nc_info.tile();
						uint8_t type1 = nctile.type(ncl_ijk);
						f(type0, type1);
                    }
                }
            }
        }
        else {
            if (ninfo.isGhost()) {
                //it's coarser
                Coord np_ijk = acc.parentCoord(ng_ijk);
                acc.findVoxel(info.mLevel - 1, np_ijk, ninfo, nl_ijk);

            }
            auto& ntile = ninfo.tile();
			uint8_t type1 = ntile.type(nl_ijk);
			f(type0, type1);
        }
    }
}

void ExtrapolateVelocity(HADeviceGrid<Tile>& grid, const int u_channel, const int num_iters);

int RefineWithValuesOneStep(HADeviceGrid<Tile>& grid, int channel, T threshold, int coarse_level, int fine_level, bool verbose);
int CoarsenWithValueneStep(HADeviceGrid<Tile>& grid, int channel, T threshold, int coarse_level, int fine_level, bool verbose);