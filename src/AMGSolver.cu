#include "AMGSolver.h"
#include "PoissonGrid.h"
#include "CPUTimer.h"
#include "PoissonSolver.h"
#include "GMGSolver.h"
//#include "PoissonSolverOptimized.h"

#include "PoissonIOFunc.h"
#include <polyscope/polyscope.h>

__hostdev__ __forceinline__ int3 localIdxToInt3(int _idx)
{
    return { _idx / 64  , (_idx / 8) % 8, _idx % 8 };
}

//it's the same as the one in PoissonSolverOptimized.cu
__forceinline__ __device__ T NegativeLaplacianCoeff(T h, uint8_t ttype0, uint8_t ttype1, uint8_t ctype0, const uint8_t ctype1) {
    ////tile types check
    ////we only calculate LEAF-GHOST and LEAF-LEAF terms
    ////and we set delta_h correspondingly

    int both_leafs = int((ttype0 & LEAF) && (ttype1 & LEAF));
    int one_leaf_one_ghost = int((ttype0 & LEAF && ttype1 & GHOST) || (ttype0 & GHOST && ttype1 & LEAF));
    //dh=h if 2 LEAFS
    //dh=1.5h if 1 LEAF 1 GHOST
    //ignore for all others
    //1.5f is important because FP64 operations will slow down the kernel

    T one_over_delta_h = (1 / h) * both_leafs + (1 / (1.5f * h)) * one_leaf_one_ghost;
    //T one_over_delta_h = ((1 / h) * (both_leafs)+(1.5f / h) * (1 - both_leafs)) * (1 - both_ghosts);
    //ignore if one of them is NEUMANN
    int has_neumann = int(ctype0 & NEUMANN || ctype1 & NEUMANN);
    //T coeff = one_over_delta_h / h * (1 - has_neumann);
    return has_neumann ? 0 : one_over_delta_h / h;
}

//coeff_channel+0,1,2: 3 off-diagonal coefficients
//coeff_channel+3: diagonal coefficient
//this function will consider all NONLEAF tiles as LEAFs, for the purpose of single-level smoothing in AMG
void CalculateAMGCoefficients(HADeviceGrid<Tile>& grid, const int coeff_channel, const uint8_t launch_tile_types) {
    grid.launchVoxelFuncOnAllTiles(
        [=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
        auto h = acc.voxelSize(info);
        Tile& tile = info.tile();

        //     if (!tile.isInterior(l_ijk)) {
                 ////only calculate coefficients for INTERIOR voxels
        //         for (int axis = 0; axis < 3; ++axis) {
        //             tile(coeff_channel + axis, l_ijk) = 0;
        //         }
        //         tile(coeff_channel + 3, l_ijk) = 0;
        //         return;
        //     }

        //uint8_t ttype0 = info.mType == NONLEAF ? LEAF : info.mType;
        uint8_t ttype0 = info.mType;
        uint8_t ctype0 = tile.type(l_ijk);

        Tile::T diag_coeff = 0;
        Tile::T off_diag_coeff[3] = { 0, 0, 0 };

        //iterate neighbors
        acc.iterateSameLevelNeighborVoxels(info, l_ijk,
            [&]__device__(const HATileInfo<Tile>&ninfo, const Coord & nl_ijk, const int axis, const int sgn) {

            uint8_t ttype1 = ttype0;
            uint8_t ctype1;
            T coeff = 0;

            if (ninfo.empty()) {
                ttype1 = ttype0;
                ctype1 = DIRICHLET;

            }
            else {
                auto& ntile = ninfo.tile();
                //ttype1 = ninfo.mType == NONLEAF ? LEAF : ninfo.mType;
                ttype1 = ninfo.mType;
                ctype1 = ntile.type(nl_ijk);
            }

            coeff = NegativeLaplacianCoeff(h, ttype0, ttype1, ctype0, ctype1) * h * h * h;



            if (sgn == -1 && ctype1 == INTERIOR)  off_diag_coeff[axis] = -coeff;
            diag_coeff += coeff;


            //{
            //    auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
            //    if (info.mLevel == 3 && g_ijk == Coord(31 + 1, 46, 62)) {
            //        printf("compute axis %d sgn %d coeff %f ttype0 %d ttype1 %d ctype0 %d ctype1 %d\n", axis, sgn, coeff, ttype0, ttype1, ctype0, ctype1);
            //    }
            //}

        });

        for (int axis = 0; axis < 3; ++axis) {
            //tile(coeff_channel + axis, l_ijk) = off_diag_coeff[axis];
            tile(coeff_channel + axis, l_ijk) = (ctype0 == INTERIOR) ? off_diag_coeff[axis] : 0;
        }
        //tile(coeff_channel + 3, l_ijk) = diag_coeff;
		tile(coeff_channel + 3, l_ijk) = (ctype0 == INTERIOR) ? diag_coeff : 0;
    }, launch_tile_types);
        //level, launch_types, mode
    //);
}

void CoarsenTypesAndAMGCoeffs(HADeviceGrid<Tile>& grid, const int coeff_channel, const T one_over_alpha) {
    // Before calling this function, cell types for LEAFs must be filled
    // This function will:
    // 1. fill GHOST cell types
    // 2. calculate off-diag and diag coeffs for LEAF and GHOST cells
    // 3. calculate cell types off-diag and diag coeffs for NONLEAF cells
	// one_over_alpha is the coefficient used for coarsening: we have P=alpha*R^T, and all non-zero entries in R are one_over_alpha

    //step 1: fill GHOST cell types
    grid.launchVoxelFuncOnAllTiles(
        [=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
        auto& tile = info.tile();
        auto g_ijk = acc.composeGlobalCoord(info.mTileCoord, l_ijk);
        auto p_g_ijk = acc.parentCoord(g_ijk);
        HATileInfo<Tile> p_info; Coord p_l_ijk;
        acc.findVoxel(info.mLevel - 1, p_g_ijk, p_info, p_l_ijk);
        if (!p_info.empty()) {
            tile.type(l_ijk) = p_info.tile().type(p_l_ijk);
        }
        else {
            tile.type(l_ijk) = DIRICHLET;
        }
    }, GHOST);

	//step 2: calculate coefficients for LEAF and GHOST cells
    CalculateAMGCoefficients(grid, coeff_channel, LEAF | GHOST);

	//step 3: update types and coefficients for NONLEAF cells
    grid.launchVoxelFunc(
        [=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
        auto& tile = info.tile();
        auto g_ijk = acc.composeGlobalCoord(info.mTileCoord, l_ijk);
        int interior_cnt = 0;
        bool has_ghost_child = false;
        T coff_diag[3][2][2][2];//8 children
        T diag_sum = 0;
        for (int ci = 0; ci < 2; ci++) {
            for (int cj = 0; cj < 2; cj++) {
                for (int ck = 0; ck < 2; ck++) {
                    Coord cg_ijk(g_ijk[0] * 2 + ci, g_ijk[1] * 2 + cj, g_ijk[2] * 2 + ck);
					HATileInfo<Tile> cinfo; Coord cl_ijk;
                    acc.findVoxel(info.mLevel + 1, cg_ijk, cinfo, cl_ijk);
                    if (!cinfo.empty()) {
						if (cinfo.mType == GHOST) has_ghost_child = true;
                        auto& ctile = cinfo.tile();
                        diag_sum += ctile(coeff_channel + 3, cl_ijk);
                        interior_cnt += (ctile.type(cl_ijk) == INTERIOR);
                        for (int axis : {0, 1, 2}) {
                            coff_diag[axis][ci][cj][ck] = ctile(coeff_channel + axis, cl_ijk);
                        }
                    }
                    else {
                        for (int axis : {0, 1, 2}) {
                            coff_diag[axis][ci][cj][ck] = 0;
                        }
                    }
                    
                }
            }
        }


        T offs[3] = { 0,0,0 };
        for (int ci = 0; ci < 2; ci++) {
            for (int cj = 0; cj < 2; cj++) {
                for (int ck = 0; ck < 2; ck++) {
					if (ci == 0) offs[0] += coff_diag[0][ci][cj][ck];//face term
                    if (ci == 1) diag_sum += 2 * coff_diag[0][ci][cj][ck];//a cross term will be count twice

					if (cj == 0) offs[1] += coff_diag[1][ci][cj][ck];
					if (cj == 1) diag_sum += 2 * coff_diag[1][ci][cj][ck];

					if (ck == 0) offs[2] += coff_diag[2][ci][cj][ck];
					if (ck == 1) diag_sum += 2 * coff_diag[2][ci][cj][ck];
                }
            }
        }

        if (info.mType == NONLEAF) {
            tile.type(l_ijk) = interior_cnt > 0 ? INTERIOR : DIRICHLET;

            for (int axis : {0, 1, 2}) {
                tile(coeff_channel + axis, l_ijk) = offs[axis] * one_over_alpha;
            }
            tile(coeff_channel + 3, l_ijk) = diag_sum * one_over_alpha;
        }
        if (info.mType == LEAF && has_ghost_child) {
            for (int axis : {0, 1, 2}) {
                tile(coeff_channel + axis, l_ijk) += offs[axis];
            }
            tile(coeff_channel + 3, l_ijk) += diag_sum;
        }
    },
        -1, NONLEAF | LEAF, LAUNCH_SUBTREE, FINE_FIRST);
}

class AMGLaplacianTileData {
public:
    static constexpr int halo = 1;
    static constexpr int SN = 8 + halo * 2;
    T x[SN][SN][SN];               // x channel
	T diag[8][8][8];			   // Diagonal coefficient
	T off_diag0[9][8][8];		  // Off-diagonal coefficient 0
	T off_diag1[8][9][8];		  // Off-diagonal coefficient 1
	T off_diag2[8][8][9];		  // Off-diagonal coefficient 2
    uint8_t ttype[SN][SN][SN];    // tile type

    //they will take actual tile local coords
    __device__ T& xValueT(Coord l_ijk) { return x[l_ijk[0] + halo][l_ijk[1] + halo][l_ijk[2] + halo]; }
	__device__ T& diagValueT(Coord l_ijk) { return diag[l_ijk[0]][l_ijk[1]][l_ijk[2]]; }
	__device__ T& offDiag0ValueT(Coord l_ijk) { return off_diag0[l_ijk[0]][l_ijk[1]][l_ijk[2]]; }
	__device__ T& offDiag1ValueT(Coord l_ijk) { return off_diag1[l_ijk[0]][l_ijk[1]][l_ijk[2]]; }
	__device__ T& offDiag2ValueT(Coord l_ijk) { return off_diag2[l_ijk[0]][l_ijk[1]][l_ijk[2]]; }
    //take the face coeff of the negative direction of actual tile local voxel l_ijk
	__device__ T& offDiagValueT(int axis, Coord l_ijk) {
		if (axis == 0) return off_diag0[l_ijk[0]][l_ijk[1]][l_ijk[2]];
		if (axis == 1) return off_diag1[l_ijk[0]][l_ijk[1]][l_ijk[2]];
        return off_diag2[l_ijk[0]][l_ijk[1]][l_ijk[2]];
	}
	__device__ uint8_t& ttypeValue(Coord l_ijk) { return ttype[l_ijk[0] + halo][l_ijk[1] + halo][l_ijk[2] + halo]; }

    //__device__ T& offDiagVal(int axis, Coord scrd) { return off_diag[axis][scrd[0]][scrd[1]][scrd[2]]; }
    //__device__ T& diagVal(Coord scrd) { return diag[scrd[0]][scrd[1]][scrd[2]]; }

    //for single-level usage, does not consider ttype
    __device__ T negativeLapSameLevel(Coord l_ijk) {
        T x0 = xValueT(l_ijk);
		T sum = x0 * diagValueT(l_ijk);

        for (int axis : {0, 1, 2}) {
            for (int sgn : {-1, 1}) {
                Coord nl_ijk = l_ijk;
                nl_ijk[axis] += sgn;
                //the "upwind" cell, with higher axis coord
                Coord ul_ijk = l_ijk;
                ul_ijk[axis] += (sgn == -1) ? 0 : 1;
                
                sum += offDiagValueT(axis, ul_ijk) * xValueT(nl_ijk);
            }
        }

        return sum;
    }

	//for cross-level usage, consider ttype
    //l_ijk range in in [0,8) it's the actual tile local coord
    __device__ T negativeLapCrossLevel(Coord l_ijk) {
        //int li = l_ijk[0], lj = l_ijk[1], lk = l_ijk[2];
        //if (ttype_s[li + halo][li + halo][lk + halo] == NONLEAF) return 0;
        if (ttypeValue(l_ijk) == NONLEAF) return 0;
        T sum = 0;
        T x0 = xValueT(l_ijk);

        for (int axis : {0, 1, 2}) {
            for (int sgn : {-1, 1}) {
                Coord nl_ijk = l_ijk;
                nl_ijk[axis] += sgn;
                //the "upwind" cell, with higher axis coord
                Coord ul_ijk = l_ijk;
				ul_ijk[axis] += (sgn == -1) ? 0 : 1;
				uint8_t ttype1 = ttypeValue(nl_ijk);
				//uint8_t ttype1 = ttype_s[nl_ijk[0] + halo][nl_ijk[1] + halo][nl_ijk[2] + halo];
                sum += (ttype1 == NONLEAF) ? 0 : offDiagValueT(axis, ul_ijk) * (xValueT(nl_ijk) - x0);
            }
        }

        return sum;
    }
};

//vi in [0,64)
__device__ static Coord FaceNeighborLocalCoord(int axis, int sgn, const int vi) {
    int axk = (sgn == -1) ? -1 : 8;
    int i = vi / 8, j = vi % 8;
    return axis == 0 ? Coord(axk, i, j) : axis == 1 ? Coord(i, axk, j) : Coord(i, j, axk);
}

__device__ static Coord FastLocalOffsetToCoord(int vi) {
    return { vi / 64, (vi / 8) % 8, vi % 8 };
}

//__inline__ __device__ void LoadSharedMemoryAMGTileData(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, SharedAMGLaplacianTileData& shared_data, const int x_channel, const int coeff_channel, int ti) {
    //constexpr int SN = 10;
    //const int shared_offset = 1;

    //auto& tile = info.tile();

    //// Load 4 voxels into the shared memory region from the main tile
    //for (int i = 0; i < 4; i++) {
    //    // Voxel index
    //    int vi = i * 128 + ti;
    //    int3 l_ijk = localIdxToInt3(vi);
    //    int si = l_ijk.x + shared_offset, sj = l_ijk.y + shared_offset, sk = l_ijk.z + shared_offset;

    //    // Load x, off-diagonal and diagonal coefficients
    //    shared_data.x[si][sj][sk] = tile(x_channel, vi);
    //    for (int axis = 0; axis < 3; axis++) {
    //        shared_data.off_diag[axis][si][sj][sk] = tile(coeff_channel + axis, vi);  // Access off-diagonal coeff by axis
    //    }
    //    shared_data.diag[si][sj][sk] = tile(coeff_channel + 3, vi);  // Load diagonal coefficient
    //}


    //Coord b_ijk = info.mTileCoord;

    //// Load boundaries for x-, y-, z- (negative) boundaries
    //if (ti < 64) {
    //    int fi = ti;
    //    for (int axis : {0, 1, 2}) {
    //        Coord nb_ijk = b_ijk; nb_ijk[axis]--;
    //        Coord nl_ijk = acc.rotateCoord(axis, Coord(7, fi / 8, fi % 8));
    //        Coord s_ijk = nl_ijk + Coord(shared_offset, shared_offset, shared_offset);
    //        s_ijk[axis] = 0;
    //        int si = s_ijk[0], sj = s_ijk[1], sk = s_ijk[2];

    //        if (tile.mNeighbors[axis].empty()) {
    //            shared_data.x[si][sj][sk] = 0;
    //            for (int ax = 0; ax < 3; ax++) {
    //                shared_data.off_diag[ax][si][sj][sk] = 0;
    //            }
    //            shared_data.diag[si][sj][sk] = 0;
    //        }
    //        else {
    //            //HATileInfo<Tile> ninfo = acc.tileInfo(info.mLevel, nb_ijk);
    //            auto ninfo = tile.mNeighbors[axis];
    //            auto& ntile = ninfo.tile();
				////T* __restrict__ nx_data = ntile.mData[x_channel];
				////T* __restrict__ noff0_data = ntile.mData[coeff_channel];
				////T* __restrict__ noff1_data = ntile.mData[coeff_channel + 1];
				////T* __restrict__ noff2_data = ntile.mData[coeff_channel + 2];
				////T* __restrict__ ndiag_data = ntile.mData[coeff_channel + 3];

				////int nvi = acc.localCoordToOffset(nl_ijk);
				////shared_data.x[si][sj][sk] = nx_data[nvi];
				////shared_data.off_diag[0][si][sj][sk] = noff0_data[nvi];
				////shared_data.off_diag[1][si][sj][sk] = noff1_data[nvi];
				////shared_data.off_diag[2][si][sj][sk] = noff2_data[nvi];
				////shared_data.diag[si][sj][sk] = ndiag_data[nvi];

    //            shared_data.x[si][sj][sk] = ntile(x_channel, nl_ijk);
    //            for (int ax = 0; ax < 3; ax++) {
    //                shared_data.off_diag[ax][si][sj][sk] = ntile(coeff_channel + ax, nl_ijk);
    //            }
    //            shared_data.diag[si][sj][sk] = ntile(coeff_channel + 3, nl_ijk);
    //        }
    //    }
    //}
    //else {
    //    // Load boundaries for x+, y+, z+ (positive) boundaries
    //    int fi = ti - 64;

    //    for (int axis : {0, 1, 2}) {
    //        Coord nb_ijk = b_ijk; nb_ijk[axis]++;
    //        Coord nl_ijk = acc.rotateCoord(axis, Coord(0, fi / 8, fi % 8));
    //        Coord s_ijk = nl_ijk + Coord(shared_offset, shared_offset, shared_offset);
    //        s_ijk[axis] = SN - 1;
    //        int si = s_ijk[0], sj = s_ijk[1], sk = s_ijk[2];

    //        if (tile.mNeighbors[axis + 3].empty()) {
    //            shared_data.x[si][sj][sk] = 0;
    //            for (int ax = 0; ax < 3; ax++) {
    //                shared_data.off_diag[ax][si][sj][sk] = 0;
    //            }
    //            shared_data.diag[si][sj][sk] = 0;
    //        }
    //        else {
    //            auto ninfo = tile.mNeighbors[axis + 3];
    //            auto& ntile = ninfo.tile();
    //            shared_data.x[si][sj][sk] = ntile(x_channel, nl_ijk);
    //            for (int ax = 0; ax < 3; ax++) {
    //                shared_data.off_diag[ax][si][sj][sk] = ntile(coeff_channel + ax, nl_ijk);
    //            }
    //            shared_data.diag[si][sj][sk] = ntile(coeff_channel + 3, nl_ijk);
    //        }
    //    }
    //}
//}

__device__ void LoadAMGLaplacianTileDataAndTileType(const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, AMGLaplacianTileData& shared_data, const int x_channel, const int coeff_channel, int ti) {
    constexpr int SN = 10;
    const int shared_offset = 1;

    auto& tile = info.tile();

    // Load 4 voxels into the shared memory region from the main tile
 //   T* __restrict__ x_data = tile.mData[x_channel];
 //   T* __restrict__ coeff0_data = tile.mData[coeff_channel];
	//T* __restrict__ coeff1_data = tile.mData[coeff_channel + 1];
	//T* __restrict__ coeff2_data = tile.mData[coeff_channel + 2];
	//T* __restrict__ diag_data = tile.mData[coeff_channel + 3];
    for (int i = 0; i < 4; i++) {
        // Voxel index
        int vi = i * 128 + ti;
        Coord l_ijk = acc.localOffsetToCoord(vi);
        //Coord l_ijk = FastLocalOffsetToCoord(vi);


        //shared_data.xValueT(l_ijk) = x_data[vi];
        //shared_data.offDiag0ValueT(l_ijk) = coeff0_data[vi];
        //shared_data.offDiag1ValueT(l_ijk) = coeff1_data[vi];
        //shared_data.offDiag2ValueT(l_ijk) = coeff2_data[vi];
        //shared_data.diagValueT(l_ijk) = diag_data[vi];

        // Load x, off-diagonal and diagonal coefficients
        shared_data.xValueT(l_ijk) = info.tile()(x_channel, vi);
        shared_data.offDiag0ValueT(l_ijk) = tile(coeff_channel, vi);
        shared_data.offDiag1ValueT(l_ijk) = tile(coeff_channel + 1, vi);
        shared_data.offDiag2ValueT(l_ijk) = tile(coeff_channel + 2, vi);

        shared_data.ttypeValue(l_ijk) = info.mType;

        shared_data.diagValueT(l_ijk) = tile(coeff_channel + 3, vi);  // Load diagonal coefficient
    }


    Coord b_ijk = info.mTileCoord;

    // Load boundaries for x-, y-, z- (negative) boundaries
    if (ti < 64) {
        int fi = ti;
        for (int axis : {0, 1, 2}) {
            Coord fl_ijk = FaceNeighborLocalCoord(axis, -1, fi);

            Coord nb_ijk = b_ijk; nb_ijk[axis]--; 
            Coord nl_ijk = fl_ijk; nl_ijk[axis] = 7;

            //HATileInfo<Tile> ninfo = acc.tileInfo(info.mLevel, nb_ijk);
            HATileInfo<Tile> ninfo = tile.mNeighbors[axis];
            bool empty = ninfo.empty();
            
            shared_data.xValueT(fl_ijk) = empty ? 0 : ninfo.tile()(x_channel, nl_ijk);

            shared_data.ttypeValue(fl_ijk) = empty ? info.mType : ninfo.mType;
            //ttype_s[fl_ijk[0] + 1][fl_ijk[1] + 1][fl_ijk[2] + 1] = empty ? info.mType : ninfo.mType;
        }
    }
    else {
        // Load boundaries for x+, y+, z+ (positive) boundaries
        int fi = ti - 64;
        for (int axis : {0, 1, 2}) {
			Coord fl_ijk = FaceNeighborLocalCoord(axis, 1, fi);

            Coord nb_ijk = b_ijk; nb_ijk[axis]++;
            Coord nl_ijk = fl_ijk; nl_ijk[axis] = 0;

            //HATileInfo<Tile> ninfo = acc.tileInfo(info.mLevel, nb_ijk);
            HATileInfo<Tile> ninfo = tile.mNeighbors[axis + 3];
            bool empty = ninfo.empty();

			shared_data.xValueT(fl_ijk) = empty ? 0 : ninfo.tile()(x_channel, nl_ijk);
			shared_data.offDiagValueT(axis, fl_ijk) = empty ? 0 : ninfo.tile()(coeff_channel + axis, nl_ijk);
            shared_data.ttypeValue(fl_ijk) = empty ? info.mType : ninfo.mType;
            //ttype_s[fl_ijk[0] + 1][fl_ijk[1] + 1][fl_ijk[2] + 1] = empty ? info.mType : ninfo.mType;
        }
    }
}

//will consider ttype for cross-level AMG calculation
__global__ void NegativeLaplacianAMG128Kernel(const HATileAccessor<Tile> acc, HATileInfo<Tile>* tiles, uint8_t launch_tile_types, int x_channel, int coeff_channel, int Ax_channel) {             
    __shared__ AMGLaplacianTileData shared_data;
    //__shared__ uint8_t ttype_s[10][10][10];

    //block idx (tile idx), distinguish from thread idx
    int bi = blockIdx.x;
    //thread idx
    int ti = threadIdx.x;

    //somehow it's faster with the reference symbol
    const auto& info = tiles[bi];

  //  int block_active = 0;
  //  if (ti % warpSize == 0) {
		//// only the 0th lane in a warp decides whether to execute the block
  //      block_active = (info.mType & launch_tile_types);
  //  }
  //  block_active = __shfl_sync(0xFFFFFFFF, block_active, 0);
//    if (!block_active) return;

	if (!(info.mType & launch_tile_types)) return;//use __shfl_sync does not accelerate
    //auto& tile = info.tile();

    LoadAMGLaplacianTileDataAndTileType(acc, info, shared_data, x_channel, coeff_channel, ti);
    //LoadAMGLaplacianTileDataAndTileTypeNaive(acc, info, shared_data, x_channel, coeff_channel, ti);

    __syncthreads();


    for (int i = 0; i < 4; i++) {
        //voxel idx
        int vi = i * 128 + ti;
		Coord l_ijk = acc.localOffsetToCoord(vi);
        //int3 l_ijk = localIdxToInt3(vi);
        //int si = l_ijk.x + shared_offset, sj = l_ijk.y + shared_offset, sk = l_ijk.z + shared_offset;

        
        //{
        //    auto g_ijk = acc.localToGlobalCoord(info, Coord(l_ijk.x, l_ijk.y, l_ijk.z));
        //    if (info.mLevel == 2 && g_ijk == Coord(5, 16, 29)) {
        //        for (int axis : {0, 1, 2}) {
        //            for (int sgn : {-1, 1}) {
        //                Coord ncrd(si, sj, sk);
        //                ncrd[axis] += sgn;
        //                Coord ccrd(si, sj, sk);
        //                ccrd[axis] += (sgn == -1) ? 0 : 1;

        //                
        //                printf("si %d sj %d sk %d axis %d sgn %d type %d ntype %d term %f\n",si,sj,sk, axis, sgn, ttype_s[si][sj][sk], ttype_s[ncrd[0]][ncrd[1]][ncrd[2]],(ttype_s[ncrd[0]][ncrd[1]][ncrd[2]] == NONLEAF) ? 0 : shared_data.offDiagVal(axis, ccrd) * (shared_data.xVal(ncrd) - shared_data.x[si][sj][sk]));
        //            }
        //        }
        //    }
        //}

        info.tile()(Ax_channel, vi) = info.tile().type(vi) == INTERIOR ? shared_data.negativeLapSameLevel(l_ijk) : 0;


        //if (cross_level) {
        //    info.tile()(Ax_channel, vi) = info.tile().type(vi) == INTERIOR ? shared_data.negativeLapCrossLevel(l_ijk) : 0;
        //}
        //else {
        //    //info.tile()(Ax_channel, vi) = shared_data.negativeLapSameLevel(l_ijk);
        //    info.tile()(Ax_channel, vi) = info.tile().type(vi) == INTERIOR ? shared_data.negativeLapSameLevel(l_ijk) : 0;
        //}
    }
}

void NegativeLaplacianAMG128(HADeviceGrid<Tile>& grid, thrust::device_vector<HATileInfo<Tile>>& tiles, int launch_tile_num, uint8_t launch_tile_types, int x_channel, int coeff_channel, int Ax_channel) {
    NegativeLaplacianAMG128Kernel << <launch_tile_num, 128 >> > (grid.deviceAccessor(), thrust::raw_pointer_cast(tiles.data()), launch_tile_types, x_channel, coeff_channel, Ax_channel);
}

//for single level smoothing, does not consider ttype
__global__ void NegativeLaplacianAndGaussSeidelSameLevel128Kernel(const HATileAccessor<Tile> acc, HATileInfo<Tile>* tiles, uint8_t launch_tile_types, int x_channel, int coeff_channel, int rhs_channel, int color) {
    __shared__ AMGLaplacianTileData shared_data;


    //block idx (tile idx), distinguish from thread idx
    int bi = blockIdx.x;
    //thread idx
    int ti = threadIdx.x;

    //somehow it's faster with the reference symbol
    const auto& info = tiles[bi];


    if (!(info.mType & launch_tile_types)) return;//use __shfl_sync does not accelerate

    LoadAMGLaplacianTileDataAndTileType(acc, info, shared_data, x_channel, coeff_channel, ti);

    __syncthreads();


    for (int i = 0; i < 4; i++) {
        //voxel idx
        int vi = i * 128 + ti;
        Coord l_ijk = acc.localOffsetToCoord(vi);

		int li = l_ijk[0], lj = l_ijk[1], lk = l_ijk[2];
        if ((li + lj + lk) % 2 == color) {
            uint8_t ctype = info.tile().type(vi);
            T b = info.tile()(rhs_channel, vi);
            T Ax = shared_data.negativeLapSameLevel(l_ijk);
			T D = shared_data.diagValueT(l_ijk);
            T delta_x = (ctype & INTERIOR) ? (b - Ax) / D : 0;
			//T delta_x = D == 0 ? 0 : (b - Ax) / D;
                //(b - Ax) / shared_data.diagValueT(l_ijk);
			T x1 = shared_data.xValueT(l_ijk) + delta_x;
			info.tile()(x_channel, vi) = x1;
        }
    }
}

void NegativeLaplacianAndGaussSeidelSameLevel128(HADeviceGrid<Tile>& grid, thrust::device_vector<HATileInfo<Tile>>& tiles, int launch_tile_num, uint8_t launch_tile_types, int x_channel, int coeff_channel, int rhs_channel, int color) {
    if (launch_tile_num > 0) {
        NegativeLaplacianAndGaussSeidelSameLevel128Kernel << <launch_tile_num, 128 >> > (grid.deviceAccessor(), thrust::raw_pointer_cast(tiles.data()), launch_tile_types, x_channel, coeff_channel, rhs_channel, color);
    }
}

////on all leafs of the tree
//void FullNegativeLaplacianAMG(HADeviceGrid<Tile>& grid, const int x_channel, const int coeff_channel, const int Ax_channel) {
//    //PropagateValues(grid, x_channel, x_channel, -1, GHOST, LAUNCH_SUBTREE);
//    PropagateValuesToGhostTiles(grid, x_channel, x_channel);
//    NegativeLaplacianAMG128(grid, grid.dAllTiles, grid.dAllTiles.size(), LEAF | GHOST, true, x_channel, coeff_channel, Ax_channel);
//    //NegativeLaplacianSameLevel128(grid, grid.dAllTiles, grid.dAllTiles.size(), -1, LEAF | GHOST, x_channel, Ax_channel, false);
//    //NegativeLaplacianSameLevel(grid, x_channel, Ax_channel, -1, LEAF | GHOST, LAUNCH_SUBTREE, false);
//    //add fine terms stored in ghost cells to parents
//    //AccumulateValues(grid, Ax_channel, Ax_channel, -1, LEAF, LAUNCH_SUBTREE, true);
//    AccumulateValuesToLeafTiles(grid, Ax_channel, Ax_channel, true);
//}

void GaussSeidelAMG(int iters, int order, HADeviceGrid<Tile>& grid, const int level, const int x_channel, const int coeff_channel, const int rhs_channel) {
    //order==0: 0,1
    //order==1: 1,0
    for (int i = 0; i < iters; i++) {
        NegativeLaplacianAndGaussSeidelSameLevel128(grid, grid.dTileArrays[level], grid.hNumTiles[level], LEAF | NONLEAF, x_channel, coeff_channel, rhs_channel, order);
        NegativeLaplacianAndGaussSeidelSameLevel128(grid, grid.dTileArrays[level], grid.hNumTiles[level], LEAF | NONLEAF, x_channel, coeff_channel, rhs_channel, 1 - order);
    }
}



//must initialize to zero 
void VCycleAMG(HADeviceGrid<Tile>& grid, const int x_channel, const int f_channel, const int tmp_channel, const int rhs_channel, const int coeff_channel, int level_iters, int coarsest_iters, const T one_over_alpha, const T prolong_coeff) {
    //int D_channel = Tile::D_channel;

    CPUTimer<std::chrono::microseconds> timer;
    //timer.start();

    //f channel remains unchanged during MG (it will be used in CG iterations)
    //rhs channel is b in SPGrid paper

    //static constexpr double alpha = 1.0;//coefficient for prolongation update

    grid.launchVoxelFunc(
        [=]__device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
        auto& tile = info.tile();
        //initial guess is 0 for LEAF | NONLEAF | GHOST cells
        tile(x_channel, l_ijk) = 0;

        if (info.mType & GHOST) {
            //rhs_channel is 0 for GHOST cells
            tile(rhs_channel, l_ijk) = 0;
        }
        else {
            //rhs_channel is b for LEAF | NONLEAF cells
            tile(rhs_channel, l_ijk) = tile(f_channel, l_ijk);
        }
    },
        -1, LEAF | NONLEAF | GHOST, LAUNCH_SUBTREE
    );

    //timer.stop("init"); timer.start();

    //downstroke
    for (int i = grid.mMaxLevel; i > 0; i--) {
        //u^l=0
        //this is already done at the beginning
        //Fill(grid, x_channel, (Tile::T)0, i, LEAF | NONLEAF, LAUNCH_LEVEL);

        //smooth(u^l, b^l)
        //prepare for the diagonal
        //NegativeLaplacianSameLevel128(grid, grid.dTileArrays[i], grid.hNumTiles[i], i, LEAF | NONLEAF, x_channel, D_channel, true);//diagonal
        //GaussSeidel(level_iters, 0, grid, i, x_channel, tmp_channel, rhs_channel, D_channel);

        GaussSeidelAMG(level_iters, 0, grid, i, x_channel, coeff_channel, rhs_channel);

        //calculate residual
        //r^l = b^l-Au^l
        //including ghost cells
		NegativeLaplacianAMG128(grid, grid.dTileArrays[i], grid.hNumTiles[i], LEAF | NONLEAF | GHOST, x_channel, coeff_channel, tmp_channel);
        //NegativeLaplacianSameLevel128(grid, grid.dTileArrays[i], grid.hNumTiles[i], i, LEAF | NONLEAF | GHOST, x_channel, tmp_channel, false);
        BinaryTransform(grid, rhs_channel, tmp_channel, tmp_channel, []__device__(Tile::T rhs, Tile::T tmp) { return rhs - tmp; }, i, LEAF | NONLEAF | GHOST, LAUNCH_LEVEL);

        //restrict residual to the next level
        //that fills NONLEAF cells of the next level
        Restrict(grid, tmp_channel, rhs_channel, i - 1, NONLEAF, one_over_alpha);
        Copy(grid, f_channel, rhs_channel, i - 1, LEAF, LAUNCH_LEVEL, INTERIOR | DIRICHLET | NEUMANN);
        //AccumulateValues(grid, tmp_channel, rhs_channel, i - 1, LEAF, LAUNCH_LEVEL, true);
        AccumulateToParents(grid, tmp_channel, rhs_channel, i - 1, LEAF, LAUNCH_LEVEL, INTERIOR | DIRICHLET | NEUMANN, 1., true);
        Fill(grid, rhs_channel, (Tile::T)0, i - 1, GHOST, LAUNCH_LEVEL, INTERIOR | DIRICHLET | NEUMANN);

        //Info("downstroke level {} total laplacian launched tiles: {} and level tiles {}", i, laplacian_total_tile_counts, grid.hNumTiles[i]);
    }

    //auto holder = grid.getHostTileHolder(LEAF | NONLEAF | GHOST);
    //polyscope::init();
    //IOFunc::AddLeveledPoissonGridCellCentersToPolyscopePointCloud(holder,
    //    {
    //    {coeff_channel,"offd0"}, {coeff_channel + 1,"offd1"} ,{coeff_channel + 2,"offd2"},{coeff_channel + 3,"diag"}, {-1,"type"},
    //    {Tile::x_channel,"x"}, {Tile::b_channel, "b"}
    //    },
    //    {}, FLT_MAX);
    //polyscope::show();

    //timer.stop("downstroke"); timer.start();

    //smooth bottom
    //NegativeLaplacianSameLevel128(grid, grid.dTileArrays[0], grid.hNumTiles[0], 0, LEAF | NONLEAF, x_channel, D_channel, true);//diagonal
    //DampedJacobiSingleLevel(coarsest_iters, grid, 0, x_channel, tmp_channel, rhs_channel, D_channel, omega);
    //GaussSeidel(coarsest_iters / 2, 0, grid, 0, x_channel, tmp_channel, rhs_channel, D_channel);
    //GaussSeidel(coarsest_iters / 2, 1, grid, 0, x_channel, tmp_channel, rhs_channel, D_channel);
     
	GaussSeidelAMG(coarsest_iters / 2, 0, grid, 0, x_channel, coeff_channel, rhs_channel);
	GaussSeidelAMG(coarsest_iters / 2, 1, grid, 0, x_channel, coeff_channel, rhs_channel);

    //timer.stop("bottom"); timer.start();

    //Info("bottom total laplacian launched tiles: {}", laplacian_total_tile_counts);

    for (int i = 1; i <= grid.mMaxLevel; i++) {
        //PropagateValues(grid, x_channel, x_channel, i, GHOST, LAUNCH_LEVEL);
        ////prolongation: fine.r=prolongate(coarse.x)
        ////prolongate to all fine tiles, which are coarse non-leafs
        //Prolongate(grid, x_channel, tmp_channel, i, LEAF | NONLEAF);

        //Axpy(grid, prolong_coeff, tmp_channel, x_channel, i, LEAF | NONLEAF, LAUNCH_LEVEL);

        ProlongateAndUpdate128(grid, x_channel, x_channel, i, prolong_coeff);

        //smoothing: fine.x = smooth(fine.b)
        //GaussSeidel(level_iters, 1, grid, i, x_channel, tmp_channel, rhs_channel, D_channel);
		GaussSeidelAMG(level_iters, 1, grid, i, x_channel, coeff_channel, rhs_channel);

        //Info("upstroke level {} total laplacian launched tiles: {} and level tiles {}", i, laplacian_total_tile_counts, grid.hNumTiles[i]);
    }

    // timer.stop("upstroke");
}

void AMGSolver::prepareTypesAndCoeffs(HADeviceGrid<Tile>& grid)
{
	CoarsenTypesAndAMGCoeffs(grid, coeff_channel, one_over_alpha);
}

std::tuple<int, double> AMGSolver::solve(HADeviceGrid<Tile>& grid, bool verbose, int max_iters, double relative_tolerance, int level_iters, int coarsest_iters, int sync_stride, bool is_pure_neumann)
{
    double rhs_norm2, threshold_norm2, last_residual_norm2;

    //Use 0 as initial guess
    //x0=0
    Fill(grid, Tile::x_channel, (T)0, -1, LEAF, LAUNCH_SUBTREE, INTERIOR | NEUMANN | DIRICHLET);//fill all leafs with 0
    //initial residual is b

    if (sync_stride > 0) {
        //rhs_norm2=r*r
        last_residual_norm2 = rhs_norm2 = Dot(grid, Tile::r_channel, Tile::r_channel, LEAF);
        if (verbose) Info("ConjugateGradient with initial norm of rhs: {}", sqrt(rhs_norm2));

        //if b is zero, just solve to zero
        if (rhs_norm2 == (double)0) {
            //d_x is zero
            //iters=0, relative_error=0
            //x0=0
            return std::make_tuple(0, (double)0);
        }
        //(epsilon*|b|)^2
        threshold_norm2 = relative_tolerance * relative_tolerance * rhs_norm2;
        threshold_norm2 = std::max(threshold_norm2, std::numeric_limits<double>::min());
    }
    if (is_pure_neumann) ReCenterLeafVoxels(grid, Tile::r_channel, mean_d, count_d);

    ////z0=Minv*r0
    //r_channel->z_channel
    //VCycleMultigrid(grid, Tile::z_channel, Tile::r_channel, Tile::Ap_channel, Tile::tmp_channel, Tile::D_channel, level_iters, coarsest_iters);
    VCycleAMG(grid, Tile::z_channel, Tile::r_channel, Tile::Ap_channel, Tile::tmp_channel, coeff_channel, level_iters, coarsest_iters, one_over_alpha, prolong_coeff);

    //p0=z0
    Copy(grid, Tile::z_channel, Tile::p_channel, -1, LEAF, LAUNCH_SUBTREE, INTERIOR | DIRICHLET | NEUMANN);


    DotAsync(gamma_d, grid, Tile::z_channel, Tile::r_channel, LEAF);


    int i = 0;
    for (i = 0; i < max_iters; i++) {
        //Ap_k=A*p_k
        //here A=-lap
        FullNegativeLaplacian(grid, Tile::p_channel, Tile::Ap_channel);
        //FullNegativeLaplacianAMG(grid, Tile::p_channel, coeff_channel, Tile::Ap_channel);

        //alpha_k=gamma_k/(p_k^T*A*p_k)
        DotAsync(fp_d, grid, Tile::p_channel, Tile::Ap_channel, LEAF);//fp_k=p_k^T*A*p_k
        //alpha = gamma / fp
        TernaryOnArray(alpha_d, gamma_d, fp_d, []__device__(double& alpha, double gamma, double fp) { alpha = gamma / fp; });

        //x_{k+1} = x_k + alpha_k * p_k
        //r_{k+1} = r_k - alpha_k * Ap_k
        auto alpha_ptr = alpha_d;
        grid.launchVoxelFuncOnAllTiles(
            [alpha_ptr]__device__(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk) {
            auto& tile = info.tile();
            int vi = acc.localCoordToOffset(l_ijk);
            if (tile.type(vi) & INTERIOR) {
                //double alpha = *alpha_ptr;
                auto alpha = float(*alpha_ptr);
                tile(Tile::x_channel, vi) += alpha * tile(Tile::p_channel, vi);
                //tile(Tile::r_channel, vi) -= alpha * tile(Tile::Ap_channel, vi);
            }

        }, LEAF, 4);
        grid.launchVoxelFuncOnAllTiles(
            [alpha_ptr]__device__(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk) {
            auto& tile = info.tile();
            int vi = acc.localCoordToOffset(l_ijk);
            if (tile.type(vi) & INTERIOR) {
                auto alpha = float(*alpha_ptr);
                //tile(Tile::x_channel, vi) += alpha * tile(Tile::p_channel, vi);
                tile(Tile::r_channel, vi) -= alpha * tile(Tile::Ap_channel, vi);
            }

        }, LEAF, 4);

        if (sync_stride > 0 && (i + 1) % sync_stride == 0) {
            double residual_norm2 = Dot(grid, Tile::r_channel, Tile::r_channel, LEAF);
            if (verbose) {
                Info("ConjugateGradient iter {} norm {:e}({:e}), convergence rate={:e}", i, sqrt(residual_norm2), sqrt(residual_norm2 / rhs_norm2), sqrt(residual_norm2 / last_residual_norm2));
                last_residual_norm2 = residual_norm2;
            }
            if (residual_norm2 < threshold_norm2) {
                return std::make_tuple(i + 1, sqrt(residual_norm2 / rhs_norm2));
            }
        }
        if (is_pure_neumann) ReCenterLeafVoxels(grid, Tile::r_channel, mean_d, count_d);

        //z_{k+1} = Minv * r_{k+1}
        //r->z
        //VCycleMultigrid(grid, Tile::z_channel, Tile::r_channel, Tile::Ap_channel, Tile::tmp_channel, Tile::D_channel, level_iters, coarsest_iters);
		VCycleAMG(grid, Tile::z_channel, Tile::r_channel, Tile::Ap_channel, Tile::tmp_channel, coeff_channel, level_iters, coarsest_iters, one_over_alpha, prolong_coeff);


        //gamma_old = gamma;
        TernaryOnArray(gamma_d, gamma_d, gamma_old_d, []__device__(double& gamma, double& _, double& gamma_old) { gamma_old = gamma; });

        //gamma_{k+1} = dot(r_{k+1}, z_{k+1})
        DotAsync(gamma_d, grid, Tile::r_channel, Tile::z_channel, LEAF);

        //beta_{k+1} = gamma_{k+1} / gamma_k
        //beta = gamma / gamma_old;
        TernaryOnArray(beta_d, gamma_d, gamma_old_d, []__device__(double& beta, double gamma, double gamma_old) { beta = gamma / gamma_old; });

        auto beta_ptr = beta_d;
        //p_{k+1} = z_{k+1} + beta_{k+1} * p_{k}
        //BinaryTransform(grid, Tile::z_channel, Tile::p_channel, Tile::p_channel, [=]__device__(Tile::T z, Tile::T p) { return z + beta * p; }, -1, LEAF, LAUNCH_SUBTREE);
        grid.launchVoxelFuncOnAllTiles(
            [beta_ptr]__device__(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk) {
            auto& tile = info.tile();
            int vi = acc.localCoordToOffset(l_ijk);
            if (tile.type(vi) & INTERIOR) {
                //double beta = *beta_ptr;
                auto beta = float(*beta_ptr);
                tile(Tile::p_channel, vi) = tile(Tile::z_channel, vi) + beta * tile(Tile::p_channel, vi);
            }
        },
            LEAF, 1
        );
    }

    if (sync_stride > 0) {
        double residual_norm2 = Dot(grid, Tile::r_channel, Tile::r_channel, LEAF);
        if (verbose) {
            Info("ConjugateGradient iter {} norm {:e}({:e}), convergence rate={:e}", i, sqrt(residual_norm2), sqrt(residual_norm2 / rhs_norm2), sqrt(residual_norm2 / last_residual_norm2));
        }
        return std::make_tuple(i, sqrt(residual_norm2 / rhs_norm2));
    }
    else {
        return std::make_tuple(i, -1);
    }
}
