#include "PoissonSolver.h"
#include "CPUTimer.h"
//#include "GMGSolver.h"

extern int laplacian_total_tile_counts;

void CalcCellTypesFromLeafs(HADeviceGrid<Tile>& grid) {
    //We already have cell types for leafs

    //step 1: update non-leafs
    grid.launchVoxelFunc(
        [=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
        auto& tile = info.tile();
        auto g_ijk = acc.composeGlobalCoord(info.mTileCoord, l_ijk);

        int dirichlet_cnt = 0;
        int interior_cnt = 0;
        int neumann_cnt = 0;

        acc.iterateChildVoxels(info, l_ijk,
            [&]__device__(const HATileInfo<Tile>&c_info, const Coord & c_ijk) {
            if (c_info.isActive()) {
                auto& tile = c_info.tile();
                auto typ = tile.type(c_ijk);
                if (typ & DIRICHLET) dirichlet_cnt++;
                else if (typ & INTERIOR) interior_cnt++;
            }
        });

        //extrapolation actually requires that a NONLEAF cell is marked as INTERIOR
        //if it has some INTERIOR children
        //therefore the face will be masked as valid
        //interior>dirichlet>neumann
        //if (interior_cnt > 0) tile.type(l_ijk) = INTERIOR;
        //else if (dirichlet_cnt > 0) tile.type(l_ijk) = DIRICHLET;
        //else tile.type(l_ijk) = NEUMANN;

        ////dirichlet>interior>neumann
        if (dirichlet_cnt > 0) tile.type(l_ijk) = DIRICHLET;
        else if (interior_cnt > 0) tile.type(l_ijk) = INTERIOR;
        else tile.type(l_ijk) = NEUMANN;
    },
        -1, NONLEAF, LAUNCH_SUBTREE, FINE_FIRST);

    //step 2: update ghosts
    grid.launchVoxelFuncOnAllTiles(
        [=] __device__(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk) {
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
        //-1, GHOST, LAUNCH_SUBTREE, COARSE_FIRST);
}



//calculate the integral over the cell
void VelocityVolumeDivergenceOnLeafs(HADeviceGrid<Tile>& grid, const int u_channel, const int div_channel) {
    //propagate velocities to ghost cells
    //for (int i : {0, 1, 2}) {
    //    PropagateValues(grid, u_channel + i, u_channel + i, -1, GHOST, LAUNCH_SUBTREE);
    //}

    grid.launchVoxelFunc(
        [=]__device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
		constexpr int level = -1;
        auto h = acc.voxelSize(info);
        Tile& tile = info.tile();
        if (!tile.isInterior(l_ijk)) {
            tile(div_channel, l_ijk) = 0;
            return;
        }

        auto ttype = info.subtreeType(level);//it must be LEAF or GHOST
        CellType ctype = (CellType)tile.type(l_ijk);//must be INTERIOR

        Tile::T sum = 0;
        acc.iterateSameLevelNeighborVoxels(info, l_ijk,
            [&]__device__(const HATileInfo<Tile>&n_info, const Coord & n_l_ijk, const int axis, const int sgn) {
            auto n_ttype = ttype;
            CellType n_ctype;
            T u0 = tile(u_channel + axis, l_ijk), u1;

            if (n_info.empty()) {
                //if the neighbor is empty, we will consider it has the same tile type and cell type as the current one
                n_ttype = ttype;
                n_ctype = ctype;
                u1 = 0;
            }
            else {
                auto& n_tile = n_info.tile();
                n_ttype = n_info.subtreeType(level);
                n_ctype = (CellType)n_tile.type(n_l_ijk);
                u1 = n_tile(u_channel + axis, n_l_ijk);
            }

            //auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
            //auto ng_ijk = acc.localToGlobalCoord(n_info, n_l_ijk);
            //if (g_ijk == Coord(1, 104, 105)) {
            //    printf("g_ijk = %d %d %d axis=%d sgn=%d ng_ijk=%d %d %d u0=%f u1=%f ttype=%d n_ttype=%d ctype=%d n_ctype=%d\n", g_ijk[0], g_ijk[1], g_ijk[2], axis, sgn, ng_ijk[0], ng_ijk[1], ng_ijk[2], u0, u1, ttype, n_ttype, ctype, n_ctype);
            //}


            //tile types check
            //we only calculate LEAF-GHOST and LEAF-LEAF terms
            //and we set delta_h correspondingly
            if (ttype == LEAF && n_ttype == LEAF) {}//pass
            else if (ttype == LEAF && n_ttype == GHOST) {}//pass
            else if (ttype == GHOST && n_ttype == LEAF) {}//pass
            else return;

            //if one of them are NEUMANN we will not count this flux
            //if (ctype == NEUMANN || n_ctype == NEUMANN) return;

			T u = (sgn == -1) ? -u0 : u1;
            sum += u * h * h;

			//auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
   //         if (info.mLevel == 2 && g_ijk == Coord(1, 104, 105)) {
			//	auto ng_ijk = acc.localToGlobalCoord(n_info, n_l_ijk);
   //             printf("g_ijk = %d %d %d axis=%d sgn=%d ng_ijk=%d %d %d u0=%f u1=%f sum=%f\n", g_ijk[0], g_ijk[1], g_ijk[2], axis, sgn, ng_ijk[0], ng_ijk[1], ng_ijk[2], u0, u1, sum);
   //         }
        });

        tile(div_channel, l_ijk) = sum;

    },
        -1, LEAF | GHOST, LAUNCH_SUBTREE
    );

    //LEAF-INTERIOR gathered here
    //AccumulateValues(grid, div_channel, div_channel, -1, GHOST, LAUNCH_SUBTREE, true);
    AccumulateToParents(grid, div_channel, div_channel, -1, LEAF, LAUNCH_SUBTREE, INTERIOR, 1.0, true);
}

void AddGradientToFaceCenters(HADeviceGrid<Tile>& grid, const int p_channel, const int u_channel) {
    //propagate p to ghost cells
    PropagateValues(grid, p_channel, p_channel, -1, GHOST, LAUNCH_SUBTREE);

    grid.launchVoxelFunc(
        [=]__device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
        constexpr int level = -1;
        auto& tile = info.tile();
        auto h = acc.voxelSize(info);
        auto p1 = tile.interiorValue(p_channel, l_ijk);

        auto ttype = info.subtreeType(level);//it must be LEAF or GHOST
        CellType ctype = (CellType)tile.type(l_ijk);
        if (ttype == GHOST) {
            for (int axis : {0, 1, 2}) {
				tile(u_channel + axis, l_ijk) = 0;
            }
        }

        acc.iterateSameLevelNeighborVoxels(info, l_ijk,
            [&]__device__(const HATileInfo<Tile>&n_info, const Coord & n_l_ijk, const int axis, const int sgn) {
            if (sgn == 1) return;

            auto n_ttype = ttype;
            CellType n_ctype;
            T p0;

            if (n_info.empty()) {
                //if the neighbor is empty, we will consider it has the same tile type and cell type as the current one
                n_ttype = ttype;
                n_ctype = ctype;
                p0 = Tile::BACKGROUND_VALUE;
            }
            else {
                auto& n_tile = n_info.tile();
                n_ttype = n_info.subtreeType(level);
                n_ctype = (CellType)n_tile.type(n_l_ijk);
                p0 = n_tile.interiorValue(p_channel, n_l_ijk);
            }

            //tile types check
            //we only calculate LEAF-GHOST and LEAF-LEAF terms
            //and we set delta_h correspondingly
            T delta_h = h;
            if (ttype & LEAF && n_ttype & LEAF) delta_h = h;
            else if (ttype & LEAF && n_ttype & GHOST) delta_h = 1.5 * h;
            else if (ttype & GHOST && n_ttype & LEAF) delta_h = 1.5 * h;
            else return;

            //if one of them are NEUMANN we will not count this flux
            if (ctype & NEUMANN || n_ctype & NEUMANN) return;
            tile(u_channel + axis, l_ijk) += (p1 - p0) / delta_h;

    //        auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
    //        if (p0 != 0 || p1 != 0) {
				//printf("vel correction g_ijk=%d %d %d axis=%d sgn=%d p0=%f p1=%f delta_h=%f term=%f\n", g_ijk[0], g_ijk[1], g_ijk[2], axis, sgn, p0, p1, delta_h, (p1 - p0) / delta_h);
    //        }
        });
    },
        -1, LEAF | GHOST, LAUNCH_SUBTREE
    );

    for (int i : {0, 1, 2}) {
        AccumulateToParents(grid, u_channel + i, u_channel + i, -1, LEAF, LAUNCH_SUBTREE, INTERIOR | DIRICHLET, 1.0 / 4.0, true);
    }
}

//THIS FUNCTION IS DEPRECATED.
//IT'S LEFT FOR REFERENCE ONLY.
//launch_types and mode exactly follow the convention of launchTileFunc
//inside these launched tiles, negative laplacian are calculated on INTERIOR cells
//other cells are set to 0
//but, during calculation, only the contribution of LEAF and GHOST neighbors are considered
//here "LEAF" is under the context of subtree at "level"
//for example, if level=2, non-leaf neighbors at level 2 will also contribute
void NegativeLaplacianSameLevel(HADeviceGrid<Tile>& grid, const int x_channel, const int Ax_channel, const int level, const uint8_t launch_types, const LaunchMode mode, bool calc_diagonal) {
    grid.launchVoxelFunc(
        [=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
        auto h = acc.voxelSize(info);
        Tile& tile = info.tile();
        if (!tile.isInterior(l_ijk)) {
            tile(Ax_channel, l_ijk) = 0;
            return;
        }

        auto ttype = info.subtreeType(level);//it must be LEAF or GHOST
        CellType ctype = (CellType)tile.type(l_ijk);//must be INTERIOR
        Tile::T x0 = tile(x_channel, l_ijk);//equal to interiorValue because it's interior
        if (calc_diagonal) x0 = 1;

        Tile::T sum = 0;
        acc.iterateSameLevelNeighborVoxels(info, l_ijk,
            [&]__device__(const HATileInfo<Tile>& n_info, const Coord& n_l_ijk, const int axis, const int sgn) {
            auto n_ttype = ttype;
            CellType n_ctype;
            T x1;

            if (n_info.empty()) {
                //if the neighbor is empty, we will consider it has the same tile type and cell type as the current one
                n_ttype = ttype;
                n_ctype = ctype;
                x1 = Tile::BACKGROUND_VALUE;
            }
            else {
                auto& n_tile = n_info.tile();
                n_ttype = n_info.subtreeType(level);
                n_ctype = (CellType)n_tile.type(n_l_ijk);
                //x1 = n_tile.interiorValue(x_channel, n_l_ijk);
				x1 = n_tile(x_channel, n_l_ijk);
            }
            if (calc_diagonal) x1 = 0;

            //{
            //    auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
            //    if (info.mLevel == 2 && g_ijk == Coord(16, 19, 28)) {
            //        printf("g_ijk = %d %d %d axis=%d sgn=%d x0=%f x1=%f ttype=%d nttype=%d ctype=%d nctype=%d\n", g_ijk[0], g_ijk[1], g_ijk[2], axis, sgn, x0, x1, ttype, n_ttype, ctype, n_ctype);
            //    }
            //}

            //tile types check
            //we only calculate LEAF-GHOST and LEAF-LEAF terms
            //and we set delta_h correspondingly
            T delta_h = h;
            if (ttype & LEAF && n_ttype & LEAF) delta_h = h;
            else if (ttype & LEAF && n_ttype & GHOST) delta_h = 1.5 * h;
            else if (ttype & GHOST && n_ttype & LEAF) delta_h = 1.5 * h;
            else return;

            //if one of them are NEUMANN we will not count this flux
            if (ctype & NEUMANN || n_ctype & NEUMANN) return;
            sum += (x0 - x1) / (delta_h * h);
            //{
            //    auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
            //    if (info.mLevel == 3 && g_ijk == Coord(16, 19, 28)) {
            //        printf("g_ijk = %d %d %d axis=%d sgn=%d x0=%f x1=%f delta_h=%f term=%f\n", g_ijk[0], g_ijk[1], g_ijk[2], axis, sgn, x0, x1, delta_h, (x0 - x1) / (delta_h * h));
            //    }
            //}
        });

        tile(Ax_channel, l_ijk) = sum * (h * h * h);
    },
        level, launch_types, mode
    );
}



//calculate the coarse level from its child fine level
void Restrict(HADeviceGrid<Tile>& grid, const uint8_t fine_channel, const uint8_t coarse_channel, const int coarse_level, const uint8_t launch_types, const T one_over_alpha) {
    using T = typename Tile::T;
    int fine_level = coarse_level + 1;
    grid.launchVoxelFunc(
        [=]__device__(HATileAccessor<Tile> &acc, HATileInfo<Tile> &info, const Coord & l_ijk) {
        auto& tile = info.tile();
        if (!tile.isInterior(l_ijk)) tile(coarse_channel, l_ijk) = 0;
        Coord c0_ijk = acc.childTileCoord(info, Coord(0, 0, 0));
        if (!acc.probeTile(fine_level, c0_ijk)) return;//if no child, then no need to restrict
        auto coarse_g_ijk = acc.localToGlobalCoord(info, l_ijk);
        T sum = 0;
        for (uint8_t s = 0; s < acc.NUMCHILDREN; s++) {
            auto fine_g_ijk = acc.childCoord(coarse_g_ijk, s);
            HATileInfo<Tile> fine_info; Coord fine_l_ijk;
            acc.findVoxel(fine_level, fine_g_ijk, fine_info, fine_l_ijk);
            if (!fine_info.empty()) {
                auto& fine_tile = fine_info.tile();
                sum += fine_tile.interiorValue(fine_channel, fine_l_ijk);
            }
        }
        //tile(coarse_channel, l_ijk) = sum * 0.5 * 2;//0.5 to balance laplacian operator, 2 is update coefficient
        tile(coarse_channel, l_ijk) = sum * one_over_alpha;
    },
        coarse_level, launch_types, LAUNCH_LEVEL
    );
}

//calculate the fine level from its parent coarse level
void Prolongate(HADeviceGrid<Tile>& grid, const uint8_t coarse_channel, const uint8_t fine_channel, const int fine_level, const uint8_t launch_types) {
    using T = typename Tile::T;
    int coarse_level = fine_level - 1;
    //Info("Prolongate from level {} to level {}", coarse_level, fine_level);
    grid.launchVoxelFunc(
        [=]__device__(HATileAccessor<Tile> &acc, HATileInfo<Tile> &info, const Coord & l_ijk) {
            //{
            //    auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
            //    if (info.mLevel == 1 && g_ijk == Coord(15, 7, 12)) {
            //        printf("g_ijk = %d %d %d\n", g_ijk[0], g_ijk[1], g_ijk[2]);
            //        //printf("fine_g_ijk = %d %d %d coarse_g_ijk = %d %d %d\n", g_ijk[0], g_ijk[1], g_ijk[2], coarse_g_ijk[0], coarse_g_ijk[1], coarse_g_ijk[2]);
            //    }
            //}

        auto& tile = info.tile();
        if (!tile.isInterior(l_ijk)) {
            tile(fine_channel, l_ijk) = 0;
            return;
        }
        auto fine_g_ijk = acc.localToGlobalCoord(info, l_ijk);
        auto coarse_g_ijk = acc.parentCoord(fine_g_ijk);
        HATileInfo<Tile> coarse_info; Coord coarse_l_ijk;
        acc.findVoxel(coarse_level, coarse_g_ijk, coarse_info, coarse_l_ijk);

            
        if (!coarse_info.empty()) {
            auto& coarse_tile = coarse_info.tile();
            tile(fine_channel, l_ijk) = coarse_tile.interiorValue(coarse_channel, coarse_l_ijk);
        }
        else tile(fine_channel, l_ijk) = 0;
    },
        fine_level, launch_types, LAUNCH_LEVEL
    );
}

//add 
void ReCenterLeafVoxels(HADeviceGrid<Tile>& grid, const int channel, double* mean_d, double* count_d) {
    //Assert(false, "ReCenterLeafVoxels not implemented");
    ////add an offset to make the mean value of all leafs 0
    MeanAsync(grid, channel, LEAF, mean_d, count_d);
    grid.launchVoxelFuncOnAllTiles(
        [=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
        auto& tile = info.tile();
        double mean = *mean_d;
        if (tile.isInterior(l_ijk)) {
            tile(channel, l_ijk) -= mean;
        }
    }, LEAF, 8
    );

    //double mean = Mean(grid, -1, channel);
    //grid.launchVoxelFunc(
    //    -1, EXEC_LEAF,
    //    [=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
    //    auto& tile = info.tile();
    //    if (tile.isActive(l_ijk)) {
    //        tile(channel, l_ijk) -= mean * pow(acc.voxelSize(info), 3);
    //    }
    //});

    //Info("after centering: {}", Mean(grid, -1, channel));
}