#include "SolverTests.h"
#include "PoissonSolver.h"
#include "GMGSolver.h"
#include "AMGSolver.h"
#include "PoissonIOFunc.h"
#include "Common.h"
#include "GPUTimer.h"
#include "Random.h"
#include <polyscope/polyscope.h>
#include <polyscope/point_cloud.h>

extern int laplacian_total_tile_counts;

__device__ int SolverTestsLevelTarget(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const int grid_case) {
    if (grid_case == 0) {
        //128^3
        return 4;
    }
    else if (grid_case == 1) {
        auto bbox = acc.tileBBox(info);
        int desired_level = 0;
        if (bbox.min()[0] <= 0.25) return 4;//slow converging, if 0.25 not converging
        else return 3;
    }
    else if (grid_case == 2) {
        int desired_level = 0;
        auto bbox = acc.tileBBox(info);
        const Vec pointSrc1(0.51, 0.49, 0.54);
        const Vec pointSrc2(0.93, 0.08, 0.91);
        if (bbox.isInside(pointSrc2)) desired_level = 6;
        if (bbox.isInside(pointSrc1)) desired_level = 7;
        //if (bbox.isInside(pointSrc2)) desired_level = 3;
        //if (bbox.isInside(pointSrc1)) desired_level = 4;
        return desired_level;
    }
    else if (grid_case == 3) {
        //refine at (0.35,0.35,0.35)
        //it's for testing the 3D deformation
        int desired_level = 0;
        auto bbox = acc.tileBBox(info);
        const Vec pointSrc1(0.35, 0.35, 0.35);
        const Vec pointSrc2(0.8, 0.2, 0.6);
        if (bbox.isInside(pointSrc2)) desired_level = 5;
        if (bbox.isInside(pointSrc1)) desired_level = 6;
        //if (bbox.isInside(pointSrc2)) desired_level = 3;
        //if (bbox.isInside(pointSrc1)) desired_level = 4;
        return desired_level;
    }
    else if (grid_case == 4) {
        //8^3
        //to test most basic case
        return 3;
    }
    else if (grid_case == 5) {
        //bottom part denser
        auto bbox = acc.tileBBox(info);
        if (bbox.min()[1] <= 0.25) return 4;//slow converging, if 0.25 not converging
        //if (bbox.max()[1] >= 0.75) return 3;//slow converging, if 0.25 not converging
        else return 2;
    }
    else if (grid_case == 6) {
        auto bbox = acc.tileBBox(info);
        if (bbox.min()[1] <= 0.5 && 0.5 <= bbox.max()[1]) return 6;
        else return 2;
    }
    else if (grid_case == 7) {
        //try to test nfm advection with 3d deformation

        int desired_level = 0;
        auto bbox = acc.tileBBox(info);
        double eps = 1e-6;
        const Vec pointSrc1(0.5 - eps, 0.5 - eps, 0.5 - eps);
        if (bbox.isInside(pointSrc1)) desired_level = 6;

        return desired_level;
    }
    else if (grid_case == 8) {
        return 3;
    }
    else if (grid_case == 9) {
        return 5;
    }
}




void TestNeumannBC(int grid_case, bool debug) {
    Info("Neumann BC test case {}", grid_case);

    uint32_t scale = 8;
    float h = 1.0 / scale;

    //0:8, 1:16, 2:32, 3:64, 4:128, 5:256, 6:512, 7:1024
    HADeviceGrid<Tile> grid(h, { 16,16,16,16,16,16,18,16,16,16 });

    //this bbox of the 0th layer will be refined in the 1st layer
    grid.setTileHost(0, nanovdb::Coord(0, 0, 0), Tile(), LEAF);
    grid.compressHost(false);
    grid.syncHostAndDevice();
    SpawnGhostTiles(grid, false);
    Info("Grid size: {}", grid.numTotalLeafTiles() * Tile::SIZE);

    IterativeRefine(grid, [=]__device__(const HATileAccessor<Tile>&acc, HATileInfo<Tile>&info) { return SolverTestsLevelTarget(acc, info, grid_case); }, false);

    Info("iterative done");

    int total_hash_bytes = grid.hashTableDeviceBytes();
    Info("Total hash table bytes: {}/G", total_hash_bytes / (1024.0 * 1024 * 1024));

    int grdt_channel = Tile::vor_channel;

    auto cell_type = [=]__device__(const HATileAccessor<Tile> &acc, const HATileInfo<Tile> &info, const nanovdb::Coord & l_ijk)->uint8_t {
        bool is_dirichlet = false;
        bool is_neumann = false;
        acc.iterateSameLevelNeighborVoxels(info, l_ijk,
            [&]__device__(const HATileInfo<Tile>&n_info, const Coord & n_l_ijk, const int axis, const int sgn) {
            if (n_info.empty()) {
                if (axis == 0 && sgn == -1) {
                    is_neumann = true;
                }
                else {
                    is_dirichlet = true;
                }
            }
        });
        if (is_neumann) return CellType::NEUMANN;
        else if (is_dirichlet) return CellType::DIRICHLET;
        else return CellType::INTERIOR;
    };
    //solution f(x)
    auto f = [=]__device__(const Vec & pos) {
        return exp(pos[0]) * sin(pos[1]);
    };
    grid.launchVoxelFunc(
        [=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
        auto& tile = info.tile();
        tile.type(l_ijk) = cell_type(acc, info, l_ijk);

    }, -1, LEAF, LAUNCH_SUBTREE
    );
    CalcCellTypesFromLeafs(grid);
    //rhs calculation needs to check the same-level neighbor types
    //that will depend on ghost cells as well
    //so we need to update ghost cells before
    grid.launchVoxelFunc(
        [=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
        auto& tile = info.tile();
        auto pos = acc.cellCenter(info, l_ijk);
        auto h = acc.voxelSize(info);
        tile(Tile::b_channel, l_ijk) = 0;
        tile(grdt_channel, l_ijk) = f(pos);

        if (tile.type(l_ijk) == INTERIOR) {
            //offset b
            T b = 0;
            acc.iterateSameLevelNeighborVoxels(info, l_ijk,
                [&]__device__(const HATileInfo<Tile>&n_info, const Coord & n_l_ijk, const int axis, const int sgn) {
                if (!n_info.empty()) {
                    auto& n_tile = n_info.tile();
                    if (n_tile.type(n_l_ijk) == NEUMANN) {
                        auto n_pos = acc.cellCenter(n_info, n_l_ijk);
                        b -= (f(pos) - f(n_pos)) * h;//it's /h^2 * h^3, laplacian and volume
                    }
                    else if (n_tile.type(n_l_ijk) == DIRICHLET) {
                        auto n_pos = acc.cellCenter(n_info, n_l_ijk);
                        b += f(n_pos) * h;
                    }
                }
            });
            tile(Tile::b_channel, l_ijk) = b;
            //tile(Tile::b_channel, l_ijk) = 1;
        }
        tile(Tile::u_channel, l_ijk) = tile.interiorValue(Tile::b_channel, l_ijk);
    }, -1, LEAF, LAUNCH_SUBTREE
    );


    //Info("b volume weighted linf: {}", VolumeWeightedNorm(grid, -1, Tile::b_channel));
    //Info("b volume weighted norm2: {}", VolumeWeightedNorm(grid, 2, Tile::b_channel));

    //FullNegativeLaplacian(grid, grdt_channel, Tile::tmp_channel);
    //Axpy(grid, -1.0, Tile::b_channel, Tile::tmp_channel, -1, LEAF, LAUNCH_SUBTREE);
    //Info("grdt residual norm2: {}", VolumeWeightedNorm(grid, 2, Tile::tmp_channel));

    //auto b_dot = Dot(grid, Tile::b_channel, Tile::b_channel, -1, LEAF, LAUNCH_SUBTREE);
    //Info("b dot: {}", b_dot);
    //auto grdt_dot = Dot(grid, grdt_channel, grdt_channel, -1, LEAF, LAUNCH_SUBTREE);
    //Info("grdt dot: {}", grdt_dot);


    //_sleep(200);

    //{
    //    int repeat_times = 1;
    //    CPUTimer<std::chrono::microseconds> timer;
    //    timer.start();
    //    for (int i = 0; i < repeat_times; i++) {
    //        //FullNegativeLaplacian(grid, grdt_channel, Tile::tmp_channel);
    //        //NegativeLaplacianSameLevel(grid, grdt_channel, Tile::tmp_channel, -1, LEAF, LAUNCH_SUBTREE, false);
    //        //NegativeLaplacianSameLevelShared(grid, grdt_channel, Tile::tmp_channel, -1, LEAF, LAUNCH_SUBTREE, false);
    //        NegativeLaplacianSameLevel128(grid, grid.dAllTiles, grid.dAllTiles.size(), -1, LEAF, grdt_channel, Tile::tmp_channel, false);
    //    }
    //    cudaDeviceSynchronize();
    //    
    //    CheckCudaError("Laplacian Shared done");
    //    float elapsed = timer.stop("Laplacian Shared") / repeat_times;
    //    int total_cells = grid.numTotalLeafTiles() * Tile::SIZE;
    //    float cells_per_second = (total_cells + 0.0) / (elapsed / 1e6);
    //    Info("Repeat {} times, Total {} tiles and {:.5}M cells, Full Laplacian speed {:.5} M cells /s", repeat_times, grid.numTotalLeafTiles(), total_cells / (1024.0 * 1024), cells_per_second / (1024.0 * 1024));
    //}

    //_sleep(200);


    //{
    //    int repeat_times = 1;
    //    CPUTimer<std::chrono::microseconds> timer;
    //    timer.start();
    //    for (int i = 0; i < repeat_times; i++) {
    //        //FullNegativeLaplacian(grid, grdt_channel, Tile::tmp_channel);
    //        NegativeLaplacianSameLevel(grid, grdt_channel, Tile::Ap_channel, -1, LEAF, LAUNCH_SUBTREE, false);
    //        //NegativeLaplacianSameLevelShared(grid, grdt_channel, Tile::tmp_channel, -1, LEAF, LAUNCH_SUBTREE, false);
    //    }
    //    cudaDeviceSynchronize();

    //    CheckCudaError("Full Laplacian done");
    //    float elapsed = timer.stop("Full Laplacian Normal") / repeat_times;
    //    int total_cells = grid.numTotalLeafTiles() * Tile::SIZE;
    //    float cells_per_second = (total_cells + 0.0) / (elapsed / 1e6);
    //    Info("Repeat {} times, Total {:.5}M cells, Full Laplacian speed {:.5} M cells /s", repeat_times, total_cells / (1024.0 * 1024), cells_per_second / (1024.0 * 1024));
    //}

    //Info("grdt dot: {}", Dot(grid, Tile::Ap_channel, Tile::Ap_channel, -1, LEAF, LAUNCH_SUBTREE));
    //Info("shared dot: {}", Dot(grid, Tile::tmp_channel, Tile::tmp_channel, -1, LEAF, LAUNCH_SUBTREE));

 //   polyscope::init();
	//auto holder = grid.getHostTileHolderForLeafs();
 //   IOFunc::AddPoissonGridCellCentersToPolyscopePointCloud(holder, { {grdt_channel, "x"}, {Tile::Ap_channel,"grdt"}, {Tile::tmp_channel,"ours"} }, {});
 //   polyscope::show();

    //_sleep(200);


    //{
    //    CPUTimer<std::chrono::microseconds> timer;
    //    timer.start();
    //    auto [iters, err] = ConjugateGradientSync(grid, true, 1000, 1, 20, 1e-6, false);
    //    CheckCudaError("VCycleMultigrid done");
    //    float elapsed = timer.stop("MGPCG Solve Synced");
    //    int total_cells = grid.numTotalLeafTiles() * Tile::SIZE;
    //    float cells_per_second = (total_cells + 0.0) / (elapsed / 1e6);
    //    Info("Total {:.5}M cells, MGPCG Sync speed {:.5} M cells /s", total_cells / (1024.0 * 1024), cells_per_second / (1024.0 * 1024));
    //    Info("MGPCG solved in {} iterations with error {}", iters, err);
    //}

    //{
    //    Copy(grid, Tile::b_channel, Tile::phi_channel, -1, LEAF, LAUNCH_SUBTREE, INTERIOR | DIRICHLET | NEUMANN);

    //    //parameter study for solver
    //    for (int level_iters : {1, 2, 3, 5, 10, 20,50}) {
    //        for (int coarse_iters : {1, 10, 20, 100}) {
    //            Copy(grid, Tile::phi_channel, Tile::b_channel, -1, LEAF, LAUNCH_SUBTREE, INTERIOR | DIRICHLET | NEUMANN);
    //            CPUTimer<std::chrono::microseconds> timer;
    //            timer.start();
    //            auto [iters, err] = solver.solve(grid, false, 1000, 1e-6, level_iters, coarse_iters, 1, false);
    //            CheckCudaError("VCycleMultigrid done");
    //            float elapsed = timer.stop("MGPCG Async");
    //            int total_cells = grid.numTotalLeafTiles() * Tile::SIZE;
    //            float cells_per_second = (total_cells + 0.0) / (elapsed / 1e6);
    //            Info("MGPCG solved in {} iterations with error {}", iters, err);
    //            Info("Total {:.5}M cells, MGPCG Async  with {} level iters and {} bottom iters speed {:.5} M cells /s", total_cells / (1024.0 * 1024), level_iters, coarse_iters, cells_per_second / (1024.0 * 1024));
    //        }
    //    }
    //}


    {
        CalculateNeighborTiles(grid);

        _sleep(200);
        GMGSolver solver(1.0, 1.0);
        //Copy(grid, Tile::b_channel, Tile::phi_channel, -1, LEAF, LAUNCH_SUBTREE, INTERIOR | DIRICHLET | NEUMANN);
        CPUTimer<std::chrono::microseconds> timer;
        timer.start();
        auto [iters, err] = solver.solve(grid, true, 1000, 1e-6, 1, 10, 1, false);
        CheckCudaError("VCycleMultigrid done");
        float elapsed = timer.stop("MGPCG Async");
        int total_cells = grid.numTotalTiles() * Tile::SIZE;
        float cells_per_second = (total_cells + 0.0) / (elapsed / 1e6);
        Info("Total {:.5}M cells, MGPCG Async speed {:.5} M cells /s", total_cells / (1024.0 * 1024), cells_per_second / (1024.0 * 1024));
        Info("MGPCG solved in {} iterations with error {}, average iteration throughput {:.5}M cell/s", iters, err, cells_per_second* iters / (1024.0 * 1024));
        _sleep(200);

    }


    //{
    //    Info("beginning tile counts: {}", laplacian_total_tile_counts);
    //    int repeat_times = 1;
    //    CPUTimer<std::chrono::microseconds> timer;
    //    timer.start();
    //    for (int i = 0; i < repeat_times; i++) {
    //        VCycleMultigrid(grid, Tile::x_channel, Tile::b_channel, Tile::tmp_channel, Tile::Ap_channel, Tile::D_channel, 1, 10);
    //        //DotAsync(solver.gamma_d, grid, Tile::b_channel, Tile::b_channel, -1, LEAF, LAUNCH_SUBTREE);
    //    }
    //    cudaDeviceSynchronize();
    //    CheckCudaError("VCycleMultigrid done");
    //    //cudaDeviceSynchronize();
    //    auto elapsed = timer.stop("Multigrid") / repeat_times;
    //    int total_cells = grid.numTotalLeafTiles() * Tile::SIZE;
    //    auto cells_per_second = (total_cells + 0.0) / (elapsed / 1e6);
    //    Info("Total {} leaf tiles and {:.5}M cells, Multigrid speed {:.5} M cells /s", grid.numTotalLeafTiles(), total_cells / (1024.0 * 1024), cells_per_second / (1024.0 * 1024));
    //    Info("Laplacian total tile counts: {}", laplacian_total_tile_counts);
    //}

    //_sleep(200);



    Axpy(grid, -1.0, Tile::x_channel, grdt_channel, -1, LEAF, LAUNCH_SUBTREE);
    auto norm2 = VolumeWeightedNorm(grid, 2, grdt_channel, -1, LEAF, LAUNCH_SUBTREE);
    Info("Neumann test error volume weighted L2 norm: {}", norm2);


    //if (debug) {

    //    fs::path output_base_dir = fs::current_path() / fmt::format("neumann_test{}", grid_case);
    //    fs::create_directories(output_base_dir);
    //    IOFunc::OutputTilesAsVTU(grid, output_base_dir / "tiles.vtu");

    //    auto holder = grid.getHostTileHolderForLeafs();
    //    IOFunc::OutputPoissonGridAsUnstructuredVTU(holder, { {Tile::x_channel,"pressure"}, {grdt_channel,"error"}, {Tile::u_channel,"rhs"} }, {}, output_base_dir / "solution.vtu");
    //}
}

void TestAMGNeumannBC(int grid_case, bool debug) {
    Info("AMG Neumann BC test case {}", grid_case);

    uint32_t scale = 8;
    float h = 1.0 / scale;

    //0:8, 1:16, 2:32, 3:64, 4:128, 5:256, 6:512, 7:1024
    HADeviceGrid<Tile> grid(h, { 16,16,16,16,16,16,18,16,16,16 });

    //this bbox of the 0th layer will be refined in the 1st layer
    grid.setTileHost(0, nanovdb::Coord(0, 0, 0), Tile(), LEAF);
    grid.compressHost(false);
    grid.syncHostAndDevice();
    SpawnGhostTiles(grid, false);
    Info("Grid size: {}", grid.numTotalLeafTiles() * Tile::SIZE);

    IterativeRefine(grid, [=]__device__(const HATileAccessor<Tile>&acc, HATileInfo<Tile>&info) { return SolverTestsLevelTarget(acc, info, grid_case); }, false);

    Info("iterative done");

    int total_hash_bytes = grid.hashTableDeviceBytes();
    Info("Total hash table bytes: {}/G", total_hash_bytes / (1024.0 * 1024 * 1024));

    int grdt_channel = Tile::vor_channel;

    auto cell_type = [=]__device__(const HATileAccessor<Tile> &acc, const HATileInfo<Tile> &info, const nanovdb::Coord & l_ijk)->uint8_t {
        bool is_dirichlet = false;
        bool is_neumann = false;
        acc.iterateSameLevelNeighborVoxels(info, l_ijk,
            [&]__device__(const HATileInfo<Tile>&n_info, const Coord & n_l_ijk, const int axis, const int sgn) {
            if (n_info.empty()) {
                if (axis == 0 && sgn == -1) {
                    is_neumann = true;
                }
                else {
                    is_dirichlet = true;
                }
            }
        });
        if (is_neumann) return CellType::NEUMANN;
        else if (is_dirichlet) return CellType::DIRICHLET;
        else return CellType::INTERIOR;
    };
    //solution f(x)
    auto f = [=]__device__(const Vec & pos) {
        return exp(pos[0]) * sin(pos[1]);
    };
    grid.launchVoxelFunc(
        [=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
        auto& tile = info.tile();
        tile.type(l_ijk) = cell_type(acc, info, l_ijk);

    }, -1, LEAF, LAUNCH_SUBTREE
    );
    CalcCellTypesFromLeafs(grid);
    //rhs calculation needs to check the same-level neighbor types
    //that will depend on ghost cells as well
    //so we need to update ghost cells before
    grid.launchVoxelFunc(
        [=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
        auto& tile = info.tile();
        auto pos = acc.cellCenter(info, l_ijk);
        auto h = acc.voxelSize(info);
        tile(Tile::b_channel, l_ijk) = 0;
        tile(grdt_channel, l_ijk) = f(pos);

        if (tile.type(l_ijk) == INTERIOR) {
            //offset b
            T b = 0;
            acc.iterateSameLevelNeighborVoxels(info, l_ijk,
                [&]__device__(const HATileInfo<Tile>&n_info, const Coord & n_l_ijk, const int axis, const int sgn) {
                if (!n_info.empty()) {
                    auto& n_tile = n_info.tile();
                    if (n_tile.type(n_l_ijk) == NEUMANN) {
                        auto n_pos = acc.cellCenter(n_info, n_l_ijk);
                        b -= (f(pos) - f(n_pos)) * h;//it's /h^2 * h^3, laplacian and volume
                    }
                    else if (n_tile.type(n_l_ijk) == DIRICHLET) {
                        auto n_pos = acc.cellCenter(n_info, n_l_ijk);
                        b += f(n_pos) * h;
                    }
                }
            });
            tile(Tile::b_channel, l_ijk) = b;
            //tile(Tile::b_channel, l_ijk) = 1;
        }
        tile(Tile::u_channel, l_ijk) = tile.interiorValue(Tile::b_channel, l_ijk);
    }, -1, LEAF, LAUNCH_SUBTREE
    );


    //Info("b volume weighted linf: {}", VolumeWeightedNorm(grid, -1, Tile::b_channel));
    //Info("b volume weighted norm2: {}", VolumeWeightedNorm(grid, 2, Tile::b_channel));

    //FullNegativeLaplacian(grid, grdt_channel, Tile::tmp_channel);
    //Axpy(grid, -1.0, Tile::b_channel, Tile::tmp_channel, -1, LEAF, LAUNCH_SUBTREE);
    //Info("grdt residual norm2: {}", VolumeWeightedNorm(grid, 2, Tile::tmp_channel));

    //auto b_dot = Dot(grid, Tile::b_channel, Tile::b_channel, -1, LEAF, LAUNCH_SUBTREE);
    //Info("b dot: {}", b_dot);
    //auto grdt_dot = Dot(grid, grdt_channel, grdt_channel, -1, LEAF, LAUNCH_SUBTREE);
    //Info("grdt dot: {}", grdt_dot);



    //
    //{
    //    AMGSolver solver(Tile::tmp_channel);
    //    //grdt channel: 10
    //    //6789: coeffs
    //    solver.prepareTypesAndCoeffs(grid);
    //    CalculateNeighborTiles(grid);


    //    _sleep(200);

    //    int repeat_times = 1;
    //    CPUTimer<std::chrono::microseconds> timer;
    //    timer.start();
    //    for (int i = 0; i < repeat_times; i++) {
    //        NegativeLaplacianAMGSameLevel128(grid, grid.dAllTiles, grid.dAllTiles.size(), LEAF, true, grdt_channel, solver.coeff_channel, Tile::x_channel);

    //        //FullNegativeLaplacian(grid, grdt_channel, Tile::x_channel);
    //        //FullNegativeLaplacianAMG(grid, grdt_channel, Tile::tmp_channel, Tile::b_channel);
    //        //NegativeLaplacianSameLevel(grid, grdt_channel, Tile::tmp_channel, -1, LEAF, LAUNCH_SUBTREE, false);
    //        //NegativeLaplacianSameLevelShared(grid, grdt_channel, Tile::tmp_channel, -1, LEAF, LAUNCH_SUBTREE, false);
    //        //NegativeLaplacianSameLevel128(grid, grid.dAllTiles, grid.dAllTiles.size(), -1, LEAF, grdt_channel, Tile::tmp_channel, false);
    //    }
    //    cudaDeviceSynchronize();

    //    //Copy(grid, Tile::b_channel, Tile::r_channel, -1, LEAF, LAUNCH_SUBTREE, INTERIOR | DIRICHLET | NEUMANN);
    //    //Axpy(grid, -1.0, Tile::x_channel, Tile::r_channel, -1, LEAF, LAUNCH_SUBTREE);
    //    //Info("diff dot: {}", Dot(grid, Tile::r_channel, Tile::r_channel, LEAF));
    //     
    //    //polyscope::init();
    //    //auto holder = grid.getHostTileHolderForLeafs();
    //    //IOFunc::AddPoissonGridCellCentersToPolyscopePointCloud(holder,
    //    //    { {grdt_channel, "x"}, {Tile::x_channel,"MG Lap"}, {Tile::r_channel,"diff"}, {Tile::b_channel,"AMG Lap"} ,
    //    //    {Tile::tmp_channel, "off0"}, {Tile::tmp_channel + 1,"off1"},{Tile::tmp_channel + 2,"off2"},{Tile::tmp_channel + 3,"diag"},
    //    //    {-1,"type"}
    //    //    },
    //    //    {});
    //    //polyscope::show();
    //    //
    //    CheckCudaError("Laplacian Shared done");
    //    float elapsed = timer.stop("Laplacian Shared") / repeat_times;
    //    int total_cells = grid.numTotalLeafTiles() * Tile::SIZE;
    //    float cells_per_second = (total_cells + 0.0) / (elapsed / 1e6);
    //    Info("Repeat {} times, Total {} tiles and {:.5}M cells, Full Laplacian speed {:.5} M cells /s", repeat_times, grid.numTotalLeafTiles(), total_cells / (1024.0 * 1024), cells_per_second / (1024.0 * 1024));
    //}

    //_sleep(200);


    //{
    //    int repeat_times = 1;
    //    CPUTimer<std::chrono::microseconds> timer;
    //    timer.start();
    //    for (int i = 0; i < repeat_times; i++) {
    //        //FullNegativeLaplacian(grid, grdt_channel, Tile::tmp_channel);
    //        NegativeLaplacianSameLevel(grid, grdt_channel, Tile::Ap_channel, -1, LEAF, LAUNCH_SUBTREE, false);
    //        //NegativeLaplacianSameLevelShared(grid, grdt_channel, Tile::tmp_channel, -1, LEAF, LAUNCH_SUBTREE, false);
    //    }
    //    cudaDeviceSynchronize();

    //    CheckCudaError("Full Laplacian done");
    //    float elapsed = timer.stop("Full Laplacian Normal") / repeat_times;
    //    int total_cells = grid.numTotalLeafTiles() * Tile::SIZE;
    //    float cells_per_second = (total_cells + 0.0) / (elapsed / 1e6);
    //    Info("Repeat {} times, Total {:.5}M cells, Full Laplacian speed {:.5} M cells /s", repeat_times, total_cells / (1024.0 * 1024), cells_per_second / (1024.0 * 1024));
    //}

    //Info("grdt dot: {}", Dot(grid, Tile::Ap_channel, Tile::Ap_channel, -1, LEAF, LAUNCH_SUBTREE));
    //Info("shared dot: {}", Dot(grid, Tile::tmp_channel, Tile::tmp_channel, -1, LEAF, LAUNCH_SUBTREE));

 //   polyscope::init();
    //auto holder = grid.getHostTileHolderForLeafs();
 //   IOFunc::AddPoissonGridCellCentersToPolyscopePointCloud(holder, { {grdt_channel, "x"}, {Tile::Ap_channel,"grdt"}, {Tile::tmp_channel,"ours"} }, {});
 //   polyscope::show();

    //_sleep(200);


    //{
    //    CPUTimer<std::chrono::microseconds> timer;
    //    timer.start();
    //    auto [iters, err] = ConjugateGradientSync(grid, true, 1000, 1, 20, 1e-6, false);
    //    CheckCudaError("VCycleMultigrid done");
    //    float elapsed = timer.stop("MGPCG Solve Synced");
    //    int total_cells = grid.numTotalLeafTiles() * Tile::SIZE;
    //    float cells_per_second = (total_cells + 0.0) / (elapsed / 1e6);
    //    Info("Total {:.5}M cells, MGPCG Sync speed {:.5} M cells /s", total_cells / (1024.0 * 1024), cells_per_second / (1024.0 * 1024));
    //    Info("MGPCG solved in {} iterations with error {}", iters, err);
    //}

    //{
    //    Copy(grid, Tile::b_channel, Tile::phi_channel, -1, LEAF, LAUNCH_SUBTREE, INTERIOR | DIRICHLET | NEUMANN);

    //    //parameter study for solver
    //    for (int level_iters : {1, 2, 3, 5, 10, 20,50}) {
    //        for (int coarse_iters : {1, 10, 20, 100}) {
    //            Copy(grid, Tile::phi_channel, Tile::b_channel, -1, LEAF, LAUNCH_SUBTREE, INTERIOR | DIRICHLET | NEUMANN);
    //            CPUTimer<std::chrono::microseconds> timer;
    //            timer.start();
    //            auto [iters, err] = solver.solve(grid, false, 1000, 1e-6, level_iters, coarse_iters, 1, false);
    //            CheckCudaError("VCycleMultigrid done");
    //            float elapsed = timer.stop("MGPCG Async");
    //            int total_cells = grid.numTotalLeafTiles() * Tile::SIZE;
    //            float cells_per_second = (total_cells + 0.0) / (elapsed / 1e6);
    //            Info("MGPCG solved in {} iterations with error {}", iters, err);
    //            Info("Total {:.5}M cells, MGPCG Async  with {} level iters and {} bottom iters speed {:.5} M cells /s", total_cells / (1024.0 * 1024), level_iters, coarse_iters, cells_per_second / (1024.0 * 1024));
    //        }
    //    }
    //}

    

    {
        CalculateNeighborTiles(grid);

        _sleep(200);
        //GMGSolver solver;
        AMGSolver solver(Tile::u_channel, 1., 1.);
		solver.prepareTypesAndCoeffs(grid);

        //auto holder = grid.getHostTileHolder(LEAF | NONLEAF);
        //polyscope::init();
        //IOFunc::AddLeveledPoissonGridCellCentersToPolyscopePointCloud(holder,
        //    {
        //    {solver.coeff_channel,"offd0"}, {solver.coeff_channel + 1,"offd1"} ,{solver.coeff_channel + 2,"offd2"},{solver.coeff_channel + 3,"diag"}, {-1,"type"}
        //    },
        //    {}, FLT_MAX);
        //polyscope::show();

        //Copy(grid, Tile::b_channel, Tile::phi_channel, -1, LEAF, LAUNCH_SUBTREE, INTERIOR | DIRICHLET | NEUMANN);
        CPUTimer<std::chrono::microseconds> timer;
        timer.start();
        auto [iters, err] = solver.solve(grid, true, 1000, 1e-6, 1, 10, 1, false);
        CheckCudaError("VCycleMultigrid done");
        float elapsed = timer.stop("MGPCG Async");
        int total_cells = grid.numTotalLeafTiles() * Tile::SIZE;
        float cells_per_second = (total_cells + 0.0) / (elapsed / 1e6);
        Info("Total {:.5}M cells, MGPCG Async speed {:.5} M cells /s", total_cells / (1024.0 * 1024), cells_per_second / (1024.0 * 1024));
        Info("MGPCG solved in {} iterations with error {}, average iteration throughput {:.5}M cell/s", iters, err, cells_per_second * iters / (1024.0 * 1024));
    }
    _sleep(200);


  //  {
  //      CalculateNeighborTiles(grid);
  //      //GMGSolver solver;
  //      AMGSolver solver(Tile::u_channel);
  //      //CalculateAMGCoefficients(grid, solver.coeff_channel, LEAF | GHOST);
  //      solver.prepareTypesAndCoeffs(grid);

  //      _sleep(200);


  //      Fill(grid, Tile::x_channel, 0, -1, LEAF | GHOST | NONLEAF, LAUNCH_SUBTREE, INTERIOR | NEUMANN | DIRICHLET);
		//Copy(grid, Tile::x_channel, Tile::Ap_channel, -1, LEAF, LAUNCH_SUBTREE, INTERIOR | DIRICHLET | NEUMANN);
  //      GaussSeidelAMG(1, 0, grid, grid.mMaxLevel, Tile::x_channel, solver.coeff_channel, Tile::b_channel);

		//int level = grid.mMaxLevel;
  //      NegativeLaplacianSameLevel128(grid, grid.dTileArrays[level], grid.hNumTiles[level], level, LEAF | NONLEAF, Tile::Ap_channel, Tile::D_channel, true);//diagonal
  //      GaussSeidelSingle(0, grid, level, Tile::Ap_channel, Tile::tmp_channel, Tile::b_channel, Tile::D_channel);
  //      GaussSeidelSingle(1, grid, level, Tile::Ap_channel, Tile::tmp_channel, Tile::b_channel, Tile::D_channel);
  //      //GaussSeidelSingle(level_iters, 0, grid, i, x_channel, tmp_channel, rhs_channel, D_channel);

		//Info("updated AMG x dot: {}", Dot(grid, Tile::x_channel, Tile::x_channel, LEAF));
  //      Info("Updated MG x dot: {}", Dot(grid, Tile::Ap_channel, Tile::Ap_channel, LEAF));

  //      polyscope::init();
  //      auto holder = grid.getHostTileHolderForLeafs();
  //      IOFunc::AddPoissonGridCellCentersToPolyscopePointCloud(holder,
  //          { {Tile::x_channel,"amg_x"}, {Tile::Ap_channel,"mg_x"}, {Tile::b_channel,"rhs"} ,
  //          {-1,"type"}
  //          },
  //          {});
  //      polyscope::show();

  //  }

    //{
    //    Info("beginning tile counts: {}", laplacian_total_tile_counts);
    //    int repeat_times = 1;
    //    CPUTimer<std::chrono::microseconds> timer;
    //    timer.start();
    //    for (int i = 0; i < repeat_times; i++) {
    //        VCycleMultigrid(grid, Tile::x_channel, Tile::b_channel, Tile::tmp_channel, Tile::Ap_channel, Tile::D_channel, 1, 10);
    //        //DotAsync(solver.gamma_d, grid, Tile::b_channel, Tile::b_channel, -1, LEAF, LAUNCH_SUBTREE);
    //    }
    //    cudaDeviceSynchronize();
    //    CheckCudaError("VCycleMultigrid done");
    //    //cudaDeviceSynchronize();
    //    auto elapsed = timer.stop("Multigrid") / repeat_times;
    //    int total_cells = grid.numTotalLeafTiles() * Tile::SIZE;
    //    auto cells_per_second = (total_cells + 0.0) / (elapsed / 1e6);
    //    Info("Total {} leaf tiles and {:.5}M cells, Multigrid speed {:.5} M cells /s", grid.numTotalLeafTiles(), total_cells / (1024.0 * 1024), cells_per_second / (1024.0 * 1024));
    //    Info("Laplacian total tile counts: {}", laplacian_total_tile_counts);
    //}

    //_sleep(200);



    //Axpy(grid, -1.0, Tile::x_channel, grdt_channel, -1, LEAF, LAUNCH_SUBTREE);
    //auto norm2 = VolumeWeightedNorm(grid, 2, grdt_channel, -1, LEAF, LAUNCH_SUBTREE);
    //Info("Neumann test error volume weighted L2 norm: {}", norm2);


    //if (debug) {

    //    fs::path output_base_dir = fs::current_path() / fmt::format("neumann_test{}", grid_case);
    //    fs::create_directories(output_base_dir);
    //    IOFunc::OutputTilesAsVTU(grid, output_base_dir / "tiles.vtu");

    //    auto holder = grid.getHostTileHolderForLeafs();
    //    IOFunc::OutputPoissonGridAsUnstructuredVTU(holder, { {Tile::x_channel,"pressure"}, {grdt_channel,"error"}, {Tile::u_channel,"rhs"} }, {}, output_base_dir / "solution.vtu");
    //}
}