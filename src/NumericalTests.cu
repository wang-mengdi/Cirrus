#include "NumericalTests.h"
#include "PoissonSolver.h"
#include "PoissonIOFunc.h"
#include "FMParticles.h"
#include "Common.h"
#include "GPUTimer.h"
#include "Random.h"
#include <polyscope/polyscope.h>
#include <polyscope/point_cloud.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

//case0: regular 128^3 lattice
//case1: fine on left and coarse on right
//case2: two sources
__device__ int NumericalTestsLevelTarget(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const int grid_case) {
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

void TestIOTime(const int grid_case) {
    uint32_t scale = 8;
    float h = 1.0 / scale;

    //0:8, 1:16, 2:32, 3:64, 4:128, 5:256, 6:512, 7:1024
    HADeviceGrid<Tile> grid(h, { 16,16,16,16,16,16,16,16,16,16 });

    grid.setTileHost(0, nanovdb::Coord(0, 0, 0), Tile(), LEAF);
    grid.compressHost();
    grid.syncHostAndDevice();
    SpawnGhostTiles(grid);

    IterativeRefine(grid, [=]__device__(const HATileAccessor<Tile>&acc, HATileInfo<Tile>&info) { return NumericalTestsLevelTarget(acc, info, grid_case); });

    grid.launchVoxelFuncOnAllTiles(
        [=] __device__(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk) {
        auto& tile = info.tile();
        tile.type(l_ijk) = INTERIOR;
        tile(Tile::x_channel, l_ijk) = l_ijk[0];
    }, LEAF, 1
    );

    CPUTimer<std::chrono::microseconds> timer;
    timer.start();
	auto holder = grid.getHostTileHolderForLeafs();
    auto base_dir = fs::current_path() / "data" / fmt::format("io_test{}", grid_case);
	fs::create_directories(base_dir);
    //IOFunc::OutputPoissonGridAsUnstructuredVTU(holder, { {Tile::x_channel, "x"} }, {}, base_dir / "fluid.vtu");
    IOFunc::OutputPoissonGridAsStructuredVTI(holder, { {Tile::x_channel, "x"} }, {}, base_dir / "fluid.vti");
    float elapsed = timer.stop("Output 1 channel");
    int total_cells = grid.numTotalLeafTiles() * Tile::SIZE;
    float cells_per_second = (total_cells + 0.0) / (elapsed / 1e6);
    Info("Total {:.5}M cells, output speed {:.5} M cells /s", total_cells / (1024.0 * 1024), cells_per_second / (1024.0 * 1024));
}

void TestMemoryUsage(void) {
    uint32_t scale = 8;
    float h = 1.0 / scale;

    HADeviceGrid<Tile> grid(h, { 16,16,16,16,16,16,16,20,22,16 });

    //for log2 hash size { 16,16,16,16,16,16,16,20,16,16 } and max 7 levels:
    //#     Hash table memory usage : 50.000MB
    //#     Total tiles : 0.094M and 48.627 % of them are leaf tiles
    //#     Tiles memory usage : 2.759GB
    //there is a redundant tmp_channel therefore the minimal memory usage is about 3.3GB

    //this bbox of the 0th layer will be refined in the 1st layer
    grid.setTileHost(0, nanovdb::Coord(0, 0, 0), Tile(), LEAF);
    grid.compressHost();
    grid.syncHostAndDevice();
    SpawnGhostTiles(grid);

    auto levelTarget = [=]__device__(const HATileAccessor<Tile> &acc, HATileInfo<Tile> &info) ->int {
        auto bbox = acc.tileBBox(info);
        //7 for 1024
        if (bbox.min()[1] <= 0.5 && 0.5 <= bbox.max()[1]) return 8;
        else return 0;
    };
    IterativeRefine(grid, levelTarget);

	double hash_bytes = grid.hashTableDeviceBytes();
    Info("Hash table memory usage: {:.3f}MB", hash_bytes / (1024 * 1024));
	double num_leaf_tiles = grid.numTotalLeafTiles();
    double num_total_tiles = grid.numTotalTiles();
    Info("Total tiles: {:.3f}M and {:.3f}% of them are leaf tiles", num_total_tiles / (1024 * 1024), num_leaf_tiles / num_total_tiles * 100.);
	double single_tile_bytes = sizeof(Tile);
	double total_tile_bytes = num_total_tiles * sizeof(Tile);
    Info("Single tile use {:.3f}M bytes and total tiles memory usage: {:.3f}GB", single_tile_bytes / (1024 * 1024), total_tile_bytes / (1024 * 1024 * 1024));

    double tile_cnt = 0;
    for (int i = 0; i <= grid.mMaxLevel; i++) {
		double level_cnt = grid.hNumTiles[i];
        tile_cnt += level_cnt;

        int repeat_x = (1 << (7 - i));
        int repeat_num = repeat_x * repeat_x;

        Info("If use levels 0...{} to construct 1024^2, we need total {:.3f}M tiles using {:.3f}GB and {:.3f}% of them are finest level tiles.", i, tile_cnt* repeat_num / (1024 * 1024), tile_cnt* repeat_num * sizeof(Tile) / (1024 * 1024 * 1024), level_cnt / tile_cnt * 100.);

    }

	auto holder = grid.getHostTileHolderForLeafs();
    auto base_dir = fs::current_path() / fmt::format("memory_test");
    fs::create_directories(base_dir);
    IOFunc::OutputTilesAsVTU(holder, base_dir / "tiles.vtu");
}

void TestRefine(void) {
    uint32_t scale = 8;
    float h = 1.0 / scale;

    HADeviceGrid<Tile> grid(h, { 16,16,16,16,16,16,16,16,16,16 });

    //this bbox of the 0th layer will be refined in the 1st layer
    grid.setTileHost(0, nanovdb::Coord(0, 0, 0), Tile(), LEAF);
    grid.compressHost();
    grid.syncHostAndDevice();
    SpawnGhostTiles(grid);

    auto levelTarget = [=]__device__(const HATileAccessor<Tile> &acc, HATileInfo<Tile> &info) ->int {
        // return 5;
        const Vec pointSrc1(0.51, 0.49, 0.54);
        const Vec pointSrc2(0.93, 0.08, 0.91);
        //const Vec pointSrc1(0.01, 0.01, 0.01);
        auto bbox = acc.tileBBox(info);
        int desired_level = 0;
        if (bbox.isInside(pointSrc2)) desired_level = 6;
        if (bbox.isInside(pointSrc1)) desired_level = 7;
        return desired_level;
    };
    IterativeRefine(grid, levelTarget);


    //these numbers are acquired from dasgrid_octree
    std::vector<int> target_tiles_nums({ 1, 8, 64, 120, 120, 96, 40, 8, 0, 0 });
    for (int i = 0; i < target_tiles_nums.size(); i++) {
        int cnt = 0;
        for (int j = 0; j < grid.hNumTiles[i]; j++) {
            if (grid.hTileArrays[i][j].isActive()) cnt++;
        }
        Assert(cnt == target_tiles_nums[i], "Level {} expected {} active tiles but only get {}", i, target_tiles_nums[i], cnt);
    }
        
    Pass("Correct number of active tiles: {}", target_tiles_nums);
}

void WriteParticlesToVTU(const thrust::host_vector<Vec>& pos, const thrust::host_vector<double>& scalar, std::string file_name) {
    //output particles with a scalar quantity
    //setup VTK
    vtkNew<vtkXMLUnstructuredGridWriter> writer;
    vtkNew<vtkUnstructuredGrid> unstructured_grid;

    vtkNew<vtkPoints> nodes;
    nodes->Allocate(pos.size());
    vtkNew<vtkDoubleArray> scalarArray;
    scalarArray->SetName("Value");
    scalarArray->SetNumberOfComponents(1);
    scalarArray->Allocate(scalar.size());

    for (int i = 0; i < pos.size(); i++) {
        auto p = pos[i];
        nodes->InsertNextPoint(p[0], p[1], p[2]);
        scalarArray->InsertNextTuple1(scalar[i]);
    }

    unstructured_grid->SetPoints(nodes);
    unstructured_grid->GetPointData()->AddArray(scalarArray);

    writer->SetFileName(file_name.c_str());
    writer->SetInputData(unstructured_grid);
    writer->Write();
}



Vec UnitVector(const double rad) {
    return Vec(cos(rad), sin(rad), 0);
}

void TestAdvection(const int grid_case) {

    uint32_t scale = 8;
    float h = 1.0 / scale;
    int refine_levels = 4;

    //0:8, 1:16, 2:32, 3:64, 4:128, 5:256, 6:512, 7:1024
    HADeviceGrid<Tile> grid(h, { 16,16,16,16,16,16,16,16,16,16 });

    grid.setTileHost(0, nanovdb::Coord(0, 0, 0), Tile(), LEAF);
    grid.compressHost();
    grid.syncHostAndDevice();
    SpawnGhostTiles(grid);

    IterativeRefine(grid, [=]__device__(const HATileAccessor<Tile>&acc, HATileInfo<Tile>&info) { return NumericalTestsLevelTarget(acc, info, grid_case); });
    grid.launchVoxelFunc(
        [=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
        auto& tile = info.tile();
        tile.type(l_ijk) = INTERIOR;
    }, -1, LEAF, LAUNCH_SUBTREE
    );
    CalcCellTypesFromLeafs(grid);

    //we use the 3D deformation test from: Unstructured un-split geometrical Volume-of-Fluid methods – A review

    double T0 = 8;
    using CommonConstants::pi;
    auto vel_func = [=] __device__(const Vec pos, const double t)->Vec {
        //return Vec(1, 0, 0);
        double t_over_T = t / T0;
        double x = pos[0], y = pos[1];
        double u = -2 * cos(pi * t_over_T) * cos(pi * y) * pow(sin(pi * x), 2) * sin(pi * y);
        double v = 2 * cos(pi * t_over_T) * cos(pi * x) * sin(pi * x) * pow(sin(pi * y), 2);
        return Vec(u, v, 0);
    };

  //  double T0 = 3;
  //  using CommonConstants::pi;
  //  auto vel_func = [=] __device__(const Vec pos, const double t)->Vec {
  //      double x = pos[0], y = pos[1], z = pos[2];
  //      double u = 2 * sin(pi * x) * sin(pi * x) * sin(2 * pi * y) * sin(2 * pi * z) * cos(pi * t / T0);
		//double v = -sin(2 * pi * x) * sin(pi * y) * sin(pi * y) * sin(2 * pi * z) * cos(pi * t / T0);
  //      double w = -sin(2 * pi * x) * sin(2 * pi * y) * sin(pi * z) * sin(pi * z) * cos(pi * t / T0);
  //      return Vec(u, v, w);
  //  };

    Vec center(0.5, 0.75, 0.5); double radius = 0.15;
    thrust::host_vector<Vec> verts_h;
    auto push = [&](thrust::host_vector<Vec>& v, const Vec& p) {v.push_back(p); };
    int vn = 1000;
    for (int i = 0; i < vn; i++) push(verts_h, center + UnitVector(2 * pi / vn * i) * radius);

    //Vec center(0.35, 0.35, 0.35); double radius = 0.15;
    //RandomGenerator rng;
    //int vn = 100000;
    //thrust::host_vector<Vec> verts_h;
    //for (int i = 0; i < vn; ++i) {
    //    double theta = rng.uniform(0, 2 * pi); // azimuthal angle in [0, 2*pi]
    //    double phi = rng.uniform(0, pi);     // polar angle in [0, pi]

    //    double x = radius * sin(phi) * cos(theta);
    //    double y = radius * sin(phi) * sin(theta);
    //    double z = radius * cos(phi);

    //    auto point = center + Vec(x, y, z);
    //    verts_h.push_back(point);
    //}

    auto rk4_forward = [=]__device__(const HATileAccessor<Tile>&acc, const Vec & pos, const double dt) {
        double c1 = 1.0 / 6.0 * dt, c2 = 1.0 / 3.0 * dt, c3 = 1.0 / 3.0 * dt, c4 = 1.0 / 6.0 * dt;
        Vec vel1 = InterpolateFaceValue(acc, pos, Tile::u_channel, 0);
        Vec pos1 = pos + vel1 * 0.5 * dt;
        Vec vel2 = InterpolateFaceValue(acc, pos1, Tile::u_channel, 0);
        Vec pos2 = pos + vel2 * 0.5 * dt;
        Vec vel3 = InterpolateFaceValue(acc, pos2, Tile::u_channel, 0);
        Vec pos3 = pos + vel3 * dt;
        Vec vel4 = InterpolateFaceValue(acc, pos3, Tile::u_channel, 0);
        return pos + c1 * vel1 + c2 * vel2 + c3 * vel3 + c4 * vel4;
    };

    auto rk4_forward_kernel = [=]__device__(const HATileAccessor<Tile>&acc, const Vec & pos, const double dt) {
        double c1 = 1.0 / 6.0 * dt, c2 = 1.0 / 3.0 * dt, c3 = 1.0 / 3.0 * dt, c4 = 1.0 / 6.0 * dt;
        Vec vel1; Eigen::Matrix3<T> _;
        VelocityAndJacobian(acc, pos, 0, vel1, _);
        Vec pos1 = pos + vel1 * 0.5 * dt;
        Vec vel2;
		VelocityAndJacobian(acc, pos1, 0, vel2, _);
        Vec pos2 = pos + vel2 * 0.5 * dt;
        Vec vel3;
		VelocityAndJacobian(acc, pos2, 0, vel3, _);
        Vec pos3 = pos + vel3 * dt;
        Vec vel4;
		VelocityAndJacobian(acc, pos3, 0, vel4, _);
        return pos + c1 * vel1 + c2 * vel2 + c3 * vel3 + c4 * vel4;
    };

    auto base_dir = fs::current_path() / "data" / fmt::format("advection_test{}", grid_case);
    fs::create_directories(base_dir);
    //IOFunc::OutputTilesAsVTU(grid, base_dir / "tiles.vtu");

    double cfl = 0.5;//max vel=1
    double dt = h / (1 << refine_levels) * cfl;
    int N = ceil(T0 / dt);
    double t = 0;
    thrust::device_vector<Vec> verts = verts_h;
    thrust::device_vector<Vec> velocity = verts;
    for (int i = 0; i <= N; i++) {


        grid.launchVoxelFunc(
            [=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
            auto& tile = info.tile();
            for (int axis = 0; axis < 3; axis++) {
                auto face_ctr = acc.faceCenter(axis, info, l_ijk);
                tile(Tile::u_channel + axis, l_ijk) = vel_func(face_ctr, t)[axis];
            }
        }, -1, LEAF, LAUNCH_SUBTREE
        );



        //for (int axis : {0, 1, 2}) {
        //    //PropagateValues(grid, Tile::u_channel + axis, Tile::u_channel + axis, -1, GHOST, LAUNCH_SUBTREE);
        //    PropagateToChildren(grid, Tile::u_channel + axis, Tile::u_channel + axis, -1, GHOST, LAUNCH_SUBTREE, INTERIOR | DIRICHLET | NEUMANN);
        //}
        CalcLeafNodeValuesFromFaceCenters(grid, Tile::u_channel, 0);

        auto t_acc = grid.deviceAccessor();
        Info("frame {}/{}", i, N);
        thrust::transform(verts.begin(), verts.end(), velocity.begin(), [=]__device__(const Vec& p) {
            return InterpolateFaceValue(t_acc, p, Tile::u_channel, 0);
        });
        //IOFunc::OutputParticlesAndVelocityAsVTU(verts, velocity, base_dir / fmt::format("particles_{:04d}.vtu", i));


        
        thrust::transform(verts.begin(), verts.end(), verts.begin(), [=]__device__(const Vec & p) {
            //return rk4_forward(t_acc, p, dt);
            return rk4_forward_kernel(t_acc, p, dt);
        });
        thrust::copy(verts.begin(), verts.end(), verts_h.begin());

        t += dt;

    }
}


void TestReduction(void) {
    const int N = 100; // 数组长度
    float h_data[N];  // 主机数据
    double h_result;   // 主机结果

    // 初始化主机数据
    for (int i = 0; i < N; ++i) {
        h_data[i] = static_cast<float>(i + 1); // 1, 2, ..., N
    }

    // 在设备上分配内存
    float* d_data;
    cudaMalloc((void**)&d_data, N * sizeof(float));

    // 将数据从主机复制到设备
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    // 计算结果所需的存储
    double* d_result; // 设备结果
    cudaMalloc((void**)&d_result, sizeof(double));

    // CUB 相关的临时存储
    void* d_temp_storage = nullptr; // 设备临时存储指针
    size_t temp_storage_bytes = 0;  // 存储大小

    // 第一次调用，计算所需的临时存储空间
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_data, d_result, N);
	printf("Temporary storage size: %lu bytes\n", temp_storage_bytes);

    // 分配临时存储
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // 第二次调用，实际执行求和
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_data, d_result, N);

    // 将结果从设备复制到主机
    cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

    // 打印结果
    std::cout << "Sum: " << h_result << std::endl; // 输出结果

    // 释放设备内存
    cudaFree(d_data);
    cudaFree(d_result);
    cudaFree(d_temp_storage);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
}

void TestReductionSize(void) {
    const int N = 1000;
    float* d_input;
    double* d_output;

    // 在设备上分配内存
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, sizeof(double));

    // 计算 Sum 的临时存储大小
    size_t temp_storage_bytes_sum = 0;
    cub::DeviceReduce::Sum(nullptr, temp_storage_bytes_sum, d_input, d_output, N);
    std::cout << "Temporary storage size for Sum: " << temp_storage_bytes_sum << " bytes" << std::endl;

    // 计算 Max 的临时存储大小
    size_t temp_storage_bytes_max = 0;
    cub::DeviceReduce::Max(nullptr, temp_storage_bytes_max, d_input, d_output, N);
    std::cout << "Temporary storage size for Max: " << temp_storage_bytes_max << " bytes" << std::endl;

    // 计算 Min 的临时存储大小
    size_t temp_storage_bytes_min = 0;
    cub::DeviceReduce::Min(nullptr, temp_storage_bytes_min, d_input, d_output, N);
    std::cout << "Temporary storage size for Min: " << temp_storage_bytes_min << " bytes" << std::endl;

    // 释放设备内存
    cudaFree(d_input);
    cudaFree(d_output);

}

void TestDeviceReducer(void) {
    size_t n = 100;
    double h_data[100];

    // 初始化数据 1...100
    for (int i = 0; i < n; ++i) {
        h_data[i] = static_cast<double>(i + 1);
    }

    // 创建一个 DeviceReducer<double> 实例
    DeviceReducer<double> reducer;
    reducer.resize(0);
    reducer.resize(1);
    reducer.resize(2);
    reducer.resize(10);
    reducer.resize(n);

    // 将数据传输到设备
    cudaMemcpy(reducer.data(), h_data, n * sizeof(double), cudaMemcpyHostToDevice);
    CheckCudaError("main: cudaMemcpy to device");

    // 分配一个设备上的 double 变量用于存储求和结果
    double* d_sum_result;
    cudaMalloc((void**)&d_sum_result, sizeof(double));

    // 调用 sumAsyncTo 进行求和
    reducer.sumAsyncTo(d_sum_result);
    reducer.printData();

    // 将结果从设备传回主机
    double h_sum_result;
    cudaMemcpy(&h_sum_result, d_sum_result, sizeof(double), cudaMemcpyDeviceToHost);
    CheckCudaError("main: cudaMemcpy from device");

    // 打印求和结果
    std::cout << "Sum result of 1 to 100: " << h_sum_result << std::endl;

    // 验证结果是否为 5050
    if (h_sum_result == 5050) {
        std::cout << "Test passed: the sum is correct." << std::endl;
    }
    else {
        std::cout << "Test failed: the sum is incorrect." << std::endl;
    }

    // 清理
    cudaFree(d_sum_result);
}



void TestVelocityExtrapolation(const int grid_case) {
    uint32_t scale = 8;
    float h = 1.0 / scale;

    //0:8, 1:16, 2:32, 3:64, 4:128, 5:256, 6:512, 7:1024
    HADeviceGrid<Tile> grid(h, { 16,16,16,16,16,16,16,16,16,16 });

    grid.setTileHost(0, nanovdb::Coord(0, 0, 0), Tile(), LEAF);
    grid.compressHost();
    grid.syncHostAndDevice();
    SpawnGhostTiles(grid);

    IterativeRefine(grid, [=]__device__(const HATileAccessor<Tile>&acc, HATileInfo<Tile>&info) { return NumericalTestsLevelTarget(acc, info, grid_case); });
    auto cellType = [=]__device__(const HATileAccessor<Tile>&acc, const HATileInfo<Tile>&info, const nanovdb::Coord & l_ijk) {
        bool is_neumann = false;
        bool is_dirichlet = false;
        acc.iterateSameLevelNeighborVoxels(info, l_ijk, [&]__device__(const HATileInfo<Tile>&n_info, const Coord & n_l_ijk, const int axis, const int sgn) {
            if (n_info.empty()) {
                //is_dirichlet = true;
                if ((axis == 1 && sgn == 1)) is_dirichlet = true;
                else is_neumann = true;
            }
        });

        //nanovdb::BBox<Vec> bbox(Vec(0.4, 0.7, 0.4), Vec(0.6, 0.9, 0.6));
        nanovdb::BBox<Vec> bbox(Vec(0.4, 0.7, 0.0), Vec(0.6, 0.9, 1.0));
        auto pos = acc.cellCenter(info, l_ijk);
        if (!bbox.isInside(pos)) is_dirichlet = true;


        if (is_neumann) return NEUMANN;
        if (is_dirichlet) return DIRICHLET;
        return INTERIOR;
    };
    auto vec_func = [=]__hostdev__(const Vec pos) ->Vec {
        return Vec(0, -1, 0);
    };
    grid.launchVoxelFunc(
        [=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
        auto& tile = info.tile();

        tile.type(l_ijk) = cellType(acc, info, l_ijk);
        for (int axis : {0, 1, 2}) {
            auto face_pos = acc.faceCenter(axis, info, l_ijk);
            tile(Tile::u_channel + axis, l_ijk) = vec_func(face_pos)[axis];
        }
    }, -1, LEAF, LAUNCH_SUBTREE
    );
    CalcCellTypesFromLeafs(grid);

    //apply boundary condition
    grid.launchVoxelFunc(
        [=] __device__(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk) {
        for (int axis : {0, 1, 2}) {
            auto type01 = FaceNeighborCellTypes(acc, info, l_ijk, axis);
            auto type0 = thrust::get<0>(type01);
            auto type1 = thrust::get<1>(type01);
            if (type0 == NEUMANN || type1 == NEUMANN) {
                info.tile()(Tile::u_channel + axis, l_ijk) = 0;
            }
            if (type0 == DIRICHLET && type1 == DIRICHLET) {
                info.tile()(Tile::u_channel + axis, l_ijk) = 0;
            }
        }
    }, -1, LEAF, LAUNCH_SUBTREE
    );

    
    ExtrapolateVelocity(grid, Tile::u_channel, 5);

    //CalcLeafNodeValuesFromFaceCenters(grid, Tile::u_channel, 0);

    polyscope::init();
    auto holder = grid.getHostTileHolderForLeafs();
    IOFunc::AddPoissonGridCellCentersToPolyscopePointCloud(holder, { {-1,"type"} }, {});
    //IOFunc::AddPoissonGridNodesToPolyscope(holder, {  }, { {0,"node vec"} });
    IOFunc::AddPoissonGridFaceCentersToPolyscopePointCloud(holder, { {Tile::u_channel,"velocity"} });
    polyscope::show();
}

//void TestVorticityCalculation(const int grid_case) {
//    uint32_t scale = 8;
//    float h = 1.0 / scale;
//
//    //0:8, 1:16, 2:32, 3:64, 4:128, 5:256, 6:512, 7:1024
//    HADeviceGrid<Tile> grid(h, { 16,16,16,16,16,16,16,16,16,16 });
//
//    grid.setTileHost(0, nanovdb::Coord(0, 0, 0), Tile(), LEAF);
//    grid.compressHost();
//    grid.syncHostAndDevice();
//    SpawnGhostTiles(grid);
//
//    IterativeRefine(grid, [=]__device__(const HATileAccessor<Tile>&acc, HATileInfo<Tile>&info) { return NumericalTestsLevelTarget(acc, info, grid_case); });
//    auto cellType = [=]__device__(const HATileAccessor<Tile>&acc, const HATileInfo<Tile>&info, const nanovdb::Coord & l_ijk) {
//        bool is_neumann = false;
//        bool is_dirichlet = false;
//        acc.iterateSameLevelNeighborVoxels(info, l_ijk, [&]__device__(const HATileInfo<Tile>&n_info, const Coord & n_l_ijk, const int axis, const int sgn) {
//            if (n_info.empty()) {
//                //is_dirichlet = true;
//                if ((axis == 2 && sgn == 1)) is_dirichlet = true;
//                else is_neumann = true;
//            }
//        });
//
//        if (is_neumann) return NEUMANN;
//        if (is_dirichlet) return DIRICHLET;
//        return INTERIOR;
//    };
//	Vec omega = Vec(0, 0, 1);
//    auto vec_func = [=]__hostdev__(const Vec pos) ->Vec {
//        return omega.cross(pos);
//    };
//    grid.launchVoxelFunc(
//        [=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
//        auto& tile = info.tile();
//
//        tile.type(l_ijk) = cellType(acc, info, l_ijk);
//        for (int axis : {0, 1, 2}) {
//            auto face_pos = acc.faceCenter(axis, info, l_ijk);
//            tile(Tile::u_channel + axis, l_ijk) = vec_func(face_pos)[axis];
//        }
//    }, -1, LEAF, LAUNCH_SUBTREE
//    );
//    CalcCellTypesFromLeafs(grid);
//
//    CalculateVorticityMagnitudeOnLeafs(grid, Tile::u_channel, 0, Tile::vor_channel);
//
//    //CalcLeafNodeValuesFromFaceCenters(grid, Tile::u_channel, 0);
//
//    polyscope::init();
//    auto holder = grid.getHostTileHolderForLeafs();
//    IOFunc::AddPoissonGridCellCentersToPolyscopePointCloud(holder, { {-1,"type"},{Tile::vor_channel,"vorticity"} }, {});
//    IOFunc::AddPoissonGridNodesToPolyscope(holder, {  }, { {0,"node vec"} });
//    //IOFunc::AddPoissonGridFaceCentersToPolyscopePointCloud(holder, { {Tile::u_channel,"velocity"} });
//    polyscope::show();
//}

double LinfErrorBetweenPointCloud(const thrust::host_vector<Vec>& pc1, const thrust::host_vector<Vec>& pc2) {
    Assert(pc1.size() == pc2.size(), "Two point clouds should have the same size.");

    double max_error = -1;
    int max_idx = -1;

    for (int i = 0; i < pc1.size(); ++i) {
        double error = (pc1[i] - pc2[i]).length(); // Compute distance (norm)
        if (error > max_error || max_idx == -1) {
            max_error = error;
            max_idx = i;
        }
    }

    Info("Max error at index {} with value {}, pc1 {} pc2 {}", max_idx, max_error, pc1[max_idx], pc2[max_idx]);

    return max_error;
}

double LinfErrorBetweenTargetMatrixForbenius2(const thrust::host_vector<Eigen::Matrix3<T>>& mats, const Eigen::Matrix3<T>& target) {
    double max_error = 0.0;
	int max_idx = -1;

    for (size_t i = 0; i < mats.size(); ++i) {
        double error = (mats[i] - target).norm(); // Frobenius norm in Eigen
		if (error > max_error) {
			max_error = error;
			max_idx = i;
		}
    }
	//Info("Max error at index {} with value {}", max_idx, max_error);

    return max_error;
}

double LinfErrorBetweenMatrixForbenius2(const thrust::host_vector<Eigen::Matrix3<T>>& mats1, const thrust::host_vector<Eigen::Matrix3<T>>& mats2) {
	Assert(mats1.size() == mats2.size(), "Two matrix arrays should have the same size.");
    double max_error = 0.0;

    for (size_t i = 0; i < mats1.size(); ++i) {
        double error = (mats1[i] - mats2[i]).norm(); // Frobenius norm in Eigen
        max_error = std::max(max_error, error);
    }

    return max_error;
}

template<class Func>
__hostdev__ Eigen::Matrix3<T> JacobianOfVectorField(Func vec_func, const Vec& pos, const double h) {
    Eigen::Matrix3<T> jacobian;
    for (int axis = 0; axis < 3; ++axis) {
        Vec pos_forward = pos;
        Vec pos_backward = pos;

        pos_forward[axis] += h;
        pos_backward[axis] -= h;

        Vec vel_forward = vec_func(pos_forward);
        Vec vel_backward = vec_func(pos_backward);

        jacobian(0, axis) = (vel_forward[0] - vel_backward[0]) / (2 * h); // du/dx, du/dy, du/dz
        jacobian(1, axis) = (vel_forward[1] - vel_backward[1]) / (2 * h); // dv/dx, dv/dy, dv/dz
        jacobian(2, axis) = (vel_forward[2] - vel_backward[2]) / (2 * h); // dw/dx, dw/dy, dw/dz
    }

    return jacobian;
}

class Deformation3D {
public:
    //3D deformation test from: Unstructured un-split geometrical Volume-of-Fluid methods – A review
    double T0 = 3;
    double pi = CommonConstants::pi;
    __hostdev__ Vec operator()(const Vec& pos, const double t) const {
        double x = pos[0], y = pos[1], z = pos[2];
        double u = 2 * sin(pi * x) * sin(pi * x) * sin(2 * pi * y) * sin(2 * pi * z) * cos(pi * t / T0);
        double v = -sin(2 * pi * x) * sin(pi * y) * sin(pi * y) * sin(2 * pi * z) * cos(pi * t / T0);
        double w = -sin(2 * pi * x) * sin(2 * pi * y) * sin(pi * z) * sin(pi * z) * cos(pi * t / T0);
        return Vec(u, v, w);
    };
    __hostdev__ Eigen::Matrix3<T> gradu(const Vec& pos, const double t)const {
        double x = pos[0], y = pos[1], z = pos[2];
        double cos_pi_t_T0 = cos(pi * t / T0);
        double sin_pi_t_T0 = sin(pi * t / T0);

        Eigen::Matrix3<T> jacobian;

        // Partial derivatives for u with respect to x, y, z
        jacobian(0, 0) = 4 * pi * sin(pi * x) * cos(pi * x) * sin(2 * pi * y) * sin(2 * pi * z) * cos_pi_t_T0;
        jacobian(0, 1) = 4 * pi * sin(pi * x) * sin(pi * x) * cos(2 * pi * y) * sin(2 * pi * z) * cos_pi_t_T0;
        jacobian(0, 2) = 4 * pi * sin(pi * x) * sin(pi * x) * sin(2 * pi * y) * cos(2 * pi * z) * cos_pi_t_T0;

        // Partial derivatives for v with respect to x, y, z
        jacobian(1, 0) = -2 * pi * cos(2 * pi * x) * sin(pi * y) * sin(pi * y) * sin(2 * pi * z) * cos_pi_t_T0;
        jacobian(1, 1) = -2 * pi * sin(2 * pi * x) * sin(pi * y) * cos(pi * y) * sin(2 * pi * z) * cos_pi_t_T0;
        jacobian(1, 2) = -2 * pi * sin(2 * pi * x) * sin(pi * y) * sin(pi * y) * cos(2 * pi * z) * cos_pi_t_T0;

        // Partial derivatives for w with respect to x, y, z
        jacobian(2, 0) = -2 * pi * cos(2 * pi * x) * sin(2 * pi * y) * sin(pi * z) * sin(pi * z) * cos_pi_t_T0;
        jacobian(2, 1) = -4 * pi * sin(2 * pi * x) * cos(2 * pi * y) * sin(pi * z) * sin(pi * z) * cos_pi_t_T0;
        jacobian(2, 2) = -2 * pi * sin(2 * pi * x) * sin(2 * pi * y) * cos(pi * z) * sin(pi * z) * cos_pi_t_T0;

        return jacobian;
    }
	__hostdev__ void velocityAndJacobian(const Vec& pos, const double t, Vec& vel, Eigen::Matrix3<T>& jacobian) const {
		vel = operator()(pos, t);
		jacobian = gradu(pos, t);
	}
};


class Deformation2D {
public:
    double T0 = 8;
    double pi = CommonConstants::pi;
    __hostdev__ Vec operator()(const Vec& pos, const double t)const {
        //return Vec(1, 0, 0);
        double t_over_T = t / T0;
        double x = pos[0], y = pos[1];
        double u = -2 * cos(pi * t_over_T) * cos(pi * y) * pow(sin(pi * x), 2) * sin(pi * y);
        double v = 2 * cos(pi * t_over_T) * cos(pi * x) * sin(pi * x) * pow(sin(pi * y), 2);
        return Vec(u, v, 0);
    }
};




//void TestVelocityJacobian(const int grid_case) {
//    Info("Test velocity jacobian with grid case {}", grid_case);
//    int test_flowmap_stride = 5; // x dt
//
//    float h = 1.0 / Tile::DIM;
//    //0:8, 1:16, 2:32, 3:64, 4:128, 5:256, 6:512, 7:1024
//    HADeviceGrid<Tile> grid(h, std::initializer_list<uint32_t>{ 16, 16, 16, 16, 16, 16, 16, 16, 16, 16 });
//    grid.setTileHost(0, nanovdb::Coord(0, 0, 0), Tile(), LEAF);
//    grid.compressHost(false);
//    grid.syncHostAndDevice();
//    SpawnGhostTiles(grid, false);
//
//    IterativeRefine(grid, [=]__device__(const HATileAccessor<Tile>&acc, HATileInfo<Tile>&info) { return NumericalTestsLevelTarget(acc, info, grid_case); }, false);
//    grid.launchVoxelFunc(
//        [=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
//        auto& tile = info.tile();
//        tile.type(l_ijk) = INTERIOR;
//    }, -1, LEAF, LAUNCH_SUBTREE
//    );
//    CalcCellTypesFromLeafs(grid);
//
//    Deformation3D vel_func;
//
//    int u_channel = Tile::u_channel;
//    int node_u_channel = 0;
//    double time = 0;
//    grid.launchVoxelFunc(
//        [=] __device__(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk) {
//        auto& tile = info.tile();
//        for (int axis = 0; axis < 3; axis++) {
//            auto face_ctr = acc.faceCenter(axis, info, l_ijk);
//            tile(u_channel + axis, l_ijk) = vel_func(face_ctr, time)[axis];
//        }
//    }, -1, LEAF, LAUNCH_SUBTREE
//    );
//    //for (int axis : {0, 1, 2}) {
//    //    PropagateToChildren(grid, Tile::u_channel + axis, Tile::u_channel + axis, -1, GHOST, LAUNCH_SUBTREE, INTERIOR | DIRICHLET | NEUMANN);
//    //}
//    CalcLeafNodeValuesFromFaceCenters(grid, u_channel, node_u_channel);
//    
//
//    RandomGenerator rng;
//    thrust::host_vector<Vec> points(1);
//    double lo = 1.0 / 16, hi = 1 - lo;
//    //double lo = 1.0 / 8, hi = 1 - lo;
//    for (auto& p : points) {
//        p = Vec(rng.uniform(lo, hi), rng.uniform(lo, hi), rng.uniform(lo, hi));
//    }
//    thrust::device_vector<Vec> points_d = points;
//
//    {
//        thrust::device_vector<Eigen::Matrix3<T>> jacobians_numerical_d(points.size());
//        thrust::device_vector<Eigen::Matrix3<T>> jacobians_finitediff_d(points.size());
//        thrust::device_vector<Eigen::Matrix3<T>> jacobians_analytical_d(points.size());
//
//		thrust::device_vector<Vec> numerical_velocities_d(points.size());
//		thrust::device_vector<Vec> analytical_velocities_d(points.size());
//
//        auto points_d_ptr = thrust::raw_pointer_cast(points_d.data());
//        auto jacobians_numerical_d_ptr = thrust::raw_pointer_cast(jacobians_numerical_d.data());
//		auto jacobians_finitediff_d_ptr = thrust::raw_pointer_cast(jacobians_finitediff_d.data());
//        auto jacobians_analytical_d_ptr = thrust::raw_pointer_cast(jacobians_analytical_d.data());
//
//		auto numerical_velocities_d_ptr = thrust::raw_pointer_cast(numerical_velocities_d.data());
//		auto analytical_velocities_d_ptr = thrust::raw_pointer_cast(analytical_velocities_d.data());
//
//        auto acc = grid.deviceAccessor();
//
//        LaunchIndexFunc([=]__device__(int i) {
//            Vec pos = points_d_ptr[i];
//            Vec vel; Eigen::Matrix3<T> jacobian;
//            KernelIntpVelocityAndJacobianMAC2(acc, pos, u_channel, vel, jacobian);
//            //VelocityAndJacobian(acc, pos, node_u_channel, vel, jacobian);
//            
//            jacobians_numerical_d_ptr[i] = jacobian;
//            //jacobians_finitediff_d_ptr[i] = gradu_finitediff(pos, time, 1e-5);
//            jacobians_finitediff_d_ptr[i] = JacobianOfVectorField([=]__hostdev__(const Vec & pos) { return vel_func(pos, time); }, pos, 1e-5);
//            jacobians_analytical_d_ptr[i] = vel_func.gradu(pos, time);
//
//			numerical_velocities_d_ptr[i] = vel;
//			analytical_velocities_d_ptr[i] = vel_func(pos, time);
//
//        }, points.size());
//
//		thrust::host_vector<Eigen::Matrix3<T>> jacobians_numerical = jacobians_numerical_d;
//		thrust::host_vector<Eigen::Matrix3<T>> jacobians_finitediff = jacobians_finitediff_d;
//		thrust::host_vector<Eigen::Matrix3<T>> jacobians_analytical = jacobians_analytical_d;
//
//		//std::cout << "jacobians numerical: " << jacobians_numerical[0] << std::endl;
//		//std::cout << "jacobians finitediff: " << jacobians_finitediff[0] << std::endl;
//		//std::cout << "jacobians analytical: " << jacobians_analytical[0] << std::endl;
//
//
//		double error = LinfErrorBetweenMatrixForbenius2(jacobians_numerical, jacobians_analytical);
//		Info("Linf error between numerical and analytical jacobians: {}", error);
//		Info("Linf error between finite difference and analytical jacobians: {}", LinfErrorBetweenMatrixForbenius2(jacobians_finitediff, jacobians_analytical));
//		Info("Linf error between numerical and finite difference jacobians: {}", LinfErrorBetweenMatrixForbenius2(jacobians_numerical, jacobians_finitediff));
//		Info("Linf error between numerical and analytical velocities: {}", LinfErrorBetweenPointCloud(numerical_velocities_d, analytical_velocities_d));
//        fmt::print("\n");
//    }
//}

void TestAnalyticalFlowMapAdvectionVariableTime(void) {
	Info("Test analytical flow map advection with variable time");
    Deformation3D vel_func;
    auto velocity_and_jacobian = [=]__hostdev__(const Vec & pos, const double t, Vec & vel, Eigen::Matrix3<T>&jacobian) {
        vel = vel_func(pos, t);
        jacobian = vel_func.gradu(pos, t);
    };

    auto rk4_forward_phi_f = [=]__hostdev__(const T time, const T dt, Vec & phi, Eigen::Matrix3<T>&F) {
        Vec u1; Eigen::Matrix3<T> gradu1;
        velocity_and_jacobian(phi, time, u1, gradu1);
        Eigen::Matrix3<T> dFdt1 = gradu1 * F;
        Vec phi1 = phi + 0.5 * dt * u1;
        Eigen::Matrix3<T> F1 = F + 0.5 * dt * dFdt1;

        Vec u2; Eigen::Matrix3<T> gradu2;
        velocity_and_jacobian(phi1, time + 0.5 * dt, u2, gradu2);
        Eigen::Matrix3<T> dFdt2 = gradu2 * F1;
        Vec phi2 = phi + 0.5 * dt * u2;
        Eigen::Matrix3<T> F2 = F + 0.5 * dt * dFdt2;

        Vec u3; Eigen::Matrix3<T> gradu3;
        velocity_and_jacobian(phi2, time + 0.5 * dt, u3, gradu3);
        Eigen::Matrix3<T> dFdt3 = gradu3 * F2;
        Vec phi3 = phi + dt * u3;
        Eigen::Matrix3<T> F3 = F + dt * dFdt3;

        Vec u4; Eigen::Matrix3<T> gradu4;
        velocity_and_jacobian(phi3, time + dt, u4, gradu4);
        Eigen::Matrix3<T> dFdt4 = gradu4 * F3;

        phi = phi + dt / 6.0 * (u1 + 2 * u2 + 2 * u3 + u4);
        F = F + dt / 6.0 * (dFdt1 + 2 * dFdt2 + 2 * dFdt3 + dFdt4);
    };

    auto rk4_forward_phi_t = [=] __hostdev__(const T time, const T dt, Vec & phi, Eigen::Matrix3<T> &matT) {
        Vec u1; Eigen::Matrix3<T> gradu1;
        velocity_and_jacobian(phi, time, u1, gradu1);
        Eigen::Matrix3<T> dTdt1 = -matT * gradu1;
        Vec phi1 = phi + 0.5 * dt * u1;
        Eigen::Matrix3<T> T1 = matT + 0.5 * dt * dTdt1;

        Vec u2; Eigen::Matrix3<T> gradu2;
        velocity_and_jacobian(phi1, time + 0.5 * dt, u2, gradu2);
        Eigen::Matrix3<T> dTdt2 = -T1 * gradu2;
        Vec phi2 = phi + 0.5 * dt * u2;
        Eigen::Matrix3<T> T2 = matT + 0.5 * dt * dTdt2;

        Vec u3; Eigen::Matrix3<T> gradu3;
        velocity_and_jacobian(phi2, time + 0.5 * dt, u3, gradu3);
        Eigen::Matrix3<T> dTdt3 = -T2 * gradu3;
        Vec phi3 = phi + dt * u3;
        Eigen::Matrix3<T> T3 = matT + dt * dTdt3;

        Vec u4; Eigen::Matrix3<T> gradu4;
        velocity_and_jacobian(phi3, time + dt, u4, gradu4);
        Eigen::Matrix3<T> dTdt4 = -T3 * gradu4;

        phi = phi + dt / 6.0 * (u1 + 2 * u2 + 2 * u3 + u4);
        matT = matT + dt / 6.0 * (dTdt1 + 2 * dTdt2 + 2 * dTdt3 + dTdt4);
    };

    auto rk2_forward_phi_f = [=]__hostdev__(const T time, const T dt, Vec & pos, Eigen::Matrix3<T>&F) {
        Vec u1; Eigen::Matrix3<T> gradu1;
        vel_func.velocityAndJacobian(pos, time, u1, gradu1);

        Vec pos1 = pos + u1 * 0.5 * dt;

        Vec u2; Eigen::Matrix3<T> gradu2;
        vel_func.velocityAndJacobian(pos1, time + 0.5 * dt, u2, gradu2);
        auto dFdt2 = gradu2 * F;
        pos = pos + dt * u2;
        F = F + dt * dFdt2;
    };

    auto rk2_forward_phi_t = [=]__hostdev__(const T time, const T dt, Vec & pos, Eigen::Matrix3<T>&matT) {
        Vec u1; Eigen::Matrix3<T> gradu1;
        vel_func.velocityAndJacobian(pos, time, u1, gradu1);

        Vec pos1 = pos + u1 * 0.5 * dt;

        Vec u2; Eigen::Matrix3<T> gradu2;
        vel_func.velocityAndJacobian(pos1, time + 0.5 * dt, u2, gradu2);
        auto dTdt2 = -matT * gradu2;
        pos = pos + dt * u2;
        matT = matT + dt * dTdt2;
    };

    RandomGenerator rng;
    thrust::host_vector<Vec> points(10000);
    double lo = 1.0 / 16, hi = 1 - lo;
    //double lo = 1.0 / 8, hi = 1 - lo;
    for (auto& p : points) {
        p = Vec(rng.uniform(lo, hi), rng.uniform(lo, hi), rng.uniform(lo, hi));
    }
    double cfl = 1.0;//max vel=1
    double dt = 1.0 / 128 * cfl;

    {
        int test_flowmap_stride = 5;
        thrust::host_vector<Eigen::Matrix3<T>> F_forward(points.size());
        thrust::host_vector<Eigen::Matrix3<T>> T_backward(points.size());
        thrust::host_vector<Eigen::Matrix3<T>> FT_flowmap_forward(points.size());
        thrust::host_vector<Vec> psi_of_phi(points.size());
        tbb::parallel_for(0, static_cast<int>(points.size()), [&](int i) {
            Vec phi = points[i]; Eigen::Matrix3<T> F = Eigen::Matrix3<T>::Identity();
            Vec phi1 = phi; Eigen::Matrix3<T> matT = Eigen::Matrix3<T>::Identity();

            for (int i = 0; i < test_flowmap_stride; i++) {
                rk4_forward_phi_f(i * dt, dt, phi, F);
                rk4_forward_phi_t(i * dt, dt, phi1, matT);
            }
            F_forward[i] = F;
            FT_flowmap_forward[i] = F * matT;

            Vec psi = phi; Eigen::Matrix3<T> T_back = Eigen::Matrix3<T>::Identity();
            for (int i = test_flowmap_stride - 1; i >= 0; i--) {
                rk4_forward_phi_t(i * dt, -dt, psi, T_back);
            }
            T_backward[i] = T_back;
            psi_of_phi[i] = psi;
            }
        );

        Info("analytical rk4 advection with variable time");
        Info("linf error between psi(phi) and original points: {}", LinfErrorBetweenPointCloud(psi_of_phi, points));
        Info("Linf error between F_forward and T_backward: {}", LinfErrorBetweenMatrixForbenius2(F_forward, T_backward));
        Info("Linf error between FT_flowmap_forward and Identity: {}", LinfErrorBetweenTargetMatrixForbenius2(FT_flowmap_forward, Eigen::Matrix3<T>::Identity()));
        Info("");
    }

    {
        int test_flowmap_stride = 5;
        thrust::host_vector<Eigen::Matrix3<T>> F_forward(points.size());
        thrust::host_vector<Eigen::Matrix3<T>> T_backward(points.size());
        thrust::host_vector<Eigen::Matrix3<T>> FT_flowmap_forward(points.size());
		thrust::host_vector<Vec> psi_of_phi(points.size());
        tbb::parallel_for(0, static_cast<int>(points.size()), [&](int i) {
            Vec phi = points[i]; Eigen::Matrix3<T> F = Eigen::Matrix3<T>::Identity();
            Vec phi1 = phi; Eigen::Matrix3<T> matT = Eigen::Matrix3<T>::Identity();

            for (int i = 0; i < test_flowmap_stride; i++) {
                rk2_forward_phi_f(i * dt, dt, phi, F);
                rk2_forward_phi_t(i * dt, dt, phi1, matT);
            }
            F_forward[i] = F;
            FT_flowmap_forward[i] = F * matT;

            Vec psi = phi; Eigen::Matrix3<T> T_back = Eigen::Matrix3<T>::Identity();
            for (int i = test_flowmap_stride - 1; i >= 0; i--) {
                rk2_forward_phi_t(i * dt, -dt, psi, T_back);
            }
            T_backward[i] = T_back;
			psi_of_phi[i] = psi;
            }
        );

        Info("analytical rk2 advection with variable time");
        Info("linf error between psi(phi) and original points: {}", LinfErrorBetweenPointCloud(psi_of_phi, points));
        Info("Linf error between F_forward and T_backward: {}", LinfErrorBetweenMatrixForbenius2(F_forward, T_backward));
        Info("Linf error between FT_flowmap_forward and Identity: {}", LinfErrorBetweenTargetMatrixForbenius2(FT_flowmap_forward, Eigen::Matrix3<T>::Identity()));
        Info("");
    }
}

void TestAnalyticalFlowMapAdvectionFixedTime(void) {
    Info("Test analytical flow map advection with fixed time");
    Deformation3D vel_func;
    auto velocity_and_jacobian = [=]__hostdev__(const Vec & pos, const double t, Vec & vel, Eigen::Matrix3<T>&jacobian) {
        vel = vel_func(pos, t);
        jacobian = vel_func.gradu(pos, t);
    };

    auto rk4_forward_phi_f = [=]__hostdev__(const T time, const T dt, Vec & phi, Eigen::Matrix3<T>&F) {
        Vec u1; Eigen::Matrix3<T> gradu1;
        velocity_and_jacobian(phi, time, u1, gradu1);
        Eigen::Matrix3<T> dFdt1 = gradu1 * F;
        Vec phi1 = phi + 0.5 * dt * u1;
        Eigen::Matrix3<T> F1 = F + 0.5 * dt * dFdt1;

        Vec u2; Eigen::Matrix3<T> gradu2;
        velocity_and_jacobian(phi1, time, u2, gradu2);
        Eigen::Matrix3<T> dFdt2 = gradu2 * F1;
        Vec phi2 = phi + 0.5 * dt * u2;
        Eigen::Matrix3<T> F2 = F + 0.5 * dt * dFdt2;

        Vec u3; Eigen::Matrix3<T> gradu3;
        velocity_and_jacobian(phi2, time, u3, gradu3);
        Eigen::Matrix3<T> dFdt3 = gradu3 * F2;
        Vec phi3 = phi + dt * u3;
        Eigen::Matrix3<T> F3 = F + dt * dFdt3;

        Vec u4; Eigen::Matrix3<T> gradu4;
        velocity_and_jacobian(phi3, time, u4, gradu4);
        Eigen::Matrix3<T> dFdt4 = gradu4 * F3;

        phi = phi + dt / 6.0 * (u1 + 2 * u2 + 2 * u3 + u4);
        F = F + dt / 6.0 * (dFdt1 + 2 * dFdt2 + 2 * dFdt3 + dFdt4);
    };

    auto rk4_forward_phi_t = [=] __hostdev__(const T time, const T dt, Vec & phi, Eigen::Matrix3<T> &matT) {
        Vec u1; Eigen::Matrix3<T> gradu1;
        velocity_and_jacobian(phi, time, u1, gradu1);
        Eigen::Matrix3<T> dTdt1 = -matT * gradu1;
        Vec phi1 = phi + 0.5 * dt * u1;
        Eigen::Matrix3<T> T1 = matT + 0.5 * dt * dTdt1;

        Vec u2; Eigen::Matrix3<T> gradu2;
        velocity_and_jacobian(phi1, time, u2, gradu2);
        Eigen::Matrix3<T> dTdt2 = -T1 * gradu2;
        Vec phi2 = phi + 0.5 * dt * u2;
        Eigen::Matrix3<T> T2 = matT + 0.5 * dt * dTdt2;

        Vec u3; Eigen::Matrix3<T> gradu3;
        velocity_and_jacobian(phi2, time, u3, gradu3);
        Eigen::Matrix3<T> dTdt3 = -T2 * gradu3;
        Vec phi3 = phi + dt * u3;
        Eigen::Matrix3<T> T3 = matT + dt * dTdt3;

        Vec u4; Eigen::Matrix3<T> gradu4;
        velocity_and_jacobian(phi3, time, u4, gradu4);
        Eigen::Matrix3<T> dTdt4 = -T3 * gradu4;

        phi = phi + dt / 6.0 * (u1 + 2 * u2 + 2 * u3 + u4);
        matT = matT + dt / 6.0 * (dTdt1 + 2 * dTdt2 + 2 * dTdt3 + dTdt4);
    };

    auto rk2_forward_phi_f = [=]__hostdev__(const T time, const T dt, Vec & pos, Eigen::Matrix3<T>&F) {
        Vec u1; Eigen::Matrix3<T> gradu1;
        vel_func.velocityAndJacobian(pos, time, u1, gradu1);

        Vec pos1 = pos + u1 * 0.5 * dt;

        Vec u2; Eigen::Matrix3<T> gradu2;
        vel_func.velocityAndJacobian(pos1, time, u2, gradu2);
        auto dFdt2 = gradu2 * F;
        pos = pos + dt * u2;
        F = F + dt * dFdt2;
    };

    auto rk2_forward_phi_t = [=]__hostdev__(const T time, const T dt, Vec & pos, Eigen::Matrix3<T>&matT) {
        Vec u1; Eigen::Matrix3<T> gradu1;
        vel_func.velocityAndJacobian(pos, time, u1, gradu1);

        Vec pos1 = pos + u1 * 0.5 * dt;

        Vec u2; Eigen::Matrix3<T> gradu2;
        vel_func.velocityAndJacobian(pos1, time, u2, gradu2);
        auto dTdt2 = -matT * gradu2;
        pos = pos + dt * u2;
        matT = matT + dt * dTdt2;
    };

    RandomGenerator rng;
    thrust::host_vector<Vec> points(10000);
    double lo = 1.0 / 16, hi = 1 - lo;
    //double lo = 1.0 / 8, hi = 1 - lo;
    for (auto& p : points) {
        p = Vec(rng.uniform(lo, hi), rng.uniform(lo, hi), rng.uniform(lo, hi));
    }

    //points[0] = Vec(0.4559023, 0.2541084, 0.74300975);
    //points.resize(1);

    double cfl = 1.0;//max vel=1
    double dt = 1.0 / 128 * cfl;

    {
        int test_flowmap_stride = 5;
        thrust::host_vector<Eigen::Matrix3<T>> F_forward(points.size());
        thrust::host_vector<Eigen::Matrix3<T>> T_backward(points.size());
        thrust::host_vector<Eigen::Matrix3<T>> FT_flowmap_forward(points.size());
        thrust::host_vector<Vec> psi_of_phi(points.size());
        tbb::parallel_for(0, static_cast<int>(points.size()), [&](int i) {
            Vec phi = points[i]; Eigen::Matrix3<T> F = Eigen::Matrix3<T>::Identity();
            Vec phi1 = phi; Eigen::Matrix3<T> matT = Eigen::Matrix3<T>::Identity();

            for (int i = 0; i < test_flowmap_stride; i++) {
                rk4_forward_phi_f(i * dt, dt, phi, F);
                rk4_forward_phi_t(i * dt, dt, phi1, matT);
            }
            F_forward[i] = F;
            FT_flowmap_forward[i] = F * matT;

            Vec psi = phi; Eigen::Matrix3<T> T_back = Eigen::Matrix3<T>::Identity();
            for (int i = test_flowmap_stride - 1; i >= 0; i--) {
                rk4_forward_phi_t(i * dt, -dt, psi, T_back);
            }
            T_backward[i] = T_back;
            psi_of_phi[i] = psi;
            }
        );

        Info("analytical rk4 advection with fixed time on {} points", points.size());
        Info("linf error between psi(phi) and original points: {}", LinfErrorBetweenPointCloud(psi_of_phi, points));
        Info("Linf error between F_forward and T_backward: {}", LinfErrorBetweenMatrixForbenius2(F_forward, T_backward));
        Info("Linf error between FT_flowmap_forward and Identity: {}", LinfErrorBetweenTargetMatrixForbenius2(FT_flowmap_forward, Eigen::Matrix3<T>::Identity()));
        Info("");
    }

    {
        int test_flowmap_stride = 5;
        thrust::host_vector<Eigen::Matrix3<T>> F_forward(points.size());
        thrust::host_vector<Eigen::Matrix3<T>> T_backward(points.size());
        thrust::host_vector<Eigen::Matrix3<T>> FT_flowmap_forward(points.size());
        thrust::host_vector<Vec> psi_of_phi(points.size());
        tbb::parallel_for(0, static_cast<int>(points.size()), [&](int i) {
            Vec phi = points[i]; Eigen::Matrix3<T> F = Eigen::Matrix3<T>::Identity();
            Vec phi1 = phi; Eigen::Matrix3<T> matT = Eigen::Matrix3<T>::Identity();
            for (int i = 0; i < test_flowmap_stride; i++) {
                rk2_forward_phi_f(i * dt, dt, phi, F);
                rk2_forward_phi_t(i * dt, dt, phi1, matT);
            }
            F_forward[i] = F;
            FT_flowmap_forward[i] = F * matT;

            Vec psi = phi; Eigen::Matrix3<T> T_back = Eigen::Matrix3<T>::Identity();
            for (int i = test_flowmap_stride - 1; i >= 0; i--) {
                rk2_forward_phi_t(i * dt, -dt, psi, T_back);
            }
            T_backward[i] = T_back;
            psi_of_phi[i] = psi;
            }
        );

        thrust::host_vector<Vec> phi_of_psi(points.size());
        tbb::parallel_for(0, static_cast<int>(points.size()), [&](int i) {
            Vec psi = points[i]; Eigen::Matrix3<T> F_back = Eigen::Matrix3<T>::Identity();
            //Info("initial psi: {}", psi);
            for (int i = test_flowmap_stride - 1; i >= 0; i--) {
                rk2_forward_phi_f(i* dt, -dt, psi, F_back);
                //Info("psi at i={}: {}", i, psi);
            }

            Vec phi = psi; Eigen::Matrix3<T> F_forward = Eigen::Matrix3<T>::Identity();

            for (int i = 0; i < test_flowmap_stride; i++) {
                rk2_forward_phi_f(i * dt, dt, phi, F_forward);
                //Info("phi at i={}: {}", i, phi);
            }

			phi_of_psi[i] = phi;
            }
        );

        Info("analytical rk2 advection with fixed time on {} points", points.size());
        Info("linf error between psi(phi) and original points: {}", LinfErrorBetweenPointCloud(psi_of_phi, points));
		Info("linf error between phi(psi) and original points: {}", LinfErrorBetweenPointCloud(phi_of_psi, points));
        Info("Linf error between F_forward and T_backward: {}", LinfErrorBetweenMatrixForbenius2(F_forward, T_backward));
        Info("Linf error between FT_flowmap_forward and Identity: {}", LinfErrorBetweenTargetMatrixForbenius2(FT_flowmap_forward, Eigen::Matrix3<T>::Identity()));
        Info("");
    }
}

//void TestFlowMapAdvection(const int grid_case) {
//	Info("Test flow map advection with grid case {}", grid_case);
//    int test_flowmap_stride = 5; // x dt
//
//    std::vector<std::shared_ptr<HADeviceGrid<Tile>>> grid_ptrs;
//    std::vector<double> time_steps;
//
//    float h = 1.0 / Tile::DIM;
//    for (int i = 0; i <= test_flowmap_stride; i++) {
//        //0:8, 1:16, 2:32, 3:64, 4:128, 5:256, 6:512, 7:1024
//        auto ptr = std::make_shared<HADeviceGrid<Tile>>(h, std::initializer_list<uint32_t>{ 16, 16, 16, 16, 16, 16, 16, 16, 16, 16 });
//		grid_ptrs.push_back(ptr);
//
//        auto& grid = *ptr;
//        grid.setTileHost(0, nanovdb::Coord(0, 0, 0), Tile(), LEAF);
//        grid.compressHost(false);
//        grid.syncHostAndDevice();
//        SpawnGhostTiles(grid, false);
//
//        IterativeRefine(grid, [=]__device__(const HATileAccessor<Tile>&acc, HATileInfo<Tile>&info) { return NumericalTestsLevelTarget(acc, info, grid_case); }, false);
//        grid.launchVoxelFunc(
//            [=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
//            auto& tile = info.tile();
//            tile.type(l_ijk) = INTERIOR;
//        }, -1, LEAF, LAUNCH_SUBTREE
//        );
//        CalcCellTypesFromLeafs(grid);
//    }
//    thrust::host_vector<HATileAccessor<Tile>> accs_h;
//    for (int i = 0; i <= test_flowmap_stride; i++) {
//        accs_h.push_back(grid_ptrs[i]->deviceAccessor());
//    }
//    thrust::device_vector<HATileAccessor<Tile>> accs_d = accs_h;
//    auto accs_d_ptr = thrust::raw_pointer_cast(accs_d.data());
//
//    Deformation3D vel_func;
//
//    auto rk4_forward_step_analytical = [=]__hostdev__(const Vec& pos, const double time, const double dt) {
//        double c1 = 1.0 / 6.0 * dt, c2 = 1.0 / 3.0 * dt, c3 = 1.0 / 3.0 * dt, c4 = 1.0 / 6.0 * dt;
//		Vec vel1 = vel_func(pos, time);
//        Vec pos1 = pos + vel1 * 0.5 * dt;
//		Vec vel2 = vel_func(pos1, time + 0.5 * dt);
//        Vec pos2 = pos + vel2 * 0.5 * dt;
//		Vec vel3 = vel_func(pos2, time + 0.5 * dt);
//        Vec pos3 = pos + vel3 * dt;
//		Vec vel4 = vel_func(pos3, time + dt);
//        return pos + c1 * vel1 + c2 * vel2 + c3 * vel3 + c4 * vel4;
//        };
//
//    auto base_dir = fs::current_path() / "data" / fmt::format("flowmap_advection_test{}", grid_case);
//    fs::create_directories(base_dir);
//
//
//    double cfl = 1.0;//max vel=1
//    //double dt = h / (1 << (grid_ptrs[0]->mMaxLevel)) * cfl;
//	double dt = h / (1 << 4) * cfl;//assume that finest level is 4
//	int u_channel = Tile::u_channel;
//	int node_u_channel = 0;
//
//    //fill velocity fields
//    for (int i = 0; i <= test_flowmap_stride; i++) {
//        double time = i * dt;
//
//		auto& grid = *grid_ptrs[i];
//        grid.launchVoxelFunc(
//            [=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
//            auto& tile = info.tile();
//            for (int axis = 0; axis < 3; axis++) {
//                auto face_ctr = acc.faceCenter(axis, info, l_ijk);
//                tile(u_channel + axis, l_ijk) = vel_func(face_ctr, time)[axis];
//
//                //{
//                //    auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
//                //    if (axis == 0 && info.mLevel == 3 && g_ijk == Coord(23, 14, 15)) {
//                //        printf("first axis %d time %f g_ijk %d %d %d vel %f\n", axis, time, g_ijk[0], g_ijk[1], g_ijk[2], tile(u_channel + axis, l_ijk));
//                //    }
//                //}
//            }
//        }, -1, LEAF, LAUNCH_SUBTREE
//        );
//        CalcLeafNodeValuesFromFaceCenters(grid, u_channel, node_u_channel);
//
//        grid.launchVoxelFuncOnAllTiles(
//            [=] __device__(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk) {
//            auto& tile = info.tile();
//            for (int axis : {0, 1, 2}) {
//                auto fpos = acc.faceCenter(axis, info, l_ijk);
//                auto vel = InterpolateFaceValue(acc, fpos, u_channel, node_u_channel);
//				tile(u_channel + axis, l_ijk) = vel[axis];
//				//tile(u_channel + axis, l_ijk) = vel_func(fpos, time)[axis];
//
//                //{
//                //    auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
//                //    if (axis==0&&info.mLevel == 3 && g_ijk == Coord(23, 14, 15)) {
//                //        printf("second axis %d time %f g_ijk %d %d %d vel %f\n", axis, time, g_ijk[0], g_ijk[1], g_ijk[2], tile(u_channel + axis, l_ijk));
//                //    }
//                //}
//            }
//        }, NONLEAF | GHOST, 4
//        );
//
//        //for (int axis : {0, 1, 2}) {
//        //    PropagateValuesToGhostTiles(grid, u_channel + axis, u_channel + axis);
//        //    AccumulateToParents(grid, u_channel + axis, u_channel + axis, -1, NONLEAF, LAUNCH_SUBTREE, INTERIOR | DIRICHLET | NEUMANN, 1. / 8, false);
//        //}
//        
//        time_steps.push_back(dt);
//    }
//
//    auto rk4_phi_analytical = [=]__hostdev__(Vec pos, int step_begin, int step_end) {
//        for (int i = step_begin; i < step_end; i++) {
//            pos = rk4_forward_step_analytical(pos, i * dt, dt);
//        }
//        return pos;
//    };
//    auto rk4_psi_analytical = [=]__hostdev__(Vec pos, int step_end, int step_begin) {
//        for (int i = step_end; i > step_begin; i--) {
//            pos = rk4_forward_step_analytical(pos, i * dt, -dt);
//        }
//        return pos;
//    };
//
//
//    RandomGenerator rng;
//    thrust::host_vector<Vec> points(10000);
//    double lo = 1.0 / 16, hi = 1 - lo;
//    //double lo = 1.0 / 8, hi = 1 - lo;
//    for (auto& p : points) {
//        p = Vec(rng.uniform(lo, hi), rng.uniform(lo, hi), rng.uniform(lo, hi));
//    }
//    //points[0] = Vec(0.37287432, 0.2518963, 0.7224257);
//    //points.resize(1);
//
//    //points[0] = points[7569];
//    //points[0] = Vec(0.4559023, 0.2541084, 0.74300975);
//    //points.resize(1);
//
//    thrust::device_vector<Vec> points_d = points;
//    
//    /*
//    {
//        //test analytical phi vs flowmap advected phi
//		thrust::host_vector<Vec> phi_analytical(points.size());
//		for (int i = 0; i < points.size(); i++) {
//            phi_analytical[i] = rk4_phi_analytical(points[i], 0, test_flowmap_stride);
//		}
//
//        thrust::device_vector<Vec> phi_flowmap_d(points.size());
//        thrust::transform(points_d.begin(), points_d.end(), phi_flowmap_d.begin(), [=]__device__(const Vec& p) {
//            Vec phi = p; Eigen::Matrix3<T> F = Eigen::Matrix3<T>::Identity();
//            for (int i = 0; i < test_flowmap_stride; i++) {
//                RK4ForwardPositionAndF(accs_d_ptr[i], dt, u_channel, node_u_channel, phi, F);
//            }
//            return phi;
//        });
//		thrust::host_vector<Vec> phi_flowmap = phi_flowmap_d;
//
//		Info("Linf error between analytical phi and flowmap advected phi: {}", LinfErrorBetweenPointCloud(phi_analytical, phi_flowmap));
//	}
//    */
//
//  //  {
//  //      //test if phi(psi)=I
//  //      thrust::host_vector<Vec> psi_analytical(points.size());
//		//thrust::host_vector<Vec> phi_of_psi_analytical(points.size());
//  //      tbb::parallel_for(0, static_cast<int>(points.size()), [&](int i) {
//  //          psi_analytical[i] = rk4_psi_analytical(points[i], test_flowmap_stride, 0);
//  //          phi_of_psi_analytical[i] = rk4_phi_analytical(psi_analytical[i], 0, test_flowmap_stride);
//  //          }
//  //      );
//
//  //      thrust::device_vector<Vec> psi_flowmap_d(points.size());
//  //      thrust::transform(points_d.begin(), points_d.end(), psi_flowmap_d.begin(), [=]__device__(const Vec & p) {
//  //          Vec psi = p; Eigen::Matrix3<T> matT = Eigen::Matrix3<T>::Identity();
//  //          for (int i = test_flowmap_stride - 1; i >= 0; i--) {
//  //              RK4ForwardPositionAndT(accs_d_ptr[i], -dt, u_channel, node_u_channel, psi, matT);
//  //          }
//  //          return psi;
//  //      });
//		//thrust::device_vector<Vec> phi_of_psi_flowmap_d(points.size());
//		//thrust::transform(psi_flowmap_d.begin(), psi_flowmap_d.end(), phi_of_psi_flowmap_d.begin(), [=]__device__(const Vec& p) {
//		//	Vec phi = p; Eigen::Matrix3<T> F = Eigen::Matrix3<T>::Identity();
//		//	for (int i = 0; i < test_flowmap_stride; i++) {
//		//		RK4ForwardPositionAndF(accs_d_ptr[i], dt, u_channel, node_u_channel, phi, F);
//		//	}
//		//	return phi;
//		//});
//		//thrust::host_vector<Vec> phi_of_psi_flowmap = phi_of_psi_flowmap_d;
//
//  //      Info("Linf error between analytical phi(psi) and original points: {}", LinfErrorBetweenPointCloud(phi_of_psi_analytical, points));
//		//Info("Linf error between flowmap advected phi(psi) and original points: {}", LinfErrorBetweenPointCloud(phi_of_psi_flowmap, points));
//  //  }
//  //  return;
//
//	auto rk2_forward_phi_f_hybrid = [=]__device__(const T time, const T dt, const HATileAccessor<Tile>&acc, const int u_channel, const int node_u_channel, Vec & pos, Eigen::Matrix3<T>&F) {
//        //Vec vel1 = InterpolateFaceValue(acc, pos, u_channel, node_u_channel);
//        Vec u1; Eigen::Matrix3<T> gradu1;
//        //VelocityAndJacobian(acc, pos, node_u_channel, u1, gradu1);
//		//vel_func.velocityAndJacobian(pos, time, u1, gradu1);
//        KernelIntpVelocityAndJacobianMAC2(acc, pos, u_channel, u1, gradu1);
//
//        Vec pos1 = pos + u1 * 0.5 * dt;
//        //Vec vel2 = InterpolateFaceValue(acc, pos1, u_channel, node_u_channel);
//        //auto gradu2 = VelocityJacobian(acc, pos1, node_u_channel);
//
//        Vec u2; Eigen::Matrix3<T> gradu2;
//        //VelocityAndJacobian(acc, pos1, node_u_channel, u2, gradu2);
//		//vel_func.velocityAndJacobian(pos1, time, u2, gradu2);
//		KernelIntpVelocityAndJacobianMAC2(acc, pos1, u_channel, u2, gradu2);
//        auto dFdt2 = gradu2 * F;
//        pos = pos + dt * u2;
//        F = F + dt * dFdt2;
//	};
//
//    auto rk2_forward_phi_t_hybrid = [=]__device__(const T time, const T dt, const HATileAccessor<Tile>&acc, const int u_channel, const int node_u_channel, Vec & pos, Eigen::Matrix3<T>&matT) {
//        //Vec vel1 = InterpolateFaceValue(acc, pos, u_channel, node_u_channel);
//        Vec u1; Eigen::Matrix3<T> gradu1;
//        //VelocityAndJacobian(acc, pos, node_u_channel, u1, gradu1);
//		vel_func.velocityAndJacobian(pos, time, u1, gradu1);
//
//        Vec pos1 = pos + u1 * 0.5 * dt;
//        //Vec vel2 = InterpolateFaceValue(acc, pos1, u_channel, node_u_channel);
//        //auto gradu2 = VelocityJacobian(acc, pos1, node_u_channel);
//
//        Vec u2; Eigen::Matrix3<T> gradu2;
//        //VelocityAndJacobian(acc, pos1, node_u_channel, u2, gradu2);
//		vel_func.velocityAndJacobian(pos1, time, u2, gradu2);
//        auto dTdt2 = -matT * gradu2;
//        pos = pos + dt * u2;
//        matT = matT + dt * dTdt2;
//    };
//
//    {
//        //test if F@T=I
//        thrust::host_vector<Vec> psi_analytical(points.size());
//        thrust::host_vector<Vec> phi_of_psi_analytical(points.size());
//        tbb::parallel_for(0, static_cast<int>(points.size()), [&](int i) {
//            psi_analytical[i] = rk4_psi_analytical(points[i], test_flowmap_stride, 0);
//            phi_of_psi_analytical[i] = rk4_phi_analytical(psi_analytical[i], 0, test_flowmap_stride);
//            }
//        );
//
//		thrust::device_vector<Vec> psi_flowmap_d = points_d;
//		thrust::device_vector<Vec> phi_of_psi_flowmap_d = points_d;
//
//        //F_forward is F
//        //T_forward is T
//        //F_back should be T
//		//T_back should be F
//		thrust::device_vector<Eigen::Matrix3<T>> F_flowmap_back_d(points.size());
//		thrust::device_vector<Eigen::Matrix3<T>> T_flowmap_back_d(points.size());
//		thrust::device_vector<Eigen::Matrix3<T>> F_flowmap_forward_d(points.size());
//        thrust::device_vector<Eigen::Matrix3<T>> T_flowmap_forward_d(points.size());
//		thrust::device_vector<Eigen::Matrix3<T>> FT_flowmap_d(points.size());//F
//
//		auto points_d_ptr = thrust::raw_pointer_cast(points_d.data());
//		auto psi_flowmap_d_ptr = thrust::raw_pointer_cast(psi_flowmap_d.data());
//		auto phi_of_psi_flowmap_d_ptr = thrust::raw_pointer_cast(phi_of_psi_flowmap_d.data());
//		auto F_flowmap_back_d_ptr = thrust::raw_pointer_cast(F_flowmap_back_d.data());
//		auto T_flowmap_back_d_ptr = thrust::raw_pointer_cast(T_flowmap_back_d.data());
//		auto F_flowmap_forward_d_ptr = thrust::raw_pointer_cast(F_flowmap_forward_d.data());
//		auto T_flowmap_forward_d_ptr = thrust::raw_pointer_cast(T_flowmap_forward_d.data());
//		auto FT_flowmap_d_ptr = thrust::raw_pointer_cast(FT_flowmap_d.data());
//        LaunchIndexFunc([=]__device__(int i) {
//            Vec psi = points_d_ptr[i]; Eigen::Matrix3<T> F_back = Eigen::Matrix3<T>::Identity();
//            Vec psi1 = psi; Eigen::Matrix3<T> T_back = Eigen::Matrix3<T>::Identity();
//
//            HATileInfo<Tile> info; Coord l_ijk; Vec frac;
//            accs_d_ptr[test_flowmap_stride - 1].findLeafVoxelAndFrac(psi, info, l_ijk, frac);
//            if (info.empty()) return;
//            int reference_level = info.mLevel;
//
//            //printf("psi initial: %f %f %f reference level %d\n", psi[0], psi[1], psi[2], reference_level);
//            for (int i = test_flowmap_stride - 1; i >= 0; i--) {
//                //RK2ForwardPositionAndF(accs_d_ptr[i], -dt, u_channel, node_u_channel, psi, F_back);
//                //RK2ForwardPositionAndT(accs_d_ptr[i], -dt, u_channel, node_u_channel, psi1, T_back);
//
//                RK4ForwardPositionAndF(accs_d_ptr[i], -dt, u_channel, node_u_channel, psi, F_back);
//                RK4ForwardPositionAndT(accs_d_ptr[i], -dt, u_channel, node_u_channel, psi1, T_back);
//
//                //RK4ForwardPositionAndFAtGivenLevel(accs_d_ptr[i], reference_level, -dt, u_channel, node_u_channel, psi, F_back);
//                //printf("psi at i=%d: %f %f %f\n", i, psi[0], psi[1], psi[2]);
//                //return;
//
//                //RK4ForwardPositionAndT(accs_d_ptr[i], -dt, u_channel, node_u_channel, psi1, T_back);
//
//				//rk2_forward_phi_f_hybrid(i*dt, -dt, accs_d_ptr[i], u_channel, node_u_channel, psi, F_back);
//				//rk2_forward_phi_t_hybrid(i*dt, -dt, accs_d_ptr[i], u_channel, node_u_channel, psi1, T_back);
//
//            }
//            psi_flowmap_d_ptr[i] = psi1;
//            F_flowmap_back_d_ptr[i] = F_back;
//            T_flowmap_back_d_ptr[i] = T_back;
//
//            Vec phi = psi; Eigen::Matrix3<T> F_forward = Eigen::Matrix3<T>::Identity();
//            Vec phi1 = psi; Eigen::Matrix3<T> T_forward = Eigen::Matrix3<T>::Identity();
//            for (int i = 0; i < test_flowmap_stride; i++) {
//                //RK2ForwardPositionAndF(accs_d_ptr[i], dt, u_channel, node_u_channel, phi, F_forward);
//                //RK2ForwardPositionAndT(accs_d_ptr[i], dt, u_channel, node_u_channel, phi1, T_forward);
//
//                RK4ForwardPositionAndF(accs_d_ptr[i], dt, u_channel, node_u_channel, phi, F_forward);
//                RK4ForwardPositionAndT(accs_d_ptr[i], dt, u_channel, node_u_channel, phi1, T_forward);
//
//                //RK4ForwardPositionAndFAtGivenLevel(accs_d_ptr[i], reference_level, dt, u_channel, node_u_channel, phi, F_forward);
//                //RK4ForwardPositionAndT(accs_d_ptr[i], dt, u_channel, node_u_channel, phi1, T_forward);
//
//				//rk2_forward_phi_f_hybrid(i*dt, dt, accs_d_ptr[i], u_channel, node_u_channel, phi, F_forward);
//				//rk2_forward_phi_t_hybrid(i*dt, dt, accs_d_ptr[i], u_channel, node_u_channel, phi1, T_forward);
//
//				//printf("phi at i=%d: %f %f %f\n", i, phi[0], phi[1], phi[2]);
//            }
//            phi_of_psi_flowmap_d_ptr[i] = phi1;
//            F_flowmap_forward_d_ptr[i] = F_forward;
//            T_flowmap_forward_d_ptr[i] = T_forward;
//            //FT_flowmap_d_ptr[i] = F_forward * T_forward;
//            FT_flowmap_d_ptr[i] = F_forward * F_back;
//
//        }, points.size(), 128
//            );
//
//        //if (grid_case == 1) {
//        //    thrust::host_vector<Vec> psi_flowmap = psi_flowmap_d;
//        //    polyscope::init();
//        //    polyscope::registerPointCloud("points", points);
//        //    polyscope::registerPointCloud("psi_flowmap", psi_flowmap);
//        //    IOFunc::AddTilesToPolyscopeVolumetricMesh(*grid_ptrs[0], LEAF, "leaf tiles");
//        //    IOFunc::AddTilesToPolyscopeVolumetricMesh(*grid_ptrs[0], GHOST, "ghost tiles");
//        //    //IOFunc::AddTilesToPolyscopeVolumetricMesh(*grid_ptrs[0], NONLEAF, "NONLEAF tiles");
//        //    polyscope::show();
//        //}
//
//        Info("Test flowmap advection on {} points", points.size());
//        Info("Linf error between analytical phi(psi) and original points: {}", LinfErrorBetweenPointCloud(phi_of_psi_analytical, points));
//        Info("Linf error between flowmap advected phi(psi) and original points: {}", LinfErrorBetweenPointCloud(phi_of_psi_flowmap_d, points));
//		Info("Linf error between F_forward and T_back: {}", LinfErrorBetweenMatrixForbenius2(F_flowmap_forward_d, T_flowmap_back_d));
//		Info("Linf error between T_forward and F_back: {}", LinfErrorBetweenMatrixForbenius2(T_flowmap_forward_d, F_flowmap_back_d));
//		Info("Linf error between flowmap F@T and Identity: {}", LinfErrorBetweenTargetMatrixForbenius2(FT_flowmap_d, Eigen::Matrix3<T>::Identity()));
//        fmt::print("\n");
//
//
//    }
//}

//gamma: vortex strength
//radius: radius of the vortex ring
//delta: thickness of the vortex ring
//center: center of the vortex ring
//unit_x: unit vector of vortex in the x direction
//unit_y: unit vector of vortex in the y direction
//num_samples: number of sample points on the vortex ring
//reference: https://github.com/zjw49246/particle-flow-maps/blob/main/3D/init_conditions.py
__device__ void addVortexRingInitialVelocityAndSmoke(const Vec& pos, double gamma, float radius, float delta, const Vec& center, const Vec& unit_x, const Vec& unit_y, int num_samples, Vec& velocity, T& smoke) {


    // Curve length per sample point
    float curve_length = (2 * M_PI * radius) / num_samples;

    // Loop through each sample point on the vortex ring
    for (int l = 0; l < num_samples; ++l) {
        float theta = l / float(num_samples) * 2 * M_PI;
        Vec p_sampled = radius * (cos(theta) * unit_x + sin(theta) * unit_y) + center;

        Vec p_diff = pos - p_sampled;
        float r = p_diff.length();

        Vec w_vector = gamma * (-sin(theta) * unit_x + cos(theta) * unit_y);
        float decay_factor = exp(-pow(r / delta, 3));

        // Biot-Savart law contribution
        velocity += curve_length * (-1 / (4 * M_PI * r * r * r)) * (1 - decay_factor) * p_diff.cross(w_vector);

        // Smoke density update based on decay factor
        smoke += (curve_length * decay_factor);
    }
}

//void TestParticleToGridTransfer(const int grid_case) {
//    uint32_t scale = 8;
//    float h = 1.0 / scale;
//
//    //0:8, 1:16, 2:32, 3:64, 4:128, 5:256, 6:512, 7:1024
//    HADeviceGrid<Tile> grid(h, { 16,16,16,16,16,16,16,16,16,16 });
//
//    grid.setTileHost(0, nanovdb::Coord(0, 0, 0), Tile(), LEAF);
//    grid.compressHost();
//    grid.syncHostAndDevice();
//    SpawnGhostTiles(grid);
//
//    //Deformation3D vel_func;
//	auto vel_func = [=]__device__(const Vec & pos, const double time) {
//        double radius = 0.21;
//        double x_gap = 0.625 * radius;
//        double x_start = 0.16;
//        double delta = 0.08 * radius;
//        double gamma = radius * 0.1;
//        int num_samples = 500;
//
//        Vec unit_x(0, 0, -1);
//        Vec unit_y(0, 1, 0);
//
//        Vec velocity(0, 0, 0);
//        T smoke = 0;
//
//
//        addVortexRingInitialVelocityAndSmoke(pos, gamma, radius, delta, Vec(x_start, 0.5, 0.5), unit_x, unit_y, num_samples, velocity, smoke);
//        addVortexRingInitialVelocityAndSmoke(pos, gamma, radius, delta, Vec(x_start + x_gap, 0.5, 0.5), unit_x, unit_y, num_samples, velocity, smoke);
//        smoke = (smoke > 0.002f) ? 1.0 : 0.0;
//        return velocity;
//	};
//
//    //IterativeRefine(grid, levelTarget);
//    IterativeRefine(grid, [=]__device__(const HATileAccessor<Tile>&acc, HATileInfo<Tile>&info) { return NumericalTestsLevelTarget(acc, info, grid_case); });
//    grid.launchVoxelFunc(
//        [=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
//        auto& tile = info.tile();
//
//        bool is_neumann = false;
//        acc.iterateSameLevelNeighborVoxels(info, l_ijk, [&]__device__(const HATileInfo<Tile>&n_info, const Coord & n_l_ijk, const int axis, const int sgn) {
//            if (n_info.empty()) {
//                is_neumann = true;
//            }
//        });
//        if (is_neumann) tile.type(l_ijk) = NEUMANN;
//        else tile.type(l_ijk) = INTERIOR;
//    }, -1, LEAF, LAUNCH_SUBTREE
//    );
//    CalcCellTypesFromLeafs(grid);
//
//    std::vector<thrust::device_vector<Particle>> level_particles;
//    auto holder_ptr = grid.getHostTileHolder(LEAF | NONLEAF | GHOST, -1);
//    //GenerateParticlesUniformlyOnFinestLevel(holder_ptr, 2, particles);
//
//    int max_level = grid.mMaxLevel;
//    level_particles.resize(max_level + 1);
//    for (int i = 0; i <= max_level; i++) {
//        GenerateParticlesUniformlyOnGivenLevel(holder_ptr, i, LEAF | NONLEAF, 2, level_particles[i]);
//
//		auto particle_ptr = thrust::raw_pointer_cast(level_particles[i].data());
//		LaunchIndexFunc(
//			[=]__device__(int idx) {
//			auto& p = particle_ptr[idx];
//			Vec pos = p.pos;
//			Vec vel = vel_func(pos, 0);
//			p.impulse = vel;
//			p.matT = Eigen::Matrix3<T>::Identity();
//			p.gradm = Eigen::Matrix3<T>::Zero();
//		}, level_particles[i].size()); 
//    }
//
//    //012: uw
//    //678: u
//    //345: ue (error of u,v,w)
//    //10: max error
//    int u_channel = Tile::u_channel;
//    int uw_channel = 0;
//    int ue_channel = 3;
//    int emax_channel = Tile::dye_channel;
//    //P2G on all levels
//    {
//        grid.launchVoxelFuncOnAllTiles(
//            [=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
//            auto& tile = info.tile();
//            for (int axis : {0, 1, 2}) {
//                tile(u_channel + axis, l_ijk) = 0;
//                tile(uw_channel + axis, l_ijk) = 0;
//            }
//        }, LEAF | GHOST | NONLEAF, 4
//        );
//
//        for (int level = 0; level <= grid.mMaxLevel; level++) {
//            auto acc = grid.deviceAccessor();
//            auto particles_ptr = thrust::raw_pointer_cast(level_particles[level].data());
//            LaunchIndexFunc(
//                [=]__device__(int idx) {
//                auto& p = particles_ptr[idx];
//                Vec pos = p.pos;
//                Vec vel = MatrixTimesVec(p.matT.transpose(), p.impulse);
//
//                for (int axis : {0, 1, 2}) {
//                    KernelScatterVelocityComponentMAC2(acc, level, axis, u_channel, uw_channel, pos, vel, p.gradm);
//                }
//
//                //KernelScatterVelocityMAC2(acc, u_channel, uw_channel, pos, vel, p.gradm);
//
//
//            }, level_particles[level].size());
//        }
//
//        grid.launchVoxelFuncOnAllTiles(
//            [=] __device__(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk) {
//            auto& tile = info.tile();
//
//            for (int axis : {0, 1, 2}) {
//                auto w = tile(uw_channel + axis, l_ijk);
//                if (w > 0) {
//                    tile(u_channel + axis, l_ijk) /= w;
//                }
//            }
//        }, LEAF | NONLEAF, 8
//        );
//
//    }
//
//
//
//    //set analytical velocity for boundary part
//    grid.launchVoxelFunc(
//        [=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
//        auto& tile = info.tile();
//        auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
//
//        for (int axis : {0, 1, 2}) {
//			auto fpos = acc.faceCenter(axis, info, l_ijk);
//            double lo = 1.0 / 16, hi = 1 - lo;
//            bool halo = false;
//            for (int ii : {0, 1, 2}) {
//                if (!(lo <= fpos[ii] && fpos[ii] <= hi)) {
//                    halo = true;
//                }
//            }
//            if (halo) {
//				tile(u_channel + axis, l_ijk) = vel_func(fpos, 0)[axis];
//            }
//
//			auto analytical_u_i = vel_func(fpos, 0)[axis];
//			auto u_i = tile(u_channel + axis, l_ijk);
//			tile(ue_channel + axis, l_ijk) = analytical_u_i - u_i;
//
//        }
//
//        tile(emax_channel, l_ijk) = max(
//            abs(tile(ue_channel + 0, l_ijk)), 
//            max(
//                abs(tile(ue_channel + 1, l_ijk)), 
//                abs(tile(ue_channel + 2, l_ijk))
//            )
//        );
//
//    }, -1, LEAF, LAUNCH_SUBTREE
//    );
//
//	//Info("Test particle to grid transfer on {} particles", particles.size());
//    Info("Linf error of velocity: {}", VelocityLinfSync(grid, ue_channel, LEAF));
//
//
//
//    //holder = grid.getHostTileHolderForLeafs();
//    //polyscope::init();
//    //IOFunc::AddPoissonGridCellCentersToPolyscopePointCloud(holder, { {-1,"type"},{emax_channel,"max error"} }, {});
//    //polyscope::show();
//
//    CalculateVorticityMagnitudeOnLeafs(grid, Tile::u_channel, 0, Tile::vor_channel);
//
//    auto base_dir = fs::current_path() / "data" / fmt::format("p2g_test{}", grid_case);
//	fs::create_directories(base_dir);
//	auto holder = grid.getHostTileHolderForLeafs();
//    IOFunc::OutputPoissonGridAsStructuredVTI(holder, { {-1,"type"},{Tile::vor_channel, "vorticity"} }, {}, base_dir / "grid.vti");
//	IOFunc::OutputTilesAsVTU(holder, base_dir / "tiles.vtu");
//    //std::vector<Vec> particle_positions(particles.size());
//    //std::vector<Vec> particle_velocities(particles.size());
//    //std::transform(particles.begin(), particles.end(), particle_positions.begin(), [](const Particle& p) {return p.pos; });
//    //std::transform(particles.begin(), particles.end(), particle_velocities.begin(), [](const Particle& p) {return p.vel; });
//    //polyscope::init();
//    //auto ps_points = polyscope::registerPointCloud("points", particle_positions);
//    //ps_points->addVectorQuantity("velocities", particle_velocities);
// //   polyscope::show();
//}

template<class FuncVT>
void SetGridTypesAndVelocityWithFunction(HADeviceGrid<Tile>& grid, const int u_channel, const int node_u_channel, const double time, FuncVT vel_func) {
    grid.launchVoxelFuncOnAllTiles(
        [=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
        auto& tile = info.tile();

        bool is_neumann = false;
        acc.iterateSameLevelNeighborVoxels(info, l_ijk, [&]__device__(const HATileInfo<Tile>&n_info, const Coord & n_l_ijk, const int axis, const int sgn) {
            if (n_info.empty()) {
                is_neumann = true;
            }
        });
        if (is_neumann) tile.type(l_ijk) = NEUMANN;
        else tile.type(l_ijk) = INTERIOR;

        for (int axis : {0, 1, 2}) {
            auto fpos = acc.faceCenter(axis, info, l_ijk);
            tile(u_channel + axis, l_ijk) = vel_func(fpos, time)[axis];
        }

    }, LEAF
    );
    CalcCellTypesFromLeafs(grid);
    InterpolateVelocitiesAtAllTiles(grid, u_channel, node_u_channel);
}

void TestBinaryFileIO(int grid_case) {
    Info("align of poissontile: {}", alignof(Tile));
    Info("align of particle: {}", alignof(Particle));
    Info("align of record: {}", alignof(ParticleRecord));

    Info("Test Binary File IO on grid case {}", grid_case);

    float h = 1.0 / 8;

    std::shared_ptr<HADeviceGrid<Tile>> grid_ptr;
    double current_time = 0;

    int u_channel = Tile::u_channel;
    int node_u_channel = 0;

    Deformation3D vel_func;

    {
        grid_ptr = std::make_shared<HADeviceGrid<Tile> >(h, std::initializer_list<uint32_t>({ 16, 16, 16, 16, 16, 16, 18, 16, 16, 16 }));


        auto& grid = *grid_ptr;
        grid.setTileHost(0, nanovdb::Coord(0, 0, 0), Tile(), LEAF);
        grid.compressHost(false);
        grid.syncHostAndDevice();
        SpawnGhostTiles(grid, false);
        //always refine to 128^3
        //IterativeRefine(grid, [=]__device__(const HATileAccessor<Tile>&acc, HATileInfo<Tile>&info) { return max_level; }, false);
        IterativeRefine(grid, [=]__device__(const HATileAccessor<Tile>&acc, HATileInfo<Tile>&info) { return NumericalTestsLevelTarget(acc, info, grid_case); }, false);

        SetGridTypesAndVelocityWithFunction(grid, u_channel, node_u_channel, current_time, vel_func);
    }

    auto& grid = *grid_ptr;
    fs::create_directories("data");
    fs::path file_path = "data/grid.bin";

    auto holder_ptr = grid.getHostTileHolder(LEAF | NONLEAF | GHOST);
    IOFunc::WriteHAHostTileHolderToFile(*holder_ptr, file_path);


	auto holder1 = IOFunc::ReadHAHostTileHolderFromFile(file_path);
    auto grid1_ptr = std::make_shared<HADeviceGrid<Tile> >(h, std::initializer_list<uint32_t>({ 16, 16, 16, 16, 16, 16, 18, 16, 16, 16 }));
	auto& grid1 = *grid1_ptr;
    grid1.setTilesFromHolder(holder1);
    grid1.compressHost(false);
    grid1.syncHostAndDevice();
	SpawnGhostTiles(grid1, false);

    //same max level
    Assert(grid.mMaxLevel == grid1.mMaxLevel, "Max levels mismatch");
    for (int i = 0; i <= grid.mMaxLevel; i++) {
        Assert(grid.hNumTiles[i] == grid1.hNumTiles[i], "NumTiles mismatch at level {}", i);
    }
    //same dot(u,v)
    auto dot1 = Dot(grid, u_channel, u_channel + 1, LEAF);
	auto dot2 = Dot(grid1, u_channel, u_channel + 1, LEAF);
	Assert(dot1 == dot2, "Dot(u,v) mismatch");
	//same linf(u,v,w)
	auto linf1 = VelocityLinfSync(grid, u_channel, LEAF);
	auto linf2 = VelocityLinfSync(grid1, u_channel, LEAF);
	Assert(linf1 == linf2, "Linf(u,v,w) mismatch");
    Pass("test passed");
}

__device__ uint8_t LeapFrogCellType(const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk) {
    //x-, x+ air
    //other walls
    bool is_neumann = false;
    bool is_dirichlet = false;
    acc.iterateSameLevelNeighborVoxels(info, l_ijk, [&]__device__(const HATileInfo<Tile>&n_info, const Coord & n_l_ijk, const int axis, const int sgn) {
        if (n_info.empty()) {
            //if (axis == 0) is_dirichlet = true;
            //else is_neumann = true;
            is_neumann = true;
        }
    });

    if (is_neumann) return NEUMANN;
    else if (is_dirichlet) return DIRICHLET;
    else return INTERIOR;
}


std::shared_ptr<HADeviceGrid<Tile>> LeapFrogGridPtr(double h, const int coarse_level, const int fine_level) {
    auto grid_ptr = std::make_shared<HADeviceGrid<Tile> >(h, std::initializer_list<uint32_t>({ 16, 16, 16, 16, 16, 16, 18, 16, 16, 16 }));

    auto& grid = *grid_ptr;
    grid.setTileHost(0, nanovdb::Coord(0, 0, 0), Tile(), LEAF);
    grid.compressHost(false);
    grid.syncHostAndDevice();
    SpawnGhostTiles(grid, false);
    //always refine to 128^3
    //IterativeRefine(grid, [=]__device__(const HATileAccessor<Tile>&acc, HATileInfo<Tile>&info) { return max_level; }, false);
    IterativeRefine(grid, [=]__device__(const HATileAccessor<Tile>&acc, HATileInfo<Tile>&info) { return coarse_level; }, false);
    while (true) {
        grid.launchVoxelFuncOnAllTiles(
            [=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
            double radius = 0.21;
            double x_gap = 0.625 * radius;
            double x_start = 0.16;
            double delta = 0.08 * radius;
            double gamma = radius * 0.1;
            int num_samples = 500;

            Vec unit_x(0, 0, -1);
            Vec unit_y(0, 1, 0);

            Vec velocity(0, 0, 0);
            T smoke = 0;


            auto pos = acc.cellCenter(info, l_ijk);

            addVortexRingInitialVelocityAndSmoke(pos, gamma, radius, delta, Vec(x_start, 0.5, 0.5), unit_x, unit_y, num_samples, velocity, smoke);
            addVortexRingInitialVelocityAndSmoke(pos, gamma, radius, delta, Vec(x_start + x_gap, 0.5, 0.5), unit_x, unit_y, num_samples, velocity, smoke);
            smoke = (smoke > 0.002f) ? 1.0 : 0.0;

            auto& tile = info.tile();
            tile.type(l_ijk) = LeapFrogCellType(acc, info, l_ijk);
            tile(Tile::dye_channel, l_ijk) = smoke;
        }, -1, LEAF
        );
        int cnt = RefineWithValuesOneStep(grid, Tile::dye_channel, 0.1, coarse_level, fine_level, false);
        if (cnt == 0) break;
    }
    return grid_ptr;
}

//void TestLeapFrogAdaptiveFlowMapAdvection(void) {
//    Info("Test Flowmap Advection with Leapfrog Grid");
//
//    int flowmap_stride = 1;
//    float h = 1.0 / 8;
//    int coarse_level = 3;
//    int fine_level = 4;
//
//    std::vector<std::shared_ptr<HADeviceGrid<Tile>>> grid_ptrs;
//    std::vector<double> time_steps;
//    double cfl = 1.0;//max vel=1
//    double dt = h / (1 << 4) * cfl;
//    double current_time = 0;
//
//    //012: node u
//    //678: face u
//    int node_u_channel = 0;
//    int u_channel = Tile::u_channel;
//
//    Deformation3D vel_func;
//
//
//    for (int i = 0; i <= flowmap_stride; i++) {
//        auto grid_ptr = LeapFrogGridPtr(h, coarse_level, fine_level);
//        grid_ptrs.push_back(grid_ptr);
//        time_steps.push_back(dt);
//
//        SetGridTypesAndVelocityWithFunction(*grid_ptr, u_channel, node_u_channel, current_time, vel_func);
//
//        current_time += dt;
//    }
//
//    RandomGenerator rng;
//    thrust::host_vector<Vec> points(10000);
//    double lo = 1.0 / 16, hi = 1 - lo;
//    //double lo = 1.0 / 8, hi = 1 - lo;
//    for (auto& p : points) {
//        p = Vec(rng.uniform(lo, hi), rng.uniform(lo, hi), rng.uniform(lo, hi));
//    }
//    //points[0] = Vec(0.37903517, 0.73239285, 0.2486223);
//    //points.resize(1);
//
//    //points[0] = points[1127];
//    //points.resize(1);
//
//    //thrust::device_vector<Vec> points_d = points;
//    thrust::host_vector<Particle> particles_h(points.size());
//    for (int i = 0; i < points.size(); i++) {
//        particles_h[i].pos = points[i];
//        particles_h[i].impulse = Vec(0, 0, 0);
//        particles_h[i].matT = Eigen::Matrix3<T>::Identity();
//        particles_h[i].gradm = Eigen::Matrix3<T>::Zero();
//    }
//    thrust::device_vector<Particle> particles_d = particles_h;
//
//  //  {
//  //      auto acc0 = grid_ptrs[0]->deviceAccessor();
//		//thrust::device_vector<Vec> w_sum_d(points.size());
//		//thrust::device_vector<T> w_min_d(points.size());
//		//auto w_sum_d_ptr = thrust::raw_pointer_cast(w_sum_d.data());
//		//auto w_min_d_ptr = thrust::raw_pointer_cast(w_min_d.data());
//		//auto particles_d_ptr = thrust::raw_pointer_cast(particles_d.data());
//  //      LaunchIndexFunc(
//  //          [=]__device__(int idx) {
//  //          Vec vel; Eigen::Matrix3<T> gradu; Vec w_sum;
//  //          Vec pos = particles_d_ptr[idx].pos;
//  //          KernelIntpVelocityAndJacobianMAC2AtGivenLevel(acc0, fine_level, pos, u_channel, vel, gradu, w_sum);
//  //          w_sum_d_ptr[idx] = w_sum;
//  //          w_min_d_ptr[idx] = min(w_sum[0], min(w_sum[1], w_sum[2]));
//  //      }, points.size(), 128
//  //          );
//		//thrust::host_vector<Vec> w_sum_h = w_sum_d;
//		//thrust::host_vector<T> w_min_h = w_min_d;
//
//  //      polyscope::init();
//		//auto pc = polyscope::registerPointCloud("points", points);
//		//pc->addVectorQuantity("w_sum", w_sum_h);
//  //      pc->addScalarQuantity("w_min", w_min_h);
//		//polyscope::show();
//
//  //  }
//    //return;
//
//    //PFM integration
//    {
//        //reset grid particles
//        ResetParticleImpulse(*grid_ptrs[0], u_channel, node_u_channel, particles_d);
//
//        for (int i = 0; i < flowmap_stride; i++) {
//            auto& mid_grid = *grid_ptrs[i];
//            ResetParticlesGradM(mid_grid, u_channel, node_u_channel, particles_d);
//            AdvectParticlesRK4Forward(mid_grid, u_channel, node_u_channel, time_steps[i], particles_d);
//        }
//
//        particles_h = particles_d;
//    }
//    //return;
//
//    //Info("points: {}", points);
//
//    //polyscope show
//    {
//        polyscope::init();
//        polyscope::registerPointCloud("initial points", points);
//        IOFunc::AddParticleSystemToPolyscope(particles_d, "particles");
//		IOFunc::AddTilesToPolyscopeVolumetricMesh(*grid_ptrs[0], LEAF, "LEAF tiles");
//        polyscope::show();
//    }
//
//    thrust::device_vector<Vec> nfm_impulse_d = points;
//    thrust::host_vector<Eigen::Matrix3<T>> nfm_matT_h(points.size());
//
//    //NFM integration
//    {
//        //get end points from particles_h
//        thrust::host_vector<Vec> end_points(points.size());
//        for (int i = 0; i < points.size(); i++) {
//            end_points[i] = particles_h[i].pos;
//        }
//        thrust::device_vector<Vec> end_points_d = end_points;
//
//        thrust::host_vector<HATileAccessor<Tile>> accs_h;
//        for (int i = 0; i < flowmap_stride; i++) {
//            accs_h.push_back(grid_ptrs[i]->deviceAccessor());
//        }
//        thrust::device_vector<HATileAccessor<Tile>> accs_d = accs_h;
//        auto accs_d_ptr = thrust::raw_pointer_cast(accs_d.data());
//
//        thrust::device_vector<double> time_steps_d = time_steps;
//        auto time_steps_d_ptr = thrust::raw_pointer_cast(time_steps_d.data());
//
//        thrust::device_vector<Eigen::Matrix3<T>> nfm_matT_d = nfm_matT_h;
//        auto nfm_matT_d_ptr = thrust::raw_pointer_cast(nfm_matT_d.data());
//
//        thrust::device_vector<Vec> psi_d = end_points_d;
//        auto psi_d_ptr = thrust::raw_pointer_cast(psi_d.data());
//        auto end_points_ptr = thrust::raw_pointer_cast(end_points_d.data());
//        auto nfm_impulse_ptr = thrust::raw_pointer_cast(nfm_impulse_d.data());
//        LaunchIndexFunc(
//            [=]__device__(int idx) {
//            Vec psi = end_points_ptr[idx];
//            Eigen::Matrix3<T> matT = Eigen::Matrix3<T>::Identity();
//            //printf("idx: %d flowmap_stride:%d\n", idx, flowmap_stride);
//            for (int i = flowmap_stride - 1; i >= 0; i--) {
//                auto mid_acc = accs_d_ptr[i];
//
//
//
//                RK4ForwardPositionAndF(mid_acc, -time_steps_d_ptr[i], u_channel, node_u_channel, psi, matT);
//
//                //Vec phi = psi; Eigen::Matrix3<T> F1 = Eigen::Matrix3<T>::Identity();
//                //RK4ForwardPositionAndF(mid_acc, time_steps_d_ptr[i], u_channel, node_u_channel, phi, F1);
//                //Vec endpoint = end_points_ptr[idx];
//                //printf("endpoint: %f %f %f phi(psi): %f %f %f\n", endpoint[0], endpoint[1], endpoint[2], phi[0], phi[1], phi[2]);
//            }
//            Vec m0 = InterpolateFaceValue(accs_d_ptr[0], psi, u_channel, node_u_channel);
//            Vec m1 = MatrixTimesVec(matT.transpose(), m0);
//            nfm_matT_d_ptr[idx] = matT;
//            nfm_impulse_ptr[idx] = m1;
//            psi_d_ptr[idx] = psi;
//        }, end_points_d.size(), 128);
//
//        nfm_matT_h = nfm_matT_d;
//        Info("Linf error between original points and psi(phi): {}", LinfErrorBetweenPointCloud(points, psi_d));
//    }
//
//    {
//        thrust::host_vector<Vec> nfm_impulse = nfm_impulse_d;
//        thrust::host_vector<Vec> pfm_impulse = points;
//        thrust::host_vector<Eigen::Matrix3<T>> pfm_matT(points.size());
//        for (int i = 0; i < points.size(); i++) {
//            pfm_impulse[i] = MatrixTimesVec(particles_h[i].matT.transpose(), particles_h[i].impulse);
//            pfm_matT[i] = particles_h[i].matT;
//        }
//
//        Info("Linf error between PFM impulse and NFM impulse: {}", LinfErrorBetweenPointCloud(pfm_impulse, nfm_impulse));
//        Info("Linf error between PFM matT and NFM matT: {}", LinfErrorBetweenMatrixForbenius2(pfm_matT, nfm_matT_h));
//    }
//
//    fmt::print("\n");
//}

//void TestLeapFrogAdaptiveNFMVorticityAdvection(void) {
//    Info("Test NFM Advection with Leapfrog Grid");
//
//    int flowmap_stride = 5;
//    float h = 1.0 / 8;
//    int coarse_level = 3;
//    int fine_level = 4;
//
//    std::vector<std::shared_ptr<HADeviceGrid<Tile>>> grid_ptrs;
//    std::vector<double> time_steps;
//    double cfl = 1.0;//max vel=1
//    double dt = h / (1 << 4) * cfl;
//    double current_time = 0;
//
//    //012: node u
//    //678: face u
//    int node_u_channel = 0;
//    int u_channel = Tile::u_channel;
//
//    Deformation2D vel_func;
//    
//
//
//    auto base_dir = fs::current_path() / "data" / fmt::format("leapforg_nfm_vorticity_test_{}_{}", coarse_level, fine_level);
//    fs::create_directories(base_dir);
//
//    for (int i = 0; i <= flowmap_stride; i++) {
//        auto grid_ptr = LeapFrogGridPtr(h, coarse_level, fine_level);
//        grid_ptrs.push_back(grid_ptr);
//        time_steps.push_back(dt);
//
//        SetGridTypesAndVelocityWithFunction(*grid_ptr, u_channel, node_u_channel, current_time, vel_func);
//
//        CalculateVorticityMagnitudeOnLeafs(*grid_ptr, u_channel, node_u_channel, Tile::vor_channel);
//
//
//
//		auto holder = grid_ptr->getHostTileHolderForLeafs();
//
//        //polyscope::init();
//        //IOFunc::AddPoissonGridCellCentersToPolyscopePointCloud(holder, { {-1,"type"},{Tile::vor_channel, "vorticity"} }, {});
//        //polyscope::show();
//
//        IOFunc::OutputPoissonGridAsStructuredVTI(holder, { {-1,"type"},{Tile::vor_channel, "vorticity"} }, {}, base_dir / fmt::format("analytical_fluid{:04d}.vti", i));
//
//        if (i == 0) {
//			IOFunc::OutputTilesAsVTU(holder, base_dir / "tiles.vtu");
//        }
//
//        current_time += dt;
//    }
//
//    thrust::host_vector<HATileAccessor<Tile>> accs_h;
//    for (int i = 0; i <= flowmap_stride; i++) {
//        accs_h.push_back(grid_ptrs[i]->deviceAccessor());
//    }
//    thrust::device_vector<HATileAccessor<Tile>> accs_d = accs_h;
//    auto accs_d_ptr = thrust::raw_pointer_cast(accs_d.data());
//
//    thrust::device_vector<double> time_steps_d = time_steps;
//    auto time_steps_d_ptr = thrust::raw_pointer_cast(time_steps_d.data());
//
//    for (int n = 0; n <= flowmap_stride; n++) {
//        auto grid_ptr = LeapFrogGridPtr(h, coarse_level, fine_level);
//
//        grid_ptr->launchVoxelFuncOnAllTiles(
//            [=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
//            auto& tile = info.tile();
//			tile.type(l_ijk) = INTERIOR;
//
//
//			for (int axis : {0, 1, 2}) {
//                Vec psi = acc.faceCenter(axis, info, l_ijk);
//                Eigen::Matrix3<T> matT = Eigen::Matrix3<T>::Identity();
//
//                for (int i = n - 1; i >= 0; i--) {
//                    auto mid_acc = accs_d_ptr[i];
//                    RK4ForwardPositionAndF(mid_acc, -time_steps_d_ptr[i], u_channel, node_u_channel, psi, matT);
//                }
//
//                Vec m0 = InterpolateFaceValue(accs_d_ptr[0], psi, u_channel, node_u_channel);
//                Vec m1 = MatrixTimesVec(matT.transpose(), m0);
//
//                tile(Tile::u_channel + axis, l_ijk) = m1[axis];
//			}
//
//        }, LEAF, 4);
//
//		InterpolateVelocitiesAtAllTiles(*grid_ptr, u_channel, node_u_channel);
//        CalculateVorticityMagnitudeOnLeafs(*grid_ptr, u_channel, node_u_channel, Tile::vor_channel);
//
//
//        auto holder = grid_ptr->getHostTileHolderForLeafs();
//        IOFunc::OutputPoissonGridAsStructuredVTI(holder, { {-1,"type"},{Tile::vor_channel, "vorticity"} }, {}, base_dir / fmt::format("nfm_fluid{:04d}.vti", n));
//
//    }
//}

void TestP2GTransfer(const int grid_case) {
    Info("Test P2G transfer with grid case {}", grid_case);

    float h = 1.0 / Tile::DIM;
    //0:8, 1:16, 2:32, 3:64, 4:128, 5:256, 6:512, 7:1024
    HADeviceGrid<Tile> grid(h, std::initializer_list<uint32_t>{ 16, 16, 16, 16, 16, 16, 16, 16, 16, 16 });
    grid.setTileHost(0, nanovdb::Coord(0, 0, 0), Tile(), LEAF);
    grid.compressHost(false);
    grid.syncHostAndDevice();
    SpawnGhostTiles(grid, false);

    IterativeRefine(grid, [=]__device__(const HATileAccessor<Tile>&acc, HATileInfo<Tile>&info) { return NumericalTestsLevelTarget(acc, info, grid_case); }, false);
    grid.launchVoxelFunc(
        [=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
        auto& tile = info.tile();
        tile.type(l_ijk) = INTERIOR;
    }, -1, LEAF, LAUNCH_SUBTREE
    );
    CalcCellTypesFromLeafs(grid);

    Deformation3D vel_func;

    thrust::device_vector<Particle> particles_d;
	auto holder = grid.getHostTileHolderForLeafs();
    GenerateParticlesUniformlyOnGivenLevel(holder, grid.mMaxLevel, LEAF, 2, particles_d);

 //   polyscope::init();
	//IOFunc::AddTilesToPolyscopeVolumetricMesh(grid, LEAF, "LEAF tiles");
	//IOFunc::AddParticleSystemToPolyscope(particles, "particles");
 //   polyscope::show();

	auto particles_d_ptr = thrust::raw_pointer_cast(particles_d.data());
    LaunchIndexFunc(
        [=]__device__(int idx) {
        auto& p = particles_d_ptr[idx];
        Vec vel = vel_func(p.pos, 0);
        p.impulse = vel;
        p.matT = Eigen::Matrix3<T>::Identity();
        p.gradm = Eigen::Matrix3<T>::Zero();
    }, particles_d.size());

    Info("particle size: {}", sizeof(Particle));

    //345: u weight
    //678: u,v,w
    int u_channel = Tile::u_channel;
    int uw_channel = 3;
    CPUTimer timer; timer.start();
    ParticleImpulseToGridMACIntp(grid, particles_d, u_channel, uw_channel);
    cudaDeviceSynchronize();
    T elapsed = timer.stop("P2G");
    T num_particles_M = particles_d.size() / (1024. * 1024.);
    Info("P2G transfer time: {} ms, {}M particles, throughput {}M particles/s", elapsed, num_particles_M, num_particles_M / elapsed * 1000);
}

double LinfErrorBetweenParticleImpulse(const thrust::host_vector<Particle>& pc1, const thrust::host_vector<Particle>& pc2) {
    Assert(pc1.size() == pc2.size(), "LinfErrorBetweenParticleImpulse pc1 size {} pc2 size {}", pc1.size(), pc2.size());
    double error = 0;
    int max_error_idx = -1;
    int both_invalid_count = 0;
    int one_invalid_count = 0;
    int total_count = pc1.size();

    for (int i = 0; i < pc1.size(); i++) {
        bool is_pc1_invalid = (pc1[i].pos == Vec(NODATA, NODATA, NODATA));
        bool is_pc2_invalid = (pc2[i].pos == Vec(NODATA, NODATA, NODATA));

        if (is_pc1_invalid && is_pc2_invalid) {
            both_invalid_count++;
            continue;
        }

        if (is_pc1_invalid || is_pc2_invalid) {
            one_invalid_count++;
            continue;
        }

        Vec impulse_1 = MatrixTimesVec(pc1[i].matT.transpose(), pc1[i].impulse);
        Vec impulse_2 = MatrixTimesVec(pc2[i].matT.transpose(), pc2[i].impulse);
        double current_error = (impulse_1 - impulse_2).length();

        if (current_error > error) {
            error = current_error;
            max_error_idx = i;
        }
    }

    Info("Maximum error {} occurred at index {}", error, max_error_idx);
    Info("Both invalid count: {} ({:.2f}%), One invalid count: {} ({:.2f}%)",
        both_invalid_count, 100.0 * both_invalid_count / total_count,
        one_invalid_count, 100.0 * one_invalid_count / total_count);
    return error;
}


void TestG2PAdvectionEffeciency(int grid_case) {
    Info("align of poissontile: {}", alignof(Tile));
    Info("align of particle: {}", alignof(Particle));
    Info("align of record: {}", alignof(ParticleRecord));

    Info("Test G2P Advection Effeciency on grid case {}", grid_case);

    float h = 1.0 / 8;

    std::shared_ptr<HADeviceGrid<Tile>> grid_ptr;
    double current_time = 0;

    //012: node u
    //678: face u
    int node_u_channel = 0;
    int u_channel = Tile::u_channel;
    int counter_channel = Tile::vor_channel;

    Deformation3D vel_func;

    {
        grid_ptr = std::make_shared<HADeviceGrid<Tile> >(h, std::initializer_list<uint32_t>({ 16, 16, 16, 16, 16, 16, 18, 16, 16, 16 }));


        auto& grid = *grid_ptr;
        grid.setTileHost(0, nanovdb::Coord(0, 0, 0), Tile(), LEAF);
        grid.compressHost(false);
        grid.syncHostAndDevice();
        SpawnGhostTiles(grid, false);
        //always refine to 128^3
        //IterativeRefine(grid, [=]__device__(const HATileAccessor<Tile>&acc, HATileInfo<Tile>&info) { return max_level; }, false);
        IterativeRefine(grid, [=]__device__(const HATileAccessor<Tile>&acc, HATileInfo<Tile>&info) { return NumericalTestsLevelTarget(acc, info, grid_case); }, false);

        SetGridTypesAndVelocityWithFunction(grid, u_channel, node_u_channel, current_time, vel_func);
    }

	auto& grid = *grid_ptr;
    double cfl = 0.5;
    auto acc_h = grid.hostAccessor();
    double dx = acc_h.voxelSize(grid.mMaxLevel);
	double dt = dx * cfl;

    thrust::device_vector<Particle> particles_d;
    auto holder = grid.getHostTileHolderForLeafs();
    GenerateParticlesUniformlyOnGivenLevel(holder, grid.mMaxLevel, LEAF, 2, particles_d);
    Info("Initial {} particles", particles_d.size());

    //particles_d[0] = particles_d[3567559];
    //particles_d.resize(1);

	ResetParticleImpulse(grid, u_channel, node_u_channel, particles_d);

    //vanilla single-level advect
    thrust::device_vector<Particle> particles_vanilla_d = particles_d;
    Info("particles_vanilla_d {} particles", particles_vanilla_d.size());
    {
        CPUTimer timer; timer.start();
        //AdvectParticlesAndSingleStepGradMRK4Forward(grid, u_channel, node_u_channel, dt, particles_d);
        AdvectParticlesAndSingleStepGradMRK4ForwardAtGivenLevel(grid, grid.mMaxLevel, u_channel, node_u_channel, dt, particles_vanilla_d, false);
		cudaDeviceSynchronize();
		T elapsed = timer.stop("G2P advection");
		T num_particles_M = particles_vanilla_d.size() / (1024. * 1024.);
		Info("G2P advection time: {} ms, {}M particles, throughput {}M particles/s", elapsed, num_particles_M, num_particles_M / elapsed * 1000);
    }

    _sleep(200);
    {
		thrust::device_vector<ParticleRecord> records_d;
        thrust::device_vector<int> tile_prefix_sum_d;

		records_d.resize(particles_d.size());
        tile_prefix_sum_d.resize(grid.dAllTiles.size());

        
        HistogramSortParticlesAtGivenLevel(grid, grid.mMaxLevel, counter_channel, particles_d, tile_prefix_sum_d, records_d);
        cudaDeviceSynchronize();
        CPUTimer timer; timer.start();
        OptimizedAdvectParticlesAndSingleStepGradMRK4ForwardAtGivenLevel(grid, grid.mMaxLevel, u_channel, node_u_channel, dt, tile_prefix_sum_d, records_d);
        //not erasing because two functions may disagree on invalid particles
        //EraseInvalidParticles(particles_d);
		cudaDeviceSynchronize();
		T elapsed = timer.stop("Histogram sort");
		T num_particles_M = particles_d.size() / (1024. * 1024.);
		Info("Histogram sort and optimized advection time: {} ms, {}M particles, throughput {}M particles/s", elapsed, num_particles_M, num_particles_M / elapsed * 1000);
	}

    _sleep(200);

	//double linf_err = LinfErrorBetweenParticleImpulse(particles_d, particles_vanilla_d);
 //   if (linf_err < 1e-6) {
 //       Pass("Linf error between optimized and vanilla: {}", linf_err);
 //   }
 //   else {
 //       Error("Linf error between optimized and vanilla: {}", linf_err);
 //   }
    fmt::print("\n");
}

void TestParticleToGridTransferEfficiency(int grid_case) {
    Info("Test Particle-to-Grid Transfer Efficiency on grid case {}", grid_case);

    Info("particle size: {}", sizeof(Particle));
    Info("record size: {}", sizeof(ParticleRecord));
    double tile_size = sizeof(Tile);
    Info("tile size {} voxel size {}", tile_size, tile_size / 512);

    float h = 1.0 / 8;

    std::shared_ptr<HADeviceGrid<Tile>> grid_ptr;
    double current_time = 0;

    // 012: node u
    // 678: face u
    int node_u_channel = 0;
    int u_channel = Tile::u_channel;
    int uw_channel = 0;
    int u1_channel = 3;
    int counter_channel = 9;

    Deformation3D vel_func;

    {
        grid_ptr = std::make_shared<HADeviceGrid<Tile>>(h, std::initializer_list<uint32_t>({ 16, 16, 16, 16, 16, 16, 18, 16, 16, 16 }));

        auto& grid = *grid_ptr;
        grid.setTileHost(0, nanovdb::Coord(0, 0, 0), Tile(), LEAF);
        grid.compressHost(false);
        grid.syncHostAndDevice();
        SpawnGhostTiles(grid, false);

        IterativeRefine(grid, [=]__device__(const HATileAccessor<Tile>&acc, HATileInfo<Tile>&info) {
            return NumericalTestsLevelTarget(acc, info, grid_case);
        }, false);

        SetGridTypesAndVelocityWithFunction(grid, u_channel, node_u_channel, current_time, vel_func);
    }

    auto& grid = *grid_ptr;
    double cfl = 0.5;
    auto acc_h = grid.hostAccessor();
    double dx = acc_h.voxelSize(grid.mMaxLevel);
    double dt = dx * cfl;

    thrust::device_vector<Particle> particles_d;
    auto holder = grid.getHostTileHolderForLeafs();
    GenerateParticlesRandomlyInVoxels(holder, grid.mMaxLevel, LEAF, 8, particles_d);
    //particles_d.resize(1);

    Info("Initial {} particles", particles_d.size());

	auto particles_d_ptr = thrust::raw_pointer_cast(particles_d.data());
	LaunchIndexFunc(
		[=]__device__(int idx) {
		auto& p = particles_d_ptr[idx];
		Vec vel = vel_func(p.pos, 0);
		p.impulse = vel;
		p.matT = Eigen::Matrix3<T>::Identity();
		p.gradm = Eigen::Matrix3<T>::Zero();
	}, particles_d.size());


    // Test Particle-to-Grid (P2G) transfer efficiency
    {
        CPUTimer timer;
        timer.start();

        ParticleImpulseToGridMACIntp(grid, particles_d, u_channel, uw_channel);

        cudaDeviceSynchronize();
        double elapsed = timer.stop("Particle-to-Grid Transfer");

        double num_particles_M = particles_d.size() / (1024. * 1024.);
        Info("Particle-to-Grid Transfer time: {} ms, {}M particles, throughput {}M particles/s", elapsed, num_particles_M, num_particles_M / elapsed * 1000);
    }

	// Run another pass of P2G transfer to measure throughput
    {
        thrust::device_vector<ParticleRecord> records_d;
        thrust::device_vector<int> tile_prefix_sum_d;

        records_d.resize(particles_d.size());
        tile_prefix_sum_d.resize(grid.dAllTiles.size());

        CPUTimer timer;
        timer.start();
        //ParticleImpulseToGridMACIntp(grid, particles_d, u1_channel, uw_channel);
        HistogramSortParticlesAtGivenLevel(grid, grid.mMaxLevel, counter_channel, particles_d, tile_prefix_sum_d, records_d);
        cudaDeviceSynchronize(); 
        double elapsed0 = timer.stop("Histogram sort"); timer.start();
        OptimizedP2GTransferAtGivenLevel(grid, grid.mMaxLevel, u1_channel, uw_channel, tile_prefix_sum_d, records_d);
        cudaDeviceSynchronize();
        double elapsed1 = timer.stop("Particle-to-Grid Transfer");
        double num_particles_M = particles_d.size() / (1024. * 1024.);
        Info("Particle-to-Grid Transfer time: {} ms, {}M particles, throughput {}M particles/s", elapsed1, num_particles_M, num_particles_M / elapsed1 * 1000);
        //Info("total: {}ms, throughput {}M particles/s", elapsed0 + elapsed1, num_particles_M / (elapsed0 + elapsed1) * 1000);
        Info("total: {}ms, throughput {}M particles/s", elapsed0 + elapsed1, num_particles_M / (elapsed0 + elapsed1) * 1000);
    }

    //test the difference
    {
        grid.launchVoxelFuncOnAllTiles(
            [=] __device__(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk) {
            auto& tile = info.tile();
            for (int axis : {0, 1, 2}) {
                tile(uw_channel + axis, l_ijk) = tile(u_channel + axis, l_ijk) - tile(u1_channel + axis, l_ijk);

                //if (abs(tile(uw_channel + axis, l_ijk)) > 1e-6) {
                //    auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
                //    printf("axis %d g_ijk: %d %d %d uw: %f u: %f u1: %f\n", axis, g_ijk[0], g_ijk[1], g_ijk[2], tile(uw_channel + axis, l_ijk), tile(u_channel + axis, l_ijk), tile(u1_channel + axis, l_ijk));
                //}
            
            }
        }, LEAF, 4);

        double err = VelocityLinfSync(grid, uw_channel, LEAF);
        if (err < 1e-5) {
            Pass("Linf error between two P2G transfers: {}", err);
        }
        else {
            Error("Linf error between two P2G transfers: {}", err);
        }
    }

    fmt::print("\n");
}



void TestCertainAdvection(int num_tiles) {
    fs::path particles_file = fs::path("data") / "particles_007_255.bin";
    fs::path grid_file = fs::path("data") / "grid_007_255.bin";

    thrust::device_vector<Particle> particles = IOFunc::ReadHostVectorFromBinary<Particle>(particles_file);
    auto holder = IOFunc::ReadHAHostTileHolderFromFile(grid_file);

    Info("particles and holder read done");

    auto run_tiles = [&](const int tile_num) {
        double h = 1.0 / 8;
        auto grid_ptr = std::make_shared<HADeviceGrid<Tile> >(h, std::initializer_list<uint32_t>({ 16, 16, 16, 16, 16, 16, 18, 16, 16, 16 }));
        auto& last_grid = *grid_ptr;
        last_grid.setTilesFromHolder(holder);
        last_grid.compressHost(false);
        last_grid.syncHostAndDevice();
        SpawnGhostTiles(last_grid, false);

        thrust::device_vector<ParticleRecord> records_d;
        thrust::device_vector<int> tile_prefix_sum_d;
        int u_channel = Tile::u_channel;//6
        int last_u_node_channel = 0;//on last_grid
        int last_tmp_channel = 3;

        HistogramSortParticlesAtGivenLevel(last_grid, 6,last_tmp_channel, particles, tile_prefix_sum_d, records_d);

        //Info("tile_prefix_sum_d: {}", tile_prefix_sum_d);
        //Info("total {} particles", particles.size());

        cudaDeviceSynchronize(); CheckCudaError("HistogramSortParticles");
        //Info("HistogramSortParticles done");
        //OptimizedAdvectParticlesAndSingleStepGradMRK4ForwardAtGivenLevel(last_grid, 6, u_channel, last_u_node_channel, 0.0002469679349881029, tile_prefix_sum_d, records_d);

        auto all_infos_ptr = thrust::raw_pointer_cast(last_grid.dAllTiles.data());
        auto tile_prefix_sum_ptr = thrust::raw_pointer_cast(tile_prefix_sum_d.data());
        auto records_ptr = thrust::raw_pointer_cast(records_d.data());
        auto acc = last_grid.deviceAccessor();

        //Info("OptimizedAdvectParticlesAndSingleStepGradMRK4ForwardAtGivenLevel launch {} tiles ", grid.dAllTiles.size());

        //{
        //    thrust::host_vector<ParticleRecord> records_h = records_d;
        //    auto particles_ptr = thrust::raw_pointer_cast(particles.data());
        //    for (int i = 16777033; i < 16778723; i++) {
        //        auto rcd = records_h[i];
        //        Info("i: {} particle idx: {}", i, rcd.ptr - particles_ptr);
        //    }
        //}

        OptimizedAdvectParticlesAndSingleStepGradMRK4ForwardAtGivenLevel128Kernel << <tile_num, 128 >> > (acc, all_infos_ptr, 6, u_channel, last_u_node_channel, .0002469679349881029, tile_prefix_sum_ptr, records_ptr, 1e-4);

        cudaDeviceSynchronize(); CheckCudaError("OptimizedAdvectParticlesAndSingleStepGradMRK4ForwardAtGivenLevel");
        // Info("adv done");

        };

    //   double h = 1.0 / 8;
    //   auto grid_ptr = std::make_shared<HADeviceGrid<Tile> >(h, std::initializer_list<uint32_t>({ 16, 16, 16, 16, 16, 16, 18, 16, 16, 16 }));
       //auto& last_grid = *grid_ptr;
    //   last_grid.setTilesFromHolder(holder);
    //   last_grid.compressHost(false);
    //   last_grid.syncHostAndDevice();
    //   SpawnGhostTiles(last_grid, false);

    num_tiles = holder.mHostTiles.size();
    Info("run {} tiles", num_tiles);
    run_tiles(num_tiles);
    //int base = 24192;
    //int tn = 1;
    //while (base+tn <= 24255) {
    //    Info("base+tn={}", base + tn);
    //    run_tiles(base + tn);
    //    //tn *= 2;
    //    tn++;
    //}
    //run_tiles(24255);

}