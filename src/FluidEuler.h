//////////////////////////////////////////////////////////////////////////
// Fluid Euler
// Copyright (c) (2022-), Bo Zhu, Mengdi Wang
// This file is part of MESO, whose distribution is governed by the LICENSE file.
//////////////////////////////////////////////////////////////////////////
#pragma once

#include "Simulator.h"
#include "FluidParams.h"
//#include "Random.h"
#include "GPUTimer.h"
#include "PoissonSolver.h"

//#include "AMGSolver.h"
#include "PoissonIOFunc.h"
#include "FMParticles.h"


#include <sys/types.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/sysinfo.h>
#endif

template<class FuncV>
__global__ void MarkInterestAreaWithPointFunction128Kernel(FuncV point_func, HATileAccessor<PoissonTile<T>> acc, HATileInfo<PoissonTile<T>>* infos, T threshold, int subtree_level, uint8_t launch_types) {
	int bi = blockIdx.x;
	int ti = threadIdx.x;
	const HATileInfo<PoissonTile<T>>& info = infos[bi];

	auto& tile = info.tile();
	if (!(info.subtreeType(subtree_level) & launch_types)) {
		if (ti == 0) tile.mIsInterestArea = false;
		return;
	}

	// 定义一个共享存储来存储 block 内的最大值
	typedef cub::BlockReduce<T, 128> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage;

	// 初始化当前线程的局部最大值为负无穷
	T local_max = -FLT_MAX;

	// 遍历 tile 中的 cell centers
	for (int cell_idx = ti; cell_idx < Tile::SIZE; cell_idx += blockDim.x) {
		Coord l_ijk = acc.localOffsetToCoord(cell_idx);
		Vec pos = acc.cellCenter(info, l_ijk);
		T value = point_func(pos);          // 使用 intp() 查询值
		local_max = max(local_max, value);             // 更新局部最大值
	}

	// 计算 block 内的最大值
	T block_max = BlockReduce(temp_storage).Reduce(local_max, cub::Max());

	if (ti == 0) {
		// 判断 block 最大值是否超过阈值
		if (block_max > threshold) {
			tile.mIsInterestArea = true;
		}
		else {
			tile.mIsInterestArea = false;
		}
	}
}

template<class FuncV>
void RefineWithPointValue(FuncV point_func, HADeviceGrid<Tile>& grid, FluidParams params, bool verbose) {
	int coarse_level = params.mCoarseLevel;
	int fine_level = params.mFineLevel;
	auto levelTarget = [fine_level, coarse_level]__device__(const HATileAccessor<Tile> &acc, const HATileInfo<Tile> &info) ->int {
		auto& tile = info.tile();
		if (tile.mIsInterestArea) return fine_level;
		return coarse_level;
	};

	while (true) {
		auto info_ptr = thrust::raw_pointer_cast(grid.dAllTiles.data());
		//calculate interest area flags on leafs
		MarkInterestAreaWithPointFunction128Kernel << <grid.dAllTiles.size(), 128 >> > (point_func, grid.deviceAccessor(), info_ptr, params.mRefineThreshold, -1, LEAF);
		auto refine_cnts = RefineLeafsOneStep(grid, levelTarget, verbose);
		SpawnGhostTiles(grid, verbose);
		if (verbose) Info("Refine {} tiles on each layer", refine_cnts);
		int cnt = std::accumulate(refine_cnts.begin(), refine_cnts.end(), 0);
		if (cnt == 0) break;
	}
}

template<class FuncV>
void CoarsenWithPointValue(FuncV point_func, HADeviceGrid<Tile>& grid, FluidParams params, bool verbose) {
	int coarse_level = params.mCoarseLevel;
	int fine_level = params.mFineLevel;
	auto levelTarget = [fine_level, coarse_level]__device__(const HATileAccessor<Tile> &acc, const HATileInfo<Tile> &info) ->int {
		auto& tile = info.tile();
		if (tile.mIsInterestArea) return fine_level;
		return coarse_level;
	};

	while (true) {
		auto info_ptr = thrust::raw_pointer_cast(grid.dAllTiles.data());
		//calculate interest area flags on leafs
		MarkInterestAreaWithPointFunction128Kernel << <grid.dAllTiles.size(), 128 >> > (point_func, grid.deviceAccessor(), info_ptr, params.mRefineThreshold, -1, LEAF);
		auto coarsen_cnts = CoarsenStep(grid, levelTarget, verbose);

		if (verbose) Info("Coarsen {} tiles on each layer", coarsen_cnts);
		int cnt = std::accumulate(coarsen_cnts.begin(), coarsen_cnts.end(), 0);
		if (cnt == 0) break;
	}
	SpawnGhostTiles(grid, verbose);

}

__device__ Vec NFMErodedAdvectionPoint(const int axis, const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const Coord& l_ijk);

//set all face velocities that has a neumann neighbor to 0
void ClearAllNeumannNeighborFaces(HADeviceGrid<Tile>& grid);
void FixIsolatedInteriorCells(HADeviceGrid<Tile>& grid, const int tmp_channel);

//void CalculateVorticityMagnitudeOnLeafs(HADeviceGrid<Tile>& grid, const int u_channel, const int tmp_u_node_channel, const int vor_channel);
void MarkOldParticlesAsInvalid(thrust::device_vector<Particle>& particles, const T current_time, const T particle_life);




//__device__ Vec RK4ForwardPosition(const HATileAccessor<Tile>& acc, const Vec& pos, const double dt, const int u_channel, const int node_u_channel);
__device__ Vec SemiLagrangianBackwardPosition(const HATileAccessor<Tile>& acc, const Vec& pos, const T dt, const int u_channel, const int node_u_channel);
//void AdvectParticlesRK2Forward(HADeviceGrid<Tile>& grid, const int u_channel, const int node_u_channel, const double dt, thrust::device_vector<Particle>& particles_d);

int LockedRefineWithNonBoundaryNeumannCellsOneStep(const T current_time, HADeviceGrid<Tile>& grid, const FluidParams params, const int tmp_channel, bool verbose);
void ReseedParticles(HADeviceGrid<Tile>& grid, const FluidParams& params, const int tmp_channel, const double current_time, const int num_particles_per_cell, thrust::device_vector<Particle>& particles);

class FluidEuler : public Simulator {
public:
	using Tile = PoissonTile<T>;
	using Coord = nanovdb::Coord;

	int mNumParticlesPerCell = 8;
	int time_step_counter = 0;

	std::shared_ptr<HADeviceGrid<Tile>> nfm_query_grid_ptr;
	std::vector<std::shared_ptr<HADeviceGrid<Tile>>> grid_ptrs;
	std::vector<double> time_steps;
	//std::shared_ptr<HADeviceGrid<Tile>> grid_ptr;
	//std::shared_ptr<HADeviceGrid<Tile>> last_grid_ptr;
	//thrust::device_vector<Particle> particles;
	//std::vector<thrust::device_vector<Particle>> level_particles;
	thrust::device_vector<Particle> particles;
	thrust::device_vector<ParticleRecord> records_d;
	thrust::device_vector<int> tile_prefix_sum_d;

	FluidParams mParams;
	std::shared_ptr<MaskGrid> mMaskGrid = nullptr;
	std::shared_ptr<SDFGrid> mSDFGrid = nullptr;

	std::shared_ptr<SDFGrid> mAnimationSDFGrid0 = nullptr;
	std::shared_ptr<SDFGrid> mAnimationSDFGrid1 = nullptr;
	std::shared_ptr<SDFGrid> mAnimationVelocityGrids[3];

	std::string animation_sdf_path = "";
	fs::path flamingo_data_file = fs::path("data") / "flamingo-sdf.bin";
	int loaded_animated_frame = -1;

	int cell_center_vel_channel = 3;

	double advance_time = 0;
	double projection_time = 0;
	double reseeding_time = 0;
	double particle_advection_time = 0;
	double adaptive_time = 0;
	double p2g_time = 0;
	double nfm_advection_time = 0;

	void applyVelocityBC(HADeviceGrid<Tile>& grid, const double time) {
		//if (mParams.mTestCase == TVORTEX || mParams.mTestCase==FORCE_) 
		{
			ClearAllNeumannNeighborFaces(grid);
			auto params = mParams;
			grid.launchVoxelFuncOnAllTiles(
				[params, time] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
				auto& tile = info.tile();
				//if (!tile.isInterior(l_ijk)) {
					params.setBoundaryCondition(acc, info, l_ijk, time);
				//}
			}, LEAF
			);
		}
	}

	void applyDyeDensityBC(HADeviceGrid<Tile>& grid, const double time) {
		//if (mParams.mTestCase == TVORTEX) {
		{
			auto params = mParams;
			grid.launchVoxelFuncOnAllTiles(
				[params] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
				params.enforceDyeDensityBoundaryCondition(acc, info, l_ijk);
			}, LEAF, 1
			);
		}
	}

	__device__ Vec freeVortexVelocityLambOseen(const Vec& position, const Vec& vortexCenter, double Gamma, double delta) {
		Vec r_vec = position - vortexCenter;
		r_vec[2] = 0;
		double r = r_vec.length();
		if (r < 1e-6) {
			return Vec(0, 0, 0);
		}

		double v_theta = (Gamma / (2 * M_PI * r)) * (1 - std::exp(-r * r / (delta * delta)));
		Vec tangentialVelocity(-r_vec[1] / r, r_vec[0] / r, 0);

		return tangentialVelocity * v_theta;
	}

	__device__ void setFreeVortexInitialLambOseen(HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const Coord& l_ijk, const Vec& vortexCenter, double gamma, double delta) {
		auto pos = acc.cellCenter(info, l_ijk);
		auto vel = freeVortexVelocityLambOseen(pos, vortexCenter, gamma, delta);
		auto& tile = info.tile();
		tile(Tile::u_channel, l_ijk) += vel[0];
		tile(Tile::u_channel + 1, l_ijk) += vel[1];
		tile(Tile::u_channel + 2, l_ijk) += vel[2];

		auto r_vec = pos - vortexCenter;
		r_vec[2] = 0;
		if (r_vec.length() < delta) {
			tile(Tile::dye_channel, l_ijk) = 1;
		}
	}

	//gamma: vortex strength
	//radius: radius of the vortex ring
	//delta: thickness of the vortex ring
	//center: center of the vortex ring
	//unit_x: unit vector of vortex in the x direction
	//unit_y: unit vector of vortex in the y direction
	//num_samples: number of sample points on the vortex ring
	//reference: https://github.com/zjw49246/particle-flow-maps/blob/main/3D/init_conditions.py
	__device__ void addVortexRingInitialAndSmoke(HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const Coord& l_ijk, double gamma, float radius, float delta, const Vec& center, const Vec& unit_x, const Vec& unit_y, int num_samples) {
		auto pos = acc.cellCenter(info, l_ijk);
		auto& tile = info.tile();

		// Curve length per sample point
		float curve_length = (2 * M_PI * radius) / num_samples;


		Vec velocity(0, 0, 0);
		float smoke = 0;
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

		// Apply smoke color if the density is high enough
		for (int axis : {0, 1, 2}) {
			tile(Tile::u_channel + axis, l_ijk) += velocity[axis];
		}
		if (smoke > 0.002f) {
			tile(Tile::dye_channel, l_ijk) = 1.0;
		}
	}

	std::tuple<int, T> getBatAnimationFrameAndFrace(const T time) {
		int bat_fps = 200;
		T bat_f = time * bat_fps;
		int bat_frame = (int)bat_f;
		T frac = bat_f - bat_frame;
		return std::make_tuple(bat_frame, frac);
	}
	fs::path getBatCyclicAnimationFile(const fs::path& animation_sdf_path, int bat_frame) {
		//int cyc_idx = bat_frame % 205 + 1;
		int cyc_idx = bat_frame;
		return animation_sdf_path / fmt::format("{:04d}.bin", cyc_idx);
	}
	fs::path getBatCyclicVelocityFile(const fs::path& animation_sdf_path, int bat_frame, int axis) {
		//int cyc_idx = bat_frame % 205 + 1;
		int cyc_idx = bat_frame;
		if (axis == 0) return animation_sdf_path / fmt::format("{:04d}_x.bin", cyc_idx);
		else if (axis == 1) return animation_sdf_path / fmt::format("{:04d}_y.bin", cyc_idx);
		else if (axis == 2) return animation_sdf_path / fmt::format("{:04d}_z.bin", cyc_idx);
		else return fs::path("");
	}

	fs::path getFlamingoAnimationFile(const fs::path& animation_sdf_path, int frame) {
		return animation_sdf_path / fmt::format("flamingo-flock{:04d}.bin", frame);
	}
	fs::path generateFlamingoFileWithPython(const fs::path& animation_sdf_path, const int frame, const T frac, const T isovalue) {
		// Conda environment name and Python script path
		std::string conda_env = "py311_env";
		std::string python_script = fmt::format(
			"C:\\Code\\HASimulator\\gen_single_flamingo.py  --mesh_path {} --frame {} --frac {} --scale 256 --isovalue {} --output_file {}",
			animation_sdf_path.string(),
			frame,
			frac,
			isovalue,
			flamingo_data_file.string()
		);
		
		// Command to activate conda environment and execute the Python script
		std::string command = fmt::format(
			"conda run -n py311_env python {}",
			python_script
		);


		Info("Executing Python script: {}", command);
		// Execute the command
		int ret_code = std::system(command.c_str());

		if (ret_code == 0) {
			fmt::print("Python script executed successfully.\n");
		}
		else {
			fmt::print("Failed to execute the Python script. Return code: {}\n", ret_code);
		}

		return flamingo_data_file;
	}


	//void fetchFrameSDF(const int frame_number) {
	//	std::string filename = fmt::format("{}/bat_ani{:04d}.bin", animation_sdf_path, frame_number);
	//	fs::path sdf_file = fs::path(filename);
	//	Assert(fs::exists(sdf_file), "SDF file {} does not exist", sdf_file.string());
	//}

	void init(json &j) {
		//Info("before init"); PrintMemoryInfo();

		std::string sdf_grid_file = Json::Value<std::string>(j, "sdf_grid_file", "");
		float sdf_isovalue = Json::Value<float>(j, "sdf_isovalue", 0.025);
		if (sdf_grid_file != "") {
			float solid_isovalue = Json::Value<float>(j, "solid_isovalue", 0);
			mMaskGrid = std::make_shared<MaskGrid>(sdf_grid_file, solid_isovalue, sdf_isovalue);
		}
		else {
			mMaskGrid = nullptr;
		}
		std::string sdf_float_grid_file = Json::Value<std::string>(j, "sdf_float_grid_file", "");
		if (sdf_float_grid_file != "") {
			float solid_ext_isovalue = Json::Value<float>(j, "solid_ext_isovalue", 0);
			float sdf_ext_isovalue = Json::Value<float>(j, "sdf_ext_isovalue", 0.025);
			mSDFGrid = std::make_shared<SDFGrid>(sdf_float_grid_file, solid_ext_isovalue, sdf_ext_isovalue);
		}
		else {
			mSDFGrid = nullptr;
		}
		animation_sdf_path = Json::Value<std::string>(j, "animation_sdf_path", "");
		Info("animation_sdf_path {}", animation_sdf_path);

		
		
		float reserve_particles_m = Json::Value<float>(j, "reserve_particles_m", 1.0);
		particles.reserve((size_t)(reserve_particles_m * 1024 * 1024));
		mParams = FluidParams(j, mMaskGrid, mSDFGrid);
		mParams.mPureSDFGenerationThreshold = sdf_isovalue;

		if (mParams.mTestCase == BAT) {
			float solid_ext_isovalue = Json::Value<float>(j, "solid_ext_isovalue", 0);
			float sdf_ext_isovalue = Json::Value<float>(j, "sdf_ext_isovalue", 0.025);
			auto file0 = getBatCyclicAnimationFile(animation_sdf_path, 0);
			mSDFGrid = std::make_shared<SDFGrid>(file0, solid_ext_isovalue, sdf_ext_isovalue);

			//for (int axis : {0, 1, 2}) {
			//	auto vel_file = getBatCyclicVelocityFile(animation_sdf_path, 0, axis);
			//	mAnimationVelocityGrids[axis] = std::make_shared<SDFGrid>(vel_file, solid_ext_isovalue, sdf_ext_isovalue);
			//}

			mParams.mSDFGridAccessor = mSDFGrid->GetDeviceAccessor();
			//auto file0 = getBatCyclicAnimationFile(animation_sdf_path, 0);
			//auto file1 = getBatCyclicAnimationFile(animation_sdf_path, 1);
			//mSDFGrid = std::make_shared<SDFGrid>(file0, solid_ext_isovalue, sdf_ext_isovalue);
			//mAnimationSDFGrid0 = std::make_shared<SDFGrid>(file0, solid_ext_isovalue, sdf_ext_isovalue);
			//mAnimationSDFGrid1 = std::make_shared<SDFGrid>(file1, solid_ext_isovalue, sdf_ext_isovalue);
			//loaded_animated_frame = 1;

			//for (int axis : {0, 1, 2}) {
			//	mParams.mSDFVelocityAccessors[axis] = mAnimationVelocityGrids[axis]->GetDeviceAccessor();
			//}
		}
		else if (mParams.mTestCase == FLAMINGO) {
			float solid_ext_isovalue = Json::Value<float>(j, "solid_ext_isovalue", 0);
			float sdf_ext_isovalue = Json::Value<float>(j, "sdf_ext_isovalue", 0.025);

			Info("generate flamingo file");

			//auto file0 = generateFlamingoFileWithPython(animation_sdf_path, 0, 0, solid_ext_isovalue);

			auto file0 = getFlamingoAnimationFile(animation_sdf_path, 0);
			mSDFGrid = std::make_shared<SDFGrid>(file0, solid_ext_isovalue, sdf_ext_isovalue);
			mParams.mSDFGridAccessor = mSDFGrid->GetDeviceAccessor();

			//auto file1 = getFlamingoAnimationFile(animation_sdf_path, 1);
			//mAnimationSDFGrid0 = std::make_shared<SDFGrid>(file1, solid_ext_isovalue, sdf_ext_isovalue);
			//mAnimationSDFGrid1 = std::make_shared<SDFGrid>(file1, solid_ext_isovalue, sdf_ext_isovalue);
			
			loaded_animated_frame = 0;
		}

		//level-resolution:
		//0:8, 1:16, 2:32, 3:64, 4:128, 5:256, 6:512, 7:1024
		double h = 1.0 / 8;
		auto grid_ptr = std::make_shared<HADeviceGrid<Tile> >(h, std::initializer_list<uint32_t>({ 16, 16, 16, 16, 16, 16, 18, 16, 16, 16 }));
		grid_ptrs.clear();
		grid_ptrs.push_back(grid_ptr);
		auto& grid = *grid_ptr;

		if (mParams.mTestCase == LEAP_FROG || mParams.mTestCase == BAT) {
			grid.setTileHost(0, nanovdb::Coord(0, 0, 0), Tile(), LEAF);
			grid.setTileHost(0, nanovdb::Coord(1, 0, 0), Tile(), LEAF);
		}
		else if(mParams.mTestCase == FISH){
			grid.setTileHost(0, nanovdb::Coord(0, 0, 0), Tile(), LEAF);
			grid.setTileHost(0, nanovdb::Coord(1, 0, 0), Tile(), LEAF);
			grid.setTileHost(0, nanovdb::Coord(2, 0, 0), Tile(), LEAF);
			grid.setTileHost(0, nanovdb::Coord(3, 0, 0), Tile(), LEAF);
		}
		else if (mParams.mTestCase == FLAMINGO) {
			grid.setTileHost(0, nanovdb::Coord(0, 0, 0), Tile(), LEAF);
			grid.setTileHost(0, nanovdb::Coord(0, 0, 1), Tile(), LEAF);
			grid.setTileHost(0, nanovdb::Coord(0, 0, 2), Tile(), LEAF);
			grid.setTileHost(0, nanovdb::Coord(0, 0, 3), Tile(), LEAF);
		}
		else if (mParams.mTestCase == NASA || mParams.mTestCase == WP3D || mParams.mTestCase == F1CAR) {
			grid.setTileHost(0, nanovdb::Coord(0, 0, 0), Tile(), LEAF);
			grid.setTileHost(0, nanovdb::Coord(0, 0, 1), Tile(), LEAF);
		}
		else {
			//1*1*1
			grid.setTileHost(0, nanovdb::Coord(0, 0, 0), Tile(), LEAF);
		}
		grid.compressHost();
		grid.syncHostAndDevice();
		SpawnGhostTiles(grid);

		{

			auto params = mParams;
			auto levelTarget = [=]__device__(const HATileAccessor<Tile> &acc, HATileInfo<Tile> &info) ->int {
				return params.initialLevelTarget(acc, info);

				//auto bbox = acc.tileBBox(info);
				//int desired_level = 0;
				//if (bbox.min()[0] <= 0.25) return 4;//slow converging, if 0.25 not converging
				//else return 3;

				//return params.mCoarseLevel;
				//return 3;//64^3 for fast estimation
				//return 5;//256^3 tentative
			};
			IterativeRefine(grid, levelTarget);

			//{
			//	grid.launchVoxelFuncOnAllTiles(
			//		[params] __device__(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk) {
			//		params.setInitialCondition(acc, info, l_ijk);
			//	}, -1, LEAF
			//	);
			//}

			while (true) {
				grid.launchVoxelFuncOnAllTiles(
					[params] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
					params.setInitialCondition(acc, info, l_ijk);
				}, -1, LEAF
				);
				int cnt = RefineWithValuesOneStep(grid, Tile::dye_channel, params.mRefineThreshold, params.mCoarseLevel, params.mFineLevel, false);
				int cnt1 = LockedRefineWithNonBoundaryNeumannCellsOneStep(0.0, grid, params, 0, false);
				if (cnt + cnt1 == 0) break;
			}

			applyVelocityBC(grid, 0.0);
			CalcCellTypesFromLeafs(grid);

			//std::swap(grid_ptr, last_grid_ptr);
			auto holder_ptr = grid.getHostTileHolderForLeafs();
			GenerateParticlesWithDyeDensity(holder_ptr, Tile::dye_channel, mParams.mRefineThreshold, mNumParticlesPerCell, particles);
		}
		CalculateVorticityMagnitudeOnLeafs(*grid_ptr, mParams.mFineLevel, mParams.mCoarseLevel, Tile::u_channel, 0, Tile::vor_channel);


		//{
		//	polyscope::init();
		//	IOFunc::AddPoissonGridCellCentersToPolyscopePointCloud(grid.getHostTileHolderForLeafs(), { { -1,"type" }, {Tile::vor_channel, "vorticity density"} }, { {Tile::u_channel,"vel"} });
		//	polyscope::show();
		//}
		//_sleep(200);

		//Info("end init"); PrintMemoryInfo();
	}

	virtual double CFL_Time(const double cfl) {
		auto& grid = *grid_ptrs.back();
		//return FLT_MAX;
		HATileAccessor<Tile> acc = grid.deviceAccessor();
		double dx = acc.voxelSize(acc.mMaxLevel);
		double max_vel = VelocityLinfSync(grid, Tile::u_channel, LEAF);
		return dx * cfl / max_vel;
	}
	virtual void Output(DriverMetaData& metadata) {
		//return;
		auto& grid = *grid_ptrs.back();
		//if (metadata.current_frame == 0) {
		//	//IOFunc::OutputCellAsPoints(grid, metadata.base_path / "boundary.vtu", true);
		//	//IOFunc::OutputCellAsPoints(grid, metadata.base_path / "grid.vtu", false);
		//	IOFunc::OutputTilesAsVTU(grid, metadata.base_path / "tiles.vtu");
		//}
		
		//{
		//	polyscope::init();
		//	auto holder = grid.getHostTileHolderForLeafs();
		//	IOFunc::AddParticleSystemToPolyscope(particles, "particles");
		//	//IOFunc::AddPoissonGridCellCentersToPolyscopePointCloud(holder, { { -1,"type" }, {Tile::dye_channel, "dye density"} }, { {Tile::u_channel,"vel"} });
		//	//IOFunc::AddPoissonGridFaceCentersToPolyscopePointCloud(holder, { {Tile::u_channel, "vel"} });
		//	polyscope::show();
		//}

		{

			auto holder = grid.getHostTileHolderForLeafs();

			//metadata.Append_Output_Thread(std::make_shared<std::thread>(IOFunc::OutputTilesAsVTU, holder, metadata.base_path / fmt::format("tiles{:04d}.vtu", metadata.current_frame)));
			//IOFunc::OutputTilesAsVTU(holder, metadata.base_path / fmt::format("tiles{:04d}.vtu", metadata.current_frame));

			metadata.Append_Output_Thread(std::make_shared<std::thread>(IOFunc::OutputPoissonGridAsStructuredVTI, holder,
				std::vector<std::pair<int, std::string>>{ {-1,"type"}, { -2, "level" }, {Tile::vor_channel, "vorticity"}},
				//std::vector<std::pair<int, std::string>>{ },
				std::vector<std::pair<int, std::string>>{ {cell_center_vel_channel, "velocity"} },
				//std::vector<std::pair<int, std::string>>{ { -1, "type" }, { Tile::vor_channel, "vorticity" }, { Tile::dye_channel, "dye_density" } },
				//std::vector<std::pair<int, std::string>>{ {Tile::u_channel, "velocity"} },
				metadata.base_path / fmt::format("fluid{:04d}.vti", metadata.current_frame)));

		}

		{
			WriteStatToFile(metadata);
		}

		//{
		//	auto change_drive_to_d = [](const fs::path& original_path) -> fs::path {
		//		if (original_path.has_root_name()) {
		//			return fs::path("D:") / original_path.relative_path();
		//		}
		//		else {
		//			return original_path;
		//		}
		//		};
		//	auto base_path_d = change_drive_to_d(metadata.base_path);
		//	//auto base_path_d = metadata.base_path;
		//	//make directory
		//	fs::create_directories(base_path_d);

		//	auto particles_h_ptr = std::make_shared<thrust::host_vector<Particle>>(particles);
		//	metadata.Append_Output_Thread(std::make_shared<std::thread>(IOFunc::OutputParticleSystemAsVTU,
		//		particles_h_ptr, base_path_d / fmt::format("particles{:04d}.vtu", metadata.current_frame)
		//	));
		//}

		//{
		//	if (metadata.current_frame == 200) {
		//		auto all_holder = grid.getHostTileHolder(LEAF | NONLEAF | GHOST, -1);
		//		IOFunc::WriteHAHostTileHolderToFile(*all_holder, metadata.base_path / "fluid_end.bin");
		//	}
		//}
	}


	void project(HADeviceGrid<Tile>& grid, int u_channel, double current_time) {

		for (int axis : {0, 1, 2}) {
			PropagateToChildren(grid, u_channel + axis, u_channel + axis, -1, GHOST, LAUNCH_SUBTREE, INTERIOR | DIRICHLET | NEUMANN);
		}
		VelocityVolumeDivergenceOnLeafs(grid, u_channel, Tile::b_channel);

		//GMGSolver solver;

		CalculateNeighborTiles(grid);
		GMGSolver solver(1., 1.);
		//AMGSolver solver(Tile::c0_channel, 1., 1.);
		//solver.prepareTypesAndCoeffs(grid);

		CPUTimer timer;
		timer.start();
		//auto [iters, err] = ConjugateGradientSync(grid, false, 1000, 2, 10, 1e-6, false);
		auto [iters, err] = solver.solve(grid, false, 1000, 1e-6, 1, 10, 1, mParams.mIsPureNeumann);
		//auto [iters, err] = solver.dampedJacobiSolve(grid, false, 10, 1e-6, 1.0);
		cudaDeviceSynchronize();
		double elapsed = timer.stop("MGPCG");
		double total_cells = grid.numTotalTiles() * Tile::SIZE;
		double cells_per_second = (total_cells + 0.0) / (elapsed / 1000.0);
		Info("Total {:.5}M cells, MGPCG speed {:.5} M cells /s at {} iters", total_cells / (1024.0 * 1024), cells_per_second / (1024.0 * 1024), iters);
		projection_time = elapsed;

		AddGradientToFaceCenters(grid, Tile::x_channel, u_channel);
		applyVelocityBC(grid, current_time);

		//VelocityVolumeDivergenceOnLeafs(grid, u_channel, Tile::b_channel);
		//Info("After velocity fix div linf {}", VolumeWeightedNorm(grid, -1, Tile::b_channel));
	}

	void adaptAndAdvect(DriverMetaData& metadata, std::vector<std::shared_ptr<HADeviceGrid<Tile>>> grid_ptrs) {
		const double dt = metadata.dt;
		const double current_time = metadata.current_time;


		CPUTimer timer; timer.start();

		//1. advect particles with last_grid
		//2. refine grid with particles
		//3. calculate type, advect dye_density and velocity on grid

		//012: temporary node velocity
		//3: temporary node dye density

		//shared by two grids
		int u_channel = Tile::u_channel;//6
		int dye_channel = Tile::dye_channel;//10

		//last_grid:
		//012: node u
		//345: u copy
		//678: face u
		//9: node dye
		//10: voxel dye
		int last_tmp_channel = 3;
		int last_u_node_channel = 0;//on last_grid
		int last_dye_node_channel = Tile::vor_channel;//on last_grid
		
		//next grid:
		//012: uw
		//3: particle counter
		//678: face u
		//10: voxel dye
		int next_uw_channel = 0;//on grid
		int next_counter_channel = 4;//on grid
		
		//saved intermediate velocities
		int n = grid_ptrs.size() - 1;
		//we only need to prepare the last grid at this time
		auto& last_grid = *grid_ptrs[n - 1];
		InterpolateVelocitiesAtAllTiles(last_grid, u_channel, last_u_node_channel);

		
		//cudaDeviceSynchronize(); timer.stop("Prepare last grid"); timer.start();
		CheckCudaError("prepare last grid");

		thrust::host_vector<HATileAccessor<Tile>> accs_h;
		for (int i = 0; i < n; i++) {
			accs_h.push_back(grid_ptrs[i]->deviceAccessor());
		}
		thrust::device_vector<HATileAccessor<Tile>> accs_d = accs_h;
		auto accs_d_ptr = thrust::raw_pointer_cast(accs_d.data());
		thrust::device_vector<double> time_steps_d = time_steps;
		auto time_steps_d_ptr = thrust::raw_pointer_cast(time_steps_d.data());

		MarkParticlesOutsideFluidRegionAsInvalid(particles, last_grid);
		MarkOldParticlesAsInvalid(particles, current_time, mParams.mParticleLife);
		EraseInvalidParticles(particles);

		ReseedParticles(last_grid, mParams, last_tmp_channel, current_time, mNumParticlesPerCell, particles);
		//cudaDeviceSynchronize(); timer.stop("Reseeding particles"); timer.start();
		



		cudaDeviceSynchronize(); reseeding_time = timer.stop("reseeding and remove particles in solid"); timer.start();
		Info("total {:.5f}M particles, time step counter {}", particles.size() / (1024 * 1024 + 0.f), time_step_counter);

		//{
		//	polyscope::init();
		//	IOFunc::AddTilesToPolyscopeVolumetricMesh(last_grid, LEAF, "leaf tiles");
		//	IOFunc::AddParticleSystemToPolyscope(particles, "particles");
		//	polyscope::show();
		//}


		//reset impulse for all particles
		if (time_step_counter % mParams.mFlowMapStride == 0) {
			//auto holder_ptr = last_grid.getHostTileHolderForLeafs();
			//GenerateParticlesUniformlyOnFinestLevel(holder_ptr, 2, particles);

			//with midpoint velocity on, we have to create a copy
			//nfm_query_grid_ptr = grid_ptrs[n - 1]->deepCopy();
			 
			//without midpoint velocity, we can directly use the last grid
			nfm_query_grid_ptr = grid_ptrs[n - 1];
			ResetParticleImpulse(last_grid, u_channel, last_u_node_channel, particles);

			cudaDeviceSynchronize(); timer.stop("reset all particles impulse"); timer.start();
		}
		else {
			int fine_level = mParams.mFineLevel;
			int coarse_level = mParams.mCoarseLevel;

			int back_traced_steps = time_step_counter % mParams.mFlowMapStride;
			int nfm_start_idx = n - back_traced_steps - 1;
			auto particles_d_ptr = thrust::raw_pointer_cast(particles.data());
			LaunchIndexFunc([=] __device__(int idx) {
				auto& particle = particles_d_ptr[idx];

				if (particle.start_time == current_time) {

					Vec psi = particle.pos;
					Vec m0; Eigen::Matrix3<T> matT;
					NFMBackQueryImpulseAndT(accs_d_ptr, fine_level, coarse_level, time_steps_d_ptr, u_channel, last_u_node_channel, nfm_start_idx, n - 1, psi, m0, matT);
					particle.impulse = m0;
					particle.matT = matT;
				}
			}, particles.size(), 128);

			cudaDeviceSynchronize(); timer.stop("reset newly sampled particles impulse"); timer.start();
		}
		//cudaDeviceSynchronize(); timer.stop("Reset particle impulse"); timer.start();

		//ResetParticlesGradM(last_grid, u_channel, last_u_node_channel, particles);
		//cudaDeviceSynchronize(); timer.stop("Reset particle gradm"); timer.start();
		//CheckCudaError("reinit");




		////midpoint velocity
		//{
		//	//012: node u
		//	//345: u copy
		//	//678: face u
		//	//9: node dye
		//	//10: voxel dye
		//	int last_u_copy_channel = 3;
		//	for (int axis : {0, 1, 2}) {
		//		Copy(last_grid, u_channel + axis, last_u_copy_channel + axis, -1, LEAF, LAUNCH_SUBTREE, INTERIOR | DIRICHLET | NEUMANN);
		//	}
		//	auto last_acc = last_grid.deviceAccessor();
		//	int fine_level = mParams.mFineLevel;
		//	int coarse_level = mParams.mCoarseLevel;
		//	last_grid.launchVoxelFuncOnAllTiles(
		//		[=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
		//		auto& tile = info.tile();
		//		for (int axis : {0, 1, 2}) {
		//			//Vec psi = last_acc.faceCenter(axis, info, l_ijk);
		//			Vec psi = NFMErodedAdvectionPoint(axis, acc, info, l_ijk);
		//			Eigen::Matrix3<T> matT = Eigen::Matrix3<T>::Identity();

		//			//we want forward T to calculate matT.T@m0
		//			//and forward T equals to backward F
		//			RK4ForwardPositionAndF(last_acc, fine_level, coarse_level, -0.5 * dt, last_u_copy_channel, last_u_node_channel, psi, matT);
		//			Vec m0 = InterpolateFaceValue(last_acc, psi, last_u_copy_channel, last_u_node_channel);

		//			Vec m1 = MatrixTimesVec(matT.transpose(), m0);

		//			tile(u_channel + axis, l_ijk) = m1[axis];
		//		}
		//	}, LEAF, 4
		//	);

		//	//AdvectGridImpulseRK4Forward(last_grid, last_u_copy_channel, last_u_node_channel, 0.5 * dt, last_grid, u_channel);
		//	applyVelocityBC(last_grid, current_time);

		//	project(last_grid, u_channel, current_time);
		//	//CalcLeafNodeValuesFromFaceCenters(last_grid, u_channel, last_u_node_channel);
		//	InterpolateVelocitiesAtAllTiles(last_grid, u_channel, last_u_node_channel);

		//	//CheckCudaError("project");
		//	cudaDeviceSynchronize(); timer.stop("Midpoint"); timer.start();
		//}

		

		//RemoveParticlesInNeumannCells(particles, last_grid);

		//if (metadata.current_frame == 7 && time_step_counter >= 253) {
		//	auto holder = last_grid.getHostTileHolder(LEAF | NONLEAF | GHOST);
		//	IOFunc::WriteHAHostTileHolderToFile(*holder, fs::path(metadata.output_base_dir) / fmt::format("grid_{:03d}_{:03d}.bin", metadata.current_frame, time_step_counter));
		//	IOFunc::WriteHostVectorToBinary<Particle>(particles, fs::path(metadata.output_base_dir) / fmt::format("particles_{:03d}_{:03d}.bin", metadata.current_frame, time_step_counter));
		//}
		

		//AdvectParticlesRK4Forward(last_grid, u_channel, last_u_node_channel, dt, particles);
		//AdvectParticlesAndSingleStepGradMRK4Forward(last_grid, u_channel, last_u_node_channel, dt, particles);
		//AdvectParticlesAndSingleStepGradMRK4ForwardAtGivenLevel(last_grid, mParams.mFineLevel, u_channel, last_u_node_channel, dt, particles, 1e-4);
		HistogramSortParticlesAtGivenLevel(last_grid, mParams.mFineLevel, last_tmp_channel, particles, tile_prefix_sum_d, records_d);
		//Info("HistogramSortParticles done");
		//cudaDeviceSynchronize(); CheckCudaError("HistogramSortParticles");
		OptimizedAdvectParticlesAndSingleStepGradMRK4ForwardAtGivenLevel(last_grid, mParams.mFineLevel, u_channel, last_u_node_channel, dt, tile_prefix_sum_d, records_d);
		//cudaDeviceSynchronize(); CheckCudaError("OptimizedAdvectParticlesAndSingleStepGradMRK4ForwardAtGivenLevel");
		//Info("OptimizedAdvectParticlesAndSingleStepGradMRK4ForwardAtGivenLevel done");
		//Info("after pfm advection max gradm {}", LinfNormOfGradMForbenius(particles));
		//Info("optimized advect {} particles", particles.size());
		EraseInvalidParticles(particles);
		//Info("after erasing max gradm {}", LinfNormOfGradMForbenius(particles));
		//Info("after erasing {} particles", particles.size());

		cudaDeviceSynchronize(); double adv_elapsed = timer.stop("Advect particles"); timer.start(); particle_advection_time = adv_elapsed;
		{
			double num_particles_M = particles.size() / (1024. * 1024.);
			Info("Particle advection time: {} ms, {}M particles, throughput {}M particles/s", adv_elapsed, num_particles_M, num_particles_M / adv_elapsed * 1000);
		}
		CheckCudaError("adv particle");

		//Info("end here");
		//exit(-1);

		//{
		//	polyscope::init();
		//	auto holder = last_grid.getHostTileHolderForLeafs();
		//	IOFunc::AddParticleSystemToPolyscope(particles, "particles");
		//	//IOFunc::AddPoissonGridCellCentersToPolyscopePointCloud(holder, { { -1,"type" } }, { {Tile::u_channel,"vel"} });
		//	//IOFunc::AddPoissonGridFaceCentersToPolyscopePointCloud(holder, { {Tile::u_channel, "vel"} });
		//	polyscope::show();
		//}

		auto& grid = *grid_ptrs[n];

		RefineWithParticles(grid, particles, mParams.mCoarseLevel, mParams.mFineLevel, next_counter_channel, false);

		//cudaDeviceSynchronize(); timer.stop("Refine with particles"); timer.start();


		CoarsenWithParticles(grid, particles, mParams.mCoarseLevel, mParams.mFineLevel, next_counter_channel, false);
		CheckCudaError("adapt with particles");


		cudaDeviceSynchronize(); adaptive_time = timer.stop("adapt with particles"); timer.start();


		//ParticleImpulseToGridMACIntp(grid, particles, u_channel, next_uw_channel);
		HistogramSortParticlesAtGivenLevel(grid, mParams.mFineLevel, next_counter_channel, particles, tile_prefix_sum_d, records_d);
		OptimizedP2GTransferAtGivenLevel(grid, mParams.mFineLevel, u_channel, next_uw_channel, tile_prefix_sum_d, records_d);
		EraseInvalidParticles(particles);

		CheckCudaError("pfm p2g");
		//Info("max impulse after pfm: {}", VelocityLinf(grid, u_channel, -1, LEAF, LAUNCH_SUBTREE));


		cudaDeviceSynchronize(); p2g_time = timer.stop("P2G"); timer.start();
		{
			double num_particles_M = particles.size() / (1024. * 1024.);
			Info("P2G time: {} ms, {}M particles, throughput {}M particles/s", p2g_time, num_particles_M, num_particles_M / p2g_time * 1000);
		}

		//advect dye and NFM
		{
			CalcLeafNodeValuesFromCellCenters(last_grid, dye_channel, last_dye_node_channel);



			auto last_acc = last_grid.deviceAccessor();
			auto nfm_query_acc = nfm_query_grid_ptr->deviceAccessor();
			auto params = mParams;
			
			int fine_level = mParams.mFineLevel;
			int coarse_level = mParams.mCoarseLevel;

			int back_traced_steps = time_step_counter % mParams.mFlowMapStride;
			int nfm_start_idx = n - back_traced_steps - 1;
			
			grid.launchVoxelFuncOnAllTiles(
				[=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
				auto& tile = info.tile();
				//if (!tile.isInterior(l_ijk)) return;

				//type
				int boundary_axis, boundary_off;
				tile.type(l_ijk) = params.cellType(current_time, acc, info, l_ijk, boundary_axis, boundary_off);

				//{
				//	auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
				//	Coord diff = g_ijk - Coord(128, 128, 130);
				//	if (abs(diff[0]) + abs(diff[1]) + abs(diff[2]) <= 1) {
				//		printf("g_ijk %d %d %d type %d\n", g_ijk[0], g_ijk[1], g_ijk[2], tile.type(l_ijk));
				//	}
				//}

				//dye density
				{
					auto pos = acc.cellCenter(info, l_ijk);
					auto pos2 = SemiLagrangianBackwardPosition(last_acc, pos, dt, u_channel, last_u_node_channel);
					auto dye2 = InterpolateCellValue(last_acc, pos2, Tile::dye_channel, last_dye_node_channel);
					tile(Tile::dye_channel, l_ijk) = dye2;

					////SL velocity advection
					//for (int axis : {0, 1, 2}) {
					//	auto fpos = acc.faceCenter(axis, info, l_ijk); //face center position for advection
					//	Vec fpos2 = SemiLagrangianBackwardPosition(last_acc, fpos, dt, u_channel, last_u_node_channel);
					//	tile(Tile::u_channel + axis, l_ijk) = InterpolateFaceValue(last_acc, fpos2, u_channel, last_u_node_channel)[axis];
					//}
				}

				{
					//grid velocity advection
					for (int axis : {0, 1, 2}) {
						if (tile(next_uw_channel + axis, l_ijk) < 1 - 1e-3)
						{
							//Vec psi = acc.faceCenter(axis, info, l_ijk);
							Vec psi = NFMErodedAdvectionPoint(axis, acc, info, l_ijk);
							Vec m0; Eigen::Matrix3<T> matT;

							//NFMBackQueryImpulseAndT(accs_d_ptr, info.mLevel, coarse_level, time_steps_d_ptr, u_channel, last_u_node_channel, nfm_start_idx, n, psi, m0, matT);
							//NFMBackQueryImpulseAndT(accs_d_ptr, fine_level, coarse_level, time_steps_d_ptr, u_channel, last_u_node_channel, nfm_start_idx, n, psi, m0, matT);

							NFMBackMarchPsiAndT(accs_d_ptr, fine_level, coarse_level, time_steps_d_ptr, u_channel, last_u_node_channel, nfm_start_idx, n, psi, matT);
							//m0 = InterpolateFaceValue(accs_d_ptr[nfm_start_idx], psi, u_channel, last_u_node_channel);
							m0 = InterpolateFaceValue(nfm_query_acc, psi, u_channel, last_u_node_channel);

							Vec m1 = MatrixTimesVec(matT.transpose(), m0);

							tile(Tile::u_channel + axis, l_ijk) = m1[axis];

							//if (m1[axis] > 1e5) {
							//	auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
							//	printf("g_ijk %d %d %d axis %d m1 %f\n", g_ijk[0], g_ijk[1], g_ijk[2], axis, m1[axis]);
							//}
						}
					}
				}
			}, LEAF, 4
			);
		}


		CalcCellTypesFromLeafs(grid);

		cudaDeviceSynchronize(); nfm_advection_time = timer.stop("NFM advection"); timer.start();
		CheckCudaError("nfm advection");

		//Info("max impulse after nfm: {}", VelocityLinf(grid, u_channel, -1, LEAF, LAUNCH_SUBTREE));
	}

	void applyExternalForce(HADeviceGrid<Tile>& grid, const double dt) {

		//int tmp_ta_node = 0;
		//CalcLeafNodeValuesFromCellCenters(grid, Tile::Ta_channel, tmp_ta_node);


		const nanovdb::Vec3R gravity = mParams.mGravity;
		auto params = mParams;
		grid.launchVoxelFunc(
			[dt, gravity, params] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
			auto& tile = info.tile();
			//if (!tile.isInterior(l_ijk)) return;
			for (int axis : {0, 1, 2}) {
				auto pos = acc.faceCenter(axis, info, l_ijk);

				//auto Ta = InterpolateCellValue(acc, pos, Tile::Ta_channel, tmp_ta_node);
				//auto buoyancy = params.mBuoyancy * (Ta - params.T_ambient);

				nanovdb::Vec3R a = gravity;
				//a[1] += buoyancy;
				tile(Tile::u_channel + axis, l_ijk) += a[axis] * dt;
			}
		}, -1, LEAF, LAUNCH_SUBTREE
		);
	}





	virtual void Advance(DriverMetaData& metadata) {
		CPUTimer timer;
		timer.start();

		if (mParams.mTestCase == BAT) {
			auto [bat_frame, frac] = getBatAnimationFrameAndFrace(metadata.current_time);
			if (loaded_animated_frame != bat_frame) {
				{
					auto sdf_file_n = getBatCyclicAnimationFile(animation_sdf_path, bat_frame);
					mSDFGrid->ReloadSDFFile(sdf_file_n);
				}

				//for (int axis : {0, 1, 2}) {
				//	auto vel_file_n = getBatCyclicVelocityFile(animation_sdf_path, bat_frame, axis);
				//	mAnimationVelocityGrids[axis]->ReloadSDFFile(vel_file_n);
				//}

				loaded_animated_frame = bat_frame;
			}

			//auto [bat_frame, frac] = getBatAnimationFrameAndFrace(metadata.current_time);
			//if (loaded_animated_frame != bat_frame + 1) {
			//	auto next_file = getBatCyclicAnimationFile(animation_sdf_path, bat_frame + 1);
			//	mAnimationSDFGrid0 = mAnimationSDFGrid1;
			//	mAnimationSDFGrid1->ReloadSDFFile(next_file);
			//	loaded_animated_frame = bat_frame + 1;
			//}
			//mSDFGrid->InterpolateFromTwoSDFGrids(*mAnimationSDFGrid0, *mAnimationSDFGrid1, frac);
		}
		else if (mParams.mTestCase == FLAMINGO) {
			int frame = metadata.current_frame;
			T time_past_frame = metadata.current_time - (frame + 0.0) / metadata.fps;
			T frac = time_past_frame * metadata.fps;

			//Info("frame {} time {} fps {} time past frame {} frac {}", frame, metadata.current_time, metadata.fps, time_past_frame, frac);
			//auto sdf_file_n = generateFlamingoFileWithPython(animation_sdf_path, frame, frac, mParams.mSDFGridAccessor.solid_isovalue);
			////auto sdf_file_n = getFlamingoAnimationFile(animation_sdf_path, frame);
			//mSDFGrid->ReloadSDFFile(sdf_file_n);

			if (loaded_animated_frame != frame) {
				auto sfile0 = getFlamingoAnimationFile(animation_sdf_path, frame);
				auto sfile1 = getFlamingoAnimationFile(animation_sdf_path, frame + 1);
				mSDFGrid->ReloadSDFFile(sfile0);
				//mAnimationSDFGrid0->ReloadSDFFile(sfile0);
				//mAnimationSDFGrid1->ReloadSDFFile(sfile1);
				loaded_animated_frame = frame;
			}
			//mSDFGrid->InterpolateFromTwoSDFGrids(*mAnimationSDFGrid0, *mAnimationSDFGrid1, frac);
		}

		//Info("before advance"); PrintMemoryInfo();

		if (grid_ptrs.size() < mParams.mFlowMapStride + 1) {
			//create a new grid
			auto nxt_ptr = grid_ptrs.back()->deepCopy();
			grid_ptrs.push_back(nxt_ptr);
			time_steps.push_back(metadata.dt);

			//polyscope::init();
			//IOFunc::AddTilesToPolyscopeVolumetricMesh(*nxt_ptr, LEAF, "next grid");
			//polyscope::show();
		}
		else {
			auto nxt_ptr = grid_ptrs[0];
			grid_ptrs.erase(grid_ptrs.begin());
			grid_ptrs.push_back(nxt_ptr);
			time_steps.erase(time_steps.begin());
			time_steps.push_back(metadata.dt);
		}

		auto& grid = *grid_ptrs.back();
		auto& last_grid = *grid_ptrs[grid_ptrs.size() - 2];

		//Info("grid ptrs size: {}", grid_ptrs.size());
		//Info("grid total leafs: {}", grid.numTotalLeafTiles());


		double dt = metadata.dt;


		Info("frame {} dt {}", metadata.current_frame, dt);

		adaptAndAdvect(metadata, grid_ptrs);

		//{

		//	polyscope::init();
		//	auto holder = grid.getHostTileHolderForLeafs();
		//	IOFunc::AddPoissonGridCellCentersToPolyscopePointCloud(holder, { { -1,"type" }, {Tile::vor_channel, "vorticity"} ,{Tile::b_channel, "div"} }
		//	, { {Tile::u_channel,"vel"} });
		//	//IOFunc::AddPoissonGridNodesToPolyscope(holder, { {0,"node_vel"} }, { });
		//	polyscope::show();
		//}

		//FixIsolatedInteriorCells(grid, Tile::b_channel);


		applyExternalForce(grid, dt);
		applyDyeDensityBC(grid, metadata.current_time);


		applyVelocityBC(grid, metadata.current_time);


		//projection
		project(grid, Tile::u_channel, metadata.current_time);

		//CalculateVorticityMagnitudeOnLeafs(grid, mParams.mFineLevel, mParams.mCoarseLevel, Tile::u_channel, 0, Tile::vor_channel);
		CalculateVelocityAndVorticityMagnitudeOnLeafFaceCenters(grid, mParams.mFineLevel, mParams.mCoarseLevel, Tile::u_channel, 0, cell_center_vel_channel, Tile::vor_channel);

		//{

		//	polyscope::init();
		//	auto holder = grid.getHostTileHolderForLeafs();
		//	IOFunc::AddPoissonGridCellCentersToPolyscopePointCloud(holder, { { -1,"type" }, {Tile::vor_channel, "vorticity"} ,{Tile::b_channel, "div"} }
		//	, { {Tile::u_channel,"vel"} });
		//	//IOFunc::AddPoissonGridNodesToPolyscope(holder, { {0,"node_vel"} }, { });
		//	polyscope::show();
		//}

		CheckCudaError("Advance");
		time_step_counter++;

		//system("pause");

		PrintMemoryInfo();
		//system("pause");

		cudaDeviceSynchronize(); advance_time = timer.stop();

		//writeIterVelocityField(metadata);
	}

	void WriteStatToFile(DriverMetaData& metadata) {
		fs::path output_path = metadata.base_path / "logs";
		fs::create_directories(output_path);
		fs::path output_file = output_path / fmt::format("simulator_stat_{:04d}.txt", metadata.current_frame);
		std::ofstream out(output_file);
		fmt::print(out,
			"frame {}\n"
			"total particles {}\n"
			"total leaf cells {}\n"
			"reseeding time {} ms\n"
			"particle_advection_time {} ms\n"
			"adaptive_time {} ms\n"
			"p2g_time {} ms\n"
			"projection_time {} ms\n"
			"nfm_advection_time {} ms\n"
			"advance_time {} ms\n",
			metadata.current_frame,
			particles.size(),
			grid_ptrs.back()->numTotalLeafTiles() * Tile::SIZE,
			reseeding_time,
			particle_advection_time,
			adaptive_time,
			p2g_time,
			projection_time,
			nfm_advection_time,
			advance_time);
		out.close();
	}

	void PrintMemoryInfo(void) {
		double M = 1024 * 1024, G = 1024 * 1024 * 1024;
		double particle_num = particles.size();
		double particle_capacity = particles.capacity();
		double total_tile_num = 0;
		for (auto grid_ptr : grid_ptrs) {
			auto& grid = *grid_ptr;
			total_tile_num += grid.numTotalTiles();
		}
		double total_num_voxels = total_tile_num * Tile::SIZE;
		double memory_size_gb = (particle_num * sizeof(Particle) + total_tile_num * sizeof(Tile)) / G;
		double capacity_size_gb = (particle_capacity * sizeof(Particle) + total_tile_num * sizeof(Tile)) / G;
		
		Info("total {:.3f}({:.3f})M particles, {:.3f}M tiles and {:.3f}M voxels, memory size {:.3f}({:.3f})G", particle_num / M, particle_capacity / M, total_tile_num / M, total_num_voxels / M, memory_size_gb, capacity_size_gb);

		size_t free_mem, total_mem;

		// 查询设备的内存信息
		cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
		if (err != cudaSuccess) {
			std::cerr << "Failed to get CUDA memory info: " << cudaGetErrorString(err) << std::endl;
			return;
		}

		// 转换为GB
		double free_mem_gb = static_cast<double>(free_mem) / (1024 * 1024 * 1024);
		double total_mem_gb = static_cast<double>(total_mem) / (1024 * 1024 * 1024);
		double used_mem_gb = total_mem_gb - free_mem_gb;
		Info("Using CUDA Memory: total {:.3f}G, free {:.3f}G, used {:.3f}G, background {:.3f}({:.3f})G", total_mem_gb, free_mem_gb, used_mem_gb, used_mem_gb - memory_size_gb, used_mem_gb - capacity_size_gb);

		// 查询 CPU 内存信息
#ifdef _WIN32
		MEMORYSTATUSEX memInfo;
		memInfo.dwLength = sizeof(MEMORYSTATUSEX);
		if (GlobalMemoryStatusEx(&memInfo)) {
			double total_physical_gb = static_cast<double>(memInfo.ullTotalPhys) / G;
			double free_physical_gb = static_cast<double>(memInfo.ullAvailPhys) / G;
			double used_physical_gb = total_physical_gb - free_physical_gb;
			Info("Using CPU Memory: total {:.3f}G, free {:.3f}G, used {:.3f}G",
				total_physical_gb, free_physical_gb, used_physical_gb);
		}
		else {
			std::cerr << "Failed to get CPU memory info on Windows." << std::endl;
		}
#else
		struct sysinfo memInfo;
		if (sysinfo(&memInfo) == 0) {
			double total_physical_gb = static_cast<double>(memInfo.totalram) * memInfo.mem_unit / G;
			double free_physical_gb = static_cast<double>(memInfo.freeram) * memInfo.mem_unit / G;
			double used_physical_gb = total_physical_gb - free_physical_gb;
			Info("Using CPU Memory: total {:.3f}G, free {:.3f}G, used {:.3f}G",
				total_physical_gb, free_physical_gb, used_physical_gb);
		}
		else {
			std::cerr << "Failed to get CPU memory info on Linux." << std::endl;
		}
#endif
	}
};