#pragma once

#include "HAAccessor.h"

#include <vector>
//to print std::vector
#include <fmt/ranges.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>

void CheckCudaError(const std::string& message);

template <typename T>
class DeviceReducer {
public:
	thrust::device_vector<T> d_data;         // 使用 device_vector 来存储数据
	thrust::device_vector<char> d_temp_sum;  // 使用 device_vector 来存储临时存储区
	thrust::device_vector<char> d_temp_max;  // 使用 device_vector 来存储最大值的临时存储区
	size_t n;                                // 数据长度
	size_t temp_bytes_sum;                   // 临时存储空间字节数
	size_t temp_bytes_max;                   // 最大值临时存储空间字节数

	DeviceReducer() : n(0), temp_bytes_sum(0), temp_bytes_max(0) {}

	T* data(void) {
		return thrust::raw_pointer_cast(d_data.data());
	}

	void resize(size_t len) {
		n = len;

		if (n > 0) {
			d_data.resize(n);

			// Allocate temp storage for sum
			cub::DeviceReduce::Sum(nullptr, temp_bytes_sum, data(), data(), n);
			d_temp_sum.resize(temp_bytes_sum);

			// Allocate temp storage for max
			cub::DeviceReduce::Max(nullptr, temp_bytes_max, data(), data(), n);
			d_temp_max.resize(temp_bytes_max);
		}
	}

	void sumAsyncTo(T* d_result) {
		if (n > 0) {
			cub::DeviceReduce::Sum(thrust::raw_pointer_cast(d_temp_sum.data()), temp_bytes_sum, data(), d_result, n);
			CheckCudaError("DeviceReducer: cub::DeviceReduce::Sum");
		}
	}

	void maxAsyncTo(T* d_result) {
		if (n > 0) {
			cub::DeviceReduce::Max(thrust::raw_pointer_cast(d_temp_max.data()), temp_bytes_max, data(), d_result, n);
			CheckCudaError("DeviceReducer: cub::DeviceReduce::Max");
		}
	}

	void printData() const {
		if (n > 0) {
			std::vector<T> h_data(n);

			cudaMemcpy(&h_data[0], thrust::raw_pointer_cast(d_data.data()), n * sizeof(T), cudaMemcpyDeviceToHost);
			CheckCudaError("DeviceReducer: cudaMemcpy for d_data");

			fmt::print("Device data: {}\n", h_data);
		}
	}
};


template<class Tile>
class HAHostTileHolder {
public:
	using Coord = typename Tile::CoordType;
	using T = typename Tile::T;

	T mH0;
	int mNumLayers;
	int mMaxLevel;

	std::vector<Tile> mHostTiles;
	std::vector<std::vector<HATileInfo<Tile>>> mHostLevels;

	HAHostTileHolder() {}
	HAHostTileHolder(const T h0, const int num_layers, const int max_level) :
		mH0(h0), mNumLayers(num_layers), mMaxLevel(max_level)
	{
		mHostLevels.resize(num_layers);
	}

	HACoordAccessor<Tile> coordAccessor(void) const {
		return HACoordAccessor<Tile>(mH0);
	}

	int numberOfLeafTiles(void)const {
		int cnt = 0;
		for (int i = 0; i <= mMaxLevel; i++) {
			for (auto& info : mHostLevels[i]) {
				if (info.isLeaf()) cnt++;
			}
		}
		return cnt;
	}

	void iterateLeafTiles(std::function<void(HATileInfo<Tile>&)> f) {
		for (int i = 0; i <= mMaxLevel; i++) {
			for (auto& info : mHostLevels[i]) {
				if (info.isLeaf()) f(info);
			}
		}
	}

	//f(info, l_ijk)
	template<class FuncBC>
	void iterateLeafCells(FuncBC f) {
		for (int level = 0; level <= mMaxLevel; level++) {
			for (auto& info : mHostLevels[level]) {
				if (info.isLeaf()) {

					for (int i = 0; i < Tile::DIM; i++) {
						for (int j = 0; j < Tile::DIM; j++) {
							for (int k = 0; k < Tile::DIM; k++) {
								f(info, Coord(i, j, k));
							}
						}
					}
				}
			}
		}
	}


	template<class FuncBC>
	void iterateLevelCells(const int level, FuncBC f) {
		for (auto& info : mHostLevels[level]) {
			for (int i = 0; i < Tile::DIM; i++) {
				for (int j = 0; j < Tile::DIM; j++) {
					for (int k = 0; k < Tile::DIM; k++) {
						f(info, Coord(i, j, k));
					}
				}
			}
		}
	}
};

//A=accessor, B=block(tile), C=coord
//FuncABC means f(acc, tile_info, ijk)

template<class Tile, class FuncABC>
__global__ void LaunchVoxelsOnAllTilesHelperKernel(HATileAccessor<Tile> acc, FuncABC f, HATileInfo<Tile>* tiles, uint8_t launch_types, int num_groups) {
	using Coord = typename Tile::Coord;
	HATileInfo<Tile> info = tiles[blockIdx.x];
	int ti = threadIdx.x;
	if (info.mType & launch_types) {
		for (int i = 0; i < num_groups; i++) {
			int vi = i * blockDim.x + ti;
			Coord l_ijk = acc.localOffsetToCoord(vi);
			f(acc, info, l_ijk);
		}

		//Coord l_ijk = Coord(threadIdx.x + i_offset, threadIdx.y, threadIdx.z);
		//f(acc, info, l_ijk);
	}
}

//A=accessor, B=block(tile), C=coord
//FuncABC means f(acc, tile_info, ijk)
template<class Tile, class FuncABC>
__global__ void LaunchVoxelsHelperKernel(HATileAccessor<Tile> acc, FuncABC f, HATileInfo<Tile>* tiles, const int subtree_level, uint8_t launch_types, int i_offset) {
	using Coord = typename Tile::Coord;
	HATileInfo<Tile> tile_info = tiles[blockIdx.x];
	Coord l_ijk = Coord(threadIdx.x + i_offset, threadIdx.y, threadIdx.z);
	//if (l_ijk == Coord(4, 5, 4)) {
	//	printf("LaunchVoxelsHelperKernel l_ijk %d %d %d\n", l_ijk[0], l_ijk[1], l_ijk[2]);
	//}
	if (tile_info.subtreeType(subtree_level) & launch_types) f(acc, tile_info, l_ijk);
}

template<class Tile, class FuncABC>
__global__ void LaunchNodesHelperKernel(HATileAccessor<Tile> acc, FuncABC f, HATileInfo<Tile>* tiles, const int subtree_level, uint8_t launch_types) {
	using Coord = typename Tile::Coord;
	//auto idx = threadIdx.x;
	for (auto idx : { threadIdx.x * 2, threadIdx.x * 2 + 1 }) {
		if (idx < (Tile::DIM + 1) * (Tile::DIM + 1) * (Tile::DIM + 1)) {
			Coord r_ijk = acc.localNodeOffsetToCoord(idx);
			HATileInfo<Tile> tile_info = tiles[blockIdx.x];
			if (tile_info.subtreeType(subtree_level) & launch_types) f(acc, tile_info, r_ijk);
		}
	}
}

template<class Tile, class FuncAIB>
__global__ void LaunchTilesHelperKernel(HATileAccessor<Tile> acc, FuncAIB f, HATileInfo<Tile>* tiles, int num_tiles, const int subtree_level, uint8_t launch_types) {
	auto idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_tiles) {
		HATileInfo<Tile> tile_info = tiles[idx];
		if (tile_info.subtreeType(subtree_level) & launch_types) f(acc, idx, tile_info);
	}
}

template<class Tile>
class HADeviceGrid {
	static constexpr DataHolder AllocSide = DEVICE;
public:
	using Coord = typename Tile::Coord;
	using T = typename Tile::T;

	cudaStream_t mStream;
	T mH0;
	uint32_t mNumLayers = 0;
	int mMaxLevel;

	bool mCompressedFlag = true;

	//all tiles are allocated on the device
	//but tile info may be sotred on host or device
	//technically, the metadata can reside on HOST/DEVICE
	//and data (tiles) can reside on HOST/DEVICE
	//the combination of DEVICE-HOST is meaningless
	//here we have HOST-DEVICE and DEVICE-DEVICE combinations
	//note that we DO NOT have HOST-HOST combination
	//therefore, all modifications to the grid are done with allocating and freeing DEVICE memory


	thrust::host_vector<uint32_t> hNumTiles;
	thrust::host_vector<uint32_t> hLog2Hashes;
	std::vector<thrust::host_vector<HATileInfo<Tile>>> hHashTables;
	std::vector<thrust::host_vector<HATileInfo<Tile>>> hTileArrays;
	//pointers to host data, helping building accessors
	//these pointers are calculated when initializing the grid
	std::vector<HATileInfo<Tile>* > hHashTablePtrs;

	bool mDeviceSyncFlag;//if device layers are synchronized with host layers


	//device data structure
	thrust::device_vector<uint32_t> dNumTiles;
	thrust::device_vector<uint32_t> dLog2Hashes;
	std::vector<thrust::device_vector<HATileInfo<Tile>>> dHashTables;
	std::vector<thrust::device_vector<HATileInfo<Tile>>> dTileArrays;
	//pointers to device data, helping building accessors
	//these pointers are calculated when synchronizing the device
	thrust::device_vector<HATileInfo<Tile>*> dHashTablePtrs;


	//auxillary
	//thrust::device_vector<HATileInfo<Tile>> dLeafTiles;
	//thrust::device_vector<HATileInfo<Tile>> dGhostTiles;
	thrust::host_vector<HATileInfo<Tile>> hAllTiles;
	thrust::device_vector<HATileInfo<Tile>> dAllTiles;
	DeviceReducer<double> dAllTilesReducer;// for dot etc

	HADeviceGrid() :
		mH0(0),
		mNumLayers(0),
		mMaxLevel(-1),
		mCompressedFlag(true)
	{
		cudaStreamCreate(&mStream);
	}
	HADeviceGrid(const T h0, const thrust::host_vector<uint32_t>& levels_log2_hash) :
		mH0(h0),
		mMaxLevel(-1),
		mCompressedFlag(true),//it's empty
		mDeviceSyncFlag(true)//empty
	{
		cudaStreamCreate(&mStream);

		mNumLayers = levels_log2_hash.size();

		hNumTiles.resize(mNumLayers);
		hLog2Hashes.resize(mNumLayers);
		hHashTables.resize(mNumLayers);
		hTileArrays.resize(mNumLayers);
		hHashTablePtrs.resize(mNumLayers);
		//hTileArrayPtrs.resize(mNumLayers);

		//hLog2Hashes = std::vector<uint32_t>(levels_log2_hash);
		hLog2Hashes = levels_log2_hash;

		for (int i = 0; i < mNumLayers; i++) {
			hNumTiles[i] = 0;
			hHashTables[i].resize(1u << hLog2Hashes[i]);
			hTileArrays[i].resize(1u << hLog2Hashes[i]);

			hHashTablePtrs[i] = thrust::raw_pointer_cast(hHashTables[i].data());
			//hTileArrayPtrs[i] = thrust::raw_pointer_cast(hTileArrays[i].data());
		}
	}
	HADeviceGrid(const T h0, const std::initializer_list<uint32_t>& levels_log2_hash) :
		HADeviceGrid(h0, thrust::host_vector<uint32_t>(levels_log2_hash))
	{
	}

	HADeviceGrid(const HADeviceGrid<Tile>&) = delete;
	HADeviceGrid& operator=(const HADeviceGrid<Tile>&) = delete;

	~HADeviceGrid() {
		Assert(mCompressedFlag, "Grid must be compressed before destruction");
		for (int i = 0; i < mNumLayers; i++) {
			for (int j = 0; j < hNumTiles[i]; j++) {
				auto info = hTileArrays[i][j];
				if (!info.empty()) {
					if (AllocSide == DEVICE) cudaFree(info.mTilePtr);
					else std::free(info.mTilePtr);
				}
			}
		}
		cudaStreamDestroy(mStream);
	}

	void setTilesFromHolder(const HAHostTileHolder<Tile>& holder) {
		Assert(mCompressedFlag, "setTilesFromHolder requires compressed grid");
		Assert(holder.mNumLayers == mNumLayers, "setTilesFromHolder: num layers mismatch");
		mH0 = holder.mH0;
		mNumLayers = holder.mNumLayers;
		mMaxLevel = holder.mMaxLevel;
		mCompressedFlag = false;
		for (int i = 0; i < mNumLayers; i++) {
			hNumTiles[i] = holder.mHostLevels[i].size();
			for (int j = 0; j < hNumTiles[i]; j++) {
				auto& info = holder.mHostLevels[i][j];
				auto& tile = info.tile();
				auto type = info.mType;
				auto coord = info.mTileCoord;
				setTileHost(i, coord, tile, type);
			}
		}
	}

	std::shared_ptr<HADeviceGrid<Tile>> deepCopy(void){
		Assert(mCompressedFlag, "deepCopy requires compressed grid");
		auto ptr = std::make_shared<HADeviceGrid<Tile>>(mH0, hLog2Hashes);
		auto& grid1 = *ptr;
		for (int i = 0; i <= mMaxLevel; i++) {
			for (int j = 0; j < hNumTiles[i]; j++) {
				auto info = hTileArrays[i][j];
				grid1.setTileHost(i, info.mTileCoord, info.getTile(DEVICE), info.mType);
				//grid1.setTileHost(i, info.mTileCoord, Tile(), info.mType);
				//Info("copyStructure: level={} coord={} type={}", i, info.mTileCoord, info.mType);
			}
		}
		grid1.compressHost();
		grid1.syncHostAndDevice();
		SpawnGhostTiles(grid1);
		return ptr;


		//double h = 1.0 / 8;
		//auto grid_ptr = std::make_shared<HADeviceGrid<Tile> >(h, std::initializer_list<uint32_t>({ 16, 16, 16, 16, 16, 16, 18, 16, 16, 16 }));
		//auto& grid = *grid_ptr;

		//	grid.setTileHost(0, nanovdb::Coord(0, 0, 0), Tile(), LEAF);
		//	grid.setTileHost(0, nanovdb::Coord(1, 0, 0), Tile(), LEAF);
		//grid.compressHost();
		//grid.syncHostAndDevice();
		//SpawnGhostTiles(grid);
		//return grid_ptr;
	}

	int numTotalTiles(void) const {
		Assert(mCompressedFlag, "numTotalTiles requires compressed grid");
		int num = 0;
		for (int i = 0; i < mNumLayers; i++) {
			num += hNumTiles[i];
		}
		return num;
	}
	int numTotalLeafTiles(void) const {
		Assert(mCompressedFlag, "numTotalLeafs requires compressed grid");
		int num = 0;
		for (int i = 0; i < mNumLayers; i++) {
			for (int j = 0; j < hNumTiles[i]; j++) {
				if (hTileArrays[i][j].isLeaf()) num++;
			}
		}
		return num;
	}

	//b_ijk for block coord, which is the beginning voxel coord / 8
	//this operation only changes the host data structure
	//need to sync
	void setTileHost(const int level, const Coord& b_ijk, const Tile& data, uint8_t type) {
		auto h_acc = hostAccessor();
		auto& info = h_acc.tileInfo(level, b_ijk);

		mCompressedFlag = false;
		mDeviceSyncFlag = false;

		if (info.empty()) {
			//create a new tile
			Tile* ptr = nullptr;
			if (AllocSide == DEVICE) cudaMalloc((void**)&ptr, sizeof(Tile));
			else ptr = (Tile*)std::malloc(sizeof(Tile));
			info = HATileInfo<Tile>(type, level, ptr, b_ijk);
		}

		//modify the existing tile
		if (AllocSide == DEVICE) cudaMemcpy(info.mTilePtr, &data, sizeof(Tile), cudaMemcpyHostToDevice);
		else std::memcpy(info.mTilePtr, &data, sizeof(Tile));
		Tile* ptr = info.mTilePtr;
		info = HATileInfo<Tile>(type, level, ptr, b_ijk);
	}

	void removeTileHost(const int level, const Coord& b_ijk) {
		auto& h_acc = hostAccessor();
		auto i = h_acc.tileIdx(level, b_ijk);
		if (hHashTables[level][i].empty()) return;
		auto j = i;
		uint32_t HASH_MASK = (1u << hLog2Hashes[level]) - 1u;
		while (true) {
			j = (j + 1) & HASH_MASK;
			if (hHashTables[level][j].empty()) break;
			auto k = h_acc.hash(hHashTables[level][j].mTileCoord, hLog2Hashes[level]);
			if ((j > i && (k <= i || k > j)) || (j < i && (k <= i && k > j))) {
				//mHostHashTable[i] = mHostHashTable[j];
				std::swap(hHashTables[level][i], hHashTables[level][j]);//we need to deconstruct j later
				i = j;
			}
		}
		if (AllocSide == DEVICE) cudaFree(hHashTables[level][i].mTilePtr);
		else std::free(hHashTables[level][i].mTilePtr);
		hHashTables[level][i] = HATileInfo<Tile>();//set to empty
		//mHostHashTable[i].mTilePtr = nullptr;
		mCompressedFlag = false;
		mDeviceSyncFlag = false;
	}

	std::shared_ptr<HAHostTileHolder<Tile>> getHostTileHolder(const uint8_t tile_types, int max_level = -1) {
		if (max_level == -1) max_level = mMaxLevel;
		auto holder = std::make_shared<HAHostTileHolder<Tile>>(mH0, mNumLayers, mMaxLevel);
		holder->mHostTiles.clear();
		int num_leafs = 0;
		for (int i = 0; i <= max_level; i++) {
			holder->mHostLevels[i].clear();
			int level_leafs = 0;
			for (int j = 0; j < hNumTiles[i]; j++) {
				HATileInfo<Tile> info = hTileArrays[i][j];
				if (info.mType & tile_types) {
					level_leafs++;
					num_leafs++;
				}
			}
			holder->mHostLevels[i].reserve(level_leafs);
		}
		holder->mHostTiles.reserve(num_leafs);
		for (int i = 0; i <= max_level; i++) {
			for (int j = 0; j < hNumTiles[i]; j++) {
				HATileInfo<Tile> info = hTileArrays[i][j];
				if (info.mType & tile_types) {
					holder->mHostTiles.push_back(info.getTile(DEVICE));
					info.mTilePtr = &(holder->mHostTiles.back());
					holder->mHostLevels[i].push_back(info);
				}
			}
		}
		return holder;
	}

	std::shared_ptr<HAHostTileHolder<Tile>> getHostTileHolderForLeafs(void) {
		return getHostTileHolder(LEAF);
	}

	int maxProbeLength(void) {
		auto h_acc = hostAccessor();
		int max_len = 0;
		for (int i = 0; i < mNumLayers; i++) {
			for (int j = 0; j < hNumTiles[i]; j++) {
				auto info = hTileArrays[i][j];
				auto hash_idx = h_acc.hash(info.mTileCoord, hLog2Hashes[i]);
				auto tile_idx = h_acc.tileIdx(i, info.mTileCoord);
				int len = tile_idx - hash_idx;
				max_len = std::max(max_len, len);
			}
		}
		return max_len;
	}

	int hashTableDeviceBytes(void) {
		int total_bytes = 0;
		for (int i = 0; i < mNumLayers; i++) {
			total_bytes += dHashTables[i].size() * sizeof(HATileInfo<Tile>);
		}
		return total_bytes;
	}

	void compressHost(bool verbose = true) {
		std::vector<int> load_rate;
		mMaxLevel = -1;
		for (int i = 0; i < mNumLayers; i++) {
			hNumTiles[i] = thrust::copy_if(
				hHashTables[i].begin(), hHashTables[i].end(),
				hTileArrays[i].begin(),
				[&](const HATileInfo<Tile>& info) ->bool {return info.mTilePtr != nullptr; }//pred
			) - hTileArrays[i].begin();
			uint32_t HASH_SIZE = (1u << hLog2Hashes[i]);
			load_rate.push_back((hNumTiles[i] + 0.0) / HASH_SIZE * 100);
			if (hNumTiles[i] != 0) mMaxLevel = i;
		}
		mCompressedFlag = true;
		if (verbose) {
			Info("Grid compressed {} layers, with each {} tiles and load rate {}/%", mNumLayers, hNumTiles, load_rate);
			Info("Max probe length: {}", maxProbeLength());
		}
	}
	void syncHostAndDevice(void) {
		Assert(mCompressedFlag, "syncHostAndDevice requires compressed grid");

		dNumTiles = hNumTiles;
		dLog2Hashes = hLog2Hashes;
		dHashTables.resize(mNumLayers);
		dTileArrays.resize(mNumLayers);
		for (int i = 0; i < mNumLayers; i++) {
			dHashTables[i] = hHashTables[i];
			dTileArrays[i] = hTileArrays[i];
		}

		thrust::host_vector<HATileInfo<Tile>*> host_layer_hash_table_ptrs;
		//thrust::host_vector<HATileInfo<Tile>*> host_layer_tile_array_ptrs;
		for (int i = 0; i < mNumLayers; i++) {
			host_layer_hash_table_ptrs.push_back(thrust::raw_pointer_cast(dHashTables[i].data()));
			//host_layer_tile_array_ptrs.push_back(thrust::raw_pointer_cast(dTileArrays[i].data()));
		}
		dHashTablePtrs = host_layer_hash_table_ptrs;
		//dTileArrayPtrs = host_layer_tile_array_ptrs;


		//create leaf and ghost tiles list for launching

		//thrust::host_vector<HATileInfo<Tile>> hostLeafTiles;
		//thrust::host_vector<HATileInfo<Tile>> hostGhostTiles;
		//thrust::host_vector<HATileInfo<Tile>> hAllTiles;

		hAllTiles.clear();
		for (int layer = 0; layer < mNumLayers; layer++) {
			for (int j = 0; j < hNumTiles[layer]; j++) {
				const HATileInfo<Tile>& info = hTileArrays[layer][j];
				//push leaf tiles
				//if (info.isLeaf()) {
				//	hostLeafTiles.push_back(info);
				//}
				////push ghost tiles
				//else if (info.isGhost()) {
				//	hostGhostTiles.push_back(info);
				//}
				hAllTiles.push_back(info);
			}
		}

		//dLeafTiles = hostLeafTiles;
		//dGhostTiles = hostGhostTiles;
		dAllTiles = hAllTiles;

		dAllTilesReducer.resize(dAllTiles.size());

		mDeviceSyncFlag = true;
	}

	//template<class FuncB>
	//void iterateLeafTiles(FuncB f) {
	//	Assert(mCompressedFlag, "iterateLeafTiles requires compressed grid");
	//	for (int level = 0; level <= mMaxLevel; level++) {
	//		for (int i = 0; i < hNumTiles[level]; i++) {
	//			HATileInfo<Tile>& info = hTileArrays[level][i];
	//			if (info.mType & LEAF) {
	//				f(info);
	//			}
	//		}
	//	}
	//}

	template<class FuncABC>
	void launchVoxelFunc(FuncABC f, int level, const uint8_t launch_types, LaunchMode mode = LAUNCH_LEVEL, const LaunchOrder order = COARSE_FIRST, const int num_groups = 1) {
		//launch a voxel function f(Accessor, TileInfo, l_ijk) on specified tiles

		//launch_types is a bit mask containing all tile types we want to launch
		//if mode is LAUNCH_LEVEL, we will authentically launch specified types
		//otherwise, all non-leafs on that level will be considered as leafs when launching

		//if mode is set to LAUNCH_LEVEL, then only the specified level will be launched
		//otherwise, we will launch on levels [0,level], which is a sub-tree with specified finest level
		//and all tiles at this finest level are regarded as leafs
		//if level is set to -1, LAUNCH_SUBTREE is forced, and all levels will be launched

		//if order is set to COARSE_FIRST, then we will launch from coarse to fine
		//otherwise, we will launch from fine to coarse

		//if num_groups=1, all 512 voxels are launched in the same block
		//if num_groups=2, 256 voxels are launched in the same block
		//...4,8

		Assert(num_groups == 1 || num_groups == 2 || num_groups == 4 || num_groups == 8, "launchVoxelFunc invalid num groups: {}", num_groups);

		if (level == -1) {
			if (mode == LAUNCH_LEVEL) {
				Info("launchVoxelFunc level=-1, overwrite mode=LAUNCH_SUBTREE");
			}
			mode = LAUNCH_SUBTREE;
			level = mNumLayers - 1;
		}
		int start, end, step;
		if (mode == LAUNCH_LEVEL) start = level, end = level + 1, step = 1;
		else {
			if (order == COARSE_FIRST) start = 0, end = level + 1, step = 1;
			else start = level, end = -1, step = -1;
		}

		int subtree_level = (mode == LAUNCH_LEVEL) ? -1 : level;
		for (int i = start; i != end; i += step) {
			for (int gi = 0; gi < num_groups; gi++) {
				int offset = gi * Tile::DIM / num_groups;
				if (hNumTiles[i] == 0) continue;
				LaunchVoxelsHelperKernel << <hNumTiles[i], dim3(Tile::DIM / num_groups, Tile::DIM, Tile::DIM) >> > (
					deviceAccessor(),
					f,
					thrust::raw_pointer_cast(dTileArrays[i].data()),
					subtree_level,
					launch_types,
					offset
					);

				CheckCudaError(fmt::format("launchVoxelFunc level={} gi={} num tiles {} num_groups={}", i, gi, hNumTiles[i], num_groups));
			}
		}
	}

	template<class FuncABC>
	void launchVoxelFuncOnTiles(FuncABC f, thrust::device_vector<HATileInfo<Tile>> &tiles, const int num_launched_tiles, const uint8_t launch_types, const int num_groups = 1) {
		//for (int gi = 0; gi < num_groups; gi++) {
			//int offset = gi * Tile::DIM / num_groups;
			if (num_launched_tiles == 0) return;
			LaunchVoxelsOnAllTilesHelperKernel << <num_launched_tiles, Tile::SIZE / num_groups >> > (
				deviceAccessor(),
				f,
				thrust::raw_pointer_cast(tiles.data()),
				launch_types,
				num_groups
				);
		//}
	}

	template<class FuncABC>
	void launchVoxelFuncOnAllTiles(FuncABC f, const uint8_t launch_types, const int num_groups = 1) {
		launchVoxelFuncOnTiles(f, dAllTiles, dAllTiles.size(), launch_types, num_groups);
	}

	template<class FuncABC>
	void launchNodeFunc(FuncABC f, int level, const uint8_t launch_types, LaunchMode mode = LAUNCH_LEVEL, const LaunchOrder order = COARSE_FIRST) {
		//same as launchVoxelFunc but we apply functions on nodes instead of voxels(cells)

		if (level == -1) {
			if (mode == LAUNCH_LEVEL) {
				Info("launchVoxelFunc level=-1, overwrite mode=LAUNCH_SUBTREE");
			}
			mode = LAUNCH_SUBTREE;
			level = mNumLayers - 1;
		}
		int start, end, step;
		if (mode == LAUNCH_LEVEL) start = level, end = level + 1, step = 1;
		else {
			if (order == COARSE_FIRST) start = 0, end = level + 1, step = 1;
			else start = level, end = -1, step = -1;
		}

		int subtree_level = (mode == LAUNCH_LEVEL) ? -1 : level;
		for (int i = start; i != end; i += step) {
			if (hNumTiles[i] == 0) continue;
			LaunchNodesHelperKernel << <hNumTiles[i], Tile::SIZE >> > (
				deviceAccessor(),
				f,
				thrust::raw_pointer_cast(dTileArrays[i].data()),
				subtree_level,
				launch_types
				);
		}
	}

	template<class FuncAIB>
	void launchTileFunc(FuncAIB f, int level, const uint8_t launch_types, LaunchMode mode = LAUNCH_LEVEL, const LaunchOrder order = COARSE_FIRST, int blockSize = 512) {
		//launch a voxel function f(Accessor, TileInfo, l_ijk) on specified tiles

		//launch_types is a bit mask containing all tile types we want to launch

		//if mode is set to LAUNCH_LEVEL, then only the specified level will be launched
		//otherwise, we will launch on levels [0,level], which is a sub-tree with specified finest level
		//and all tiles at this finest level are regarded as leafs
		//if level is set to -1, LAUNCH_SUBTREE is forced, and all levels will be launched

		//if order is set to COARSE_FIRST, then we will launch from coarse to fine
		//otherwise, we will launch from fine to coarse

		if (level == -1) {
			mode = LAUNCH_SUBTREE;
			level = mNumLayers - 1;
		}
		int start, end, step;
		if (mode == LAUNCH_LEVEL) start = level, end = level + 1, step = 1;
		else {
			if (order == COARSE_FIRST) start = 0, end = level + 1, step = 1;
			else start = level, end = -1, step = -1;
		}

		int subtree_level = (mode == LAUNCH_LEVEL) ? -1 : level;
		for (int i = start; i != end; i += step) {
			if (hNumTiles[i] == 0) continue;
			LaunchTilesHelperKernel << <(hNumTiles[i] + blockSize - 1) / blockSize, blockSize >> > (
				deviceAccessor(),
				f,
				thrust::raw_pointer_cast(dTileArrays[i].data()),
				hNumTiles[i],
				subtree_level,
				launch_types
				);
		}
	}

	HATileAccessor<Tile> hostAccessor(void) {
		return HATileAccessor<Tile>(
			mH0,
			mNumLayers,
			mMaxLevel,
			thrust::raw_pointer_cast(hNumTiles.data()),
			thrust::raw_pointer_cast(hLog2Hashes.data()),
			thrust::raw_pointer_cast(hHashTablePtrs.data())//,
			//thrust::raw_pointer_cast(hTileArrayPtrs.data())
		);
	}

	//this can't be a const member function because that requires the pointers to be const
	HATileAccessor<Tile> deviceAccessor(void) {
		return HATileAccessor<Tile>(
			mH0,
			mNumLayers,
			mMaxLevel,
			thrust::raw_pointer_cast(dNumTiles.data()),
			thrust::raw_pointer_cast(dLog2Hashes.data()),
			thrust::raw_pointer_cast(dHashTablePtrs.data())//,
			//thrust::raw_pointer_cast(dTileArrayPtrs.data())
		);
	}
};


