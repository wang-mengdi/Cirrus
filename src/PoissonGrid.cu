#include "PoissonGrid.h"

#include <cub/cub.cuh>
#include <cub/block/block_reduce.cuh>
#include <thrust/execution_policy.h>


template<class T>
__global__ void FillArrayKernel(T* arr, const T val, const int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        arr[idx] = val;
    }
}

template<class T>
void FillArray(T* d_array, const int n, const T val, const int block_size = 512) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error before FillArray: " << cudaGetErrorString(err) << std::endl;
    }

    int numBlocks = (n + block_size - 1) / block_size;

    FillArrayKernel << <numBlocks, block_size >> > (d_array, val, n);
}



__host__ void SpawnGhostTiles(HADeviceGrid<Tile>& grid, bool verbose) {
    using Acc = HACoordAccessor<Tile>;
    //note that we will NOT spawn ghost tiles on level 0
    //because it's the root level, and they will not have parents
    Assert(grid.mCompressedFlag, "Grid must be compressed and synced before spawn ghost tiles");
    auto h_acc = grid.hostAccessor();
    for (int level = grid.mMaxLevel; level > 0; level--) {
        for (int i = 0; i < grid.hNumTiles[level]; i++) {
            auto& info = grid.hTileArrays[level][i];
            if (info.isActive()) {
                Coord b_ijk = info.mTileCoord;
                Acc::iterateNeighborCoords(b_ijk, [&](const Coord& n_ijk) {
                    auto& n_info = h_acc.tileInfo(level, n_ijk);
                    //auto& n_info = grid.mHostLayers[level].tileInfo(n_ijk);
                    if (n_info.empty()) {
                        Coord p_ijk = Acc::parentCoord(n_ijk);
                        auto& p_info = h_acc.tileInfo(level - 1, p_ijk);
                        //auto& p_info = grid.mHostLayers[level - 1].tileInfo(p_ijk);
                        if (p_info.isLeaf()) {
                            grid.setTileHost(level, n_ijk, Tile(), GHOST);
                            //grid.mHostLayers[level].setTile(n_ijk, Tile(), level, GHOST);
                        }
                    }
                    }
                );
            }
        }
    }
    grid.compressHost(verbose);
    grid.syncHostAndDevice();
}

void Copy(HADeviceGrid<Tile>& grid, const int in_channel, const int out_channel, const int level, const uint8_t launch_types, const LaunchMode mode, const uint8_t cell_types) {
    UnaryTransform(grid, in_channel, out_channel, []__device__(Tile::T x) { return x; }, level, launch_types, mode, cell_types);
}

//will fill all voxels specified by level and exec_policy
//no matter if the voxel is active or not
void Fill(HADeviceGrid<Tile>& grid, const int channel, const Tile::T val, const int level, const uint8_t launch_types, const LaunchMode mode, const uint8_t cell_types) {
    UnaryTransform(grid, channel, channel, [=]__device__(Tile::T x) { return val; }, level, launch_types, mode, cell_types);
}


//running on all leafs (level=-1) or a specific level
//out[i] += alpha * in[i]
void Axpy(HADeviceGrid<Tile>& grid, const Tile::T alpha, const uint8_t in_channel, const uint8_t out_channel, const int level, const uint8_t launch_types, const LaunchMode mode) {
    BinaryTransform(grid, in_channel, out_channel, out_channel, [=]__device__(Tile::T x, Tile::T y) { return y + alpha * x; }, level, launch_types, mode);
}

template<typename T>
T CUBDeviceArraySum(const T* device_ptr, const int n) {
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    T* d_result;
    cudaMalloc(&d_result, sizeof(T));
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, device_ptr, d_result, n);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, device_ptr, d_result, n);
    T result;
    cudaMemcpy(&result, d_result, sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(d_temp_storage);
    cudaFree(d_result);
    return result;
}

template<typename T1, typename T2>
void CUBDeviceArraySumAsync(const T1* device_ptr, const int n, T2* d_result) {
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, device_ptr, d_result, n);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, device_ptr, d_result, n);
    cudaFree(d_temp_storage);
}

template<typename T>
T CUBDeviceArrayMax(const T* device_ptr, const int n) {
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    T* d_result;
    cudaMalloc(&d_result, sizeof(T));
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, device_ptr, d_result, n);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, device_ptr, d_result, n);
    T result;
    cudaMemcpy(&result, d_result, sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(d_temp_storage);
    cudaFree(d_result);
    return result;
}

void CalculateNeighborTiles(HADeviceGrid<Tile>& grid) {
    grid.launchTileFunc(
        [=] __device__(HATileAccessor<Tile> acc, const int _, HATileInfo<Tile>&info) {
        auto& tile = info.tile();
        for (int axis : {0, 1, 2}) {
            Coord b_ijk = info.mTileCoord;
            Coord nb_ijk = b_ijk; nb_ijk[axis]--;
            Coord np_ijk = b_ijk; np_ijk[axis]++;

            tile.mNeighbors[axis + 0] = acc.tileInfo(info.mLevel, nb_ijk);
            tile.mNeighbors[axis + 3] = acc.tileInfo(info.mLevel, np_ijk);
        }
    }, -1, LEAF | GHOST | NONLEAF, LAUNCH_SUBTREE);
}

////follow the same launch convention as launchVoxelFunc
//template<class T>
//__global__ void DotKernel(HATileAccessor<PoissonTile<T>> acc, HATileInfo<PoissonTile<T>>* infos, const uint8_t in1_channel, const uint8_t in2_channel, double* sum, int subtree_level, uint8_t launch_types) {
//    const HATileInfo<PoissonTile<T>>& info = infos[blockIdx.x];
//    Coord l_ijk = Coord(threadIdx.x, threadIdx.y, threadIdx.z);
//
//    if (!(info.subtreeType(subtree_level) & launch_types)) {
//        if (l_ijk == Coord(0, 0, 0)) sum[blockIdx.x] = 0;
//        return;
//    }
//
//    auto& tile = info.tile();
//    double thread_dot = tile.interiorValue(in1_channel, l_ijk) * tile.interiorValue(in2_channel, l_ijk);
//
//    //printf("l_ijk=%d %d %d thread_dot=%f\n", l_ijk[0], l_ijk[1], l_ijk[2], thread_dot);
//
//    typedef cub::BlockReduce<double, Tile::DIM, cub::BLOCK_REDUCE_WARP_REDUCTIONS, Tile::DIM, Tile::DIM> BlockReduce;
//    __shared__ typename BlockReduce::TempStorage temp_storage;
//    double block_sum = BlockReduce(temp_storage).Sum(thread_dot);
//
//    if (l_ijk == Coord(0, 0, 0)) {
//        sum[blockIdx.x] = block_sum;
//
//        //auto b_ijk = info.mTileCoord;
//        //printf("launch dotkernel tile bijk %d %d %d sum %f value %f type %d interiorValue %f\n", b_ijk[0], b_ijk[1], b_ijk[2], block_sum, tile(in1_channel, l_ijk), tile.type(l_ijk), tile.interiorValue(in1_channel, l_ijk));
//    }
//}

//follow the same launch convention as launchVoxelFunc
__global__ void Dot128Kernel(HATileAccessor<PoissonTile<T>> acc, HATileInfo<PoissonTile<T>>* infos, const uint8_t in1_channel, const uint8_t in2_channel, double* sum, int subtree_level, uint8_t launch_types) {
    int bi = blockIdx.x;
    int ti = threadIdx.x;
    const HATileInfo<PoissonTile<T>>& info = infos[bi];
    

    if (!(info.subtreeType(subtree_level) & launch_types)) {
        if (ti == 0) sum[bi] = 0;
        return;
    }

    auto& tile = info.tile();
    auto data1AsFloat4 = reinterpret_cast<float4*>(tile.mData[in1_channel]);
	auto data2AsFloat4 = reinterpret_cast<float4*>(tile.mData[in2_channel]);
	float4 data1 = data1AsFloat4[ti];
	float4 data2 = data2AsFloat4[ti];
	double thread_dot = data1.x * data2.x + data1.y * data2.y + data1.z * data2.z + data1.w * data2.w;
    

    //double thread_dot = tile.interiorValue(in1_channel, l_ijk) * tile.interiorValue(in2_channel, l_ijk);

    //printf("l_ijk=%d %d %d thread_dot=%f\n", l_ijk[0], l_ijk[1], l_ijk[2], thread_dot);

    typedef cub::BlockReduce<double, 128> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    double block_sum = BlockReduce(temp_storage).Sum(thread_dot);

    if (ti==0) {
        sum[bi] = block_sum;

    }
}

//follow the launch convention of launchVoxelFunc
void DotAsync(double* d_result, HADeviceGrid<Tile>& grid, const uint8_t in1_channel, const uint8_t in2_channel, const uint8_t launch_tile_types) {
    int num_tiles = grid.dAllTiles.size();
    FillArray(grid.dAllTilesReducer.data(), grid.dAllTiles.size(), 0.0);
    if (num_tiles > 0) {
        Dot128Kernel << <num_tiles, 128 >> > (
            grid.deviceAccessor(),
            thrust::raw_pointer_cast(grid.dAllTiles.data()),
            in1_channel, in2_channel,
            grid.dAllTilesReducer.data(),
            -1, launch_tile_types
            );
		grid.dAllTilesReducer.sumAsyncTo(d_result);
    }

    CheckCudaError("DotAsync end");
}

//follow the launch convention of launchVoxelFunc
double Dot(HADeviceGrid<Tile>& grid, const uint8_t in1_channel, const uint8_t in2_channel, const uint8_t launch_tile_types) {
    double* d_result;
    cudaMalloc(&d_result, sizeof(double));
    
	DotAsync(d_result, grid, in1_channel, in2_channel, launch_tile_types);

    double result;
    cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    return result;
}

//follow the same launch convention as launchVoxelFunc
__global__ void VoxelSum128Kernel(HATileAccessor<PoissonTile<T>> acc, HATileInfo<PoissonTile<T>>* infos, const int in_channel, double* d_sum, int subtree_level, uint8_t launch_types) {
    int bi = blockIdx.x;
    int ti = threadIdx.x;
    const HATileInfo<PoissonTile<T>>& info = infos[bi];


    if (!(info.subtreeType(subtree_level) & launch_types)) {
        if (ti == 0) d_sum[bi] = 0;
        return;
    }

    auto& tile = info.tile();
    auto data1AsFloat4 = reinterpret_cast<float4*>(tile.mData[in_channel]);
    float4 data1 = data1AsFloat4[ti];
    double thread_sum = data1.x + data1.y + data1.z + data1.w;


    //double thread_dot = tile.interiorValue(in1_channel, l_ijk) * tile.interiorValue(in2_channel, l_ijk);

    //printf("l_ijk=%d %d %d thread_dot=%f\n", l_ijk[0], l_ijk[1], l_ijk[2], thread_dot);

    typedef cub::BlockReduce<double, 128> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    double block_sum = BlockReduce(temp_storage).Sum(thread_sum);

    if (ti == 0) {
        d_sum[bi] = block_sum;
    }
}

//follow the same launch convention as launchVoxelFunc
__global__ void InteriorCount128Kernel(HATileAccessor<PoissonTile<T>> acc, HATileInfo<PoissonTile<T>>* infos, double* d_count, int subtree_level, uint8_t launch_types) {
    int bi = blockIdx.x;
    int ti = threadIdx.x;
    const HATileInfo<PoissonTile<T>>& info = infos[bi];


    if (!(info.subtreeType(subtree_level) & launch_types)) {
        if (ti == 0) d_count[bi] = 0;
        return;
    }

    auto& tile = info.tile();
    auto typeAsuchar4 = reinterpret_cast<uchar4*>(tile.mCellType);
    auto type4 = typeAsuchar4[ti];
    double thread_interior_cnt =
        (type4.x & INTERIOR) +
        (type4.y & INTERIOR) +
        (type4.z & INTERIOR) +
        (type4.w & INTERIOR);


    //double thread_dot = tile.interiorValue(in1_channel, l_ijk) * tile.interiorValue(in2_channel, l_ijk);

    //printf("l_ijk=%d %d %d thread_dot=%f\n", l_ijk[0], l_ijk[1], l_ijk[2], thread_dot);

    typedef cub::BlockReduce<double, 128> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    double block_count = BlockReduce(temp_storage).Sum(thread_interior_cnt);

    if (ti == 0) {
        d_count[bi] = block_count;
    }
}

//follow the launch convention of launchVoxelFunc
void MeanAsync(HADeviceGrid<Tile>& grid, const int in_channel, const uint8_t launch_tile_types, double* d_mean, double* d_count) {
    int num_tiles = grid.dAllTiles.size();
    FillArray(grid.dAllTilesReducer.data(), grid.dAllTiles.size(), 0.0);
    if (num_tiles > 0) {
        VoxelSum128Kernel << <num_tiles, 128 >> > (
            grid.deviceAccessor(),
            thrust::raw_pointer_cast(grid.dAllTiles.data()),
            in_channel,
            grid.dAllTilesReducer.data(),
            -1, launch_tile_types
            );
        grid.dAllTilesReducer.sumAsyncTo(d_mean);
    }

    FillArray(grid.dAllTilesReducer.data(), grid.dAllTiles.size(), 0.0);
    if (num_tiles > 0) {
        InteriorCount128Kernel << <num_tiles, 128 >> > (
            grid.deviceAccessor(),
            thrust::raw_pointer_cast(grid.dAllTiles.data()),
            grid.dAllTilesReducer.data(),
            -1, launch_tile_types
            );
        grid.dAllTilesReducer.sumAsyncTo(d_count);
    }

    TernaryOnArray(d_mean, d_count, d_count, []__device__(double& mean, double count, double _) { mean = mean / count; });
    
    CheckCudaError("DotAsync end");
}


////follow the same launch convention as launchVoxelFunc
//__global__ void VelocityLinfKernel(HATileAccessor<PoissonTile<T>> acc, HATileInfo<PoissonTile<T>>* infos, const uint8_t u_channel, double* blk_max, int subtree_level, uint8_t launch_types) {
//    const HATileInfo<PoissonTile<T>>& info = infos[blockIdx.x];
//    Coord l_ijk = Coord(threadIdx.x, threadIdx.y, threadIdx.z);
//
//    if (!(info.subtreeType(subtree_level) & launch_types)) {
//        if (l_ijk == Coord(0, 0, 0)) blk_max[blockIdx.x] = 0;
//        return;
//    }
//
//    auto& tile = info.tile();
//    double v1 = abs(tile(u_channel + 0, l_ijk));
//    double v2 = abs(tile(u_channel + 1, l_ijk));
//    double v3 = abs(tile(u_channel + 2, l_ijk));
//    double thread_val = max(v1, max(v2, v3));
//
//    typedef cub::BlockReduce<double, Tile::DIM, cub::BLOCK_REDUCE_WARP_REDUCTIONS, Tile::DIM, Tile::DIM> BlockReduce;
//    __shared__ typename BlockReduce::TempStorage temp_storage;
//    double block_max = BlockReduce(temp_storage).Reduce(thread_val, thrust::maximum<T>());
//
//    if (l_ijk == Coord(0, 0, 0)) {
//        blk_max[blockIdx.x] = block_max;
//
//        //auto b_ijk = info.mTileCoord;
//        //printf("launch dotkernel tile bijk %d %d %d sum %f value %f type %d interiorValue %f\n", b_ijk[0], b_ijk[1], b_ijk[2], block_sum, tile(in1_channel, l_ijk), tile.type(l_ijk), tile.interiorValue(in1_channel, l_ijk));
//    }
//}
//
////follow the launch convention of launchVoxelFunc
//double VelocityLinf(HADeviceGrid<Tile>& grid, const uint8_t u_channel, int level, const uint8_t launch_types, LaunchMode mode) {
//    //static thrust::device_vector<double> device_sum(1, 0);
//    if (level == -1) {
//        mode = LAUNCH_SUBTREE;
//        level = grid.mNumLayers - 1;
//    }
//    int start, end, step;
//    if (mode == LAUNCH_LEVEL) start = level, end = level + 1, step = 1;
//    else start = 0, end = level + 1, step = 1;
//    int subtree_level = (mode == LAUNCH_LEVEL) ? -1 : level;
//
//    double linf = 0;
//    //thrust::device_vector<double> device_mx;
//    for (int i = start; i != end; i += step) {
//        Assert(0 <= i && i < grid.mNumLayers, "Dot invalid level {}", i);
//
//        if (grid.hNumTiles[i] > 0) {
//			//somehow, if we use a thrust::device_vector here, it will cause a crash
//            //I don't know why
//            //seems thrust::device_vector has some bugs
//            double* device_mx_ptr;
//            cudaMalloc(&device_mx_ptr, grid.hNumTiles[i] * sizeof(double));
//            VelocityLinfKernel << <grid.hNumTiles[i], dim3(Tile::DIM, Tile::DIM, Tile::DIM) >> > (
//                grid.deviceAccessor(),
//                thrust::raw_pointer_cast(grid.dTileArrays[i].data()),
//                u_channel,
//                device_mx_ptr,
//                subtree_level, launch_types
//                );
//			double level_linf = CUBDeviceArrayMax(device_mx_ptr, grid.hNumTiles[i]);
//            linf = std::max(linf, level_linf);
//            cudaFree(device_mx_ptr);
//        }
//    }
//    return linf;
//}

// Kernel to compute the maximum absolute value (Linf norm) across three channels
__global__ void VelocityLinfKernel(HATileAccessor<PoissonTile<T>> acc, HATileInfo<PoissonTile<T>>* infos, const int u_channel, double* max_values, int subtree_level, uint8_t launch_types) {
    int bi = blockIdx.x;
    int ti = threadIdx.x;
    const HATileInfo<PoissonTile<T>>& info = infos[bi];

    if (!(info.subtreeType(subtree_level) & launch_types)) {
        if (ti == 0) max_values[bi] = 0.0;
        return;
    }

    auto& tile = info.tile();
    auto uAsFloat4 = reinterpret_cast<float4*>(tile.mData[u_channel]);
    auto vAsFloat4 = reinterpret_cast<float4*>(tile.mData[u_channel + 1]);
    auto wAsFloat4 = reinterpret_cast<float4*>(tile.mData[u_channel + 2]);

    float4 u = uAsFloat4[ti];
    float4 v = vAsFloat4[ti];
    float4 w = wAsFloat4[ti];

    // Compute max abs value for this thread
    double thread_max = max(max(fabs(u.x), fabs(u.y)), max(fabs(u.z), fabs(u.w)));
    thread_max = max(thread_max, max(max(fabs(v.x), fabs(v.y)), max(fabs(v.z), fabs(v.w))));
    thread_max = max(thread_max, max(max(fabs(w.x), fabs(w.y)), max(fabs(w.z), fabs(w.w))));

    // Use CUB to perform block-wide reduction to find the maximum
    typedef cub::BlockReduce<double, 128> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    double block_max = BlockReduce(temp_storage).Reduce(thread_max, thrust::maximum<T>());

    if (ti == 0) {
        max_values[bi] = block_max;
    }
}

// Launch VelocityLinfKernel asynchronously
void VelocityLinfAsync(double* d_result, HADeviceGrid<Tile>& grid, const int u_channel, const uint8_t launch_tile_types) {
    int num_tiles = grid.dAllTiles.size();
    FillArray(grid.dAllTilesReducer.data(), grid.dAllTiles.size(), 0.0);
    if (num_tiles > 0) {
        VelocityLinfKernel << <num_tiles, 128 >> > (
            grid.deviceAccessor(),
            thrust::raw_pointer_cast(grid.dAllTiles.data()),
            u_channel,
            grid.dAllTilesReducer.data(),
            -1, launch_tile_types
            );
        grid.dAllTilesReducer.maxAsyncTo(d_result);
    }

    CheckCudaError("VelocityLinfAsync end");
}

// Synchronous function to compute Linf norm
double VelocityLinfSync(HADeviceGrid<Tile>& grid, const int u_channel, const uint8_t launch_tile_types) {
    double* d_result;
    cudaMalloc(&d_result, sizeof(double));

    VelocityLinfAsync(d_result, grid, u_channel, launch_tile_types);

    double result;
    cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    return result;
}


//follow the same launch convention as launchVoxelFunc
//order=-1 means L-infinity norm
__global__ void VolumeWeightedNormKernel(HATileAccessor<Tile> acc, HATileInfo<Tile>* infos, const int order, const int in_channel, double* weighted_value_sum, double* weights_sum, int subtree_level, uint8_t launch_types) {
    const HATileInfo<PoissonTile<T>>& info = infos[blockIdx.x];
    Coord l_ijk = Coord(threadIdx.x, threadIdx.y, threadIdx.z);
    if (!(info.subtreeType(subtree_level) & launch_types)) {
        if (l_ijk == Coord(0, 0, 0)) {
            weighted_value_sum[blockIdx.x] = 0;
            weights_sum[blockIdx.x] = 0;
        }
        return;
    }

    auto& tile = info.tile();
    auto h = acc.voxelSize(info);
    auto v = tile.interiorValue(in_channel, l_ijk);

	//printf("l_ijk=%d %d %d v=%f\n", l_ijk[0], l_ijk[1], l_ijk[2], v);

    if (order == 1 || order == -1) {
        v = abs(v);
    }
    else if (order == 2) {
        v = v * v;
    }
    else {
        v = pow(abs(v), order);
    }
    
    auto w = h * h * h;
    typedef cub::BlockReduce<T, Tile::DIM, cub::BLOCK_REDUCE_WARP_REDUCTIONS, Tile::DIM, Tile::DIM> BlockReduce;
    __shared__ typename BlockReduce::TempStorage ws_storage;
    __shared__ typename BlockReduce::TempStorage w_storage;

    double ws_sum = 0, w_sum = 0;
    if (order == -1) {
        ws_sum = BlockReduce(ws_storage).Reduce(v, thrust::maximum<T>());
        w_sum = w;
    }
    else {
        ws_sum = BlockReduce(ws_storage).Reduce(v * w, thrust::plus<T>());
        w_sum = BlockReduce(w_storage).Reduce(w, thrust::plus<T>());
    }
    if (l_ijk == Coord(0, 0, 0)) {
        //if (order == -1) printf("block %d ws_sum %f w_sum %f v %f\n", blockIdx.x, ws_sum, w_sum, v);
        weighted_value_sum[blockIdx.x] = ws_sum;
        weights_sum[blockIdx.x] = w_sum;
    }

}

//follow the launch convention of launchVoxelFunc
//see eqn (12) in Gerris: a tree-based adaptive solver for the incompressible Euler equations in complex geometries
//their a is VOF fraction and 2D
//we don't have VOF and we're 3D
//if order=-1, it's L-infinity norm, which is single point, or the maximum cell-center value in the whole tree
//and the returned ws is h*h*h for the corresponding cell
std::tuple<double, double> VolumeWeightedSumAndVolume(HADeviceGrid<Tile>& grid, const int order, const int in_channel, int level, const uint8_t launch_types, LaunchMode mode) {
    if (level == -1) {
        mode = LAUNCH_SUBTREE;
        level = grid.mNumLayers - 1;
    }
    int start, end, step;
    if (mode == LAUNCH_LEVEL) start = level, end = level + 1, step = 1;
    else start = 0, end = level + 1, step = 1;
    int subtree_level = (mode == LAUNCH_LEVEL) ? -1 : level;

    double ws_sum = 0, w_sum = 0;
    for (int i = start; i != end; i += step) {
        Assert(0 <= i && i < grid.mNumLayers, "Dot invalid level {}", i);

        if (grid.hNumTiles[i] > 0) {
            double* ws_d;
            double* w_d;
            cudaMalloc(&ws_d, grid.hNumTiles[i] * sizeof(double));
            cudaMalloc(&w_d, grid.hNumTiles[i] * sizeof(double));

            //Info("dot at level {} with {} tiles", i, grid.hNumTiles[i]);
            VolumeWeightedNormKernel << <grid.hNumTiles[i], dim3(Tile::DIM, Tile::DIM, Tile::DIM) >> > (
                grid.deviceAccessor(),
                thrust::raw_pointer_cast(grid.dTileArrays[i].data()),
                order, in_channel,
                ws_d, w_d,
                subtree_level, launch_types
                );
            if (order == -1) {
                ////do not pass 0 to the initial value because it will be resolved as int
                auto level_linf = CUBDeviceArrayMax(ws_d, grid.hNumTiles[i]);
                if (level_linf > ws_sum) {
                    ws_sum = level_linf;
                    cudaMemcpy(&w_sum, w_d, sizeof(T), cudaMemcpyDeviceToHost);
                }
            }
            else {
                ws_sum += CUBDeviceArraySum(ws_d, grid.hNumTiles[i]);
                w_sum += CUBDeviceArraySum(w_d, grid.hNumTiles[i]);
            }

			cudaFree(ws_d);
			cudaFree(w_d);
        }
    }
    return std::make_tuple(ws_sum, w_sum);
}

double VolumeWeightedNorm(HADeviceGrid<Tile>& grid, const int order, const int in_channel, int level, const uint8_t launch_types, LaunchMode mode) {
    auto [ws, w] = VolumeWeightedSumAndVolume(grid, order, in_channel, level, launch_types, mode);
    if (order == -1) return ws;
    else {
        auto norm = ws / w;
        if (order == 1) {

        }
        else if (order == 2) {
			norm = sqrt(norm);
        }
        else {
			norm = pow(norm, 1.0 / order);
        }
        return norm;
    }
}

void PropagateValuesToGhostTiles(HADeviceGrid<Tile>& grid, const int coarse_channel, const int fine_channel) {
    grid.launchVoxelFuncOnAllTiles(
        [=]__device__(HATileAccessor<Tile> &acc, HATileInfo<Tile> &info, const Coord & l_ijk) {
        auto& tile = info.tile();
        if (!tile.isInterior(l_ijk)) {
            tile(fine_channel, l_ijk) = Tile::BACKGROUND_VALUE;
            return;
        }
        auto fine_g_ijk = acc.localToGlobalCoord(info, l_ijk);
        auto coarse_g_ijk = acc.parentCoord(fine_g_ijk);
        HATileInfo<Tile> coarse_info; Coord coarse_l_ijk;
        acc.findVoxel(info.mLevel - 1, coarse_g_ijk, coarse_info, coarse_l_ijk);
        if (!coarse_info.empty()) {
            auto& coarse_tile = coarse_info.tile();
            tile(fine_channel, l_ijk) = coarse_tile.interiorValue(coarse_channel, coarse_l_ijk);
        }
        else tile(fine_channel, l_ijk) = Tile::BACKGROUND_VALUE;
    },
        GHOST
    );
}

//copy values from parents for tiles specified by propagate_tile_types
//for example, GHOST will propagate ghost values from parents
//this is actually prolongation with sum kernel
//we call it propagate following the convention in SPGrid paper
void PropagateValues(HADeviceGrid<Tile>& grid, const int coarse_channel, const int fine_channel, const int fine_level, const uint8_t propagated_tile_types, const LaunchMode mode) {
    //can be used to propagate one single level or iterating through the whole tree
    //if it's the whole tree, must propagate from coarse to fine
    grid.launchVoxelFunc(
        [=]__device__(HATileAccessor<Tile> &acc, HATileInfo<Tile> &info, const Coord & l_ijk) {
        auto& tile = info.tile();
        if (!tile.isInterior(l_ijk)) {
            tile(fine_channel, l_ijk) = Tile::BACKGROUND_VALUE;
            return;
        }
        auto fine_g_ijk = acc.localToGlobalCoord(info, l_ijk);
        auto coarse_g_ijk = acc.parentCoord(fine_g_ijk);
        HATileInfo<Tile> coarse_info; Coord coarse_l_ijk;
        acc.findVoxel(info.mLevel - 1, coarse_g_ijk, coarse_info, coarse_l_ijk);
        if (!coarse_info.empty()) {
            auto& coarse_tile = coarse_info.tile();
            tile(fine_channel, l_ijk) = coarse_tile.interiorValue(coarse_channel, coarse_l_ijk);
        }
        else tile(fine_channel, l_ijk) = Tile::BACKGROUND_VALUE;
    },
        fine_level, propagated_tile_types, mode, COARSE_FIRST);
}


void PropagateToChildren(HADeviceGrid<Tile>& grid, const int coarse_channel, const int fine_channel, const int target_subtree_level, const uint8_t target_tile_types, const LaunchMode mode, const uint8_t cell_types) {
    //can be used to propagate one single level or iterating through the whole tree
    //if it's the whole tree, must propagate from coarse to fine
    grid.launchVoxelFunc(
        [=]__device__(HATileAccessor<Tile> &acc, HATileInfo<Tile> &info, const Coord & fine_l_ijk) {
        auto& tile = info.tile();
        if (!(tile.type(fine_l_ijk) & cell_types)) {
            //tile(fine_channel, fine_l_ijk) = Tile::BACKGROUND_VALUE;
            return;
        }
        auto fine_g_ijk = acc.localToGlobalCoord(info, fine_l_ijk);
        auto coarse_g_ijk = acc.parentCoord(fine_g_ijk);
        HATileInfo<Tile> coarse_info; Coord coarse_l_ijk;
        acc.findVoxel(info.mLevel - 1, coarse_g_ijk, coarse_info, coarse_l_ijk);

        auto val = Tile::BACKGROUND_VALUE;
        if (!coarse_info.empty()) {
            auto& coarse_tile = coarse_info.tile();
            val = coarse_tile(coarse_channel, coarse_l_ijk);
            if (!(coarse_tile.type(coarse_l_ijk) & cell_types)) {
                val = Tile::BACKGROUND_VALUE;
            }
        }
        tile(fine_channel, fine_l_ijk) = val;
    },
        target_subtree_level, target_tile_types, mode, COARSE_FIRST);
}

//will be called on fine tiles
__global__ void AccumulateToParents128Kernel(HATileAccessor<Tile> acc, HATileInfo<Tile>* fine_tiles, int fine_subtree_level, uint8_t fine_tile_types, int fine_channel, int coarse_channel, Tile::T coeff, bool additive, uint8_t cell_types) {
    __shared__ T data[8 * 8 * 8];
	int bi = blockIdx.x;
	int ti = threadIdx.x;
	auto& finfo = fine_tiles[bi];

	if (!(finfo.subtreeType(fine_subtree_level) & fine_tile_types)) {
		return;
	}
	auto& ftile = finfo.tile();
    
    for (int i = 0; i < 4; i++) {
        int vi = i * 128 + ti;
		uint8_t ctype = ftile.type(vi);
        data[vi] = (ctype & cell_types) ? ftile(fine_channel, vi) : 0;
    }
    __syncthreads();

    if (ti < 64) {
        Coord fl_ijk(ti / 16 * 2, (ti / 4) % 4 * 2, ti % 4 * 2);
        Coord fg_ijk = acc.localToGlobalCoord(finfo, fl_ijk);
        Coord cg_ijk = acc.parentCoord(fg_ijk);
        HATileInfo<Tile> cinfo; Coord cl_ijk;//coarse tile and local coord
        acc.findVoxel(finfo.mLevel - 1, cg_ijk, cinfo, cl_ijk);
        if (!cinfo.empty()) {
			auto& ctile = cinfo.tile();
            T sum = 0;
            for (int ii : {0, 1}) {
                for (int jj : {0, 1}) {
                    for (int kk : {0, 1}) {
                        Coord fl1_ijk = fl_ijk + Coord(ii, jj, kk);
                        int vi1 = acc.localCoordToOffset(fl1_ijk);
                        sum += data[vi1];
                    }
                }
            }
            //ghost accumulate
            if (additive) {
                ctile(coarse_channel, cl_ijk) += sum * coeff;
            }
            else {
                ctile(coarse_channel, cl_ijk) = sum * coeff;
            }
        }

    }
}

void AccumulateToParents128(HADeviceGrid<Tile>& grid, const int fine_channel, const int coarse_channel, const uint8_t fine_tile_types, const Tile::T coeff, bool additive, uint8_t cell_types) {
	int num_fine_tiles = grid.dAllTiles.size();
    AccumulateToParents128Kernel << <num_fine_tiles, 128 >> > (
        grid.deviceAccessor(),
        thrust::raw_pointer_cast(grid.dAllTiles.data()),
        -1, fine_tile_types, fine_channel, coarse_channel, coeff, additive, cell_types
        );
}

void AccumulateToParents(HADeviceGrid<Tile>& grid, const int fine_channel, const int coarse_channel, const int target_subtree_level, const uint8_t target_tile_types, const LaunchMode mode, const uint8_t cell_types, const Tile::T coeff, bool additive) {
    grid.launchVoxelFunc(
        [=]__device__(HATileAccessor<Tile> &acc, HATileInfo<Tile> &info, const Coord & coarse_l_ijk) {
        auto& tile = info.tile();
        auto coarse_g_ijk = acc.localToGlobalCoord(info, coarse_l_ijk);

        //if (info.mLevel == 2 && coarse_g_ijk == Coord(6, 31, 15)) {
        //    printf("accumulating to parent level %d g_ijk %d %d %d type %d update types %d\n", info.mLevel, coarse_g_ijk[0], coarse_g_ijk[1], coarse_g_ijk[2], tile.type(coarse_l_ijk), cell_types);
        //}

        if (!(tile.type(coarse_l_ijk) & cell_types)) {
            return;
        }
        Tile::T sum = 0;

        for (int cid = 0; cid < acc.NUMCHILDREN; cid++) {
            auto fine_g_ijk = acc.childCoord(coarse_g_ijk, cid);
            HATileInfo<Tile> fine_info; Coord fine_l_ijk;
            acc.findVoxel(info.mLevel + 1, fine_g_ijk, fine_info, fine_l_ijk);
            if (!fine_info.empty()) {
                auto& fine_tile = fine_info.tile();
                if (fine_tile.type(fine_l_ijk) & cell_types) {
                    sum += fine_tile(fine_channel, fine_l_ijk);
                }
                //sum += fine_tile.interiorValue(fine_channel, fine_l_ijk);
            }
        }



        if (additive) tile(coarse_channel, coarse_l_ijk) += sum * coeff;
		else tile(coarse_channel, coarse_l_ijk) = sum * coeff;
    },
        target_subtree_level, target_tile_types, mode, FINE_FIRST);
}

void CalcLeafNodeValuesFromFaceCenters(HADeviceGrid<Tile>& grid, const int u_channel, const int node_u_channel) {
    //for (int axis : {0, 1, 2}) {
    //    PropagateValues(grid, u_channel + axis, u_channel + axis, -1, GHOST, LAUNCH_SUBTREE);
    //}

    auto calcNode = [=]__device__(HATileAccessor<Tile>&acc, HATileInfo<Tile> &info, const Coord & l_ijk, int axis) -> Tile::T {
        HATileInfo<Tile> n_info; Coord n_l_ijk;
        Tile::T sum = 0, weight_sum = 0;


        //Coord off(0, 0, 0);
        //int axis1 = acc.rotateAxis(axis, 1);
        //int axis2 = acc.rotateAxis(axis, 2);
        for (int offj : {-1, 0}) {
            for (int offk : {-1, 0}) {
                Coord off_ijk = acc.rotateCoord(axis, Coord(0, offj, offk));
                //off[axis1] = off1;
                //off[axis2] = off2;
                acc.findNodeNeighborLeaf(info, l_ijk, off_ijk, n_info, n_l_ijk);
				//acc.findNodeNeighborLeafOrGhost(info, l_ijk, off_ijk, n_info, n_l_ijk);

                if (!n_info.empty()) {
                    auto h = acc.voxelSize(n_info);
                    auto weight = 1.0 / h;
                    //in order to be valid, the node must be a corner of this neighbor cell
                    //which means, if this neighbor cell is coarser, the node coords must be divisible by 2,4,8...
					//for the vector intp case, the maximum difference of levels is 2, so the worst case is that it's divisible by 2 but not by 4
                    if (n_info.mLevel < info.mLevel) {
                        int k = (1 << (info.mLevel - n_info.mLevel));
                        if (l_ijk[0] % k != 0 || l_ijk[1] % k != 0 || l_ijk[2] % k != 0) {
                            return NODATA;
                        }
                    }
                    auto& n_tile = n_info.tile();
                    //sum += n_tile.interiorValue(u_channel + axis, n_l_ijk) * weight;
					sum += n_tile(u_channel + axis, n_l_ijk) * weight;
                    weight_sum += weight;

                }
            }
        }
        return weight_sum > 0 ? sum / weight_sum : 0;
    };

    grid.launchNodeFunc(
        [=]__device__(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& r_ijk) {
        auto& tile = info.tile();
        for (int axis : {0, 1, 2}) {
            tile.node(node_u_channel + axis, r_ijk) = calcNode(acc, info, r_ijk, axis);
        }
    },
        -1, LEAF, LAUNCH_SUBTREE
    );

    //fix all nodes that can be divided by k and not valid
    auto fixJunctions = [&](int level, int k) {
        grid.launchNodeFunc(
            [=]__device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & r_ijk) {
            auto& tile = info.tile();
            for (int axis : {0, 1, 2}) {
                if (tile.node(node_u_channel + axis, r_ijk) == NODATA && r_ijk[0] % k == 0 && r_ijk[1] % k == 0 && r_ijk[2] % k == 0) {
                    Tile::T sum = 0;
                    for (int offi : {-1, 1}) {
                        int n_i = (r_ijk[0] % (2 * k) != 0) ? r_ijk[0] + offi * k : r_ijk[0];
                        for (int offj : {-1, 1}) {
                            int n_j = (r_ijk[1] % (2 * k) != 0) ? r_ijk[1] + offj * k : r_ijk[1];
                            for (int offk : {-1, 1}) {
                                int n_k = (r_ijk[2] % (2 * k) != 0) ? r_ijk[2] + offk * k : r_ijk[2];
                                Coord n_r_ijk(n_i, n_j, n_k);
                                sum += tile.node(node_u_channel + axis, n_r_ijk);
                            }
                        }
                    }

                    tile.node(node_u_channel + axis, r_ijk) = sum / 8;
                }
            }
        },
            level, LEAF, LAUNCH_SUBTREE
        );
        };

    for (int i = 0; i <= grid.mMaxLevel; i++) {
		fixJunctions(i, 2);
        fixJunctions(i, 1);
    }
}

void CalcLeafNodeValuesFromCellCenters(HADeviceGrid<Tile>& grid, const int cell_channel, const int node_channel) {
    //PropagateValues(grid, cell_channel, cell_channel, -1, GHOST, LAUNCH_SUBTREE);
    PropagateToChildren(grid, cell_channel, cell_channel, -1, GHOST, LAUNCH_SUBTREE, INTERIOR | DIRICHLET | NEUMANN);

    auto calcNode = [=]__device__(HATileAccessor<Tile>&acc, HATileInfo<Tile> &info, const Coord & l_ijk) -> Tile::T {
        auto g_ijk = acc.localToGlobalCoord(info, l_ijk);

        HATileInfo<Tile> n_info; Coord n_l_ijk;
        Tile::T sum = 0, weight_sum = 0;
        for (int offx : {-1, 0}) {
            for (int offy : {-1, 0}) {
                for (int offz : {-1, 0}) {
                    Coord off(offx, offy, offz);
                    acc.findNodeNeighborLeaf(info, l_ijk, off, n_info, n_l_ijk);

                    if (!n_info.empty()) {
                        auto h = acc.voxelSize(n_info);
                        auto weight = 1.0 / h;

                        if (n_info.mLevel < info.mLevel) {
                            int k = (1 << (info.mLevel - n_info.mLevel));
                            if (l_ijk[0] % k != 0 || l_ijk[1] % k != 0 || l_ijk[2] % k != 0) {
                                return NODATA;
                            }
                        }

                        auto& n_tile = n_info.tile();
                        sum += n_tile(cell_channel, n_l_ijk) * weight;
                        weight_sum += weight;
                    }

                }
            }
        }

        return weight_sum > 0 ? sum / weight_sum : 0;
    };

    grid.launchVoxelFunc(
        [=]__device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
        auto& tile = info.tile();
        int thread_idx = acc.localCoordToOffset(l_ijk);
        for (int idx : {thread_idx * 2, thread_idx * 2 + 1}) {
            if (0 <= idx && idx < Tile::NODESIZE) {

                Coord node_l_ijk = acc.localNodeOffsetToCoord(idx);
                tile.node(node_channel, node_l_ijk) = calcNode(acc, info, node_l_ijk);
            }
        }
    },
        -1, LEAF, LAUNCH_SUBTREE
    );

    //fix all nodes that can be divided by k and not valid
    auto fixJunctions = [&](int level, int k) {
        grid.launchNodeFunc(
            [=]__device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & r_ijk) {
            auto& tile = info.tile();
            if (tile.node(node_channel, r_ijk) == NODATA && r_ijk[0] % k == 0 && r_ijk[1] % k == 0 && r_ijk[2] % k == 0) {
                Tile::T sum = 0;
                for (int offi : {-1, 1}) {
                    int n_i = (r_ijk[0] % (2 * k) != 0) ? r_ijk[0] + offi * k : r_ijk[0];
                    for (int offj : {-1, 1}) {
                        int n_j = (r_ijk[1] % (2 * k) != 0) ? r_ijk[1] + offj * k : r_ijk[1];
                        for (int offk : {-1, 1}) {
                            int n_k = (r_ijk[2] % (2 * k) != 0) ? r_ijk[2] + offk * k : r_ijk[2];
                            Coord n_r_ijk(n_i, n_j, n_k);
                            sum += tile.node(node_channel, n_r_ijk);
                        }
                    }
                }

                tile.node(node_channel, r_ijk) = sum / 8;
            }
        },
            level, LEAF, LAUNCH_SUBTREE
        );
        };

    for (int i = 0; i <= grid.mMaxLevel; i++) {
        fixJunctions(i, 4);
        fixJunctions(i, 2);
        fixJunctions(i, 1);
    }
}


__device__ Tile::T InterpolateCellValue(const HATileAccessor<Tile>& acc, const Vec& pos, const int cell_channel, const int node_channel) {
    HATileInfo<Tile> info; Coord l_ijk; Vec frac;
    if (acc.findLeafVoxelAndFrac(pos, info, l_ijk, frac)) {
        auto b_ijk = info.mTileCoord;
        auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
        // printf("find leaf at level %d g_ijk %d %d %d l_ijk %d %d %d frac %f %f %f\n", info.mLevel, g_ijk[0], g_ijk[1], g_ijk[2], l_ijk[0], l_ijk[1], l_ijk[2], frac[0], frac[1], frac[2]);
        auto& tile = info.tile();
        return tile.cellInterp(cell_channel, node_channel, l_ijk, frac);
    }
    else return Tile::BACKGROUND_VALUE;
}

__device__ Vec InterpolateFaceValue(const HATileAccessor<Tile>& acc, const Vec& pos, const int u_channel, const int node_u_channel) {
    Vec vec;

    HATileInfo<Tile> info; Coord l_ijk; Vec frac;
    acc.findLeafVoxelAndFrac(pos, info, l_ijk, frac);
    for (int axis = 0; axis < 3; axis++) {
        //printf("=============================interpolate face value at axis=%d\n", axis);
        //printf("pos: %f %f %f\n", pos[0], pos[1], pos[2]);
        //printf("pos0*12800: %lf\n", pos[0] * 12800);

        Tile::T v0, v1, w;
        w = frac[axis];
        if (!info.empty()) {
            auto& tile = info.tile();
            v0 = tile.faceInterp(u_channel, node_u_channel, axis, l_ijk, frac);
        }
        else v0 = 0;

		auto cell_ctr = acc.cellCenter(info, l_ijk);
		auto n_pos = pos; n_pos[axis] += (1 - frac[axis]) * acc.voxelSize(info);

        HATileInfo<Tile> n_info; Coord n_l_ijk; Vec n_frac;
        acc.findPlusFaceIntpVoxel(pos, axis, info, l_ijk, n_info, n_l_ijk, n_frac);
        if (!n_info.empty()) {
            auto& n_tile = n_info.tile();

            v1 = n_tile.faceInterp(u_channel, node_u_channel, axis, n_l_ijk, n_frac);
        }
        else v1 = 0;

        //printf("calc: %lf\n", (1 - w) * 10300 + w * 10350);
        vec[axis] = (1 - w) * v0 + w * v1;
    }
    return vec;
}

__device__ thrust::tuple<uint8_t, uint8_t> FaceNeighborCellTypes(const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const Coord& l_ijk, const int axis) {
    auto& tile = info.tile();
    uint8_t type0 = tile.type(l_ijk);

    auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
    auto ng_ijk = g_ijk; ng_ijk[axis]--;
    HATileInfo<Tile> n_info; Coord nl_ijk;
    acc.findVoxel(info.mLevel, ng_ijk, n_info, nl_ijk);
    uint8_t type1;
    if (!n_info.empty()) {
        auto& n_tile = n_info.tile();
        type1 = n_tile.type(nl_ijk);
    }
    else type1 = DIRICHLET;
    return thrust::make_tuple(type0, type1);
}

void ExtrapolateVelocity(HADeviceGrid<Tile>& grid, const int u_channel, const int num_iters) {

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
                info.tile()(Tile::u_channel + axis, l_ijk) = NODATA;
            }
        }
    }, -1, LEAF, LAUNCH_SUBTREE
    );

    for (int i = 0; i < num_iters; i++) {
        grid.launchVoxelFunc(
            [=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
            auto& tile = info.tile();

            for (int axis : {0, 1, 2}) {

                if (tile(u_channel + axis, l_ijk) == NODATA) {
                    T ws = 0;
                    T vws = 0;
                    //acc.iterateNeighborLeafFaces(axis, info, l_ijk, [&]__device__(const HATileInfo<Tile>&ninfo, const Coord & nl_ijk, const int axj, const int sgn) {
                    acc.iterateNeighborLeafCells(info, l_ijk, [&]__device__(const HATileInfo<Tile>&ninfo, const Coord & nl_ijk, const int _, const int sgn) {
                        auto& ntile = ninfo.tile();
                        auto v = ntile(u_channel + axis, nl_ijk);
                        auto h = acc.voxelSize(ninfo);
                        if (v != NODATA) {
                            ws += h * h * h;
                            vws += v * h * h * h;
                        }
                    });
                    if (ws > 0) {
                        tile(u_channel + axis, l_ijk) = vws / ws;
                    }
                }
            }
        }, -1, LEAF, LAUNCH_SUBTREE
        );
    }

    grid.launchVoxelFunc(
        [=] __device__(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk) {
        auto& tile = info.tile();
        for (int axis : {0, 1, 2}) {
            if (tile(u_channel + axis, l_ijk) == NODATA) {
                tile(u_channel + axis, l_ijk) = 0;
            }
        }
    }, -1, LEAF, LAUNCH_SUBTREE
    );
}

__global__ void MarkInterestAreaWithValue128Kernel(HATileAccessor<PoissonTile<T>> acc, HATileInfo<PoissonTile<T>>* infos, int channel, T threshold, int subtree_level, uint8_t launch_types) {
    int bi = blockIdx.x;
    int ti = threadIdx.x;
    const HATileInfo<PoissonTile<T>>& info = infos[bi];

    auto& tile = info.tile();
    if (!(info.subtreeType(subtree_level) & launch_types)) {
        if (ti == 0) tile.mIsInterestArea = false;
        return;
    }

    auto data1AsFloat4 = reinterpret_cast<float4*>(tile.mData[channel]);
    float4 data1 = data1AsFloat4[ti];
    float max_data = fmaxf(fmaxf(data1.x, data1.y), fmaxf(data1.z, data1.w));

    typedef cub::BlockReduce<T, 128> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    T block_max = BlockReduce(temp_storage).Reduce(max_data, thrust::maximum<T>());

    if (ti == 0) {
        if (block_max > threshold) {
            tile.mIsInterestArea = true;
        }
        else
        {
            tile.mIsInterestArea = false;
        }

    }
}

int RefineWithValuesOneStep(HADeviceGrid<Tile>& grid, int channel, T threshold, int coarse_level, int fine_level, bool verbose) {
    auto levelTarget = [fine_level, coarse_level]__device__(const HATileAccessor<Tile> &acc, const HATileInfo<Tile> &info) ->int {
        auto& tile = info.tile();
        if (tile.mIsInterestArea) return fine_level;
        return coarse_level;
    };
    auto info_ptr = thrust::raw_pointer_cast(grid.dAllTiles.data());
    //calculate interest area flags on leafs
    MarkInterestAreaWithValue128Kernel << <grid.dAllTiles.size(), 128 >> > (grid.deviceAccessor(), info_ptr, channel, threshold, -1, LEAF);
    auto refine_cnts = RefineLeafsOneStep(grid, levelTarget, verbose);
    SpawnGhostTiles(grid, verbose);
    if (verbose) Info("Refine {} tiles on each layer", refine_cnts);
    int cnt = std::accumulate(refine_cnts.begin(), refine_cnts.end(), 0);
    return cnt;
}

int CoarsenWithValueneStep(HADeviceGrid<Tile>& grid, int channel, T threshold, int coarse_level, int fine_level, bool verbose) {
    auto levelTarget = [fine_level, coarse_level]__device__(const HATileAccessor<Tile> &acc, const HATileInfo<Tile> &info) ->int {
        auto& tile = info.tile();
        if (tile.mIsInterestArea) return fine_level;
        return coarse_level;
    };
    auto info_ptr = thrust::raw_pointer_cast(grid.dAllTiles.data());
    //calculate interest area flags on leafs
    MarkInterestAreaWithValue128Kernel << <grid.dAllTiles.size(), 128 >> > (grid.deviceAccessor(), info_ptr, channel, threshold, -1, LEAF);
    auto coarsen_cnts = CoarsenStep(grid, levelTarget, verbose);

    if (verbose) Info("Coarsen {} tiles on each layer", coarsen_cnts);
    int cnt = std::accumulate(coarsen_cnts.begin(), coarsen_cnts.end(), 0);
    return cnt;
}