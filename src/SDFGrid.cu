
#include "SDFGrid.h"


// Kernel to compute two masks (SDF <= 0 and SDF <= isovalue)
__global__ void ComputeMasksFromSDF(const float* sdf, uint32_t* mask_0, uint32_t* mask_iso, float solid_isovalue, float gen_isovalue, int num_cells) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_cells) {
        uint32_t sdf_mask_0 = (sdf[idx] <= solid_isovalue) ? 1 : 0;
        uint32_t sdf_mask_iso = (sdf[idx] <= gen_isovalue) ? 1 : 0;

        // Update the masks using bitset-like access (we divide by 32 to get the uint32_t index)
        int int_idx = idx / 32;  // We use uint32_t to store the mask, so divide by 32
        uint8_t bit_idx = idx % 32;  // Each uint32_t holds 32 bits

   //     {
   //         int nx = 256, ny = 256, nz = 256;
			//int i = idx / (ny * nz);
			//int j = (idx % (ny * nz)) / nz;
			//int k = idx % nz;
   //         int diff = abs(i - 161) + abs(j - 112) + abs(k - 209);
   //         if (diff <= 1) {
			//	printf("i: %d, j: %d, k: %d, diff: %d sdf: %f sdf_mask_0: %d sdf_mask_iso: %d\n", i, j, k, diff, sdf[idx], sdf_mask_0, sdf_mask_iso);
   //         }
   //     }

        // Perform atomic operations on uint32_t data type
        atomicOr(&mask_0[int_idx], sdf_mask_0 << bit_idx);
        atomicOr(&mask_iso[int_idx], sdf_mask_iso << bit_idx);
    }
}

// Method to convert position `pos` (a Vec) to grid index `idx`
__hostdev__ int MaskGridAccessor::GetIndexFromPosition(const Vec& pos) const {
    // Calculate the relative position of the point in the grid's local space
    float relative_x = (pos[0] - origin_x) / cell_size_x;
    float relative_y = (pos[1] - origin_y) / cell_size_y;
    float relative_z = (pos[2] - origin_z) / cell_size_z;

    // Ensure that the indices are within bounds
    if (relative_x < 0 || relative_x >= nx ||
        relative_y < 0 || relative_y >= ny ||
        relative_z < 0 || relative_z >= nz) {
        // In device code, throwing exceptions is not allowed, so you may need to handle it differently
        return -1;  // Return an invalid index or handle error appropriately
    }

    // Compute the index in 1D based on the 3D coordinates
    //int idx = (int)relative_x + nx * ((int)relative_y + ny * (int)relative_z);
    //row-major as in numpy
	int idx = (int)relative_x * ny * nz + (int)relative_y * nz + (int)relative_z;
    return idx;
}

// Generalized function to access mask values at a specific index
__hostdev__ bool MaskGridAccessor::GetMask(uint32_t* mask, int idx) const {
    if (idx == -1) return false;

    int int_idx = idx / 32;  // We use uint32_t to store the mask, so divide by 32
    int bit_idx = idx % 32;  // Each uint32_t holds 32 bits

    // Directly access the element and apply the bitmask
    uint32_t result = mask[int_idx];
    return (result >> bit_idx) & 1;
}

__hostdev__ bool MaskGridAccessor::GetMask0(const Vec& pos) const {
	int idx = GetIndexFromPosition(pos);
	return GetMask(mask_0, idx);
}
__hostdev__ bool MaskGridAccessor::GetMaskIso(const Vec& pos) const {
	int idx = GetIndexFromPosition(pos);
	return GetMask(mask_iso, idx);
}

// Constructor: initializes the grid based on the input file
MaskGrid::MaskGrid(const fs::path& filename, float solid_isovalue, float gen_isovalue) {
    // Read grid info and SDF values from binary file
    std::vector<float> sdf_data;
    ReadSDFFile(filename, sdf_data);

    // Allocate device memory for the bitset masks (use uint32_t instead of uint8_t)
    // Each uint32_t holds 32 bits, so we need to divide num_cells by 32 to get the number of uint32_t elements.
    cudaMalloc(&mask_0, (num_cells + 31) / 32 * sizeof(uint32_t)); // Round up to handle any remainder cells
    cudaMalloc(&mask_iso, (num_cells + 31) / 32 * sizeof(uint32_t)); // Round up to handle any remainder cells

    // Copy the SDF data to device memory (for future processing if needed)
    float* d_sdf;
    cudaMalloc(&d_sdf, num_cells * sizeof(float));
    cudaMemcpy(d_sdf, sdf_data.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize the masks based on SDF thresholds
    ComputeMasksFromSDF << <(num_cells + 255) / 256, 256 >> > (d_sdf, mask_0, mask_iso, solid_isovalue, gen_isovalue, num_cells);
    cudaDeviceSynchronize();

    cudaFree(d_sdf);

    // Post-process to update mask_0 based on neighbor values
    //UpdateMaskNeighbors << <(num_cells + 255) / 256, 256 >> > (mask_0, mask_iso, nx, ny, nz);
    //cudaDeviceSynchronize();
}

// Destructor to free allocated memory
MaskGrid::~MaskGrid() {
    cudaFree(mask_0);
    cudaFree(mask_iso);
}


MaskGridAccessor MaskGrid::GetDeviceAccessor() {
    MaskGridAccessor acc;
	acc.mask_0 = mask_0;
	acc.mask_iso = mask_iso;
	acc.nx = nx;
	acc.ny = ny;
	acc.nz = nz;
	acc.cell_size_x = cell_size_x;
	acc.cell_size_y = cell_size_y;
	acc.cell_size_z = cell_size_z;
	acc.origin_x = origin_x;
	acc.origin_y = origin_y;
	acc.origin_z = origin_z;
	return acc;
}

// Method to read SDF data from file
void MaskGrid::ReadSDFFile(const fs::path& filename, std::vector<float>& sdf_data) {
    FILE* file = fopen(filename.string().c_str(), "rb");
    if (!file) {
        Assert(false, "Failed to open the file {}", filename.string());
        return;
    }
    float sx, sy, sz;

    // Read grid dimensions and resolution
    float grid_info[6];
    fread(grid_info, sizeof(float), 6, file);
    sx = grid_info[0];
    sy = grid_info[1];
    sz = grid_info[2];
    nx = static_cast<int>(grid_info[3]);
    ny = static_cast<int>(grid_info[4]);
    nz = static_cast<int>(grid_info[5]);

    // Calculate the total number of cells
    num_cells = nx * ny * nz;

    // Calculate the size of each cell
    cell_size_x = sx / nx;
    cell_size_y = sy / ny;
    cell_size_z = sz / nz;

    // Calculate the origin of the grid (assumed to be the center of the bounding box)
    origin_x = 0.f;
    origin_y = 0.f;
    origin_z = 0.f;

    // Allocate memory for SDF data and read it from file
    sdf_data.resize(num_cells);
    fread(sdf_data.data(), sizeof(float), num_cells, file);
    fclose(file);
}

SDFGrid::SDFGrid(const fs::path& filename, float _solid_isovalue, float _gen_isovalue):
	solid_isovalue(_solid_isovalue), gen_isovalue(_gen_isovalue)
{
    // Read grid info and SDF values from binary file
    std::vector<float> sdf_data;
    ReadSDFFile(filename, sdf_data);

    // Copy the SDF data to device memory (for future processing if needed)
    cudaMalloc(&sdf_field, num_cells * sizeof(float));
    cudaMemcpy(sdf_field, sdf_data.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
}

SDFGrid::~SDFGrid() {
	cudaFree(sdf_field);
}

SDFGridAccessor SDFGrid::GetDeviceAccessor() {
	SDFGridAccessor acc;
	acc.sdf_field = sdf_field;
	acc.solid_isovalue = solid_isovalue;
	acc.gen_isovalue = gen_isovalue;
	acc.nx = nx;
	acc.ny = ny;
	acc.nz = nz;
	acc.cell_size_x = cell_size_x;
	acc.cell_size_y = cell_size_y;
	acc.cell_size_z = cell_size_z;
	acc.origin_x = origin_x;
	acc.origin_y = origin_y;
	acc.origin_z = origin_z;
	return acc;
}

// Method to read SDF data from file
void SDFGrid::ReadSDFFile(const fs::path& filename, std::vector<float>& sdf_data) {
    FILE* file = fopen(filename.string().c_str(), "rb");
	Info("Reading SDF file: {}", filename.string());
    if (!file) {
        Assert(false, "Failed to open the file {}", filename.string());
        return;
    }
    float sx, sy, sz;

    // Read grid dimensions and resolution
    float grid_info[6];
    fread(grid_info, sizeof(float), 6, file);
    sx = grid_info[0];
    sy = grid_info[1];
    sz = grid_info[2];
    nx = static_cast<int>(grid_info[3]);
    ny = static_cast<int>(grid_info[4]);
    nz = static_cast<int>(grid_info[5]);

    // Calculate the total number of cells
    num_cells = nx * ny * nz;

    // Calculate the size of each cell
    cell_size_x = sx / nx;
    cell_size_y = sy / ny;
    cell_size_z = sz / nz;

    // Calculate the origin of the grid (assumed to be the center of the bounding box)
    origin_x = 0.f;
    origin_y = 0.f;
    origin_z = 0.f;

    // Allocate memory for SDF data and read it from file
    sdf_data.resize(num_cells);
    fread(sdf_data.data(), sizeof(float), num_cells, file);
    fclose(file);
}

void SDFGrid::ReloadSDFFile(const fs::path& filename)
{
    // Read grid info and SDF values from binary file
    //std::vector<float> sdf_data;
    ReadSDFFile(filename, sdf_data);

    // Copy the SDF data to device memory (for future processing if needed)
    cudaMemcpy(sdf_field, sdf_data.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
}

__global__ void InterpolateCellValue(const float* sdf_field_0, const float* sdf_field_1, float* sdf_field_out, float alpha, int num_cells) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_cells) {
		sdf_field_out[idx] = (1 - alpha) * sdf_field_0[idx] + alpha * sdf_field_1[idx];
	}
}

void SDFGrid::InterpolateFromTwoSDFGrids(const SDFGrid& sdf_grid_0, const SDFGrid& sdf_grid_1, float alpha)
{
	InterpolateCellValue << <(num_cells + 255) / 256, 256 >> > (sdf_grid_0.sdf_field, sdf_grid_1.sdf_field, sdf_field, alpha, num_cells);
	cudaDeviceSynchronize();
}
