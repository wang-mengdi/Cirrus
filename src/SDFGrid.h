#pragma once
#include "PoissonGrid.h"// for cuda, Vec etc

class MaskGridAccessor {
public:
    // Method to convert position `pos` (a Vec) to grid index `idx`
    __hostdev__ int GetIndexFromPosition(const Vec& pos)const;

    // Generalized function to access mask values at a specific index
    __hostdev__ bool GetMask(uint32_t* mask, int idx)const;

    // Function to access mask_0 at a specific position
    __hostdev__ bool GetMask0(const Vec& pos)const;

    // Function to access mask_iso at a specific position
    __hostdev__ bool GetMaskIso(const Vec& pos)const;

public:
    // Device pointers for the two masks and SDF data
    uint32_t* mask_0 = nullptr;
    uint32_t* mask_iso = nullptr;
    int nx = 0, ny = 0, nz = 0;
    float cell_size_x, cell_size_y, cell_size_z;
    float origin_x, origin_y, origin_z; // Origin of the grid
};

class MaskGrid {
public:
    // Constructor: initializes the grid based on the input file
    MaskGrid(const fs::path& filename, float solid_isovalue, float gen_isovalue);

    // Destructor to free allocated memory
    ~MaskGrid();

    // Returns a device accessor for the SDF grid
    MaskGridAccessor GetDeviceAccessor();

private:
    // Method to read SDF data from file
    void ReadSDFFile(const fs::path& filename, std::vector<float>& sdf_data);

public:
    // Device pointers for the two masks and SDF data
    uint32_t* mask_0;
    uint32_t* mask_iso;

    // Host variables for grid dimensions and SDF data
    float cell_size_x, cell_size_y, cell_size_z;
    float origin_x, origin_y, origin_z; // Origin of the grid
    int nx, ny, nz;
    int num_cells; // Total number of cells in the grid
};

class SDFGridAccessor {
public:
    // (i, j, k) -> idx
    __hostdev__ int idx(int i, int j, int k) const {
		return i * ny * nz + j * nz + k;
    }
    __hostdev__ float linearInterpolate(float x, float y, float z) const {
        float gx = (x - (origin_x + 0.5f * cell_size_x)) / cell_size_x;
        float gy = (y - (origin_y + 0.5f * cell_size_y)) / cell_size_y;
        float gz = (z - (origin_z + 0.5f * cell_size_z)) / cell_size_z;

        int i = static_cast<int>(floor(gx));
        int j = static_cast<int>(floor(gy));
        int k = static_cast<int>(floor(gz));

        float dx = gx - i;
        float dy = gy - j;
        float dz = gz - k;

        if (i < 0 || i >= nx - 1 || j < 0 || j >= ny - 1 || k < 0 || k >= nz - 1) {
            return FLT_MAX;
        }

        float c000 = sdf_field[idx(i, j, k)];
        float c100 = sdf_field[idx(i + 1, j, k)];
        float c010 = sdf_field[idx(i, j + 1, k)];
        float c110 = sdf_field[idx(i + 1, j + 1, k)];
        float c001 = sdf_field[idx(i, j, k + 1)];
        float c101 = sdf_field[idx(i + 1, j, k + 1)];
        float c011 = sdf_field[idx(i, j + 1, k + 1)];
        float c111 = sdf_field[idx(i + 1, j + 1, k + 1)];

        float c00 = c000 * (1 - dx) + c100 * dx;
        float c01 = c001 * (1 - dx) + c101 * dx;
        float c10 = c010 * (1 - dx) + c110 * dx;
        float c11 = c011 * (1 - dx) + c111 * dx;

        float c0 = c00 * (1 - dy) + c10 * dy;
        float c1 = c01 * (1 - dy) + c11 * dy;

        float c = c0 * (1 - dz) + c1 * dz;

        return c;
    }

public:
	// Device pointer for SDF data
	float* sdf_field = nullptr;
	float solid_isovalue, gen_isovalue;

	int nx = 0, ny = 0, nz = 0;
	float cell_size_x, cell_size_y, cell_size_z;
	float origin_x, origin_y, origin_z; // Origin of the grid
};

class SDFGrid {
public:
	SDFGrid(const fs::path& filename, float _solid_isovalue, float _gen_isovalue);
	~SDFGrid();

	// Returns a device accessor for the SDF grid
	SDFGridAccessor GetDeviceAccessor();

public:
	// Method to read SDF data from file
	void ReadSDFFile(const fs::path& filename, std::vector<float>& sdf_data);
	void ReloadSDFFile(const fs::path& filename);

	//grid_0 * (1 - alpha) + grid_1 * alpha
	void InterpolateFromTwoSDFGrids(const SDFGrid& sdf_grid_0, const SDFGrid& sdf_grid_1, float alpha);

public:
	// Device pointers for sdf data
	float* sdf_field;
	float solid_isovalue, gen_isovalue;

    // Host variables for grid dimensions and SDF data
    float cell_size_x, cell_size_y, cell_size_z;
    float origin_x, origin_y, origin_z; // Origin of the grid
    int nx, ny, nz;
    int num_cells; // Total number of cells in the grid

    std::vector<float> sdf_data;
};