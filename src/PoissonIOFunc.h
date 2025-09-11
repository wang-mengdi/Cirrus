#pragma once

#include "PoissonGrid.h"
#include "FMParticles.h"
#include "CPUTimer.h"

#include <fmt/ostream.h>

#include <vtkHexahedron.h>
#include <vtkNew.h>
#include <vtkSmartPointer.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkUnstructuredGrid.h>
#include <vtkStructuredGrid.h>
#include <vtkXMLStructuredGridWriter.h>

namespace IOFunc {
    //plain binary I/O
    void WriteHAHostTileHolderToFile(const HAHostTileHolder<Tile>& holder, const fs::path& filepath);

    HAHostTileHolder<Tile> ReadHAHostTileHolderFromFile(const fs::path& filepath);
    template <typename T> void WriteHostVectorToBinary(const thrust::host_vector<T>& vec, const fs::path& filepath) {
        std::ofstream outFile(filepath.string(), std::ios::binary);
        if (!outFile) {
            throw std::runtime_error("Failed to open file for writing: " + filepath.string());
        }

        size_t size = vec.size();
        outFile.write(reinterpret_cast<const char*>(&size), sizeof(size));

        outFile.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(T));

        outFile.close();
    }

    template <typename T> thrust::host_vector<T> ReadHostVectorFromBinary(const fs::path& filepath) {
        std::ifstream inFile(filepath.string(), std::ios::binary);
        if (!inFile) {
            throw std::runtime_error("Failed to open file for reading: " + filepath.string());
        }

        // 读取向量大小
        size_t size;
        inFile.read(reinterpret_cast<char*>(&size), sizeof(size));

        // 读取向量内容
        thrust::host_vector<T> vec(size);
        inFile.read(reinterpret_cast<char*>(vec.data()), size * sizeof(T));

        inFile.close();
        return vec;
    }

    //particle system
    void OutputParticleSystemAsVTU(std::shared_ptr<thrust::host_vector<Particle>> particles_ptr, fs::path path);

    //tiles
    void OutputTilesAsVTU(std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr, const fs::path& path);

    //poisson grid
    //each pair in scalar_channels is (channel, name), if channel=-1 that means type
    //each pair in vec_channels is (u_channel, name)
    void OutputPoissonGridAsUnstructuredVTU(std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr, const std::vector<std::pair<int, std::string>> scalar_channels, std::vector<std::pair<int, std::string>> vec_channels, const fs::path& path);
    void OutputPoissonGridAsStructuredVTI(std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr, const std::vector<std::pair<int, std::string>> scalar_channels, std::vector<std::pair<int, std::string>> vec_channels, const fs::path& path);

    ////polyscope visualization
    //void AddParticleSystemToPolyscope(thrust::device_vector<Particle> particles_d, std::string name);
    //void AddTilesToPolyscopeVolumetricMesh(HADeviceGrid<Tile>& grid, const uint8_t types, std::string name);
    //void AddPoissonGridCellCentersToPolyscopePointCloud(std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr, const std::vector<std::pair<int, std::string>> scalar_channels, std::vector<std::pair<int, std::string>> vec_channels, const double invalid_value = FLT_MAX);
    //void AddLeveledPoissonGridCellCentersToPolyscopePointCloud(std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr, const std::vector<std::pair<int, std::string>> scalar_channels, std::vector<std::pair<int, std::string>> vec_channels, const double invalid_value);
    //void AddPoissonGridFaceCentersToPolyscopePointCloud(std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr, std::vector<std::pair<int, std::string>> vec_channels, const double invalid_value = FLT_MAX);
    //void AddPoissonGridNodesToPolyscope(std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr, const std::vector<std::pair<int, std::string>> scalar_channels, std::vector<std::pair<int, std::string>> vec_channels, const double invalid_value = FLT_MAX);
}