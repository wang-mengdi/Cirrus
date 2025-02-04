#include "PoissonIOFunc.h"


//#include <vtkNew.h>
//#include <vtkPoints.h>
#include <vtkFloatArray.h>
#include <vtkDoubleArray.h>
#include <vtkAMRBox.h>
//#include <vtkCellData.h>
//#include <vtkUnstructuredGrid.h>

#include <vtkOverlappingAMR.h>
#include <vtkNonOverlappingAMR.h>
#include <vtkUniformGridAMR.h>
#include <vtkXMLUniformGridAMRWriter.h>
#include <vtkXMLUniformGridAMRReader.h>
#include <vtkUniformGrid.h>
#include <vtkAMRBox.h>
#include "vtkCompositeDataWriter.h"
#include "vtkXMLHierarchicalBoxDataWriter.h"
#include <vtkXMLImageDataWriter.h>

#include <zlib.h>

//#include <AMReX.H>
//#include <AMReX_AmrCore.H>
//#include <AMReX_AmrLevel.H>
//#include <AMReX_FArrayBox.H>
//#include <AMReX_PlotFileUtil.H>
//#include <AMReX_MultiFab.H>
//#include <AMReX_BoxArray.H>
//#include <AMReX_DistributionMapping.H>
//#include <AMReX_Geometry.H>
//#include <AMReX_ParallelDescriptor.H>
//#include <AMReX_Vector.H>
//#include <AMReX_PlotFileUtil.H>

//#include <hdf5.h>

#include <polyscope/polyscope.h>
#include <polyscope/point_cloud.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/volume_mesh.h>


namespace IOFunc {
//#include <vector>
//#include <fstream>
//#include <boost/filesystem.hpp>

    //namespace bf = boost::filesystem;
    //namespace bf = 

    void WriteHAHostTileHolderToFile(const HAHostTileHolder<Tile>& holder, const fs::path& filepath) {
        std::ofstream out(filepath.string(), std::ios::binary);
        if (!out) {
            throw std::runtime_error("Failed to open file for writing: " + filepath.string());
        }

        // 写入简单成员变量
        out.write(reinterpret_cast<const char*>(&holder.mH0), sizeof(holder.mH0));
        out.write(reinterpret_cast<const char*>(&holder.mNumLayers), sizeof(holder.mNumLayers));
        out.write(reinterpret_cast<const char*>(&holder.mMaxLevel), sizeof(holder.mMaxLevel));

        // 写入 mHostTiles
        size_t numTiles = holder.mHostTiles.size();
        out.write(reinterpret_cast<const char*>(&numTiles), sizeof(numTiles));
        for (const auto& tile : holder.mHostTiles) {
            out.write(reinterpret_cast<const char*>(&tile), sizeof(tile));
        }

        // 写入 mHostLevels
        size_t numLevels = holder.mHostLevels.size();
        out.write(reinterpret_cast<const char*>(&numLevels), sizeof(numLevels));
        for (const auto& level : holder.mHostLevels) {
            size_t levelSize = level.size();
            out.write(reinterpret_cast<const char*>(&levelSize), sizeof(levelSize));
            for (const auto& tileInfo : level) {
                // 写入 mTileCoord 和其他成员变量
                out.write(reinterpret_cast<const char*>(&tileInfo.mTileCoord), sizeof(tileInfo.mTileCoord));
                out.write(reinterpret_cast<const char*>(&tileInfo.mLevel), sizeof(tileInfo.mLevel));
                out.write(reinterpret_cast<const char*>(&tileInfo.mType), sizeof(tileInfo.mType));

                // 写入 mTilePtr 的索引
                intptr_t tileIndex = tileInfo.mTilePtr ? (tileInfo.mTilePtr - holder.mHostTiles.data()) : -1;
                out.write(reinterpret_cast<const char*>(&tileIndex), sizeof(tileIndex));
            }
        }

        out.close();
    }

    

    HAHostTileHolder<Tile> ReadHAHostTileHolderFromFile(const fs::path& filepath) {
        std::ifstream in(filepath.string(), std::ios::binary);
        if (!in) {
            throw std::runtime_error("Failed to open file for reading: " + filepath.string());
        }

        HAHostTileHolder<Tile> holder;

        // 读取简单成员变量
        in.read(reinterpret_cast<char*>(&holder.mH0), sizeof(holder.mH0));
        in.read(reinterpret_cast<char*>(&holder.mNumLayers), sizeof(holder.mNumLayers));
        in.read(reinterpret_cast<char*>(&holder.mMaxLevel), sizeof(holder.mMaxLevel));

        // 读取 mHostTiles
        size_t numTiles;
        in.read(reinterpret_cast<char*>(&numTiles), sizeof(numTiles));
        holder.mHostTiles.resize(numTiles);
        for (auto& tile : holder.mHostTiles) {
            in.read(reinterpret_cast<char*>(&tile), sizeof(tile));
        }

        // 读取 mHostLevels
        size_t numLevels;
        in.read(reinterpret_cast<char*>(&numLevels), sizeof(numLevels));
        holder.mHostLevels.resize(numLevels);
        for (auto& level : holder.mHostLevels) {
            size_t levelSize;
            in.read(reinterpret_cast<char*>(&levelSize), sizeof(levelSize));
            level.resize(levelSize);
            for (auto& tileInfo : level) {
                // 读取 mTileCoord 和其他成员变量
                in.read(reinterpret_cast<char*>(&tileInfo.mTileCoord), sizeof(tileInfo.mTileCoord));
                in.read(reinterpret_cast<char*>(&tileInfo.mLevel), sizeof(tileInfo.mLevel));
                in.read(reinterpret_cast<char*>(&tileInfo.mType), sizeof(tileInfo.mType));

                // 读取 mTilePtr 的索引并重建指针
                intptr_t tileIndex;
                in.read(reinterpret_cast<char*>(&tileIndex), sizeof(tileIndex));
                tileInfo.mTilePtr = (tileIndex >= 0) ? &holder.mHostTiles[tileIndex] : nullptr;
            }
        }

        in.close();

        return holder;
    }

    void WriteUChannelsToCompressedFile(const std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr, const fs::path& filepath) {
        auto& holder = *holder_ptr;

        // Open the file for writing in binary mode
        std::ofstream out(filepath.string(), std::ios::binary);
        if (!out) {
            throw std::runtime_error("Failed to open file for writing: " + filepath.string());
        }

        // Create a buffer to store uncompressed data
        std::vector<uint8_t> uncompressedData;

        // Traverse each level in mHostLevels
        for (const auto& level : holder.mHostLevels) {
            for (const auto& tileInfo : level) {
                // Extract level and coord information
                int level = tileInfo.mLevel;
                int coordX = tileInfo.mTileCoord[0];
                int coordY = tileInfo.mTileCoord[1];
                int coordZ = tileInfo.mTileCoord[2];

                // Append level and coordinates to the uncompressed buffer
                uncompressedData.insert(uncompressedData.end(), reinterpret_cast<const uint8_t*>(&level), reinterpret_cast<const uint8_t*>(&level) + sizeof(level));
                uncompressedData.insert(uncompressedData.end(), reinterpret_cast<const uint8_t*>(&coordX), reinterpret_cast<const uint8_t*>(&coordX) + sizeof(coordX));
                uncompressedData.insert(uncompressedData.end(), reinterpret_cast<const uint8_t*>(&coordY), reinterpret_cast<const uint8_t*>(&coordY) + sizeof(coordY));
                uncompressedData.insert(uncompressedData.end(), reinterpret_cast<const uint8_t*>(&coordZ), reinterpret_cast<const uint8_t*>(&coordZ) + sizeof(coordZ));

                // Find the corresponding tile using mTilePtr in tileInfo
                if (tileInfo.mTilePtr == nullptr) {
                    throw std::runtime_error("Tile pointer is null.");
                }
                const auto& tile = *tileInfo.mTilePtr;

                // Write data from u_channel, u_channel + 1, and u_channel + 2
                for (int channelOffset = 0; channelOffset < 3; ++channelOffset) {
                    const auto& channelData = tile.mData[Tile::u_channel + channelOffset];
                    uncompressedData.insert(uncompressedData.end(),
                        reinterpret_cast<const uint8_t*>(channelData),
                        reinterpret_cast<const uint8_t*>(channelData + Tile::SIZE));
                }
            }
        }

        // Compress the uncompressed data
        uLongf compressedSize = compressBound(uncompressedData.size());
        std::vector<uint8_t> compressedData(compressedSize);

        int ret = compress(compressedData.data(), &compressedSize, uncompressedData.data(), uncompressedData.size());
        if (ret != Z_OK) {
            throw std::runtime_error("Compression failed with error code: " + std::to_string(ret));
        }

        // Write the compressed data to the file
        out.write(reinterpret_cast<const char*>(compressedData.data()), compressedSize);

        // Close the file
        out.close();
    }

    void OutputParticleSystemAsVTU(std::shared_ptr<thrust::host_vector<Particle>> particles_ptr, fs::path path) {
        fmt::print("Output Particle System to vtu file: {}\n", path.string());
        auto& particles = *particles_ptr;

        // setup VTK
        vtkNew<vtkXMLUnstructuredGridWriter> writer;
        vtkNew<vtkUnstructuredGrid> unstructured_grid;

        // Use vtkFloatArray for position data (float instead of double)
        vtkNew<vtkFloatArray> positions;
        positions->SetNumberOfComponents(3);  // 3D points
        positions->SetNumberOfTuples(particles.size());
        positions->SetName("Positions");

        // Use vtkTypeInt64Array for global_idx
        vtkNew<vtkTypeInt64Array> global_idx_array;
        global_idx_array->SetName("global_idx");
        global_idx_array->SetNumberOfComponents(1);  // Scalar data
        global_idx_array->SetNumberOfTuples(particles.size());

        for (int i = 0; i < particles.size(); i++) {
            auto p = particles[i].pos;  // Assume pos is a float array or convertible to float
            auto global_idx = particles[i].global_idx;

            positions->SetTuple3(i, p[0], p[1], p[2]);  // Add float position data
            global_idx_array->SetTuple1(i, global_idx); // Add global_idx data
        }

        // Set points for the grid
        vtkNew<vtkPoints> nodes;
        nodes->SetData(positions);
        unstructured_grid->SetPoints(nodes);

        // Add global_idx array as point data
        unstructured_grid->GetPointData()->AddArray(global_idx_array);

        // Write the output file
        writer->SetFileName(path.string().c_str());
        writer->SetInputData(unstructured_grid);
        writer->SetDataModeToBinary();  // Optional: Use binary mode for smaller file size
        writer->Write();
    }


    void OutputParticleSystemAsVTU(const thrust::device_vector<Particle>& particles_d, const fs::path& path) {
        fmt::print("Output Particle System to vtu file: {}\n", path.string());
		auto particles_ptr = std::make_shared<thrust::host_vector<Particle>>(particles_d);
		OutputParticleSystemAsVTU(particles_ptr, path);
    }

    void AddParticleSystemToPolyscope(thrust::device_vector<Particle> particles_d, std::string name) {
		thrust::host_vector<Particle> particles = particles_d;

        std::vector<Vec> positions;
        std::vector<Vec> impulses;

        positions.reserve(particles.size());
		impulses.reserve(particles.size());

        for (int i = 0; i < particles.size(); i++) {
            auto p = particles[i].pos;
            auto v = particles[i].impulse;
            positions.emplace_back(p[0], p[1], p[2]);
            impulses.emplace_back(v[0], v[1], v[2]);
        }

        polyscope::PointCloud* psCloud = polyscope::registerPointCloud(name, positions);
        psCloud->addVectorQuantity("impulse", impulses);
    }

    void OutputTilesAsVTU(std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr, const fs::path& path) {
        using Coord = typename Tile::CoordType;

        fmt::print("Output Leaf Tiles in DA Device Grid to vtu file: {}\n", path.string());
        // Create VTK unstructured grid
        vtkSmartPointer<vtkUnstructuredGrid> unstructuredGrid = vtkSmartPointer<vtkUnstructuredGrid>::New();
        // Create VTK points
        vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();


        int hex_offs[8][3] = {
            {0,0,0},{1,0,0},{1,1,0},{0,1,0},
            {0,0,1},{1,0,1},{1,1,1},{0,1,1} };

		auto& holder = *holder_ptr;
        auto c_acc = holder.coordAccessor();
        holder.iterateLeafTiles(
            [&](const HATileInfo<Tile>& info) {
                vtkNew<vtkHexahedron> hex;
                auto bbox = c_acc.tileBBox(info);
                for (int s = 0; s < 8; s++) {
                    Coord off(hex_offs[s][0], hex_offs[s][1], hex_offs[s][2]);
                    auto p = bbox.min() + Vec(off[0], off[1], off[2]) * bbox.dim();
                    auto idx = points->InsertNextPoint(p[0], p[1], p[2]);
                    hex->GetPointIds()->SetId(s, idx);
                }
                unstructuredGrid->InsertNextCell(hex->GetCellType(), hex->GetPointIds());
            }
        );

        unstructuredGrid->SetPoints(points);

        // Write to vtu file
        vtkSmartPointer<vtkXMLUnstructuredGridWriter> writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
        writer->SetFileName(path.string().c_str());
        writer->SetInputData(unstructuredGrid);
        writer->SetDataModeToBinary();
        writer->Write();
    }

    void OutputPoissonGridAsUnstructuredVTU(std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr, const std::vector<std::pair<int, std::string>> scalar_channels, std::vector<std::pair<int, std::string>> vec_channels, const fs::path& path) {
        CPUTimer<std::chrono::milliseconds> timer;
        timer.start();

        auto& holder = *holder_ptr;

        std::vector<HATileInfo<Tile> > leaf_infos;
		for (int i = 0; i <= holder.mMaxLevel; i++) {
			for (auto& info : holder.mHostLevels[i]) {
				if (info.isLeaf()) {
                    leaf_infos.push_back(info);
				}
			}
		}
		int num_leafs = leaf_infos.size();
		int num_leaf_voxels = num_leafs * Tile::SIZE;
		int num_leaf_nodes = num_leafs * Tile::NODESIZE;

        using Coord = typename Tile::CoordType;

        auto acc = holder.coordAccessor();

        fmt::print("Start output Poisson Grid to vtu file: {}\n", path.string());


        vtkNew<vtkPoints> points;
        //load points
        {
            points->SetNumberOfPoints(num_leaf_nodes);
            for (int i = 0; i < leaf_infos.size(); i++) {
                auto& info = leaf_infos[i];
                int base_off = i * Tile::NODESIZE;
                for (int j = 0; j < Tile::NODESIZE; j++) {

                    auto r_ijk = acc.localNodeOffsetToCoord(j);

                    auto pos = acc.cellCorner(info, r_ijk);
                    points->SetPoint(base_off + j, pos[0], pos[1], pos[2]);
                }
            }
        }

        // Create VTK unstructured grid
        vtkNew<vtkIdTypeArray> hex_vert_ids;
        //load hex vert ids
        {
            hex_vert_ids->SetNumberOfValues(num_leafs * Tile::SIZE * (1 + 8));
            int hex_offs[8][3] = {
    {0,0,0},{1,0,0},{1,1,0},{0,1,0},
    {0,0,1},{1,0,1},{1,1,1},{0,1,1} };
            for (int i = 0; i < leaf_infos.size(); i++) {
                int tile_node_off = i * Tile::NODESIZE;
                for (int j = 0; j < Tile::SIZE; j++) {
                    Coord l_ijk = acc.localOffsetToCoord(j);
                    hex_vert_ids->SetValue(i * Tile::SIZE * 9 + j * 9 + 0, 8);

                    for (int s = 0; s < 8; s++) {
                        Coord off(hex_offs[s][0], hex_offs[s][1], hex_offs[s][2]);
                        auto idx = acc.localNodeCoordToOffset(l_ijk + off);
						hex_vert_ids->SetValue(i * Tile::SIZE * 9 + j * 9 + s + 1, tile_node_off + idx);
                        //hex_vert_ids->SetValue(i * Tile::SIZE * 8 + j * 8 + s, tile_node_off + idx);
                    }
                }
            }
        }

        vtkNew<vtkCellArray> hexagons;
        {
            hexagons->SetCells(num_leaf_voxels, hex_vert_ids);
        }
        vtkNew<vtkUnstructuredGrid> unstructuredGrid;
		unstructuredGrid->SetPoints(points);
		unstructuredGrid->SetCells(VTK_HEXAHEDRON, hexagons);


        std::vector<vtkSmartPointer<vtkFloatArray>> scalar_data;
        std::vector<vtkSmartPointer<vtkFloatArray>> vec_data;

        for (int i = 0; i < scalar_channels.size(); i++) {
            auto data = vtkSmartPointer<vtkFloatArray>::New();
            data->SetName(scalar_channels[i].second.c_str());
            data->SetNumberOfComponents(1);
            data->Allocate(num_leaf_voxels);
            scalar_data.push_back(data);
        }

        for (int i = 0; i < vec_channels.size(); i++) {
            auto data = vtkSmartPointer<vtkFloatArray>::New();
            data->SetName(vec_channels[i].second.c_str());
            data->SetNumberOfComponents(3);
            data->Allocate(num_leaf_voxels);
            vec_data.push_back(data);
        }

        for (int info_idx = 0; info_idx < leaf_infos.size(); info_idx++) {
			int node_base_off = info_idx * Tile::NODESIZE;
			int cell_base_off = info_idx * Tile::SIZE;

            auto& info = leaf_infos[info_idx];
            if (info.isLeaf()) {

                auto& tile = info.tile();
                for (int cell_idx = 0; cell_idx < Tile::SIZE; cell_idx++) {
					Coord l_ijk = acc.localOffsetToCoord(cell_idx);
                    auto g_ijk = acc.localToGlobalCoord(info, l_ijk);

                    //type_data->InsertNextValue(tile.type(l_ijk));
                    for (int i = 0; i < scalar_channels.size(); i++) {
                        int channel = scalar_channels[i].first;
                        T f = (channel == -1) ? tile.type(l_ijk) : tile(scalar_channels[i].first, l_ijk);
                        scalar_data[i]->InsertNextValue(f);
                    }
                    for (int i = 0; i < vec_channels.size(); i++) {
                        int u_channel = vec_channels[i].first;
                        auto u = tile(u_channel, l_ijk);
                        auto v = tile(u_channel + 1, l_ijk);
                        auto w = tile(u_channel + 2, l_ijk);
                        vec_data[i]->InsertNextTuple3(u, v, w);
                    }
                }
            }
        }

        for (int i = 0; i < scalar_channels.size(); i++) {
			unstructuredGrid->GetCellData()->AddArray(scalar_data[i]);
        }
        for (int i = 0; i < vec_channels.size(); i++) {
            unstructuredGrid->GetCellData()->AddArray(vec_data[i]);
        }

        //timer.stop("build output structure");
        //timer.start();


        // Write to vtu file
        vtkSmartPointer<vtkXMLUnstructuredGridWriter> writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
        writer->SetFileName(path.string().c_str());
        writer->SetInputData(unstructuredGrid);
        writer->SetDataModeToBinary();
        writer->SetCompressorTypeToZLib();
        writer->Write();

        timer.stop(fmt::format("Output grid to unstructured .vtu file: {}", path.string()));
    }

    void OutputPoissonGridAsStructuredVTI(std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr, const std::vector<std::pair<int, std::string>> scalar_channels, std::vector<std::pair<int, std::string>> vec_channels, const fs::path& path) {
        CPUTimer<std::chrono::milliseconds> timer;
        timer.start();

        auto& holder = *holder_ptr;
        std::vector<HATileInfo<Tile>> leaf_infos;

        for (int i = 0; i <= holder.mMaxLevel; i++) {
            for (auto& info : holder.mHostLevels[i]) {
                if (info.isLeaf()) {
                    leaf_infos.push_back(info);
                }
            }
        }

        using Coord = typename Tile::CoordType;
        auto acc = holder.coordAccessor();
        int max_level = holder.mMaxLevel;

        // Step 1: Find the global coordinate range at the finest level
        Coord global_min = Coord(
            std::numeric_limits<int>::max(),
            std::numeric_limits<int>::max(),
            std::numeric_limits<int>::max()
        );
        Coord global_max = Coord(
            std::numeric_limits<int>::lowest(),
            std::numeric_limits<int>::lowest(),
            std::numeric_limits<int>::lowest()
        );

        for (auto& info : leaf_infos) {
            if (info.isLeaf()) {
                auto& tile = info.tile();
                int level = info.mLevel;
                int shift = holder.mMaxLevel - level;

                for (Coord l_ijk : {Coord(0, 0, 0), Coord(Tile::DIM - 1, Tile::DIM - 1, Tile::DIM - 1)}) {
                    Coord g_ijk = acc.localToGlobalCoord(info, l_ijk);

                    Coord min_fine_coord(g_ijk[0] << shift, g_ijk[1] << shift, g_ijk[2] << shift);
                    Coord max_fine_coord((g_ijk[0] + 1) << shift, (g_ijk[1] + 1) << shift, (g_ijk[2] + 1) << shift);

                    global_min[0] = std::min(global_min[0], min_fine_coord[0]);
                    global_min[1] = std::min(global_min[1], min_fine_coord[1]);
                    global_min[2] = std::min(global_min[2], min_fine_coord[2]);

                    global_max[0] = std::max(global_max[0], max_fine_coord[0]);
                    global_max[1] = std::max(global_max[1], max_fine_coord[1]);
                    global_max[2] = std::max(global_max[2], max_fine_coord[2]);
                }
            }
        }


        // Step 2: Determine the structured grid dimensions and allocate points
        Coord grid_dimensions = (global_max - global_min);
        vtkNew<vtkImageData> structuredGrid;
        structuredGrid->SetDimensions(grid_dimensions[0], grid_dimensions[1], grid_dimensions[2]);
        structuredGrid->SetOrigin(global_min[0], global_min[1], global_min[2]);
		auto h = acc.voxelSize(max_level);
        structuredGrid->SetSpacing(h, h, h);

        // Step 3: Allocate scalar and vector data arrays for structured grid
        std::vector<vtkSmartPointer<vtkFloatArray>> scalar_data;
        std::vector<vtkSmartPointer<vtkFloatArray>> vec_data;

        for (const auto& scalar_channel : scalar_channels) {
            auto data = vtkSmartPointer<vtkFloatArray>::New();
            data->SetName(scalar_channel.second.c_str());
            data->SetNumberOfComponents(1);
            data->SetNumberOfTuples(grid_dimensions[0] * grid_dimensions[1] * grid_dimensions[2]);
            scalar_data.push_back(data);
        }

        for (const auto& vec_channel : vec_channels) {
            auto data = vtkSmartPointer<vtkFloatArray>::New();
            data->SetName(vec_channel.second.c_str());
            data->SetNumberOfComponents(3);
            data->SetNumberOfTuples(grid_dimensions[0] * grid_dimensions[1] * grid_dimensions[2]);
            vec_data.push_back(data);
        }

        for (auto& info : leaf_infos) {
            if (info.isLeaf()) {
                auto& tile = info.tile();
                int level = info.mLevel;
                int shift = holder.mMaxLevel - level;

                for (int cell_idx = 0; cell_idx < Tile::SIZE; cell_idx++) {
                    Coord l_ijk = acc.localOffsetToCoord(cell_idx);
                    Coord g_ijk = acc.localToGlobalCoord(info, l_ijk);

                    Coord min_fine_coord(g_ijk[0] << shift, g_ijk[1] << shift, g_ijk[2] << shift);
                    Coord max_fine_coord((g_ijk[0] + 1) << shift, (g_ijk[1] + 1) << shift, (g_ijk[2] + 1) << shift);

                    for (int x = min_fine_coord[0]; x < max_fine_coord[0]; ++x) {
                        for (int y = min_fine_coord[1]; y < max_fine_coord[1]; ++y) {
                            for (int z = min_fine_coord[2]; z < max_fine_coord[2]; ++z) {
                                int flat_index = (x - global_min[0]) +
                                    (y - global_min[1]) * grid_dimensions[0] +
                                    (z - global_min[2]) * grid_dimensions[0] * grid_dimensions[1];


                                for (int i = 0; i < scalar_channels.size(); i++) {
                                    int channel = scalar_channels[i].first;
                                    T value;
                                    if (channel == -1) value = tile.type(l_ijk);
                                    else if (channel == -2) value = level;
                                    else value = tile(channel, l_ijk);

                                    scalar_data[i]->SetValue(flat_index, value);
                                }

                                for (int i = 0; i < vec_channels.size(); i++) {
                                    int u_channel = vec_channels[i].first;
                                    auto u = tile(u_channel, l_ijk);
                                    auto v = tile(u_channel + 1, l_ijk);
                                    auto w = tile(u_channel + 2, l_ijk);
                                    vec_data[i]->SetTuple3(flat_index, u, v, w);
                                }
                            }
                        }
                    }
                }
            }
        }


        // Step 5: Add the scalar and vector data to the structured grid
        for (auto& data : scalar_data) {
            structuredGrid->GetPointData()->AddArray(data);
        }
        for (auto& data : vec_data) {
            structuredGrid->GetPointData()->AddArray(data);
        }

        // Step 6: Write the structured grid to a .vti file
        vtkSmartPointer<vtkXMLImageDataWriter> writer = vtkSmartPointer<vtkXMLImageDataWriter>::New();
        writer->SetFileName(path.string().c_str());
        writer->SetInputData(structuredGrid);
        writer->SetDataModeToBinary();
        writer->SetCompressorTypeToZLib();
        writer->Write();

        timer.stop(fmt::format("Output grid to structured .vti file: {}", path.string()));
    }

    void AddTilesToPolyscopeVolumetricMesh(HADeviceGrid<Tile>& grid, const uint8_t types, std::string name) {
        using Coord = typename Tile::CoordType;

        int hex_offs[8][3] = {
            {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0},
            {0,0,1}, {1,0,1}, {1,1,1}, {0,1,1}
        };

        std::vector<Vec> vertices;
        std::vector<std::array<size_t, 8>> hexCells;
        std::vector<int> cellLevels;
        std::vector<Coord> cellCoords;
        std::vector<int> cellInterestFlags;

        auto acc = grid.hostAccessor();

        for (int level = 0; level < grid.mNumLayers; level++) {
            for (int i = 0; i < grid.hNumTiles[level]; i++) {
                auto& info = grid.hTileArrays[level][i];
                if (info.mType & types) {
                    auto bbox = acc.tileBBox(info);
                    std::array<size_t, 8> hexCell;

                    for (int s = 0; s < 8; s++) {
                        Coord off(hex_offs[s][0], hex_offs[s][1], hex_offs[s][2]);
                        auto p = bbox.min() + Vec(off[0], off[1], off[2]) * bbox.dim();
                        Vec vertex(p[0], p[1], p[2]);

                        auto it = std::find(vertices.begin(), vertices.end(), vertex);
                        if (it != vertices.end()) {
                            hexCell[s] = std::distance(vertices.begin(), it);
                        }
                        else {
                            vertices.push_back(vertex);
                            hexCell[s] = vertices.size() - 1;
                        }
                    }

                    hexCells.push_back(hexCell);
                    cellLevels.push_back(level);
                    cellCoords.push_back(info.mTileCoord);

                    auto tile = info.getTile(DEVICE);
                    cellInterestFlags.push_back(tile.mIsInterestArea);
                }
            }
        }

        auto mesh = polyscope::registerVolumeMesh(name, vertices, hexCells);
        mesh->addCellScalarQuantity("Level", cellLevels);
        mesh->addCellVectorQuantity("Tile Coord", cellCoords);
		mesh->addCellScalarQuantity("Interest Flag", cellInterestFlags);
        mesh->setTransparency(0.2);
    }

    void AddLeveledPoissonGridCellCentersToPolyscopePointCloud(std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr, const std::vector<std::pair<int, std::string>> scalar_channels, std::vector<std::pair<int, std::string>> vec_channels, const double invalid_value) {
        auto& holder = *holder_ptr;
        using Coord = typename Tile::CoordType;

        auto acc = holder.coordAccessor();
        auto add_data = [&](const int level, const uint8_t tile_types, const std::string name) {
            std::vector<Vec> points;
            std::vector<int> levels;
            std::vector<int> ttypes;
            std::vector<Coord> ijk_values;

            std::vector<std::vector<float>> scalar_data(scalar_channels.size());
            std::vector<std::vector<Vec>> vec_data(vec_channels.size());

            for (auto& info : holder.mHostLevels[level]) {
                if (!(info.mType & tile_types)) continue;
                auto& tile = info.tile();

                for (int i = 0; i < Tile::DIM; i++) {
                    for (int j = 0; j < Tile::DIM; j++) {
                        for (int k = 0; k < Tile::DIM; k++) {
                            Coord l_ijk(i, j, k);
                            auto pos = acc.cellCenter(info, l_ijk); // 使用cellCenter代替cellCorner
                            points.push_back(pos);
                            levels.push_back(level);
                            ttypes.push_back(info.mType);

                            auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
                            ijk_values.push_back(g_ijk);

                            for (int s = 0; s < scalar_channels.size(); s++) {
                                int channel = scalar_channels[s].first;
                                T f = (channel == -1) ? tile.type(l_ijk) : tile(scalar_channels[s].first, l_ijk);
                                if (isnan(f) || isinf(f)) f = invalid_value;
                                scalar_data[s].push_back(f);
                            }

                            for (int t = 0; t < vec_channels.size(); t++) {
                                int u_channel = vec_channels[t].first;
                                auto u = tile(u_channel, l_ijk);
                                auto v = tile(u_channel + 1, l_ijk);
                                auto w = tile(u_channel + 2, l_ijk);
                                vec_data[t].push_back({ u, v, w });
                            }
                        }
                    }
                }
            }

            if (points.empty()) return;
                
            auto ps = polyscope::registerPointCloud(name, points);
            ps->addScalarQuantity("Level", levels);
            ps->addVectorQuantity("IJK", ijk_values);
            ps->addScalarQuantity("Tile Type", ttypes);

            for (int s = 0; s < scalar_channels.size(); s++) {
                ps->addScalarQuantity(scalar_channels[s].second, scalar_data[s]);
            }

            for (int v = 0; v < vec_channels.size(); v++) {
                ps->addVectorQuantity(vec_channels[v].second, vec_data[v]);
            }
            };

        for (int level = 0; level <= holder.mMaxLevel; level++) {
            add_data(level, LEAF, fmt::format("Level{}LEAF", level));
            add_data(level, GHOST, fmt::format("Level{}GHOST", level));
            add_data(level, NONLEAF, fmt::format("Level{}NONLEAF", level));
        }
    }

    void AddPoissonGridCellCentersToPolyscopePointCloud(std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr, const std::vector<std::pair<int, std::string>> scalar_channels, std::vector<std::pair<int, std::string>> vec_channels, const double invalid_value) {
        auto& holder = *holder_ptr;
        using Coord = typename Tile::CoordType;

        std::vector<Vec> points;
        std::vector<int> levels;
        std::vector<Coord> ijk_values;

        std::vector<std::vector<float>> scalar_data(scalar_channels.size());
        std::vector<std::vector<Vec>> vec_data(vec_channels.size());

        auto acc = holder.coordAccessor();
        for (int level = 0; level <= holder.mMaxLevel; level++) {
            for (auto& info : holder.mHostLevels[level]) {
                if (info.isLeaf()) {
                    auto& tile = info.tile();

                    for (int i = 0; i < Tile::DIM; i++) {
                        for (int j = 0; j < Tile::DIM; j++) {
                            for (int k = 0; k < Tile::DIM; k++) {
                                Coord l_ijk(i, j, k);
                                auto pos = acc.cellCenter(info, l_ijk); // 使用cellCenter代替cellCorner
                                points.push_back(pos);
                                levels.push_back(level);

								auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
                                ijk_values.push_back(g_ijk);

                                for (int s = 0; s < scalar_channels.size(); s++) {
                                    int channel = scalar_channels[s].first;
                                    T f = (channel == -1) ? tile.type(l_ijk) : tile(scalar_channels[s].first, l_ijk);
                                    //if(channel!=-1) Info("channel {} l_ijk: {}, f: {}", channel, l_ijk, f);
                                    if (isnan(f) || isinf(f)) f = invalid_value;
                                    scalar_data[s].push_back(f);
                                }

                                for (int t = 0; t < vec_channels.size(); t++) {
                                    int u_channel = vec_channels[t].first;
                                    auto u = tile(u_channel, l_ijk);
                                    auto v = tile(u_channel + 1, l_ijk);
                                    auto w = tile(u_channel + 2, l_ijk);
                                    vec_data[t].push_back({ u, v, w });
                                }
                            }
                        }
                    }
                }
            }
        }

        polyscope::registerPointCloud("Poisson Grid", points);
        polyscope::getPointCloud("Poisson Grid")->addScalarQuantity("Level", levels);
        polyscope::getPointCloud("Poisson Grid")->addVectorQuantity("IJK", ijk_values);

        for (int s = 0; s < scalar_channels.size(); s++) {
            polyscope::getPointCloud("Poisson Grid")->addScalarQuantity(scalar_channels[s].second, scalar_data[s]);
        }

        for (int v = 0; v < vec_channels.size(); v++) {
            polyscope::getPointCloud("Poisson Grid")->addVectorQuantity(vec_channels[v].second, vec_data[v]);
        }
    }

    void AddPoissonGridFaceCentersToPolyscopePointCloud(std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr, std::vector<std::pair<int, std::string>> vec_channels, const double invalid_value) {
        auto& holder = *holder_ptr;
        using Coord = typename Tile::CoordType;

        std::vector<Vec> points[3];
        std::vector<int> levels[3];
        std::vector<Coord> ijk_values[3];

   //     std::vector<std::vector<Vec>> vec_data[3];
   //     for (int axis : {0, 1, 2}) {
			//vec_data[axis].resize(vec_channels.size());
   //     }

        std::vector<std::vector<float>> face_center_data[3];
        for (int axis : {0, 1, 2}) {
            face_center_data[axis].resize(vec_channels.size());
        }

        auto acc = holder.coordAccessor();
        holder.iterateLeafCells([&](const HATileInfo<Tile>& info, const Coord& l_ijk) {
            auto& tile = info.tile();

            for (int axis : {0, 1, 2}) {
                auto pos = acc.faceCenter(axis, info, l_ijk);
                points[axis].push_back(pos);
                levels[axis].push_back(info.mLevel);

                auto g_ijk = holder.coordAccessor().localToGlobalCoord(info, l_ijk);
                ijk_values[axis].push_back(g_ijk);

                for (int t = 0; t < vec_channels.size(); t++) {
                    int u_channel = vec_channels[t].first;
     //               Vec vec(0, 0, 0);
					//vec[axis] = tile(u_channel + axis, l_ijk);
     //               vec_data[axis][t].push_back(vec);
                    face_center_data[axis][t].push_back(tile(u_channel + axis, l_ijk));
                }
            }
            });

        for (int axis : {0, 1, 2}) {
			auto pc = polyscope::registerPointCloud(fmt::format("Poisson Grid Face Centers Axis {}", axis), points[axis]);
			pc->addScalarQuantity("Level", levels[axis]);
			pc->addVectorQuantity("IJK", ijk_values[axis]);
			for (int v = 0; v < vec_channels.size(); v++) {
				//pc->addVectorQuantity(vec_channels[v].second, vec_data[axis][v]);
				pc->addScalarQuantity(vec_channels[v].second, face_center_data[axis][v]);
			}
        }
    }

    void AddPoissonGridNodesToPolyscope(std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr, const std::vector<std::pair<int, std::string>> scalar_channels, std::vector<std::pair<int, std::string>> vec_channels, const double invalid_value) {
        auto& holder = *holder_ptr;
        using Coord = typename Tile::CoordType;

        auto acc = holder.coordAccessor();
        //polyscope::init();

        for (int level = 0; level <= holder.mMaxLevel; level++) {
            std::vector<Vec> points;
            std::vector<Coord> tile_ijks, local_ijks;

            std::vector<std::vector<float>> scalar_data(scalar_channels.size());
            std::vector<std::vector<Vec>> vec_data(vec_channels.size());

            std::vector<Vec> tile_vertices;
            std::vector<std::array<size_t, 4>> tile_faces;

            for (auto& info : holder.mHostLevels[level]) {
                if (info.isLeaf()) {
                    auto& tile = info.tile();

                    for (int i = 0; i <= Tile::DIM; i++) {
                        for (int j = 0; j <= Tile::DIM; j++) {
                            for (int k = 0; k <= Tile::DIM; k++) {
                                Coord r_ijk(i, j, k);
                                auto pos = acc.cellCorner(info, r_ijk);
                                points.push_back(pos);

                                tile_ijks.push_back(info.mTileCoord);
                                local_ijks.push_back(r_ijk);

                                for (int s = 0; s < scalar_channels.size(); s++) {
                                    int channel = scalar_channels[s].first;
                                    T f = tile.node(scalar_channels[s].first, r_ijk);
                                    if (isnan(f) || isinf(f)) f = invalid_value;
                                    scalar_data[s].push_back(f);
                                }

                                for (int t = 0; t < vec_channels.size(); t++) {
                                    int u_channel = vec_channels[t].first;
                                    auto u = tile.node(u_channel, r_ijk);
                                    auto v = tile.node(u_channel + 1, r_ijk);
                                    auto w = tile.node(u_channel + 2, r_ijk);
                                    vec_data[t].push_back({ u, v, w });
                                }
                            }
                        }
                    }

                    // Define the 8 vertices of the tile (cube)
                    std::vector<Vec> cube_vertices;
                    for (int dx = 0; dx <= 1; ++dx) {
                        for (int dy = 0; dy <= 1; ++dy) {
                            for (int dz = 0; dz <= 1; ++dz) {
                                Coord l_ijk(dx * Tile::DIM, dy * Tile::DIM, dz * Tile::DIM);
                                cube_vertices.push_back(acc.cellCorner(info, l_ijk));
                            }
                        }
                    }

                    // Add vertices to the global list
                    auto base_idx = tile_vertices.size();
                    tile_vertices.insert(tile_vertices.end(), cube_vertices.begin(), cube_vertices.end());

                    // Define the faces of the cube
                    std::array<size_t, 4> faces[6] = {
                        {base_idx + 0, base_idx + 1, base_idx + 3, base_idx + 2},
                        {base_idx + 4, base_idx + 5, base_idx + 7, base_idx + 6},
                        {base_idx + 0, base_idx + 1, base_idx + 5, base_idx + 4},
                        {base_idx + 2, base_idx + 3, base_idx + 7, base_idx + 6},
                        {base_idx + 0, base_idx + 2, base_idx + 6, base_idx + 4},
                        {base_idx + 1, base_idx + 3, base_idx + 7, base_idx + 5}
                    };

                    // Add faces to the global list
                    tile_faces.insert(tile_faces.end(), std::begin(faces), std::end(faces));

                }
            }

            auto level_cloud_name = fmt::format("Level {} Nodes", level);
            auto pc = polyscope::registerPointCloud(level_cloud_name, points);
            pc->addVectorQuantity("b_ijk", tile_ijks);
			pc->addVectorQuantity("r_ijk", local_ijks);

            for (int s = 0; s < scalar_channels.size(); s++) {
                pc->addScalarQuantity(scalar_channels[s].second, scalar_data[s]);
            }

            for (int v = 0; v < vec_channels.size(); v++) {
                pc->addVectorQuantity(vec_channels[v].second, vec_data[v]);
            }

            // Register the tile mesh for this level
            //auto tile_mesh_name = fmt::format("Level {} Tiles", level);
            //polyscope::registerSurfaceMesh(tile_mesh_name, tile_vertices, tile_faces);
        }

        //polyscope::show();
    }

}