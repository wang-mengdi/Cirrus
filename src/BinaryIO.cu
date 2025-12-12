#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <limits>
#include <cstdint>
#include <nlohmann/json.hpp>

#include "BinaryIO.h"
#include "CPUTimer.h"

namespace BinaryIO {

    void OutputPoissonGridAsJsonAndBin(std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr, const std::vector<std::pair<int, std::string>>& scalar_channels, const std::vector<std::pair<int, std::string>>& vec_channels, const std::filesystem::path& json_path)
    {
        namespace fs = std::filesystem;

        CPUTimer<std::chrono::milliseconds> timer;
        timer.start();

        fs::path bin_path = json_path.parent_path()
            / (json_path.stem().string() + ".bin");

        auto& holder = *holder_ptr;
        std::vector<HATileInfo<Tile>> leaf_infos;

        for (int i = 0; i <= holder.mMaxLevel; ++i)
            for (auto& info : holder.mHostLevels[i])
                if (info.isLeaf()) leaf_infos.push_back(info);

        using Coord = typename Tile::CoordType;
        auto acc = holder.coordAccessor();
        int max_level = holder.mMaxLevel;

        // ------------------------------------------------------------
        // Step 1: global finest-level bounds
        // ------------------------------------------------------------
        Coord global_min(
            std::numeric_limits<int>::max(),
            std::numeric_limits<int>::max(),
            std::numeric_limits<int>::max());
        Coord global_max(
            std::numeric_limits<int>::lowest(),
            std::numeric_limits<int>::lowest(),
            std::numeric_limits<int>::lowest());

        for (auto& info : leaf_infos) {
            int shift = max_level - info.mLevel;
            for (Coord l : {Coord(0, 0, 0), Coord(Tile::DIM - 1, Tile::DIM - 1, Tile::DIM - 1)}) {
                Coord g = acc.localToGlobalCoord(info, l);
                Coord lo(g[0] << shift, g[1] << shift, g[2] << shift);
                Coord hi((g[0] + 1) << shift, (g[1] + 1) << shift, (g[2] + 1) << shift);

                for (int d = 0; d < 3; ++d) {
                    global_min[d] = std::min(global_min[d], lo[d]);
                    global_max[d] = std::max(global_max[d], hi[d]);
                }
            }
        }

        Coord dims = global_max - global_min;
        int64_t nx = dims[0], ny = dims[1], nz = dims[2];
        int64_t num_points = nx * ny * nz;
        double h = acc.voxelSize(max_level);

        // ------------------------------------------------------------
        // Step 2: channel layout
        // ------------------------------------------------------------
        struct Channel {
            int id;
            std::string name;
            int components;
            uint64_t offset;
            uint64_t bytes;
            bool is_vec;
        };

        std::vector<Channel> channels;
        uint64_t cursor = 0;

        auto push = [&](int id, const std::string& name, int c, bool v) {
            Channel ch;
            ch.id = id;
            ch.name = name;
            ch.components = c;
            ch.is_vec = v;
            ch.offset = cursor;
            ch.bytes = num_points * c * sizeof(float);
            cursor += ch.bytes;
            channels.push_back(ch);
            };

        for (auto& s : scalar_channels) push(s.first, s.second, 1, false);
        for (auto& v : vec_channels)    push(v.first, v.second, 3, true);

        // ------------------------------------------------------------
        // Step 3: write binary
        // ------------------------------------------------------------
        std::ofstream bout(bin_path, std::ios::binary);
        std::vector<float> buffer;

        auto flat_index = [&](int x, int y, int z) -> int64_t {
            return (z - global_min[2])
                + (y - global_min[1]) * nz
                + (x - global_min[0]) * nz * ny;
            };

        auto write_scalar = [&](int cid) {
            buffer.assign(num_points, 0.0f);
            for (auto& info : leaf_infos) {
                int shift = max_level - info.mLevel;
                auto& tile = info.tile();
                for (int c = 0; c < Tile::SIZE; ++c) {
                    Coord l = acc.localOffsetToCoord(c);
                    Coord g = acc.localToGlobalCoord(info, l);
                    Coord lo(g[0] << shift, g[1] << shift, g[2] << shift);
                    Coord hi((g[0] + 1) << shift, (g[1] + 1) << shift, (g[2] + 1) << shift);

                    float v =
                        (cid == -1) ? (float)tile.type(l) :
                        (cid == -2) ? (float)info.mLevel :
                        (float)tile(cid, l);

                    for (int x = lo[0]; x < hi[0]; ++x)
                        for (int y = lo[1]; y < hi[1]; ++y)
                            for (int z = lo[2]; z < hi[2]; ++z)
                                buffer[flat_index(x, y, z)] = v;
                }
            }
            bout.write((char*)buffer.data(), buffer.size() * sizeof(float));
            };

        auto write_vec = [&](int cid) {
            buffer.assign(num_points * 3, 0.0f);
            for (auto& info : leaf_infos) {
                int shift = max_level - info.mLevel;
                auto& tile = info.tile();
                for (int c = 0; c < Tile::SIZE; ++c) {
                    Coord l = acc.localOffsetToCoord(c);
                    Coord g = acc.localToGlobalCoord(info, l);
                    Coord lo(g[0] << shift, g[1] << shift, g[2] << shift);
                    Coord hi((g[0] + 1) << shift, (g[1] + 1) << shift, (g[2] + 1) << shift);

                    float u = tile(cid + 0, l);
                    float v = tile(cid + 1, l);
                    float w = tile(cid + 2, l);

                    for (int x = lo[0]; x < hi[0]; ++x)
                        for (int y = lo[1]; y < hi[1]; ++y)
                            for (int z = lo[2]; z < hi[2]; ++z) {
                                int64_t idx = flat_index(x, y, z) * 3;
                                buffer[idx + 0] = u;
                                buffer[idx + 1] = v;
                                buffer[idx + 2] = w;
                            }
                }
            }
            bout.write((char*)buffer.data(), buffer.size() * sizeof(float));
            };

        for (auto& ch : channels) {
            if (ch.is_vec) write_vec(ch.id);
            else           write_scalar(ch.id);
        }
        bout.close();

        // ------------------------------------------------------------
        // Step 4: JSON
        // ------------------------------------------------------------
        nlohmann::json j;
        j["binary_file"] = bin_path.filename().string();
        j["grid"]["dimensions"] = { nx, ny, nz };
        j["grid"]["origin"] = { global_min[0], global_min[1], global_min[2] };
        j["grid"]["spacing"] = { h, h, h };
        j["layout"]["index_order"] = "z_fastest";
        j["layout"]["components_order"] = "AoS";
        j["layout"]["dtype"] = "float32";

        j["channels"] = nlohmann::json::array();
        for (auto& ch : channels) {
            j["channels"].push_back({
                {"name", ch.name},
                {"id", ch.id},
                {"components", ch.components},
                {"offset_bytes", ch.offset},
                {"bytes", ch.bytes}
                });
        }

        std::ofstream jout(json_path);
        jout << j.dump(2) << "\n";

        timer.stop(fmt::format("Output grid to json+bin (z-fastest): {}", json_path.string()));
    }

}