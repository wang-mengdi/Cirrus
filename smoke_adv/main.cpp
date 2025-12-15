// smoke_adv/main.cpp
#include <Alembic/AbcCoreOgawa/All.h>
#include <Alembic/AbcGeom/All.h>

#include <fmt/core.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>


namespace fs = std::filesystem;
using json = nlohmann::json;

using namespace Alembic::Abc;
using namespace Alembic::AbcGeom;

static const char* get_arg(int argc, char** argv, const char* key, const char* defval = nullptr) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == key) return argv[i + 1];
    }
    return defval;
}

struct DriverConfig {
    int first_frame = 0;
    int last_frame = 0;
    int fps = 24;
    fs::path output_base_dir;
};

static DriverConfig load_driver_config(const fs::path& input_dir) {
    fs::path cfg_path = input_dir / "config.json";
    std::ifstream fin(cfg_path);
    if (!fin) {
        throw std::runtime_error("Cannot open config.json: " + cfg_path.string());
    }

    json j;
    fin >> j;

    DriverConfig cfg;

    auto& d = j.at("driver");

    cfg.first_frame = d.value("first_frame", 0);
    cfg.last_frame = d.value("last_frame", cfg.first_frame);
    cfg.fps = d.value("fps", 24);
	cfg.output_base_dir = input_dir;

    cfg.fps = std::max(1, cfg.fps);
    if (cfg.last_frame < cfg.first_frame) std::swap(cfg.first_frame, cfg.last_frame);

    return cfg;
}


// ------------------------------------------------------------
// Main
// ------------------------------------------------------------
int main(int argc, char** argv)
{
    // ================================
    // Only argument: --input_dir
    // ================================
    const char* input_dir_c = get_arg(argc, argv, "--input_dir", nullptr);
    if (!input_dir_c) {
		fmt::print("Usage: {} --input_dir <input_directory>\n", argv[0]);
        return 1;
    }
    fs::path input_dir(input_dir_c);

    DriverConfig dc = load_driver_config(input_dir);
    const int first_frame = dc.first_frame;
    const int last_frame = dc.last_frame;
    const int fps = dc.fps;

    const int frames = (last_frame - first_frame + 1);

    fs::path out_path = dc.output_base_dir / "cpp_smoke_particles.abc";

    constexpr int kNumParticles = 200000;
    const int npts = kNumParticles;



    // --------------------------------------------------------
    // 1) Create Alembic archive (Ogawa)
    // --------------------------------------------------------
    OArchive archive(Alembic::AbcCoreOgawa::WriteArchive(), out_path.string());
    OObject  top_obj = archive.getTop();

    // --------------------------------------------------------
    // 2) Create OPoints schema with time sampling
    // --------------------------------------------------------
    // dt = 1/fps, start time = 0
    TimeSamplingPtr ts(new TimeSampling(1.0 / double(fps), 0.0));
    const uint32_t ts_index = archive.addTimeSampling(*ts);

    OPoints points_obj(top_obj, "particles");
    OPointsSchema& schema = points_obj.getSchema();
    schema.setTimeSampling(ts_index);

    // Optional: set some stable per-particle IDs
    std::vector<uint64_t> ids((size_t)npts);
    for (int i = 0; i < npts; ++i) ids[(size_t)i] = (uint64_t)i;

    // Initialize positions (in some small box)
    std::vector<V3f> pos((size_t)npts);
    for (int i = 0; i < npts; ++i) {
        // deterministic-ish random without <random>
        const float fx = float((i * 16807u) % 10000u) / 10000.0f;
        const float fy = float((i * 48271u) % 10000u) / 10000.0f;
        const float fz = float((i * 69621u) % 10000u) / 10000.0f;
        pos[(size_t)i] = V3f(
            0.45f + 0.10f * (fx - 0.5f),
            0.50f + 0.10f * (fy - 0.5f),
            0.18f + 0.10f * (fz - 0.5f)
        );
    }

    // --------------------------------------------------------
    // 3) Write frames
    //    (Replace this motion with your RK4 advection later.)
    // --------------------------------------------------------
    for (int f = 0; f < frames; ++f) {
        const float t = float(f) / float(fps);

        // Simple swirling motion around center (0.5, 0.5, 0.2)
        const V3f center(0.5f, 0.5f, 0.2f);
        const float ang = 1.0f * t;          // radians/sec
        const float cs = std::cos(ang);
        const float sn = std::sin(ang);

        for (int i = 0; i < npts; ++i) {
            V3f p = pos[(size_t)i] - center;

            // rotate in XY plane
            V3f pr;
            pr.x = cs * p.x - sn * p.y;
            pr.y = sn * p.x + cs * p.y;
            pr.z = p.z;

            // add a tiny vertical wobble
            pr.z += 0.01f * std::sin(2.0f * t + 0.001f * float(i));

            pos[(size_t)i] = pr + center;
        }

        // Fill Alembic sample
        Alembic::AbcGeom::V3fArraySample positions_sample(pos.data(), (size_t)npts);
        Alembic::AbcGeom::UInt64ArraySample id_sample(ids.data(), (size_t)npts);

        Alembic::AbcGeom::OPointsSchema::Sample sample;
        sample.setPositions(positions_sample);
        sample.setIds(id_sample);

        schema.set(sample);

        if ((f % std::max(1, frames / 10)) == 0) {
            fmt::print("  frame {}/{}\n", f, frames - 1);
        }
    }

    fmt::print("Done. Wrote: {}\n", out_path.string());
    return 0;
}
