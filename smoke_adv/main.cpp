// smoke_adv/main.cpp
// ------------------------------------------------------------
// Simple particle advection tool for smoke visualization.
// - Reads velocity fields from json + binary files
// - Spawns passive particles every frame
// - Advects all particles using RK4 with temporal interpolation
// - Writes particle positions to Alembic (.abc)
// ------------------------------------------------------------

#include <Alembic/AbcCoreOgawa/All.h>
#include <Alembic/AbcGeom/All.h>

#include <fmt/core.h>
#include <nlohmann/json.hpp>

#include <tbb/parallel_for_each.h>
#include <tbb/parallel_for.h>
#include <chrono>


#include <algorithm>
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

using namespace Alembic::Abc;
using namespace Alembic::AbcGeom;

using Clock = std::chrono::high_resolution_clock;


// ------------------------------------------------------------
// Minimal command-line argument helper
// ------------------------------------------------------------
static const char* get_arg(int argc, char** argv,
    const char* key,
    const char* defval = nullptr)
{
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == key)
            return argv[i + 1];
    }
    return defval;
}

// ------------------------------------------------------------
// Driver configuration (loaded from input_dir/config.json)
// ------------------------------------------------------------
struct DriverConfig {
    int first_frame = 0;
    int last_frame = 0;
    int fps = 24;
    fs::path output_base_dir;
};

// Load simulation driver parameters from config.json.
// Only a small subset is required for particle advection.
static DriverConfig load_driver_config(const fs::path& input_dir)
{
    fs::path cfg_path = input_dir / "config.json";
    std::ifstream fin(cfg_path);
    if (!fin) {
        throw std::runtime_error("Cannot open config.json: " +
            cfg_path.string());
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
    if (cfg.last_frame < cfg.first_frame)
        std::swap(cfg.first_frame, cfg.last_frame);

    return cfg;
}

// ------------------------------------------------------------
// Lightweight deterministic RNG (LCG)
// ------------------------------------------------------------
// Used only for particle spawning. Deterministic across runs.
struct LCG {
    uint32_t state = 1u;
    explicit LCG(uint32_t seed = 1u) : state(seed) {}

    uint32_t next_u32() {
        state = 1664525u * state + 1013904223u;
        return state;
    }

    // Returns a float in [0,1)
    float next_f01() {
        return (next_u32() >> 8) * (1.0f / 16777216.0f);
    }
};

// ------------------------------------------------------------
// Velocity field loading (json + binary)
// Layout assumptions:
//   - float32
//   - AoS (vx, vy, vz)
//   - z_fastest indexing
// ------------------------------------------------------------
struct GridMeta {
    int   nx = 0, ny = 0, nz = 0;
    float origin[3] = { 0, 0, 0 };
    float spacing[3] = { 1, 1, 1 };
};

struct VelocityField {
    GridMeta meta;
    std::vector<float> v; // nx*ny*nz*3 floats
    bool initialized = false;

    bool valid() const {
        return meta.nx > 1 && meta.ny > 1 && meta.nz > 1 &&
            (int)v.size() == meta.nx * meta.ny * meta.nz * 3;
    }

    // Reusable loader: overwrite internal buffer without resizing after the first time.
    void load_from_json(const fs::path& frame_json_path) {
        std::ifstream fin(frame_json_path);
        if (!fin) {
            throw std::runtime_error("Cannot open frame json: " + frame_json_path.string());
        }

        json j;
        fin >> j;

        // Grid metadata
        GridMeta new_meta;
        auto& g = j.at("grid");
        auto dims = g.at("dimensions");
        new_meta.nx = dims.at(0).get<int>();
        new_meta.ny = dims.at(1).get<int>();
        new_meta.nz = dims.at(2).get<int>();

        auto org = g.at("origin");
        for (int i = 0; i < 3; ++i) new_meta.origin[i] = org.at(i).get<float>();

        auto sp = g.at("spacing");
        for (int i = 0; i < 3; ++i) new_meta.spacing[i] = sp.at(i).get<float>();

        // Validate layout assumptions
        auto& layout = j.at("layout");
        if (layout.value("dtype", "") != "float32" ||
            layout.value("components_order", "") != "AoS" ||
            layout.value("index_order", "") != "z_fastest")
        {
            throw std::runtime_error("Unsupported layout in " + frame_json_path.string());
        }

        // Locate velocity channel
        uint64_t offset_bytes = 0;
        uint64_t bytes = 0;
        bool found = false;

        for (auto& ch : j.at("channels")) {
            if (ch.value("name", "") == "velocity") {
                offset_bytes = ch.at("offset_bytes").get<uint64_t>();
                bytes = ch.at("bytes").get<uint64_t>();
                if (ch.at("components").get<int>() != 3) {
                    throw std::runtime_error("Velocity must have 3 components");
                }
                found = true;
                break;
            }
        }
        if (!found) throw std::runtime_error("Velocity channel not found");

        const uint64_t expected_bytes =
            uint64_t(new_meta.nx) * new_meta.ny * new_meta.nz * 3ull * 4ull;
        if (bytes != expected_bytes) {
            throw std::runtime_error("Velocity byte size mismatch");
        }

        const size_t expected_floats = size_t(expected_bytes / 4ull);

        // First time: adopt meta and allocate once
        if (!initialized) {
            meta = new_meta;
            v.resize(expected_floats);
            initialized = true;
        }
        else {
            // Subsequent: ensure consistent (you said all frames match)
            if (new_meta.nx != meta.nx || new_meta.ny != meta.ny || new_meta.nz != meta.nz ||
                new_meta.origin[0] != meta.origin[0] || new_meta.origin[1] != meta.origin[1] || new_meta.origin[2] != meta.origin[2] ||
                new_meta.spacing[0] != meta.spacing[0] || new_meta.spacing[1] != meta.spacing[1] || new_meta.spacing[2] != meta.spacing[2])
            {
                throw std::runtime_error("Grid meta changed across frames: " + frame_json_path.string());
            }
            // v.size() should already match; no resize here
            if (v.size() != expected_floats) {
                throw std::runtime_error("Internal buffer size mismatch");
            }
        }

        // Read binary payload directly into existing buffer
        fs::path bin_path = frame_json_path.parent_path() / j.at("binary_file").get<std::string>();
        std::ifstream fb(bin_path, std::ios::binary);
        if (!fb) throw std::runtime_error("Cannot open binary file: " + bin_path.string());

        fb.seekg((std::streamoff)offset_bytes, std::ios::beg);
        fb.read(reinterpret_cast<char*>(v.data()), (std::streamsize)expected_bytes);

        if (!valid()) throw std::runtime_error("Invalid velocity field after read");
    }
};


// ------------------------------------------------------------
// Trilinear interpolation utilities
// ------------------------------------------------------------

// AoS indexing with z-fastest layout
static inline size_t v_index(const GridMeta& m,
    int x, int y, int z, int c)
{
    return ((((size_t)x * m.ny + y) * m.nz + z) * 3u + c);
}

// Trilinear sampling of velocity field in world space
static inline V3f sample_velocity_trilerp(const VelocityField& vf,
    const V3f& p)
{
    const GridMeta& m = vf.meta;

    float gx = (p.x - m.origin[0]) / m.spacing[0];
    float gy = (p.y - m.origin[1]) / m.spacing[1];
    float gz = (p.z - m.origin[2]) / m.spacing[2];

    gx = std::clamp(gx, 0.0f, float(m.nx - 1) - 1e-6f);
    gy = std::clamp(gy, 0.0f, float(m.ny - 1) - 1e-6f);
    gz = std::clamp(gz, 0.0f, float(m.nz - 1) - 1e-6f);

    int x0 = std::min(int(std::floor(gx)), m.nx - 2);
    int y0 = std::min(int(std::floor(gy)), m.ny - 2);
    int z0 = std::min(int(std::floor(gz)), m.nz - 2);

    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;

    float tx = gx - x0;
    float ty = gy - y0;
    float tz = gz - z0;

    auto getV = [&](int xi, int yi, int zi) {
        size_t i0 = v_index(m, xi, yi, zi, 0);
        return V3f(vf.v[i0], vf.v[i0 + 1], vf.v[i0 + 2]);
        };

    auto lerp = [](const V3f& a, const V3f& b, float t) {
        return a * (1.0f - t) + b * t;
        };

    V3f c00 = lerp(getV(x0, y0, z0), getV(x1, y0, z0), tx);
    V3f c10 = lerp(getV(x0, y1, z0), getV(x1, y1, z0), tx);
    V3f c01 = lerp(getV(x0, y0, z1), getV(x1, y0, z1), tx);
    V3f c11 = lerp(getV(x0, y1, z1), getV(x1, y1, z1), tx);

    V3f c0 = lerp(c00, c10, ty);
    V3f c1 = lerp(c01, c11, ty);

    return lerp(c0, c1, tz);
}

// Velocity with temporal interpolation between two frames
static inline V3f velocity_at(const VelocityField& v0,
    const VelocityField& v1,
    const V3f& p,
    float alpha)
{
    return sample_velocity_trilerp(v0, p) * (1.0f - alpha) +
        sample_velocity_trilerp(v1, p) * alpha;
}

// One RK4 integration step over dt
static inline V3f rk4_step(const VelocityField& v0,
    const VelocityField& v1,
    const V3f& p,
    float dt)
{
    const float half_dt = 0.5f * dt;
    const float inv6_dt = dt / 6.0f;

    V3f k1 = velocity_at(v0, v1, p, 0.0f);
    V3f k2 = velocity_at(v0, v1, p + k1 * half_dt, 0.5f);
    V3f k3 = velocity_at(v0, v1, p + k2 * half_dt, 0.5f);
    V3f k4 = velocity_at(v0, v1, p + k3 * dt, 1.0f);

    return p + (k1 + k2 * 2.0f + k3 * 2.0f + k4) * inv6_dt;
}


// ------------------------------------------------------------
// Global simulation constants (tune later)
// ------------------------------------------------------------
static constexpr int   kSpawnPerFrame = 20000;
static constexpr float kLifeSeconds = 5.f;
static constexpr float kZClamp = 1.0f;

static const V3f kSourceCenter(0.5f, 0.5f, 0.18f);
static const V3f kSourceBox(0.02f, 0.02f, 0.02f);

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------
int main(int argc, char** argv)
{
    const char* input_dir_c = get_arg(argc, argv, "--input_dir", nullptr);
    if (!input_dir_c) {
        fmt::print("Usage: {} --input_dir <directory>\n", argv[0]);
        return 1;
    }

    fs::path input_dir(input_dir_c);
    DriverConfig cfg = load_driver_config(input_dir);

    const int first = cfg.first_frame;
    const int last = cfg.last_frame;
    const int fps = cfg.fps;
    const float dt = 1.0f / float(fps);

    fs::path out_path = cfg.output_base_dir / "smoke_particles.abc";

    // Create Alembic archive
    OArchive archive(Alembic::AbcCoreOgawa::WriteArchive(),
        out_path.string());
    OObject top = archive.getTop();

    TimeSamplingPtr ts(new TimeSampling(1.0 / fps, 0.0));
    uint32_t ts_idx = archive.addTimeSampling(*ts);

    OPoints points(top, "particles");
    auto& schema = points.getSchema();
    schema.setTimeSampling(ts_idx);
    OFloatGeomParam age_param(
        schema.getArbGeomParams(),
        "age",                       // attribute name
        false,                       // not indexed
        kVertexScope,                // per-point
        1                            // 1 float per point
    );
    age_param.setTimeSampling(ts_idx);

    std::vector<uint64_t> ids;
    std::vector<V3f>      positions;
    std::vector<float>    birth_time;
    std::vector<float>    age;

    uint64_t next_id = 0;
    LCG rng(42);

    // ------------------------------------------------------------
    // Preload velocity fields (sliding window)
    // ------------------------------------------------------------
    VelocityField v_curr, v_next;
    v_curr.load_from_json(input_dir / fmt::format("frame{:04d}.json", first));
    v_next.load_from_json(input_dir / fmt::format("frame{:04d}.json", std::min(first + 1, last)));

    // Main simulation loop
    for (int f = first; f <= last; ++f) {
        auto frame_begin = Clock::now();
        float t = (f - first) * dt;

        // Spawn new particles (unchanged)
        for (int i = 0; i < kSpawnPerFrame; ++i) {
            V3f jitter(
                (rng.next_f01() - 0.5f) * kSourceBox.x,
                (rng.next_f01() - 0.5f) * kSourceBox.y,
                (rng.next_f01() - 0.5f) * kSourceBox.z
            );
            positions.push_back(kSourceCenter + jitter);
            ids.push_back(next_id++);
            birth_time.push_back(t);
        }

        // Remove dead or out-of-range particles (unchanged)
        size_t w = 0;
        for (size_t i = 0; i < positions.size(); ++i) {
            if ((t - birth_time[i]) <= kLifeSeconds &&
                positions[i].z <= kZClamp)
            {
                positions[w] = positions[i];
                ids[w] = ids[i];
                birth_time[w] = birth_time[i];
                ++w;
            }
        }
        positions.resize(w);
        ids.resize(w);
        birth_time.resize(w);

        auto remove_time = Clock::now();
        std::chrono::duration<double> remove_dur = remove_time - frame_begin;
        fmt::print("Frame {:04d}: Spawned {}, {} alive after removal, time {:.3f} s\n",
            f, kSpawnPerFrame, positions.size(), remove_dur.count());

        // Advect using the preloaded fields
        tbb::parallel_for_each(
            positions.begin(),
            positions.end(),
            [&](auto& p) {
                p = rk4_step(v_curr, v_next, p, dt);
            }
        );
		age.resize(positions.size());
        tbb::parallel_for(size_t(0), positions.size(), [&](size_t i) {
			age[i] = std::clamp(t - birth_time[i], 0.0f, kLifeSeconds);
            });

        auto advect_time = Clock::now();
        std::chrono::duration<double> advect_dur = advect_time - remove_time;
        fmt::print("Frame {:04d}: Advected particles, time {:.3f} s\n",
            f, advect_dur.count());

        // ---- write points ----
        OPointsSchema::Sample psamp{
            V3fArraySample(positions),
            UInt64ArraySample(ids)
        };
        schema.set(psamp);

        // ---- write age ----
        OFloatGeomParam::Sample asamp(
            FloatArraySample(age),
            kVertexScope
        );
        age_param.set(asamp);




        auto write_time = Clock::now();
        std::chrono::duration<double> write_dur = write_time - advect_time;
        fmt::print("Frame {:04d}: Wrote Alembic sample, time {:.3f} s\n",
            f, write_dur.count());

        // ------------------------------------------------------------
        // Slide the window: load ONLY ONE new frame per iteration
        // ------------------------------------------------------------
        // After processing frame f, advance:
        //   v_curr <- v_next
        //   v_next <- frame(f+2)   (clamped to last)
        if (f < last) {
            std::swap(v_curr, v_next);
            int f2 = std::min(f + 2, last);
            v_next.load_from_json(input_dir / fmt::format("frame{:04d}.json", f2));

            auto load_time = Clock::now();
            std::chrono::duration<double> load_dur = load_time - write_time;
            fmt::print("Frame {:04d}: Loaded next velocity field (frame {:04d}), time {:.3f} s\n",
                f, f2, load_dur.count());
        }

        auto frame_end = Clock::now();
        std::chrono::duration<double> frame_dur = frame_end - frame_begin;
        fmt::print("Frame {:04d}: {} particles, time {:.3f} s, sim time {:.3f} s\n\n",
            f, positions.size(), frame_dur.count(), t);
    }


    fmt::print("Finished writing {}\n", out_path.string());
    return 0;
}
