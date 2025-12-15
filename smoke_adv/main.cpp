// smoke_adv/main.cpp
#include <Alembic/AbcCoreOgawa/All.h>
#include <Alembic/AbcGeom/All.h>

#include <fmt/core.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#ifdef _WIN32
#  include <windows.h>
#endif

using namespace Alembic::Abc;
using namespace Alembic::AbcGeom;

// ------------------------------------------------------------
// Tiny arg parser (keeps deps minimal)
// ------------------------------------------------------------
static const char* get_arg(int argc, char** argv, const char* key, const char* defval = nullptr)
{
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == key) return argv[i + 1];
    }
    return defval;
}

static int get_arg_int(int argc, char** argv, const char* key, int defval)
{
    const char* v = get_arg(argc, argv, key, nullptr);
    return v ? std::atoi(v) : defval;
}

static double get_arg_double(int argc, char** argv, const char* key, double defval)
{
    const char* v = get_arg(argc, argv, key, nullptr);
    return v ? std::atof(v) : defval;
}

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------
int main(int argc, char** argv)
{

    const char* out_path = get_arg(argc, argv, "--out", "smoke_particles.abc");
    const int   frames = std::max(1, get_arg_int(argc, argv, "--frames", 120));
    const int   fps = std::max(1, get_arg_int(argc, argv, "--fps", 24));
    const int   npts = std::max(1, get_arg_int(argc, argv, "--n", 200000));

    fmt::print("Writing Alembic:\n  out    = {}\n  frames = {}\n  fps    = {}\n  npts   = {}\n",
        out_path, frames, fps, npts);

    // --------------------------------------------------------
    // 1) Create Alembic archive (Ogawa)
    // --------------------------------------------------------
    OArchive archive(Alembic::AbcCoreOgawa::WriteArchive(), out_path);
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

    fmt::print("Done. Wrote: {}\n", out_path);
    return 0;
}
