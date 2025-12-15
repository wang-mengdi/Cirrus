import os, json, argparse
import numpy as np
import sys
import bpy
import time

# ============================================================
# Global constants (edit here)
# ============================================================
# Frame file naming inside input_dir:
# e.g. input_dir/frame0000.json + input_dir/frame0000.bin
FRAME_JSON_PATTERN = "frame{frame:04d}.json"

# Channel name in json
VELOCITY_CHANNEL_NAME = "velocity"

# Particle settings
NUM_PARTICLES_MAX = 200000
RNG_SEED = 1

# Emitter (world space)
SOURCE_CENTER = np.array([0.5, 0.5, 0.18], dtype=np.float32)
SOURCE_BOX    = np.array([0.01, 0.01, 0.01], dtype=np.float32)
EMIT_PER_FRAME = 2000  # 0 to disable
#EMIT_PER_FRAME = 1  # 0 to disable

# Advection settings
SUBSTEPS = 10
# dt is derived from fps: dt = 1 / fps

# Output settings (if you want a fixed path, set OUT_ABC_PATH explicitly)
# If None: use config["driver"]["output_base_dir"]/particles.abc under input_dir
OUT_ABC_PATH = None

# ============================================================
# I/O: read json+bin channels
# ============================================================
def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def frame_json_path(input_dir: str, frame: int) -> str:
    return os.path.join(input_dir, FRAME_JSON_PATTERN.format(frame=frame))

def load_channel_as_numpy(meta: dict, json_path: str, channel_name: str):
    # Find channel
    ch = None
    for c in meta["channels"]:
        if c["name"] == channel_name:
            ch = c
            break
    if ch is None:
        raise KeyError(f"Channel '{channel_name}' not found in {json_path}")

    layout = meta["layout"]
    if layout["dtype"] != "float32":
        raise ValueError(f"Only float32 supported, got {layout['dtype']}")
    if layout["components_order"] != "AoS":
        raise ValueError(f"Only AoS supported, got {layout['components_order']}")
    if layout["index_order"] != "z_fastest":
        raise ValueError(f"Only z_fastest supported, got {layout['index_order']}")

    grid = meta["grid"]
    nx, ny, nz = grid["dimensions"]
    comps = int(ch["components"])
    dtype = np.float32

    bin_rel = meta["binary_file"]
    bin_path = os.path.join(os.path.dirname(json_path), bin_rel)

    offset = int(ch["offset_bytes"])
    n_floats = int(ch["bytes"]) // np.dtype(dtype).itemsize
    arr1d = np.memmap(bin_path, dtype=dtype, mode="r", offset=offset, shape=(n_floats,))

    # z_fastest => linear ((x*ny + y)*nz + z); AoS => last dim is comps contiguous
    if comps == 1:
        out = np.asarray(arr1d).reshape((nx, ny, nz), order="C")
    else:
        out = np.asarray(arr1d).reshape((nx, ny, nz, comps), order="C")

    origin = np.array(grid["origin"], dtype=np.float32)
    spacing = np.array(grid["spacing"], dtype=np.float32)
    return out, origin, spacing

def sample_velocity_trilerp_two(v0, v1, origin, spacing, p, alpha):
    nx, ny, nz, _ = v0.shape
    g = (p - origin[None, :]) / spacing[None, :]
    gx, gy, gz = g[:, 0], g[:, 1], g[:, 2]

    x0 = np.clip(np.floor(gx).astype(np.int32), 0, nx - 2)
    y0 = np.clip(np.floor(gy).astype(np.int32), 0, ny - 2)
    z0 = np.clip(np.floor(gz).astype(np.int32), 0, nz - 2)

    tx = (gx - x0).astype(np.float32)[:, None]
    ty = (gy - y0).astype(np.float32)[:, None]
    tz = (gz - z0).astype(np.float32)[:, None]

    x1 = x0 + 1; y1 = y0 + 1; z1 = z0 + 1

    # v0 corners
    c0000 = v0[x0, y0, z0]; c1000 = v0[x1, y0, z0]
    c0100 = v0[x0, y1, z0]; c1100 = v0[x1, y1, z0]
    c0010 = v0[x0, y0, z1]; c1010 = v0[x1, y0, z1]
    c0110 = v0[x0, y1, z1]; c1110 = v0[x1, y1, z1]

    # v1 corners
    c0001 = v1[x0, y0, z0]; c1001 = v1[x1, y0, z0]
    c0101 = v1[x0, y1, z0]; c1101 = v1[x1, y1, z0]
    c0011 = v1[x0, y0, z1]; c1011 = v1[x1, y0, z1]
    c0111 = v1[x0, y1, z1]; c1111 = v1[x1, y1, z1]

    def trilerp(c000, c100, c010, c110, c001, c101, c011, c111):
        c00 = c000 * (1 - tx) + c100 * tx
        c10 = c010 * (1 - tx) + c110 * tx
        c01 = c001 * (1 - tx) + c101 * tx
        c11 = c011 * (1 - tx) + c111 * tx
        c0 = c00 * (1 - ty) + c10 * ty
        c1 = c01 * (1 - ty) + c11 * ty
        return c0 * (1 - tz) + c1 * tz

    v0i = trilerp(c0000,c1000,c0100,c1100,c0010,c1010,c0110,c1110)
    v1i = trilerp(c0001,c1001,c0101,c1101,c0011,c1011,c0111,c1111)

    a = np.float32(alpha)
    return v0i * (1.0 - a) + v1i * a


# ============================================================
# RK4 advection with time interpolation between frames
# ============================================================
def rk4_step(p, dt, v0, v1, origin, spacing, alpha0, alpha1):
    amid = 0.5 * (alpha0 + alpha1)
    k1 = sample_velocity_trilerp_two(v0, v1, origin, spacing, p, alpha0)
    k2 = sample_velocity_trilerp_two(v0, v1, origin, spacing, p + 0.5*dt*k1, amid)
    k3 = sample_velocity_trilerp_two(v0, v1, origin, spacing, p + 0.5*dt*k2, amid)
    k4 = sample_velocity_trilerp_two(v0, v1, origin, spacing, p + 1.0*dt*k3, alpha1)
    return p + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# ============================================================
# Blender helpers: point mesh + shapekeys
# ============================================================
def make_point_mesh_object(name: str, max_points: int):
    mesh = bpy.data.meshes.new(name + "_mesh")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    verts = [(0.0, 0.0, 0.0)] * max_points
    mesh.from_pydata(verts, [], [])
    mesh.update()

    obj.shape_key_add(name="Basis", from_mix=False)
    return obj

def set_shape_key_positions(obj, key_name: str, positions: np.ndarray):
    sk = obj.shape_key_add(name=key_name, from_mix=False)
    kb = sk.data
    for i in range(len(kb)):
        x, y, z = positions[i]
        kb[i].co = (float(x), float(y), float(z))

    sk.value = 1.0
    sk.keyframe_insert(data_path="value")

    for other in obj.data.shape_keys.key_blocks:
        if other.name != "Basis" and other.name != key_name:
            other.value = 0.0
            other.keyframe_insert(data_path="value")

def get_blender_args():
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1:]

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)

    args = parser.parse_args(get_blender_args())

    input_dir = args.input_dir
    cfg_path = os.path.join(input_dir, "config.json")
    cfg = read_json(cfg_path)

    driver = cfg["driver"]
    first_frame = int(driver["first_frame"])
    last_frame  = int(driver["last_frame"])
    fps         = float(driver["fps"])
    dt_frame    = 1.0 / fps

    # Decide output path
    if OUT_ABC_PATH is not None:
        out_abc = OUT_ABC_PATH
    else:
        # output_base_dir is a path like "./output/smokesphere"
        # We'll interpret it relative to input_dir (simple & predictable)
        out_base = str(driver.get("output_base_dir", "./output"))
        out_dir = os.path.normpath(os.path.join(input_dir, out_base))
        os.makedirs(out_dir, exist_ok=True)
        out_abc = os.path.join(out_dir, "particles.abc")

    print(f"[config] frames: {first_frame} .. {last_frame}, fps={fps}, dt={dt_frame}")
    print(f"[out] {out_abc}")

    rng = np.random.default_rng(RNG_SEED)

    # Load first frame velocity (also gives origin/spacing)
    meta0 = read_json(frame_json_path(input_dir, first_frame))
    v0, origin, spacing = load_channel_as_numpy(meta0, frame_json_path(input_dir, first_frame), VELOCITY_CHANNEL_NAME)

    # Particle buffer
    maxP = int(NUM_PARTICLES_MAX)
    P = np.zeros((maxP, 3), dtype=np.float32)
    alive = 0

    def emit(n):
        nonlocal alive
        if n <= 0 or alive >= maxP:
            return
        n = min(n, maxP - alive)
        start = SOURCE_CENTER - 0.5 * SOURCE_BOX
        newp = rng.random((n, 3), dtype=np.float32) * SOURCE_BOX + start
        P[alive:alive+n] = newp
        alive += n

    # Blender scene
    scene = bpy.context.scene
    scene.frame_start = first_frame
    scene.frame_end = last_frame
    # Keep Blender fps consistent (optional, but nice)
    scene.render.fps = int(round(fps))

    obj = make_point_mesh_object("particles", maxP)

    # Initial emit
    emit(EMIT_PER_FRAME)

    for i in range(first_frame, last_frame + 1):
        start_time = time.time()
        print(f"=== Frame {i} ===")
        scene.frame_set(i)

        meta_i = read_json(frame_json_path(input_dir, i))
        v_i, origin_i, spacing_i = load_channel_as_numpy(meta_i, frame_json_path(input_dir, i), VELOCITY_CHANNEL_NAME)

        if i < last_frame:
            meta_ip1 = read_json(frame_json_path(input_dir, i + 1))
            v_ip1, origin_ip1, spacing_ip1 = load_channel_as_numpy(meta_ip1, frame_json_path(input_dir, i + 1), VELOCITY_CHANNEL_NAME)
        else:
            v_ip1 = v_i

        # Update origin/spacing if needed
        origin = origin_i
        spacing = spacing_i

        # Emit new particles each frame
        emit(EMIT_PER_FRAME)

        # Advect from i -> i+1 using substeps within the frame
        if alive > 0 and i < last_frame:
            dt_sub = dt_frame / float(SUBSTEPS)
            for s in range(SUBSTEPS):
                a0 = (s / SUBSTEPS)
                a1 = ((s + 1) / SUBSTEPS)
                P[:alive] = rk4_step(P[:alive], dt_sub, v_i, v_ip1, origin, spacing, a0, a1).astype(np.float32)

        # Write positions to a shapekey
        key_name = f"f{i:04d}"
        set_shape_key_positions(obj, key_name, P)

        print(f"[frame {i}] alive={alive}")
        end_time = time.time()
        print(f"  time: {end_time - start_time:.3f} sec")

    bpy.ops.wm.alembic_export(
        filepath=out_abc,
        selected=False,
        start=first_frame,
        end=last_frame,
        xsamples=1,
        gsamples=1
    )
    print("Exported:", out_abc)

if __name__ == "__main__":
    main()
