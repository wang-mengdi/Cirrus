import bpy
import os
from mathutils import Vector

# ============================================================
# User settings
# ============================================================
ABC_PATH = r"C:\Code\Cirrus\output\smokesphere\smoke_particles.abc"  # TODO
abc_dir = os.path.dirname(ABC_PATH)
OUTPUT_DIR = os.path.join(abc_dir, "render")
RENDER_RES = (1920, 1080)

# Domain & sim resolution (given by you)
DOMAIN_MIN = Vector((0.0, 0.0, 0.0))
DOMAIN_MAX = Vector((1.0, 1.0, 1.0))
SIM_RES    = 512
DX         = 1.0 / SIM_RES

# ============================================================
# Defaults tuned for [0,1]^3 and <=200k points
# ============================================================
VOXEL_SIZE    = 1.5 * DX      # ~0.00293
VOLUME_RADIUS = 5.0 * DX      # ~0.00977

DENSITY_SCALE  = 2.0
ANISOTROPY     = 0.35
NOISE_SCALE    = 6.0
NOISE_STRENGTH = 0.60

CYCLES_SAMPLES    = 256
VOLUME_STEPS_RATE = 0.5
VOLUME_MAX_STEPS  = 1024
USE_GPU           = True

# ============================================================
# Helpers
# ============================================================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def set_cycles_gpu():
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    prefs = bpy.context.preferences
    cprefs = prefs.addons['cycles'].preferences
    for t in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
        try:
            cprefs.compute_device_type = t
            break
        except:
            pass
    for d in cprefs.devices:
        d.use = True
    scene.cycles.device = 'GPU'

def look_at(obj, target=Vector((0,0,0))):
    direction = target - obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    obj.rotation_euler = rot_quat.to_euler()

def import_alembic(filepath: str):
    """
    Blender 4.5+: Ask importer to set scene frame range (no CacheFile.frame_start).
    """
    scene = bpy.context.scene
    old_start, old_end = scene.frame_start, scene.frame_end

    # IMPORTANT: set_frame_range=True
    bpy.ops.wm.alembic_import(filepath=filepath, set_frame_range=True)

    print(f"[Alembic] scene frame range: {scene.frame_start} -> {scene.frame_end} (was {old_start}->{old_end})")

def create_black_smoke_volume_material(name="M_BlackSmoke"):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (820, 0)

    pv = nodes.new("ShaderNodeVolumePrincipled")
    pv.location = (520, 0)
    pv.inputs["Color"].default_value = (0.02, 0.02, 0.02, 1.0)
    pv.inputs["Absorption Color"].default_value = (0.02, 0.02, 0.02, 1.0)
    pv.inputs["Anisotropy"].default_value = ANISOTROPY

    # --- Noise chain (3D) ---
    texcoord = nodes.new("ShaderNodeTexCoord")
    texcoord.location = (-780, 0)

    mapping = nodes.new("ShaderNodeMapping")
    mapping.location = (-600, 0)
    mapping.inputs["Scale"].default_value = (NOISE_SCALE, NOISE_SCALE, NOISE_SCALE)

    noise = nodes.new("ShaderNodeTexNoise")
    noise.location = (-400, 0)
    noise.inputs["Detail"].default_value = 4.0
    noise.inputs["Roughness"].default_value = 0.6

    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.location = (-200, 0)
    ramp.color_ramp.elements[0].position = 0.35
    ramp.color_ramp.elements[1].position = 0.80

    # Convert ramp output to a scalar in [0,1]
    # Blender 4.5 ColorRamp outputs "Color" (and sometimes "Alpha"), not "Fac".
    sep = nodes.new("ShaderNodeSeparateRGB")
    sep.location = (0, 0)

    mult = nodes.new("ShaderNodeMath")
    mult.location = (200, -140)
    mult.operation = 'MULTIPLY'
    mult.inputs[1].default_value = NOISE_STRENGTH

    add = nodes.new("ShaderNodeMath")
    add.location = (380, -140)
    add.operation = 'ADD'
    add.inputs[1].default_value = 1.0 - (NOISE_STRENGTH * 0.5)

    dens_mult = nodes.new("ShaderNodeMath")
    dens_mult.location = (380, 0)
    dens_mult.operation = 'MULTIPLY'
    dens_mult.inputs[1].default_value = DENSITY_SCALE

    # Wiring
    links.new(texcoord.outputs["Object"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])

    # ramp -> scalar
    links.new(ramp.outputs["Color"], sep.inputs["Image"])
    links.new(sep.outputs["R"], mult.inputs[0])

    links.new(mult.outputs["Value"], add.inputs[0])
    links.new(add.outputs["Value"], dens_mult.inputs[0])
    links.new(dens_mult.outputs["Value"], pv.inputs["Density"])

    links.new(pv.outputs["Volume"], out.inputs["Volume"])
    return mat


def _find_socket_by_name_like(sockets, keywords):
    """Return first socket whose name contains any keyword (case-insensitive)."""
    kws = [k.lower() for k in keywords]
    for s in sockets:
        name = (s.name or "").lower()
        if any(k in name for k in kws):
            return s
    return None

def _float_input_sockets(node):
    """All float-like input sockets (exclude geometry)."""
    floats = []
    for s in node.inputs:
        # In practice float sockets show up as NodeSocketFloat
        if s.type == 'VALUE':
            floats.append(s)
    return floats

def create_points_to_volume_gn(name="GN_PointsToVolume", volume_mat=None):
    ng = bpy.data.node_groups.new(name, 'GeometryNodeTree')
    nodes = ng.nodes
    links = ng.links

    # Blender 4.5+: use ng.interface
    ng.interface.new_socket(name="Geometry", in_out='INPUT',  socket_type='NodeSocketGeometry')
    ng.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')

    inp = nodes.new("NodeGroupInput")
    inp.location = (-600, 0)
    out = nodes.new("NodeGroupOutput")
    out.location = (600, 0)

    p2v = nodes.new("GeometryNodePointsToVolume")
    p2v.location = (-150, 0)

    # Connect geometry -> points input (this name is stable: "Points" is almost always present)
    # If "Points" is missing in some build, fall back to the first geometry input.
    pts_in = _find_socket_by_name_like(p2v.inputs, ["points"])
    if pts_in is None:
        # fallback: first geometry input
        for s in p2v.inputs:
            if s.type == 'GEOMETRY':
                pts_in = s
                break
    if pts_in is None:
        raise RuntimeError("Points to Volume node: cannot find a geometry/points input socket.")

    links.new(inp.outputs["Geometry"], pts_in)

    # ---- Robustly set voxel size / radius / density ----
    # 1) Try name-like matches
    voxel_sock = _find_socket_by_name_like(p2v.inputs, ["voxel"])
    radius_sock = _find_socket_by_name_like(p2v.inputs, ["radius"])
    density_sock = _find_socket_by_name_like(p2v.inputs, ["density"])

    # 2) Fallback heuristics if names are missing/renamed
    floats = _float_input_sockets(p2v)

    # Density: if not found, choose a float socket that looks like density:
    # Usually there's exactly one density-ish socket; safest fallback: the one whose default is 1.0.
    if density_sock is None:
        cand = None
        for s in floats:
            try:
                if abs(float(s.default_value) - 1.0) < 1e-6:
                    cand = s
                    break
            except Exception:
                pass
        density_sock = cand

    # If voxel/radius are missing, pick from remaining float sockets:
    if voxel_sock is None or radius_sock is None:
        remaining = [s for s in floats if s is not density_sock]
        # If we have 2 float sockets left, assume smaller one = voxel, larger one = radius
        if len(remaining) >= 2:
            # sort by current default_value if possible; otherwise keep order
            def _safe_default(s):
                try:
                    return float(s.default_value)
                except Exception:
                    return 0.0
            remaining_sorted = sorted(remaining, key=_safe_default)
            # The problem: defaults may be identical; we still want voxel < radius
            # We'll just assign by intended values if possible.
            # Pick two sockets:
            a, b = remaining_sorted[0], remaining_sorted[1]
            # Decide which should be voxel based on which assignment preserves voxel < radius
            # (our intended values already satisfy VOXEL_SIZE < VOLUME_RADIUS)
            if voxel_sock is None:
                voxel_sock = a
            if radius_sock is None:
                radius_sock = b

    # Final safety: if still missing, just set by index among float sockets
    if density_sock is None and len(floats) >= 1:
        density_sock = floats[0]
    if voxel_sock is None and len(floats) >= 2:
        voxel_sock = floats[1]
    if radius_sock is None and len(floats) >= 3:
        radius_sock = floats[2]

    # Apply values (guarded)
    if density_sock is not None:
        try:
            density_sock.default_value = 1.0
        except Exception:
            pass

    if voxel_sock is not None:
        try:
            voxel_sock.default_value = VOXEL_SIZE
        except Exception:
            pass

    if radius_sock is not None:
        try:
            radius_sock.default_value = VOLUME_RADIUS
        except Exception:
            pass

    # Set Material
    setmat = nodes.new("GeometryNodeSetMaterial")
    setmat.location = (250, 0)
    if volume_mat is not None:
        setmat.inputs["Material"].default_value = volume_mat

    # Connect volume output to material and group output
    # Output socket name can vary too; try common ones then fallback to first geometry output.
    vol_out = _find_socket_by_name_like(p2v.outputs, ["volume"])
    if vol_out is None:
        # fallback: first geometry output
        for s in p2v.outputs:
            if s.type == 'GEOMETRY':
                vol_out = s
                break
    if vol_out is None:
        raise RuntimeError("Points to Volume node: cannot find a volume/geometry output socket.")

    links.new(vol_out, setmat.inputs["Geometry"])
    links.new(setmat.outputs["Geometry"], out.inputs["Geometry"])

    # Helpful debug print (won't hurt in background mode)
    try:
        print("[GN] PointsToVolume inputs:", [s.name for s in p2v.inputs])
        print("[GN] Chosen sockets:",
              "voxel=", getattr(voxel_sock, "name", None),
              "radius=", getattr(radius_sock, "name", None),
              "density=", getattr(density_sock, "name", None))
    except Exception:
        pass

    return ng


def setup_camera_and_lights():
    scene = bpy.context.scene
    center = (DOMAIN_MIN + DOMAIN_MAX) * 0.5

    # Camera
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    scene.camera = cam
    cam.location = Vector((1.7, -1.4, 1.2))
    look_at(cam, center)
    cam_data.lens = 50

    # Key backlight
    key_data = bpy.data.lights.new(name="KeyBack", type='AREA')
    key = bpy.data.objects.new(name="KeyBack", object_data=key_data)
    bpy.context.collection.objects.link(key)
    key.location = Vector((-0.3, 0.5, 1.3))
    look_at(key, center)
    key_data.energy = 2000
    key_data.size = 1.2

    # Fill
    fill_data = bpy.data.lights.new(name="Fill", type='AREA')
    fill = bpy.data.objects.new(name="Fill", object_data=fill_data)
    bpy.context.collection.objects.link(fill)
    fill.location = Vector((1.3, 0.2, 0.8))
    look_at(fill, center)
    fill_data.energy = 250
    fill_data.size = 1.0

def pick_imported_point_object(objs):
    for obj in reversed(objs):
        if obj.type == "POINTCLOUD":
            return obj
    for obj in reversed(objs):
        if obj.type in {"MESH", "CURVE"}:
            return obj
    return None

# ============================================================
# Main
# ============================================================
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

clear_scene()

ensure_dir(OUTPUT_DIR)

# Import Alembic (sets scene frame range)
import_alembic(ABC_PATH)

# Find imported object
imported = list(bpy.context.selected_objects) or list(bpy.data.objects)
pts_obj = pick_imported_point_object(imported)
if pts_obj is None:
    raise RuntimeError("Could not find imported point object. Please inspect imported objects and select manually.")

# Material & GN
mat = create_black_smoke_volume_material()
gn = create_points_to_volume_gn(volume_mat=mat)

mod = pts_obj.modifiers.new(name="PointsToVolume", type='NODES')
mod.node_group = gn

# Render settings
scene = bpy.context.scene
scene.render.resolution_x = RENDER_RES[0]
scene.render.resolution_y = RENDER_RES[1]
scene.render.engine = 'CYCLES'

if USE_GPU:
    try:
        set_cycles_gpu()
    except Exception as e:
        print("GPU setup failed, falling back to CPU:", e)

scene.cycles.samples = CYCLES_SAMPLES
scene.cycles.use_denoising = True
scene.cycles.volume_step_rate = VOLUME_STEPS_RATE
scene.cycles.volume_max_steps = VOLUME_MAX_STEPS

scene.view_settings.view_transform = 'Filmic'
scene.view_settings.look = 'Medium High Contrast'
scene.view_settings.exposure = 0.0

scene.render.film_transparent = True
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = os.path.join(OUTPUT_DIR, "smoke_")

setup_camera_and_lights()

print("Done. Render animation with:")
print("  bpy.ops.render.render(animation=True)")
print(f"Params: VOXEL_SIZE={VOXEL_SIZE:.6f}, RADIUS={VOLUME_RADIUS:.6f}, DENSITY_SCALE={DENSITY_SCALE}")
print(f"Scene frames: {scene.frame_start} -> {scene.frame_end}")

# --- Actually render ---
scene = bpy.context.scene

# Make sure output path ends with a separator + base name
# e.g. C:\...\render_out\smoke_
scene.render.filepath = os.path.join(OUTPUT_DIR, "smoke_")

# # Render animation (writes frames to disk)
# bpy.ops.render.render(animation=True)

# Render last frame for debug
scene.frame_set(scene.frame_end)
bpy.ops.render.render(write_still=True)

print("Render finished.")
