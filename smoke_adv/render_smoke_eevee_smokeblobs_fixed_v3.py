import bpy

def _set_attr_safe(obj, name, value):
    """Set attribute if it exists; ignore otherwise (Blender API differs across versions)."""
    if hasattr(obj, name):
        try:
            setattr(obj, name, value)
        except Exception:
            pass
import os
import math
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
VOXEL_SIZE    = 0.75 * DX
VOLUME_RADIUS = 1.5 * DX

DENSITY_SCALE  = 34.0
ANISOTROPY     = 0.65
NOISE_SCALE    = 6.0
NOISE_STRENGTH = 0.35

CYCLES_SAMPLES    = 512
VOLUME_STEPS_RATE = 0.15
VOLUME_MAX_STEPS  = 1024
USE_GPU           = True

# ============================================================
# Helpers
# ============================================================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def set_eevee_gpu(scene):
    """Force Eevee (GPU realtime). Note: Eevee uses GPU by design."""
    # Blender 4.x: try Eevee Next first, fallback to legacy Eevee
    try:
        scene.render.engine = 'BLENDER_EEVEE_NEXT'
    except Exception:
        scene.render.engine = 'BLENDER_EEVEE'

    # Eevee settings (safe defaults for alpha-smoke blobs)
    ee = scene.eevee
    try:
        ee.use_soft_shadows = True
        ee.shadow_cube_size = '1024'
        ee.shadow_cascade_size = '1024'
    except Exception:
        pass

    # Color management (photographic contrast)
    scene.view_settings.view_transform = 'Filmic'
    scene.view_settings.look = 'High Contrast'
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0


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

    # print full world transform matrix for ALL objects in the scene
    print("=== [DBG] Objects in scene ===")
    for obj in sorted(scene.objects, key=lambda o: o.name):
        mw = obj.matrix_world
        print(f"[Alembic][Matrix] {obj.name}  type={obj.type}")
        print(f"  [{mw[0][0]: .6f} {mw[0][1]: .6f} {mw[0][2]: .6f} {mw[0][3]: .6f}]")
        print(f"  [{mw[1][0]: .6f} {mw[1][1]: .6f} {mw[1][2]: .6f} {mw[1][3]: .6f}]")
        print(f"  [{mw[2][0]: .6f} {mw[2][1]: .6f} {mw[2][2]: .6f} {mw[2][3]: .6f}]")
        print(f"  [{mw[3][0]: .6f} {mw[3][1]: .6f} {mw[3][2]: .6f} {mw[3][3]: .6f}]")

def rotate_all_pointclouds_x_minus_90():
    """
    Rotate all POINTCLOUD objects:
    clockwise 90 degrees around X axis (i.e. -90 deg).
    """
    angle = -math.pi * 0.5

    for obj in bpy.data.objects:
        if obj.type == 'POINTCLOUD':
            # Apply rotation in object space
            rx, ry, rz = obj.rotation_euler
            obj.rotation_euler = (rx + angle, ry, rz)

            print(f"[AxisFix] Rotated POINTCLOUD '{obj.name}' by -90° around X")

def create_black_smoke_volume_material(name="MAT_Smoke_Eevee"):
    """Eevee-friendly 'smoke blob' material: dark BSDF with noisy alpha (HASHED).
    This is not true volume; it is a fast GPU approximation using many instanced spheres.
    """
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    # Eevee transparency settings (required for noisy alpha)
    _set_attr_safe(mat, "blend_method", "BLEND")
    _set_attr_safe(mat, "shadow_method", "NONE")  # optional; Blender 4.5 may not have this
    mat.use_backface_culling = False

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (600, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (200, 40)
    bsdf.inputs["Base Color"].default_value = (0.03, 0.03, 0.03, 1.0)  # black smoke
    bsdf.inputs["Roughness"].default_value = 0.95
    # Blender 4.x renamed "Specular" to "Specular IOR Level"
    if "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = 0.0
    elif "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.0

    transp = nodes.new("ShaderNodeBsdfTransparent")
    transp.location = (200, -140)

    mix = nodes.new("ShaderNodeMixShader")
    mix.location = (420, -40)

    # Noisy alpha mask (world-space)
    texcoord = nodes.new("ShaderNodeTexCoord"); texcoord.location = (-700, -200)
    mapping  = nodes.new("ShaderNodeMapping");  mapping.location  = (-520, -200)
    noise    = nodes.new("ShaderNodeTexNoise"); noise.location    = (-320, -200)
    noise.inputs["Scale"].default_value = 10.0
    noise.inputs["Detail"].default_value = 4.0
    noise.inputs["Roughness"].default_value = 0.55

    ramp = nodes.new("ShaderNodeValToRGB"); ramp.location = (-120, -200)
    # Make a puffy mask: mostly transparent, with soft blobs
    ramp.color_ramp.elements[0].position = 0.44
    ramp.color_ramp.elements[1].position = 0.56

    # Use sphere's facing to soften edges (camera-facing fade)
    layerw = nodes.new("ShaderNodeLayerWeight"); layerw.location = (-320, -20)
    layerw.inputs["Blend"].default_value = 0.6
    invert = nodes.new("ShaderNodeInvert"); invert.location = (-120, -20)

    mul = nodes.new("ShaderNodeMath"); mul.location = (80, -110)
    mul.operation = 'MULTIPLY'
    mul.inputs[1].default_value = 1.0

    # Wiring
    links.new(texcoord.outputs["Object"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(layerw.outputs["Facing"], invert.inputs["Color"])
    links.new(ramp.outputs["Color"], mul.inputs[0])
    links.new(invert.outputs["Color"], mul.inputs[1])

    links.new(mul.outputs["Value"], mix.inputs["Fac"])
    links.new(transp.outputs["BSDF"], mix.inputs[1])
    links.new(bsdf.outputs["BSDF"], mix.inputs[2])
    links.new(mix.outputs["Shader"], out.inputs["Surface"])

    # Eevee alpha settings
    _set_attr_safe(mat, "blend_method", "BLEND")
    _set_attr_safe(mat, "shadow_method", "NONE")  # optional; Blender 4.5 may not have this
    mat.use_backface_culling = False
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

def create_points_to_volume_gn(name="GN_SmokeBlobs", volume_mat=None):
    """Replace Points->Volume with fast Eevee approximation:
    instance low-poly spheres on points, with a noisy-alpha smoke material.
    """
    ng = bpy.data.node_groups.new(name, 'GeometryNodeTree')
    nodes = ng.nodes
    links = ng.links

    ng.interface.new_socket(name="Geometry", in_out='INPUT',  socket_type='NodeSocketGeometry')
    ng.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')

    inp = nodes.new("NodeGroupInput");  inp.location = (-700, 0)
    out = nodes.new("NodeGroupOutput"); out.location = (700, 0)

    # Icosphere prototype (very cheap)
    ico = nodes.new("GeometryNodeMeshIcoSphere"); ico.location = (-520, -160)
    ico.inputs["Subdivisions"].default_value = 1
    ico.inputs["Radius"].default_value = VOLUME_RADIUS * 0.9  # puff size

    # Instance on points
    inst = nodes.new("GeometryNodeInstanceOnPoints"); inst.location = (-260, 0)
    inst.inputs["Pick Instance"].default_value = False

    # Slight random scale per point (adds grain)
    rand = nodes.new("FunctionNodeRandomValue"); rand.location = (-520, 120)
    rand.data_type = 'FLOAT'
    rand.inputs["Min"].default_value = 0.75
    rand.inputs["Max"].default_value = 1.25

    scale = nodes.new("ShaderNodeMath"); scale.location = (-300, 120)
    scale.operation = 'MULTIPLY'
    scale.inputs[1].default_value = 1.0

    # Convert random to vector scale
    comb = nodes.new("ShaderNodeCombineXYZ"); comb.location = (-90, 120)

    # Set material (on instances is fine; Eevee will shade per-instance)
    setmat = nodes.new("GeometryNodeSetMaterial"); setmat.location = (260, 0)
    if volume_mat is not None:
        setmat.inputs["Material"].default_value = volume_mat

    # Optional: realize instances for stable shading (costly). Keep off by default.
    realize = nodes.new("GeometryNodeRealizeInstances"); realize.location = (60, 0)

    # Wiring
    links.new(inp.outputs["Geometry"], inst.inputs["Points"])
    links.new(ico.outputs["Mesh"], inst.inputs["Instance"])
    links.new(rand.outputs["Value"], scale.inputs[0])
    links.new(scale.outputs["Value"], comb.inputs["X"])
    links.new(scale.outputs["Value"], comb.inputs["Y"])
    links.new(scale.outputs["Value"], comb.inputs["Z"])
    links.new(comb.outputs["Vector"], inst.inputs["Scale"])

    links.new(inst.outputs["Instances"], realize.inputs["Geometry"])
    links.new(realize.outputs["Geometry"], setmat.inputs["Geometry"])
    links.new(setmat.outputs["Geometry"], out.inputs["Geometry"])

    return ng

def setup_camera_and_lights():
    scene = bpy.context.scene

    # --- Eevee look: light background to showcase black smoke ---
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    world = scene.world
    world.use_nodes = True
    wn = world.node_tree
    bg = wn.nodes.get("Background")
    if bg:
        # light grey background (like overcast sky / studio)
        bg.inputs[0].default_value = (0.86, 0.88, 0.90, 1.0)
        bg.inputs[1].default_value = 1.0

    # Reduce Eevee artifacts that look like "black chunks"
    ee = getattr(scene, "eevee", None)
    if ee is not None:
        if hasattr(ee, "use_gtao"):
            ee.use_gtao = False
        if hasattr(ee, "use_ssr"):
            ee.use_ssr = False
        if hasattr(ee, "use_bloom"):
            ee.use_bloom = False

        # --- known bounding box ---
        bb_min = DOMAIN_MIN
        bb_max = DOMAIN_MAX
        center = (bb_min + bb_max) * 0.5
        radius = (bb_max - bb_min).length * 0.5  # ~= 0.866 for unit cube

        # --- create camera ---
        cam_data = bpy.data.cameras.new("Camera")
        cam = bpy.data.objects.new("Camera", cam_data)
        scene.collection.objects.link(cam)
        scene.camera = cam

        # --- camera parameters ---
        cam_data.lens = 28.0              # wide enough for volume
        cam_data.clip_start = 0.001       # critical for volume
        cam_data.clip_end = 20.0

        # --- place camera (diagonal view, classic smoke shot) ---
        view_dir = Vector((1.0, -1.0, 0.8)).normalized()

        distance = 1.5 * radius           # safe distance
        cam.location = center + view_dir * distance

        # --- look at center ---
        direction = center - cam.location
        cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

        print("[Camera] location:", cam.location)
        print("[Camera] looking at:", center)

        # Key backlight
        key_data = bpy.data.lights.new(name="KeyBack", type='AREA')
        key = bpy.data.objects.new(name="KeyBack", object_data=key_data)
        bpy.context.collection.objects.link(key)
        key.location = Vector((-0.3, 0.5, 1.3))
        look_at(key, center)
        if hasattr(key_data, "use_shadow"):
            key_data.use_shadow = False
        key_data.energy = 2000
        key_data.size = 1.2

        # Fill
        fill_data = bpy.data.lights.new(name="Fill", type='AREA')
        fill = bpy.data.objects.new(name="Fill", object_data=fill_data)
        bpy.context.collection.objects.link(fill)
        fill.location = Vector((1.3, 0.2, 0.8))
        look_at(fill, center)
        if hasattr(fill_data, "use_shadow"):
            fill_data.use_shadow = False
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

rotate_all_pointclouds_x_minus_90()

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
#set_eevee_gpu(scene)
scene.render.engine = 'CYCLES'

# [Eevee skip] scene.cycles.samples = CYCLES_SAMPLES
# [Eevee skip] scene.cycles.use_denoising = False
# [Eevee skip] scene.cycles.volume_step_rate = VOLUME_STEPS_RATE
# [Eevee skip] scene.cycles.volume_max_steps = VOLUME_MAX_STEPS

scene.view_settings.view_transform = 'Filmic'
scene.view_settings.look = 'High Contrast'
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
