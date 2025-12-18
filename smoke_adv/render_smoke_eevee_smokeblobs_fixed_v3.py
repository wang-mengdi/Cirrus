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

RENDER_ENGINE = "EEVEE" # "EEVEE" or "CYCLES"

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

def create_black_smoke_volume_material(name="MAT_Smoke_Eevee_Black_Unlit"):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    mat.use_nodes = True

    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    # Eevee transparency
    if hasattr(mat, "blend_method"):
        mat.blend_method = "BLEND"
    if hasattr(mat, "shadow_method"):
        mat.shadow_method = "NONE"
    mat.use_backface_culling = False

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (820, 0)

    # --- Alpha density: Object coords -> Noise -> Ramp ---
    texcoord = nodes.new("ShaderNodeTexCoord")
    mapping  = nodes.new("ShaderNodeMapping")
    noise    = nodes.new("ShaderNodeTexNoise")
    ramp     = nodes.new("ShaderNodeValToRGB")

    texcoord.location = (-900, -220)
    mapping.location  = (-700, -220)
    noise.location    = (-500, -220)
    ramp.location     = (-280, -220)

    noise.inputs["Scale"].default_value = 10.0
    noise.inputs["Detail"].default_value = 4.0
    noise.inputs["Roughness"].default_value = 0.55

    # 让密度更“实”，避免大面积穿孔
    ramp.color_ramp.elements[0].position = 0.40
    ramp.color_ramp.elements[1].position = 0.64

    links.new(texcoord.outputs["Object"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])

    # --- Edge factor (Facing) for subtle rim look (still dark) ---
    lw     = nodes.new("ShaderNodeLayerWeight")
    inv    = nodes.new("ShaderNodeInvert")
    lw.location  = (-500, 40)
    inv.location = (-280, 40)
    lw.inputs["Blend"].default_value = 0.6
    links.new(lw.outputs["Facing"], inv.inputs["Color"])

    # --- Alpha = max(density * edge, MIN_ALPHA) ---
    mul = nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mul.location = (-60, -120)
    # Convert ramp color -> float
    ramp_bw = nodes.new("ShaderNodeRGBToBW")
    ramp_bw.location = (-120, -220)
    links.new(ramp.outputs["Color"], ramp_bw.inputs["Color"])

    # Convert inv color -> float
    inv_bw = nodes.new("ShaderNodeRGBToBW")
    inv_bw.location = (-120, 40)
    links.new(inv.outputs["Color"], inv_bw.inputs["Color"])

    # Use floats for math
    links.new(ramp_bw.outputs["Val"], mul.inputs[0])
    links.new(inv_bw.outputs["Val"],  mul.inputs[1])


    # Read "age" attribute (0..1 recommended)
    age = nodes.new("ShaderNodeAttribute")
    age.location = (-500, -420)
    age.attribute_name = "age"

    # Remap age -> fade curve using ColorRamp
    # 0: newly born -> dense
    # 1: old -> fade out
    age_ramp = nodes.new("ShaderNodeValToRGB")
    age_ramp.location = (-280, -420)
    age_ramp.color_ramp.elements[0].position = 0.00
    age_ramp.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
    age_ramp.color_ramp.elements[1].position = 0.85
    age_ramp.color_ramp.elements[1].color = (0.10, 0.10, 0.10, 1.0)
    links.new(age.outputs["Fac"], age_ramp.inputs["Fac"])

    alpha_fade = nodes.new("ShaderNodeMath")
    alpha_fade.operation = "MULTIPLY"
    alpha_fade.location = (80, -200)
    links.new(mul.outputs["Value"], alpha_fade.inputs[0])
    # FIX: use Fac output (float), not Color
    age_bw = nodes.new("ShaderNodeRGBToBW")
    age_bw.location = (-120, -420)
    links.new(age_ramp.outputs["Color"], age_bw.inputs["Color"])
    links.new(age_bw.outputs["Val"], alpha_fade.inputs[1])



    min_alpha = nodes.new("ShaderNodeMath")
    min_alpha.operation = "MAXIMUM"
    min_alpha.location = (160, -120)
    min_alpha.inputs[1].default_value = 0.03
    links.new(alpha_fade.outputs["Value"], min_alpha.inputs[0])

    # -- Unlit smoke color: Emission (dark) ---
    emit = nodes.new("ShaderNodeEmission")
    emit.location = (160, 80)
    #emit.inputs["Color"].default_value = (0.055, 0.055, 0.060, 1.0)

    # Slightly brighten smoke as it ages (looks more natural)
    col_ramp = nodes.new("ShaderNodeValToRGB")
    col_ramp.location = (-280, 220)
    col_ramp.color_ramp.elements[0].position = 0.0
    col_ramp.color_ramp.elements[0].color = (0.04, 0.04, 0.045, 1.0)
    col_ramp.color_ramp.elements[1].position = 1.0
    col_ramp.color_ramp.elements[1].color = (0.14, 0.14, 0.15, 1.0)
    links.new(age.outputs["Fac"], col_ramp.inputs["Fac"])
    links.new(col_ramp.outputs["Color"], emit.inputs["Color"])




    # Emission 强度也用 edge 稍微抬一下（模拟轮廓）
    strength = nodes.new("ShaderNodeMath")
    strength.operation = "ADD"
    strength.location = (-60, 160)
    strength.inputs[0].default_value = 1.4
    # edge * 0.25
    edge_scale = nodes.new("ShaderNodeMath")
    edge_scale.operation = "MULTIPLY"
    edge_scale.location = (-280, 160)
    edge_scale.inputs[1].default_value = 0.65
    links.new(inv.outputs["Color"], edge_scale.inputs[0])
    links.new(edge_scale.outputs["Value"], strength.inputs[1])
    links.new(strength.outputs["Value"], emit.inputs["Strength"])

    # --- Mix Transparent with Emission by alpha ---
    transp = nodes.new("ShaderNodeBsdfTransparent")
    transp.location = (160, -260)

    mix = nodes.new("ShaderNodeMixShader")
    mix.location = (520, -80)
    links.new(min_alpha.outputs["Value"], mix.inputs["Fac"])

    # Fac=0 -> input1, Fac=1 -> input2
    links.new(transp.outputs["BSDF"], mix.inputs[1])
    links.new(emit.outputs["Emission"], mix.inputs[2])

    links.new(mix.outputs["Shader"], out.inputs["Surface"])

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
    """Eevee smoke blobs:
    - Instance small spheres on Alembic points
    - Preserve per-point float attribute 'age' through instancing
      by (1) Capture Attribute with capture_items, then (2) Store Named Attribute after Realize
    """
    # Recreate node group each run
    old = bpy.data.node_groups.get(name)
    if old is not None:
        bpy.data.node_groups.remove(old)

    ng = bpy.data.node_groups.new(name, 'GeometryNodeTree')
    nodes = ng.nodes
    links = ng.links

    # Interface
    ng.interface.new_socket(name="Geometry", in_out='INPUT',  socket_type='NodeSocketGeometry')
    ng.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')

    inp = nodes.new("NodeGroupInput");  inp.location = (-900, 0)
    out = nodes.new("NodeGroupOutput"); out.location = (900, 0)

    # ------------------------------------------------------------
    # Prototype sphere
    # ------------------------------------------------------------
    ico = nodes.new("GeometryNodeMeshIcoSphere"); ico.location = (-720, -220)
    ico.inputs["Subdivisions"].default_value = 1
    ico.inputs["Radius"].default_value = VOLUME_RADIUS * 0.9

    # ------------------------------------------------------------
    # Read incoming point attribute "age"
    # ------------------------------------------------------------
    age_attr = nodes.new("GeometryNodeInputNamedAttribute"); age_attr.location = (-720, -20)
    age_attr.data_type = 'FLOAT'
    age_attr.inputs["Name"].default_value = "age"

    # ------------------------------------------------------------
    # Capture Attribute (Blender 4.5 requires capture_items to create sockets)
    # ------------------------------------------------------------
    cap = nodes.new("GeometryNodeCaptureAttribute"); cap.location = (-520, -140)
    cap.domain = 'POINT'

    # IMPORTANT: add a capture item so the Value socket exists
    item = cap.capture_items.new('FLOAT', "age")  # creates inputs/outputs named "age"
    # (Some builds may ignore the args; enforce)
    item.data_type = 'FLOAT'
    item.name = "age"

    links.new(inp.outputs["Geometry"], cap.inputs["Geometry"])

    # Now the socket should exist as cap.inputs["age"] and cap.outputs["age"]
    if "age" not in cap.inputs or "age" not in cap.outputs:
        # Fallback: find first non-Geometry input/output (covers odd naming)
        cap_val_in = None
        for s in cap.inputs:
            if s.name != "Geometry":
                cap_val_in = s
                break
        cap_val_out = None
        for s in cap.outputs:
            if s.name != "Geometry":
                cap_val_out = s
                break
        if cap_val_in is None or cap_val_out is None:
            raise RuntimeError("Capture Attribute sockets not found; Blender node API changed.")
    else:
        cap_val_in = cap.inputs["age"]
        cap_val_out = cap.outputs["age"]

    links.new(age_attr.outputs["Attribute"], cap_val_in)

    # ------------------------------------------------------------
    # Random scale * (1 + age*k)  (optional, helps smoke expand)
    # ------------------------------------------------------------
    rand = nodes.new("FunctionNodeRandomValue"); rand.location = (-720, 220)
    rand.data_type = 'FLOAT'
    rand.inputs["Min"].default_value = 0.75
    rand.inputs["Max"].default_value = 1.25

    age_expand = nodes.new("ShaderNodeMath"); age_expand.location = (-520, 320)
    age_expand.operation = 'MULTIPLY'
    age_expand.inputs[1].default_value = 0.65
    links.new(age_attr.outputs["Attribute"], age_expand.inputs[0])

    age_add = nodes.new("ShaderNodeMath"); age_add.location = (-320, 320)
    age_add.operation = 'ADD'
    age_add.inputs[0].default_value = 1.0
    links.new(age_expand.outputs["Value"], age_add.inputs[1])

    scale_mul = nodes.new("ShaderNodeMath"); scale_mul.location = (-520, 220)
    scale_mul.operation = 'MULTIPLY'
    links.new(rand.outputs["Value"], scale_mul.inputs[0])
    links.new(age_add.outputs["Value"], scale_mul.inputs[1])

    comb = nodes.new("ShaderNodeCombineXYZ"); comb.location = (-320, 220)
    links.new(scale_mul.outputs["Value"], comb.inputs["X"])
    links.new(scale_mul.outputs["Value"], comb.inputs["Y"])
    links.new(scale_mul.outputs["Value"], comb.inputs["Z"])

    # ------------------------------------------------------------
    # Instance on points (use captured geometry as Points input)
    # ------------------------------------------------------------
    inst = nodes.new("GeometryNodeInstanceOnPoints"); inst.location = (-120, 0)
    inst.inputs["Pick Instance"].default_value = False
    links.new(cap.outputs["Geometry"], inst.inputs["Points"])
    links.new(ico.outputs["Mesh"], inst.inputs["Instance"])
    links.new(comb.outputs["Vector"], inst.inputs["Scale"])

    # ------------------------------------------------------------
    # Realize instances
    # ------------------------------------------------------------
    realize = nodes.new("GeometryNodeRealizeInstances"); realize.location = (120, 0)
    links.new(inst.outputs["Instances"], realize.inputs["Geometry"])

    # ------------------------------------------------------------
    # Store Named Attribute "age" on realized geometry
    # ------------------------------------------------------------
    store = nodes.new("GeometryNodeStoreNamedAttribute"); store.location = (340, -120)
    store.domain = 'POINT'
    store.data_type = 'FLOAT'
    store.inputs["Name"].default_value = "age"
    links.new(realize.outputs["Geometry"], store.inputs["Geometry"])

    # Store node's value socket name varies a bit; find first float-like input that isn't Geometry/Name
    store_val_in = None
    for s in store.inputs:
        if s.name not in ("Geometry", "Name"):
            store_val_in = s
            break
    if store_val_in is None:
        raise RuntimeError("Store Named Attribute has no value input socket (unexpected).")

    links.new(cap_val_out, store_val_in)

    # ------------------------------------------------------------
    # Set material
    # ------------------------------------------------------------
    setmat = nodes.new("GeometryNodeSetMaterial"); setmat.location = (560, 0)
    if volume_mat is not None:
        setmat.inputs["Material"].default_value = volume_mat

    links.new(store.outputs["Geometry"], setmat.inputs["Geometry"])
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
        bg.inputs[0].default_value = (0.02, 0.02, 0.025, 1.0)
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
    fill_data.energy = 450
    fill_data.size = 1.0

    rim_data = bpy.data.lights.new(name="RimTop", type='AREA')
    rim = bpy.data.objects.new(name="RimTop", object_data=rim_data)
    bpy.context.collection.objects.link(rim)

    rim.location = Vector((-0.6, 0.9, 1.8))   # 后上方
    look_at(rim, center)

    rim_data.energy = 1200
    rim_data.size = 2.0
    if hasattr(rim_data, "use_shadow"):
        rim_data.use_shadow = False


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
if RENDER_ENGINE.upper() == "EEVEE":
    set_eevee_gpu(scene)
else:
    scene.render.engine = "CYCLES"

# [Eevee skip] scene.cycles.samples = CYCLES_SAMPLES
# [Eevee skip] scene.cycles.use_denoising = False
# [Eevee skip] scene.cycles.volume_step_rate = VOLUME_STEPS_RATE
# [Eevee skip] scene.cycles.volume_max_steps = VOLUME_MAX_STEPS

scene.view_settings.view_transform = 'Filmic'
scene.view_settings.look = 'High Contrast'
scene.view_settings.exposure = 0.0

scene.render.film_transparent = False
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
