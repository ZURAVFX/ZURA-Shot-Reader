import math
from fractions import Fraction
from pathlib import Path

import bpy
from mathutils import Vector


def number(value, fallback=0.0):
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return fallback


def vector(value, defaults):
    if not isinstance(value, list) or len(value) != len(defaults):
        return defaults
    return tuple(number(component, default) for component, default in zip(value, defaults))


def records(value):
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def surface(name, colour, metallic=0.0, roughness=0.65):
    existing = bpy.data.materials.get(name)
    if existing:
        return existing
    material = bpy.data.materials.new(name)
    material.diffuse_color = colour
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = colour
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = roughness
    return material


def box(collection, name, size, colour):
    x, y, z = (max(0.01, component) for component in size)
    vertices = [
        (u * x / 2, v * y / 2, w * z)
        for w in (0, 1)
        for u, v in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]
    faces = [(3, 2, 1, 0), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    obj = bpy.data.objects.new(name, mesh)
    obj.color = colour
    collection.objects.link(obj)
    return obj


def actor(collection, label, size, sitting, library=None):
    library = Path(library) if library else Path(__file__).with_name("character.blend")
    with bpy.data.libraries.load(str(library), link=False) as (_, loaded):
        loaded.collections = ["Shot Reader Character"]
    character = loaded.collections[0]
    if character is None:
        raise ValueError("The character library needs a Shot Reader Character collection.")
    character.name = f"{label} · Character"
    collection.children.link(character)
    rig = next(obj for obj in character.objects if obj.type == "ARMATURE")
    rig.name = f"{label} · {rig.get('shot_reader_rig_name', 'Control Rig')}"
    root = bpy.data.objects.new(label, None)
    root.empty_display_size = 0.12
    character.objects.link(root)
    rig.parent = root
    height = max(0.05, size[2])
    if sitting:
        height = max(height, 1.6)
    scale = height / rig["shot_reader_height_m"]
    rig.scale *= scale
    rig.location.z -= rig["shot_reader_floor_m"] * scale
    forward = rig.get("shot_reader_forward", -1)
    if forward < 0:
        rig.rotation_euler.z += math.pi
    if sitting:
        shifts = {"Ctrl_Hips": (0, 0, -0.40)}
        for side in ("Left", "Right"):
            shifts[f"Ctrl_Foot_IK_{side}"] = (0, 0.38 * forward, 0)
            shifts[f"Ctrl_Hand_IK_{side}"] = (0, 0.14 * forward, -0.27)
        for name, shift in shifts.items():
            control = rig.pose.bones.get(name)
            if control:
                control.location += control.bone.matrix_local.to_3x3().inverted() @ Vector(shift)
    return root


def place_subject(collection, subject, fps, duration, library=None):
    label = str(subject.get("id") or subject.get("class") or "Subject")
    size = vector(subject.get("size_m"), (0.5, 0.5, 1.7))
    kind = str(subject.get("class", ""))
    is_actor = kind.casefold() in {"person", "human"}
    if is_actor:
        obj = actor(collection, label, size, subject.get("posture") == "sit", library)
    else:
        obj = box(collection, label, size, (0.24, 0.48, 0.65, 1))
        obj.data.materials.append(surface("Shot Reader • Prop", obj.color))
    obj["shot_reader_class"] = kind
    keys = {}
    for key in records(subject.get("path")):
        position = key.get("pos_m")
        if not isinstance(position, list) or len(position) != 2:
            continue
        time = max(0.0, min(duration, number(key.get("t"))))
        keys[time] = (vector(position, (0.0, 0.0)), key.get("facing_deg"))
    if not keys:
        return obj
    yaw = 0.0
    for time, ((x, y), facing) in sorted(keys.items()):
        obj.location = (x, y, 0)
        yaw = number(facing, yaw)
        obj.rotation_euler.z = math.radians(yaw - 90 if is_actor else yaw)
        if len(keys) > 1:
            frame = 1 + time * fps
            obj.keyframe_insert(data_path="location", frame=frame)
            obj.keyframe_insert(data_path="rotation_euler", index=2, frame=frame)
    if obj.animation_data:
        animation = obj.animation_data
        for layer in animation.action.layers:
            for strip in layer.strips:
                for channel in strip.channelbag(animation.action_slot).fcurves:
                    for point in channel.keyframe_points:
                        point.interpolation = "LINEAR"
    return obj


def resolve_reflection(subjects, items):
    """Resolve an apparent duplicate only when the shot sheet gives mirror geometry and a close match."""
    mirrors = [item for item in items if str(item.get("class", "")).casefold().split(":")[-1] == "mirror"]
    people = [item for item in subjects if str(item.get("class", "")).casefold() in {"person", "human"}]
    if len(mirrors) != 1 or len(people) != 2:
        return subjects, items
    mirror = mirrors[0]
    centre = Vector(vector(mirror.get("pos_m"), (0.0, 0.0)))
    width, thickness, _height = vector(mirror.get("size_m"), (1.0, 0.05, 1.0))
    yaw = math.radians(number(mirror.get("yaw_deg")))
    paths = [records(person.get("path")) for person in people]
    if not all(paths):
        return subjects, items
    pairs = sorted(
        zip(people, (path[0] for path in paths)),
        key=lambda pair: Vector(vector(pair[1].get("pos_m"), (0.0, 0.0))).length,
    )
    (person, person_key), (candidate, reflected_key) = pairs
    near = Vector(vector(person_key.get("pos_m"), (0.0, 0.0)))
    far = Vector(vector(reflected_key.get("pos_m"), (0.0, 0.0)))
    heights = [vector(item.get("size_m"), (0.5, 0.5, 1.7))[2] for item in (person, candidate)]
    facing = abs((number(person_key.get("facing_deg")) - number(reflected_key.get("facing_deg")) + 180) % 360 - 180)
    if far.y <= 0.05 or abs(heights[0] - heights[1]) > 0.18 or facing < 120:
        return subjects, items
    tangent = Vector((math.cos(yaw), math.sin(yaw)))
    normal = Vector((-tangent.y, tangent.x))
    if normal.dot(centre) < 0:
        normal.negate()
    near_depth, far_depth = normal.dot(near - centre), normal.dot(far - centre)
    ray = far.normalized()
    component = ray.dot(normal)
    if component <= 0.05:
        return subjects, items
    reflection_point = ray * (centre.dot(normal) / component)
    if near_depth >= -0.25 or far_depth < -0.35 or abs((reflection_point - centre).dot(tangent)) > width / 2:
        return subjects, items

    # Fit the mirror angle to the reflected subject's sightline, within the uncertainty of the input.
    pivot = centre - normal * max(0.025, thickness / 2)
    for wall in items:
        if str(wall.get("class", "")).casefold() != "wall":
            continue
        wall_pos = Vector(vector(wall.get("pos_m"), (0.0, 0.0)))
        wall_width, wall_depth, _height = vector(wall.get("size_m"), (1.0, 0.2, 2.5))
        if abs((wall_pos - centre).dot(normal)) <= (wall_depth + thickness) / 2 and abs((wall_pos - centre).dot(tangent)) < wall_width / 2:
            pivot = centre + normal * ((wall_pos - centre).dot(normal) - wall_depth / 2 - 0.005)
            break

    def error(delta):
        angle = yaw + delta
        direction = Vector((-math.sin(angle), math.cos(angle)))
        virtual = near - 2 * (near - pivot).dot(direction) * direction
        return abs(math.atan2(virtual.x, virtual.y) - math.atan2(far.x, far.y)) + abs(delta) * 0.02

    correction = min((math.radians(degrees) for degrees in range(-35, 36)), key=error)
    corrected_items = []
    for item in items:
        kind = str(item.get("class", "")).casefold().split(":")[-1]
        position = Vector(vector(item.get("pos_m"), (0.0, 0.0)))
        if kind in {"mirror", "wall", "mirror_cabinet"} and abs((position - centre).dot(normal)) < 0.25:
            dx, dy = position - pivot
            cos, sin = math.cos(correction), math.sin(correction)
            position = pivot + Vector((dx * cos - dy * sin, dx * sin + dy * cos))
            item = {**item, "pos_m": list(position), "yaw_deg": number(item.get("yaw_deg")) + math.degrees(correction)}
        corrected_items.append(item)
    return [item for item in subjects if item is not candidate], corrected_items


def blocks_subject(x, y, base, size, yaw, camera_height, target):
    if not target:
        return False
    c, s = math.cos(yaw), math.sin(yaw)
    tx, ty, tz = target
    camera_side = s * x - c * y
    direction = c * ty - s * tx
    if abs(direction) < 1e-5:
        return False
    t = -camera_side / direction
    if not 0 <= t < 0.95:
        return False
    crossing = c * (t * tx - x) + s * (t * ty - y)
    height = camera_height + t * (tz - camera_height)
    return abs(crossing) <= size[0] / 2 and base <= height <= base + size[2]


def place_set(collection, items, camera_height, target):
    walls = [item for item in items if str(item.get("class", "")).casefold() == "wall"]
    mirrors = [item for item in items if str(item.get("class", "")).casefold().split(":")[-1] == "mirror"]
    counters = any("counter" in str(item.get("class", "")).casefold() for item in items)
    for item in items:
        kind = str(item.get("class") or "Set piece")
        category = kind.casefold().split(":")[-1]
        base = number(item.get("base_m"))
        x, y = vector(item.get("pos_m"), (0.0, 0.0))
        size = vector(item.get("size_m"), (1.0, 1.0, 1.0))
        yaw = math.radians(number(item.get("yaw_deg")))
        is_mirror = item in mirrors
        if category == "mirror_cabinet" and any(
            math.dist((x, y), vector(mirror.get("pos_m"), (0.0, 0.0))) < 0.2 for mirror in mirrors
        ):
            continue
        if mirrors and base < 0.3 and category in {"sink", "faucet"}:
            base = 0.84 if category == "sink" else 0.88
        if mirrors and category == "sink" and not counters:
            vanity = box(collection, "Vanity", (max(1.1, size[0] * 1.6), max(0.5, size[1]), base),
                         (0.6, 0.58, 0.55, 1))
            vanity.location = (x + 0.18 * math.cos(yaw), y + 0.18 * math.sin(yaw), 0)
            vanity.rotation_euler.z = yaw
            vanity.data.materials.append(surface("Shot Reader • Vanity", vanity.color))
        if is_mirror:
            # Place a wall-mounted mirror just clear of a parallel wall when their volumes overlap.
            tangent = (math.cos(yaw), math.sin(yaw))
            normal = (-tangent[1], tangent[0])
            sign = 1 if x * normal[0] + y * normal[1] < 0 else -1
            for wall in walls:
                wx, wy = vector(wall.get("pos_m"), (0.0, 0.0))
                ws = vector(wall.get("size_m"), (1.0, 0.2, 2.5))
                dx, dy = x - wx, y - wy
                parallel = abs(math.sin(yaw - math.radians(number(wall.get("yaw_deg"))))) < 0.1
                depth = dx * normal[0] + dy * normal[1]
                if parallel and abs(depth) <= (ws[1] + size[1]) / 2 and abs(dx * tangent[0] + dy * tangent[1]) < ws[0] / 2:
                    offset = sign * ((ws[1] + size[1]) / 2 + 0.005) - depth
                    x, y = x + offset * normal[0], y + offset * normal[1]
                    break
        colour = (0.48, 0.67, 0.71, 1) if is_mirror else (0.64, 0.57, 0.49, 1)
        obj = box(collection, kind, size, colour)
        obj.location = (x, y, base)
        obj.rotation_euler.z = yaw
        mat = surface("Shot Reader • Mirror", (0.94, 0.94, 0.94, 1), 1.0, 0.015) if is_mirror else surface("Shot Reader • Set", colour)
        obj.data.materials.append(mat)
        obj["shot_reader_class"] = kind
        if category == "wall" and blocks_subject(x, y, base, size, yaw, camera_height, target):
            obj.display_type = "WIRE"
            obj.hide_render = True
            obj["shot_reader_camera_occluder"] = True
    return bool(mirrors)


def lighting(scene):
    world = bpy.data.worlds.new("Shot Reader Studio")
    world.use_nodes = True
    world.node_tree.nodes.get("Background").inputs["Color"].default_value = (0.45, 0.5, 0.57, 1)
    world.node_tree.nodes.get("Background").inputs["Strength"].default_value = 0.45
    scene.world = world
    data = bpy.data.lights.new("Soft Light", "AREA")
    data.energy = 220
    data.shape = "DISK"
    data.size = 4
    obj = bpy.data.objects.new("Soft Light", data)
    scene.collection.objects.link(obj)
    obj.location = (0, 0.8, 2.42)


def build_blockout(sheet, title, source_render, library=None):
    if not isinstance(sheet, dict) or not isinstance(sheet.get("camera"), dict):
        raise ValueError("Expected a Shot Reader JSON object with a camera section.")
    spec = sheet["camera"]
    timing = sheet.get("shot")
    if not isinstance(timing, dict):
        timing = {}
    source_format = sheet.get("source_format", {})
    width, height = source_render.resolution_x, source_render.resolution_y
    pixel_aspect = source_render.pixel_aspect_x / source_render.pixel_aspect_y
    fps_integer, fps_base = source_render.fps, source_render.fps_base
    if isinstance(source_format, dict) and source_format:
        width = int(number(source_format.get("width"), width))
        height = int(number(source_format.get("height"), height))
        pixel_aspect = number(source_format.get("pixel_aspect"), pixel_aspect)
        try:
            rate = float(Fraction(source_format.get("fps", "0")))
            if 1 <= rate <= 32767:
                fps_integer = round(rate)
                fps_base = fps_integer / rate
        except (ValueError, TypeError, ZeroDivisionError):
            pass
    fps = fps_integer / fps_base
    duration = max(0, min(1048573 / fps, number(timing.get("end")) - number(timing.get("start"))))
    scene = bpy.data.scenes.new(title.removesuffix(" · Shot Sheet.json") + " · Blockout")
    scene.unit_settings.system = "METRIC"
    scene.render.fps = fps_integer
    scene.render.fps_base = fps_base
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.pixel_aspect_x = max(1, pixel_aspect)
    scene.render.pixel_aspect_y = 1 / max(0.01, min(1, pixel_aspect))
    scene.frame_end = max(1, math.ceil(fps * duration - 1e-7))

    data = bpy.data.cameras.new("Reference Camera")
    data.sensor_fit = "HORIZONTAL"
    fov = max(1.0, min(175.0, number(spec.get("fov_deg"), 50.0)))
    data.lens = data.sensor_width / (2 * math.tan(math.radians(fov) / 2))
    camera = bpy.data.objects.new("Reference Camera", data)
    scene.collection.objects.link(camera)
    camera.location.z = number(spec.get("height_m"), 1.5)
    camera.rotation_euler.x = math.radians(90 + number(spec.get("pitch_deg")))
    camera["shot_reader_move"] = str(spec.get("move") or "unspecified")
    scene.camera = camera

    subjects, set_items = resolve_reflection(records(sheet.get("subjects")), records(sheet.get("set")))
    for subject in subjects:
        place_subject(scene.collection, subject, fps, duration, library)
    target = None
    primary = next((item for item in subjects if item.get("id") == spec.get("subject")), None)
    if primary is None and subjects:
        primary = subjects[0]
    if primary:
        path = records(primary.get("path"))
        if path and isinstance(path[0].get("pos_m"), list):
            x, y = vector(path[0]["pos_m"], (0.0, 0.0))
            if y > 0:
                height = vector(primary.get("size_m"), (0.5, 0.5, 1.7))[2]
                target = (x, y, height * 0.65)
    has_mirror = place_set(scene.collection, set_items, camera.location.z, target)
    floor = box(scene.collection, "Ground", (40, 40, 0.05), (0.43, 0.45, 0.49, 1))
    floor.location.z = -0.05
    floor.data.materials.append(surface("Shot Reader • Ground", floor.color))
    lighting(scene)
    if has_mirror:
        scene.render.engine = "CYCLES"
        scene.cycles.samples = 24
        scene.cycles.use_denoising = True
    scene.frame_set(1)
    return scene
