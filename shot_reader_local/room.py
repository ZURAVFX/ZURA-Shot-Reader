"""Create an editable room depth reference from the linked plate and active camera."""

import json
from pathlib import Path
import subprocess
import tempfile

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

from .calibration import camera_intrinsics


job = None
status = ""


def update_preview_walls(settings, context):
    for obj in context.scene.objects:
        saved = obj.get('shot_reader_preview_wall_rays')
        if saved is not None:
            obj.visible_camera = bool(saved[0]) and settings.show_preview_walls
            obj.visible_glossy = bool(saved[1]) and settings.show_preview_walls


def update_preview_fixtures(settings, context):
    for obj in context.scene.objects:
        saved = obj.get('shot_reader_preview_fixture_ray')
        if saved is not None:
            obj.visible_camera = bool(saved) and settings.show_preview_fixtures


def restore_preview_fixtures(scene):
    for obj in scene.objects:
        if 'shot_reader_preview_fixture_ray' in obj:
            obj.visible_camera = bool(obj.pop('shot_reader_preview_fixture_ray'))
        if 'shot_reader_preview_fixture_viewport' in obj:
            obj.hide_set(bool(obj.pop('shot_reader_preview_fixture_viewport')))


def fixture_category(obj):
    category = str(obj.get('shot_reader_class', '')).casefold().split(':')[-1]
    if not category and obj.name.split('.')[0] == 'Vanity' and any(
            mat and mat.name.startswith('Shot Reader • Vanity') for mat in obj.data.materials):
        return 'vanity'
    return category


def fixture_observations(scene, camera, world, wall):
    supports, taps = [], []
    support_base = None
    for obj in scene.objects:
        if obj.type != 'MESH':
            continue
        category = fixture_category(obj)
        if category in {'vanity', 'sink', 'counter', 'countertop', 'faucet'}:
            corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
            points = [world_to_camera_view(scene, camera, corner) for corner in corners]
            if min(p.z for p in points) > camera.data.clip_start:
                (taps if category == 'faucet' else supports).extend(p.x for p in points)
                if category == 'vanity' and max(p.x for p in points) > 0. and min(p.x for p in points) < 1.:
                    base = min(corner.z for corner in corners)
                    support_base = base if support_base is None else min(support_base, base)
    if not supports or max(supports) <= 0. or min(supports) >= 1.:
        return None
    normal = Vector(wall['normal'])
    return {'world': [list(row) for row in world], 'counter_u': [min(supports), max(supports)],
            'reference_bottom': max(point[1] for point in wall['rim']),
            'wall_point': list(world @ (normal * wall['distance'])),
            'wall_toward_camera': list(world.to_3x3() @ normal),
            **({'support_base': support_base} if support_base is not None else {}),
            **({'faucet_u': [min(taps), max(taps)]} if taps else {})}


class ShotReaderRoomSettings(bpy.types.PropertyGroup):
    exclusions: bpy.props.PointerProperty(
        name="Exclude Polygons (JSON)", type=bpy.types.Text,
        description="Optional array of polygons in normalised top-left plate coordinates. Exclude reflections and people.",
    )
    use_scale_anchors: bpy.props.BoolProperty(
        name="Scale from Landmarks",
        description="Use measured static landmarks on this frame to set the reconstructed depth scale",
    )
    clean_exclusions: bpy.props.PointerProperty(
        name="Fixed Exclusions", type=bpy.types.Text,
        description="Optional normalised JSON polygons for other mirrors or items that move independently",
    )
    room: bpy.props.PointerProperty(type=bpy.types.Object)
    reflected_wall: bpy.props.PointerProperty(type=bpy.types.Object)
    reflected_depth: bpy.props.FloatProperty(name='Wall Behind Camera', default=.65, min=.05, soft_max=8., subtype='DISTANCE')
    show_preview_walls: bpy.props.BoolProperty(
        name='Render Overlapping Blockout Walls', update=update_preview_walls,
        description='Show the approximate wall surfaces where a measured plate projection is available',
    )
    show_preview_fixtures: bpy.props.BoolProperty(
        name='Render Fixture Blockout', update=update_preview_fixtures,
        description='Show modeled blockout boxes where the linked plate supplies measured detail',
    )


def isolate_preview_walls(scene, camera):
    from .reflected_room import mirror_polygon

    mirror = scene.shot_reader.body_mirror
    if not mirror:
        return
    aperture = mirror_polygon(scene, mirror, required=False)
    if not aperture:
        return
    x0, y0 = map(min, zip(*aperture))
    x1, y1 = map(max, zip(*aperture))
    area = (x1 - x0) * (y1 - y0)
    show = scene.shot_reader_room.show_preview_walls
    for obj in scene.objects:
        if obj.type != 'MESH' or str(obj.get('shot_reader_class', '')).casefold().split(':')[-1] != 'wall':
            continue
        points = [world_to_camera_view(scene, camera, obj.matrix_world @ Vector(corner)) for corner in obj.bound_box]
        visible = [(p.x, 1. - p.y) for p in points if p.z > 0.]
        if not visible:
            continue
        left, top = map(min, zip(*visible))
        right, bottom = map(max, zip(*visible))
        overlap = max(0., min(x1, right) - max(x0, left)) * max(0., min(y1, bottom) - max(y0, top))
        if area > 0 and overlap >= area * .25:
            saved = obj.get('shot_reader_preview_wall_rays')
            if saved is None:
                saved = [int(obj.visible_camera), int(obj.visible_glossy)]
                obj['shot_reader_preview_wall_rays'] = saved
            obj.visible_camera = bool(saved[0]) and show
            obj.visible_glossy = bool(saved[1]) and show


def isolate_preview_fixtures(scene, camera, projection):
    """Suppress preview boxes where the measured plate and room mesh cover their footprint."""
    import numpy as np

    restore_preview_fixtures(scene)
    physical = projection.parent
    if not physical or physical.type != 'MESH':
        return {}
    source = next((node.image for node in projection.data.materials[0].node_tree.nodes
                   if node.type == 'TEX_IMAGE' and node.image), None)
    if not source or source.channels != 4:
        return {}
    width, height = source.size
    raster = np.empty(width * height * 4, dtype=np.float32)
    source.pixels.foreach_get(raster)
    measured = raster.reshape(height, width, 4)[:, :, 3]
    fx, fy, cx, cy = camera_intrinsics(scene, camera.data)
    rw, rh = scene.render.resolution_x, scene.render.resolution_y
    camera_matrix = camera.matrix_world.normalized()
    room_view = physical.matrix_world.inverted() @ camera_matrix
    room_origin = room_view.translation
    room_axes = room_view.to_3x3()
    show = scene.shot_reader_room.show_preview_fixtures
    evidence = {}
    for obj in scene.objects:
        if obj.type != 'MESH':
            continue
        tag = 'shot_reader_preview_fixture_ray'
        category = fixture_category(obj)
        if category not in {'vanity', 'sink', 'faucet', 'counter', 'countertop'} or not obj.visible_camera:
            continue
        corners = [world_to_camera_view(scene, camera, obj.matrix_world @ Vector(corner)) for corner in obj.bound_box]
        if min(point.z for point in corners) <= camera.data.clip_start:
            continue
        left, bottom = np.clip(np.min([(p.x, p.y) for p in corners], axis=0), 0., 1.)
        right, top = np.clip(np.max([(p.x, p.y) for p in corners], axis=0), 0., 1.)
        if min((right - left) * width, (top - bottom) * height) < 3.:
            continue
        local = obj.matrix_world.inverted() @ camera_matrix
        origin = local.translation
        axes = local.to_3x3()
        tested = covered = supported = 0
        for y in np.linspace(bottom, top, 15):
            for x in np.linspace(left, right, 15):
                ray = Vector(((x * rw - cx) / fx, (cy - (1. - y) * rh) / fy, -1.))
                if not obj.ray_cast(origin, axes @ ray)[0]:
                    continue
                tested += 1
                if measured[min(height - 1, int(y * height)), min(width - 1, int(x * width))] >= .98:
                    covered += 1
                    supported += physical.ray_cast(room_origin, room_axes @ ray)[0]
        if tested >= 20 and covered >= .85 * tested and supported >= .65 * tested:
            obj[tag] = int(obj.visible_camera)
            obj.visible_camera = show
            evidence[obj.name] = {'plate_coverage': covered / tested, 'mesh_support': supported / tested,
                                  'checked_rays': tested}
    return evidence


def set_mesh(name, xyz, faces, uv):
    import numpy as np

    used, indices = np.unique(faces, return_inverse=True)
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(xyz[used].tolist(), [], indices.reshape(-1, 3).tolist())
    mesh.update()
    mesh.uv_layers.new(name='Plate').uv.foreach_set('vector', uv[faces.ravel()].ravel())
    return mesh


def add_faucet_profile(scene, physical, report):
    profile = bpy.data.curves.new('Faucet Gooseneck', 'CURVE')
    profile.dimensions = '3D'
    profile.bevel_depth = report['radius']
    profile.bevel_resolution = 4
    profile.use_fill_caps = True
    line = profile.splines.new('POLY')
    line.points.add(len(report['path']) - 1)
    inverse = physical.matrix_world.inverted()
    for point, position in zip(line.points, report['path']):
        point.co = (*tuple(inverse @ Vector(position)), 1.)
    pipe = bpy.data.objects.new(profile.name, profile)
    scene.collection.objects.link(pipe)
    pipe.parent = physical
    metal = bpy.data.materials.new('Fixture Chrome')
    metal.diffuse_color = (.46, .47, .49, 1.)
    profile.materials.append(metal)
    pipe['shot_reader_fixture_proxy'] = True
    pipe.hide_render = True


def add_fixture_blockout(scene, physical, name, kind, corners, faces, colour, thickness=None):
    inverse = physical.matrix_world.inverted()
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([inverse @ Vector(point) for point in corners], [], faces)
    mesh.update()
    material = bpy.data.materials.new(name)
    material.diffuse_color = colour
    mesh.materials.append(material)
    wall = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(wall)
    wall.parent = physical
    wall['shot_reader_fixture_proxy'] = kind
    wall.hide_render = True
    if thickness:
        shell = wall.modifiers.new('Wall Thickness', 'SOLIDIFY')
        shell.thickness = thickness


def separate_fixture_geometry(scene, physical, directory, reports):
    import numpy as np

    with np.load(directory / 'room.npz', allow_pickle=False) as arrays:
        xyz, faces, uv = arrays['xyz'], arrays['faces'], arrays['uv']
    labels = np.load(directory / 'fixture-groups.npy', allow_pickle=False)
    if labels.shape != (len(faces),) or not (labels == 0).any():
        return
    material = physical.data.materials[0]
    old = physical.data
    physical.data = set_mesh(old.name, xyz, faces[labels == 0], uv)
    physical.data.materials.append(material)
    if not old.users:
        bpy.data.meshes.remove(old)
    for group, kind, name, thickness, viewport_colour in (
            (1, 'counter', 'Measured Counter Surface', .022, (.58, .57, .52, 1.)),
            (2, 'faucet', 'Measured Faucet Surface', .012, (.44, .46, .49, 1.)),
            (3, 'faucet_profile', 'Faucet Plate Surface', 0., (.44, .46, .49, 1.)),
            (4, 'fixture_depth_bridges', 'Fixture Plate Boundaries', 0., (.44, .46, .49, 1.)),
            (5, 'backsplash', 'Backsplash Plate Surface', 0., (.36, .32, .29, 1.))):
        subset = faces[labels == group]
        if not len(subset):
            continue
        mesh = set_mesh(name, xyz, subset, uv)
        mat = material.copy()
        mat.name = name
        mat.diffuse_color = viewport_colour
        mesh.materials.append(mat)
        obj = bpy.data.objects.new(name, mesh)
        scene.collection.objects.link(obj)
        obj.parent = physical
        obj['shot_reader_fixture_surface'] = kind
        obj.visible_diffuse = obj.visible_glossy = obj.visible_transmission = False
        obj.visible_shadow = obj.visible_volume_scatter = False
        if thickness:
            shell = obj.modifiers.new('Fixture Thickness', 'SOLIDIFY')
            shell.thickness = thickness
            shell.show_render = False
        else:
            obj.hide_set(True)
        driver = obj.driver_add('hide_render').driver
        driver.type = 'AVERAGE'
        target = driver.variables.new().targets[0]
        target.id, target.data_path = physical, 'hide_render'
    if 'faucet_profile' in reports and (labels == 3).any():
        add_faucet_profile(scene, physical, reports['faucet_profile'])
    if 'backsplash' in reports and (labels == 5).any():
        add_fixture_blockout(scene, physical, 'Fitted Backsplash', 'backsplash', reports['backsplash']['corners'],
                             [(0, 1, 2, 3)], (.36, .32, .29, 1.), thickness=.035)
    if 'counter_support' in reports:
        faces = [(3, 2, 1, 0), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        add_fixture_blockout(scene, physical, 'Estimated Vanity Cabinet', 'counter_support',
                             reports['counter_support']['corners'], faces, (.28, .27, .24, 1.))
    for obj in scene.objects:
        if obj.type != 'MESH' or 'shot_reader_preview_fixture_ray' not in obj:
            continue
        category = fixture_category(obj)
        if category != 'faucet' or (labels == 2).any():
            obj['shot_reader_preview_fixture_viewport'] = int(obj.hide_get())
            obj.hide_set(True)


def add_set_projection(scene, room, path, info, *, estimated=False):
    fx, fy, cx, cy = info['normalized_intrinsics']
    depth = min(float(info['depth_scene_units']), scene.camera.data.clip_end * .97)
    corners = [(0., 1.), (1., 1.), (1., 0.), (0., 0.)]
    vertices = [((u - cx) * depth / fx, (cy - v) * depth / fy, -depth) for u, v in corners]
    name = 'Estimated Set Continuation' if estimated else 'Observed Set Projection'
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], [(0, 1, 2, 3)])
    mesh.update()
    layer = mesh.uv_layers.new(name='Plate')
    layer.uv.foreach_set('vector', [value for u, v in corners for value in (u, 1. - v)])
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    obj.parent = room
    obj.hide_set(True)
    obj.visible_diffuse = obj.visible_glossy = obj.visible_transmission = False
    obj.visible_shadow = obj.visible_volume_scatter = False
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.cycles.emission_sampling = 'NONE'
    nodes = material.node_tree.nodes
    nodes.clear()
    link = material.node_tree.links.new
    source = nodes.new('ShaderNodeTexImage')
    source.image = bpy.data.images.load(str(path), check_existing=False)
    source.image.name = f'{name} Colour and Coverage'
    source.image.pack()
    source.extension = 'CLIP'
    alpha = nodes.new('ShaderNodeMath')
    alpha.operation = 'MULTIPLY'
    ray = nodes.new('ShaderNodeLightPath')
    link(ray.outputs['Is Camera Ray'], alpha.inputs[0])
    link(source.outputs['Alpha'], alpha.inputs[1])
    glow = nodes.new('ShaderNodeEmission')
    link(source.outputs['Color'], glow.inputs['Color'])
    clear = nodes.new('ShaderNodeBsdfTransparent')
    mix = nodes.new('ShaderNodeMixShader')
    link(alpha.outputs[0], mix.inputs[0])
    link(clear.outputs[0], mix.inputs[1])
    link(glow.outputs[0], mix.inputs[2])
    link(mix.outputs[0], nodes.new('ShaderNodeOutputMaterial').inputs['Surface'])
    mesh.materials.append(material)
    driver = obj.driver_add('hide_render').driver
    variable = driver.variables.new()
    variable.name = 'hide_set_projection'
    variable.targets[0].id = room
    variable.targets[0].data_path = 'hide_render'
    driver.type = 'AVERAGE'
    key = 'shot_reader_estimated_projection' if estimated else 'shot_reader_observed_projection'
    obj[key] = True
    room[key] = obj


def add_room(scene, camera_matrix, directory, info):
    import numpy as np

    with np.load(directory / "room.npz", allow_pickle=False) as arrays:
        xyz, faces, uv = arrays["xyz"], arrays["faces"], arrays["uv"]
    mesh = bpy.data.meshes.new("Plate Room Surface")
    mesh.from_pydata(xyz.tolist(), [], faces.tolist())
    mesh.update()
    mesh.uv_layers.new(name="Plate").uv.foreach_set("vector", uv[faces.ravel()].ravel())

    texture = bpy.data.images.load(str(directory / "plate.png"), check_existing=False)
    projection_only = 'clean_set' in info or 'parallax' in info
    texture.name = "Clean Set Reference" if "clean_set" in info else "Room Reference Plate"
    texture.pack()
    mat = bpy.data.materials.new("Room Plate Projection")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    source = nodes.new("ShaderNodeTexImage")
    source.image = texture
    shader = nodes.new("ShaderNodeEmission")
    output = nodes.new("ShaderNodeOutputMaterial")
    mat.node_tree.links.new(source.outputs["Color"], shader.inputs["Color"])
    if projection_only:
        # A plate projection represents the reference camera's view of the observed surfaces.
        ray = nodes.new("ShaderNodeLightPath")
        transparent = nodes.new("ShaderNodeBsdfTransparent")
        mix = nodes.new("ShaderNodeMixShader")
        mat.node_tree.links.new(ray.outputs["Is Camera Ray"], mix.inputs[0])
        mat.node_tree.links.new(transparent.outputs[0], mix.inputs[1])
        mat.node_tree.links.new(shader.outputs[0], mix.inputs[2])
        mat.node_tree.links.new(mix.outputs[0], output.inputs["Surface"])
    else:
        mat.node_tree.links.new(shader.outputs["Emission"], output.inputs["Surface"])
    mesh.materials.append(mat)
    label = 'Parallax Set' if 'parallax' in info else 'Clean Visible Set' if 'clean_set' in info else 'Observed Room Surfaces'
    room = bpy.data.objects.new(label, mesh)
    scene.collection.objects.link(room)
    room.matrix_world = camera_matrix
    room.display_type = "WIRE"
    room.hide_render = True
    if projection_only:
        room.visible_diffuse = room.visible_glossy = room.visible_transmission = False
        room.visible_shadow = room.visible_volume_scatter = False
    room["shot_reader_room_report"] = json.dumps(info)
    room["shot_reader_plate_frame"] = scene.frame_current
    clean_set = info.get('clean_set', {})
    projection = clean_set.get('observed_projection')
    if projection:
        add_set_projection(scene, room, directory / 'observed-set.png', projection)
    estimate = clean_set.get('estimated_projection')
    if estimate:
        add_set_projection(scene, room, directory / 'estimated-set.png', estimate, estimated=True)
    scene.shot_reader_room.room = room
    return room


class ShotReaderBuildRoom(bpy.types.Operator):
    bl_idname = "shot_reader.build_room"
    bl_label = "Reconstruct Visible Set"
    bl_description = "Estimate depth and project the linked frame on an editable mesh"
    bl_options = {"UNDO"}

    @classmethod
    def poll(cls, context):
        from . import reference_background

        camera = context.scene.camera
        return bool(job is None and camera and camera.data.type == "PERSP" and reference_background(context.scene))

    def execute(self, context):
        from . import ADDON, reference_background, worker_environment

        scene, prefs = context.scene, context.preferences.addons[ADDON].preferences
        reference = reference_background(scene).clip
        camera = scene.camera
        state = scene.shot_reader_room
        model = Path(bpy.path.abspath(prefs.lens_model))
        if not model.is_file():
            self.report({"ERROR"}, "Set the MoGe-2 ONNX model in the add-on preferences.")
            return {"CANCELLED"}
        try:
            masks = json.loads(state.exclusions.as_string()) if state.exclusions else []
            if not isinstance(masks, list):
                raise ValueError("Exclusions must be a JSON array of polygons.")
            seconds = (scene.frame_current - reference.frame_start + reference.frame_offset) / (
                scene.render.fps / scene.render.fps_base)
            rw, rh = scene.render.resolution_x, scene.render.resolution_y
            if not reference.size[1] or abs(reference.size[0] / reference.size[1] - rw/rh) > .005:
                raise ValueError("Match the render aspect ratio to the source clip before reconstructing.")
            fx, fy, cx, cy = camera_intrinsics(scene, camera.data)
            world = camera.matrix_world.normalized()
            anchors = []
            cal = scene.shot_reader_calibration
            if state.use_scale_anchors and cal.frame == scene.frame_current:
                inverse = world.inverted()
                anchors = [[*point.uv, -(inverse @ point.object.matrix_world @ Vector(point.local)).z]
                           for point in cal.points if point.object and not point.holdout]
            request = {
                "model": str(model), "video": bpy.path.abspath(reference.filepath, library=reference.library),
                "time": seconds, "intrinsics": [fx / rw, fy / rh, cx / rw, cy / rh],
                "exclude": masks, "scale_anchors": anchors,
            }
            with tempfile.TemporaryDirectory(prefix="shot-reader-room-") as scratch:
                worker = subprocess.run([
                    bpy.path.abspath(prefs.python), str(Path(__file__).with_name("room_depth.py")), scratch,
                ], input=json.dumps(request), capture_output=True, encoding="utf-8", timeout=55,
                    env=worker_environment(prefs), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                if worker.returncode:
                    raise ValueError(worker.stderr.strip() or "Depth reconstruction failed.")
                info = json.loads(worker.stdout)
                add_room(scene, world, Path(scratch), info)
        except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Reconstructed {info['triangles']:,} visible surface triangles.")
        return {"FINISHED"}


def finish_clean_room():
    global job, status
    from . import refresh

    process, temp, scene, camera, world, frame = job
    if process.poll() is None:
        return .35
    try:
        path = Path(temp.name)
        if process.returncode:
            log = (path / 'run.log').read_text(encoding='utf-8', errors='replace').strip()
            raise ValueError(log.splitlines()[-1].removeprefix('ValueError: ') if log else 'Clean set reconstruction failed.')
        if scene.camera != camera:
            raise ValueError('The active camera changed. Run the clean set reconstruction again.')
        info = json.loads((path / 'room.json').read_text(encoding='utf-8'))
        previous = scene.shot_reader_room.room
        room = add_room(scene, world, path, info)
        room['shot_reader_plate_frame'] = frame
        restore_preview_fixtures(scene)
        if previous and (previous.get('shot_reader_clean_set') or previous.get('shot_reader_parallax_set')):
            room.hide_render, room.display_type = previous.hide_render, previous.display_type
            previous.hide_render = True
        if 'parallax' in info:
            room['shot_reader_parallax_set'] = True
            status = f"Parallax set · {info['triangles']:,} triangles"
        else:
            room['shot_reader_clean_set'] = True
            if 'wall_registration' in info and previous:
                previous.hide_set(True)
                for part in previous.children:
                    if part.get('shot_reader_fixture_surface') or part.get('shot_reader_fixture_proxy'):
                        part.hide_set(True)
            if room.get('shot_reader_observed_projection'):
                isolate_preview_walls(scene, camera)
                fixtures = isolate_preview_fixtures(scene, camera, room['shot_reader_observed_projection'])
                if fixtures:
                    info['fixture_plate_coverage'] = fixtures
                    room['shot_reader_room_report'] = json.dumps(info)
                    if 'fixture_geometry' in info and (path / 'fixture-groups.npy').is_file():
                        separate_fixture_geometry(scene, room, path, info['fixture_geometry'])
            status = f"Clean set · {info['clean_set']['newly_visible_pixels']:,} newly visible pixels"
    except (OSError, RuntimeError, ValueError, ReferenceError, KeyError) as error:
        status = str(error)
    finally:
        temp.cleanup()
        job = None
        refresh()
    return None


def stop_job():
    global job, status
    if job:
        process, temp, *_ = job
        if process.poll() is None:
            process.kill()
            process.wait()
        temp.cleanup()
        job = None
    if bpy.app.timers.is_registered(finish_clean_room):
        bpy.app.timers.unregister(finish_clean_room)
    status = ''


class ShotReaderBuildCleanRoom(bpy.types.Operator):
    bl_idname = "shot_reader.build_clean_room"
    bl_label = "Reconstruct Clean Visible Set"
    bl_description = "Recover static surfaces revealed as the foreground performer moves in a locked shot"

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and ShotReaderBuildRoom.poll(context)

    def execute(self, context):
        from . import ADDON, reference_background, start_worker, worker_environment

        global job, status
        scene, settings = context.scene, context.scene.shot_reader
        prefs = context.preferences.addons[ADDON].preferences
        reference = reference_background(scene).clip
        camera, state = scene.camera, scene.shot_reader_room
        frame = scene.frame_current
        rw, rh = scene.render.resolution_x, scene.render.resolution_y
        if (not reference.size[1] or abs(reference.size[0] / reference.size[1] - rw / rh) > .005
                or scene.render.pixel_aspect_x != scene.render.pixel_aspect_y):
            self.report({'ERROR'}, 'Match the render aspect ratio to the source and use square pixels.')
            return {'CANCELLED'}
        seed = settings.foreground_region_frame
        x0, y0, x1, y1 = settings.foreground_region
        u, v = settings.body_direct_head
        if not (x0 < u < x1 and y0 < v < y1 and seed == settings.body_anchor_frame
                and scene.frame_start <= seed <= scene.frame_end and scene.frame_end - scene.frame_start >= 6):
            self.report({'ERROR'}, 'Pin the head and frame the foreground performer on the same frame within the scene range.')
            return {'CANCELLED'}
        weights, model = (Path(bpy.path.abspath(getattr(prefs, name))) for name in ('matte_weights', 'lens_model'))
        if not weights.is_file() or not model.is_file():
            self.report({'ERROR'}, 'Set the foreground matte and MoGe-2 checkpoints in Shot Reader preferences.')
            return {'CANCELLED'}
        fps = scene.render.fps / scene.render.fps_base
        clip_start = reference.frame_start - reference.frame_offset
        duration = scene.frame_end - scene.frame_start + 1
        frames = sorted({*range(scene.frame_start, scene.frame_end + 1, max(1, (duration + 16) // 17)),
                         scene.frame_end, frame})
        world = camera.matrix_world.normalized()
        temp = tempfile.TemporaryDirectory(prefix='shot-reader-clean-set-')
        try:
            masks = json.loads(state.clean_exclusions.as_string()) if state.clean_exclusions else []
            if not isinstance(masks, list):
                raise ValueError('Fixed exclusions must be a JSON array of polygons.')
            mirror = settings.body_mirror
            wall = None
            if mirror:
                from .mirror_refine import mirror_basis
                from .reflected_room import mirror_polygon

                rim = mirror_polygon(scene, mirror, required=False)
                if rim:
                    masks.append(rim)
                    normal, distance = mirror_basis(world.inverted() @ mirror.matrix_world,
                                                    min(point[1] for point in mirror.bound_box))
                    if distance < -.05:
                        wall = {'rim': rim, 'normal': list(normal), 'distance': distance - .04}
            fx, fy, cx, cy = camera_intrinsics(scene, camera.data)
            request = {'video': bpy.path.abspath(reference.filepath, library=reference.library), 'model': str(model),
                       'weights': str(weights), 'size': [rw, rh], 'intrinsics': [fx / rw, fy / rh, cx / rw, cy / rh],
                       'exclude': masks, 'wall_anchor': wall, 'time': (frame - clip_start) / fps,
                       'times': [(item - clip_start) / fps for item in frames],
                       'selection': {'time': (seed - clip_start) / fps, 'head': [u, v], 'region': [x0, y0, x1, y1]}}
            if wall:
                request['fixtures'] = fixture_observations(scene, camera, world, wall)
                if request['fixtures']:
                    request['fixtures']['intrinsics'] = request['intrinsics']
            (Path(temp.name) / 'request.json').write_text(json.dumps(request), encoding='utf-8')
            command = [bpy.path.abspath(prefs.body_python), str(Path(__file__).with_name('room_clean.py')), temp.name]
            process = start_worker(command, worker_environment(prefs), temp)
        except (ValueError, RuntimeError, OSError) as error:
            temp.cleanup()
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        job = process, temp, scene, camera, world, frame
        status = 'Reconstructing clean visible set…'
        bpy.app.timers.register(finish_clean_room, first_interval=.35)
        return {'FINISHED'}


class ShotReaderBuildParallaxRoom(bpy.types.Operator):
    bl_idname = 'shot_reader.build_parallax_room'
    bl_label = 'Reconstruct Set from Camera Movement'
    bl_description = 'Recover visible set depth from a tracked or calibrated translating camera'

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and ShotReaderBuildRoom.poll(context)

    def execute(self, context):
        from . import ADDON, reference_background, start_worker, worker_environment

        global job, status
        import numpy as np

        scene, camera = context.scene, context.scene.camera
        clip = reference_background(scene).clip
        prefs = context.preferences.addons[ADDON].preferences
        frame, subframe = scene.frame_current, scene.frame_subframe
        rw, rh = scene.render.resolution_x, scene.render.resolution_y
        if (not scene.frame_start <= frame <= scene.frame_end or scene.frame_end - scene.frame_start < 6
                or not clip.size[1] or abs(clip.size[0] / clip.size[1] - rw / rh) > .005
                or scene.render.pixel_aspect_x != scene.render.pixel_aspect_y):
            self.report({'ERROR'}, 'Place the reference frame inside the shot and match the plate aspect ratio with square pixels.')
            return {'CANCELLED'}
        state = scene.shot_reader_room
        scale = min(1., 768 / max(rw, rh))
        width, height = round(rw * scale), round(rh * scale)
        frames = sorted({frame, *map(int, np.linspace(scene.frame_start, scene.frame_end, min(19, scene.frame_end - scene.frame_start + 1)))})
        temp = tempfile.TemporaryDirectory(prefix='shot-reader-parallax-')
        try:
            scene.frame_set(frame)
            reference = camera.matrix_world.normalized()
            masks = json.loads(state.clean_exclusions.as_string()) if state.clean_exclusions else []
            if not isinstance(masks, list):
                raise ValueError('Fixed exclusions must be a JSON array of polygons.')
            from .reflected_room import mirror_polygon

            settings = scene.shot_reader
            reference_masks = json.loads(state.exclusions.as_string()) if state.exclusions else []
            if not isinstance(reference_masks, list):
                raise ValueError('Reference exclusions must be a JSON array of polygons.')
            depths = []
            if state.room:
                matrix = reference.inverted() @ state.room.matrix_world
                vertices = state.room.data.vertices
                depths = [-(matrix @ vertices[index].co).z for index in range(0, len(vertices), 10)]
                depths = [depth for depth in depths if depth > .05]
            hint = (np.percentile(depths, [4, 96]) * [.48, 1.9]).tolist() if len(depths) > 15 else None
            fps = scene.render.fps / scene.render.fps_base
            origin = clip.frame_start - clip.frame_offset
            poses, lenses, exclusions = [], [], []
            for item in frames:
                scene.frame_set(item)
                relative = camera.matrix_world.normalized().inverted() @ reference
                poses.append([list(row) for row in relative])
                fx, fy, cx, cy = camera_intrinsics(scene, camera.data)
                lenses.append([fx * width / rw, fy * height / rh, cx * width / rw, cy * height / rh])
                mirror = mirror_polygon(scene, settings.body_mirror, required=False) if settings.body_mirror else None
                polygons = masks + ([mirror] if mirror else [])
                exclusions.append(polygons + reference_masks if item == frame else polygons)
            request = {'clip': bpy.path.abspath(clip.filepath, library=clip.library), 'frame': frame,
                       'size': [width, height], 'times': [(item - origin) / fps for item in frames], 'seed': frames.index(frame),
                       'relative': poses, 'lenses': lenses, 'exclusions': exclusions, 'depth_hint': hint}
            (Path(temp.name) / 'request.json').write_text(json.dumps(request), encoding='utf-8')
            command = [bpy.path.abspath(prefs.python), str(Path(__file__).with_name('room_parallax.py')), temp.name]
            process = start_worker(command, worker_environment(prefs), temp)
        except (ValueError, RuntimeError, OSError) as error:
            temp.cleanup()
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        finally:
            scene.frame_set(frame, subframe=subframe)
        job = process, temp, scene, camera, reference, frame
        status = 'Measuring room depth from parallax…'
        bpy.app.timers.register(finish_clean_room, first_interval=.35)
        return {'FINISHED'}


class SHOT_READER_PT_room(bpy.types.Panel):
    bl_label = "Room Depth"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_parent_id = "SHOT_READER_PT_panel"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        from . import reflected_room

        state = context.scene.shot_reader_room
        layout = self.layout
        layout.prop(state, "exclusions")
        layout.prop(state, "use_scale_anchors")
        layout.operator("shot_reader.build_room", icon="MESH_GRID")
        layout.operator('shot_reader.build_parallax_room', icon='TRACKER')
        clean = layout.box()
        clean.prop(state, "clean_exclusions")
        row = clean.row(align=True)
        row.operator("shot_reader.pick_body_region", text="Pin Head", icon="PIVOT_CURSOR").pin_head = True
        row.operator("shot_reader.pick_body_region", text="Frame Foreground", icon="SELECT_SET").track_foreground = True
        clean.operator("shot_reader.build_clean_room", icon="IMAGE_DATA")
        if status:
            clean.label(text=status)
        if state.room:
            layout.prop(state.room, "display_type", text="Viewport")
            layout.prop(state.room, "hide_render", text="Hide in Render")
            projection = state.room.get('shot_reader_observed_projection')
            if projection:
                layout.prop(projection, 'visible_camera', text='Fill Set Gaps from Plate')
                if any(obj.get('shot_reader_preview_wall_rays') for obj in context.scene.objects):
                    layout.prop(state, 'show_preview_walls')
                if any(obj.get('shot_reader_preview_fixture_ray') for obj in context.scene.objects):
                    layout.prop(state, 'show_preview_fixtures')
            estimate = state.room.get('shot_reader_estimated_projection')
            if estimate:
                layout.prop(estimate, 'visible_camera', text='Show Estimated Set Fill')
        if context.scene.shot_reader.body_mirror:
            wall = layout.box()
            wall.prop(state, 'reflected_depth')
            wall.operator('shot_reader.build_reflected_room', icon='MESH_PLANE')
            if state.reflected_wall:
                wall.prop(state.reflected_wall, 'hide_render', text='Hide Reflected Wall in Render')
            if reflected_room.status:
                wall.label(text=reflected_room.status)


CLASSES = (ShotReaderRoomSettings, ShotReaderBuildRoom, ShotReaderBuildCleanRoom, ShotReaderBuildParallaxRoom,
           SHOT_READER_PT_room)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.shot_reader_room = bpy.props.PointerProperty(type=ShotReaderRoomSettings)


def unregister():
    del bpy.types.Scene.shot_reader_room
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
