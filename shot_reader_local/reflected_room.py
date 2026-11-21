"""Place recovered mirror background colours on a separate set surface."""

import json
from pathlib import Path
import tempfile

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

from .calibration import camera_intrinsics
from .colour_bake import plate_views
from .mirror_refine import fitted_rig, mirror_basis, reflection


job = None
status = ''


def mirror_polygon(scene, mirror, *, required=True):
    low, high = tuple(map(min, zip(*mirror.bound_box))), tuple(map(max, zip(*mirror.bound_box)))
    front = low[1]
    face = next((face for face in mirror.data.polygons if len(face.vertices) == 4 and
                 all(abs(mirror.data.vertices[i].co.y - front) < .00002 for i in face.vertices)), None)
    corners = ([mirror.data.vertices[i].co for i in face.vertices] if face else
               [Vector((x, front, z)) for x, z in ((low[0], high[2]), (high[0], high[2]), (high[0], low[2]), (low[0], low[2]))])
    image = [world_to_camera_view(scene, scene.camera, mirror.matrix_world @ coord) for coord in corners]
    if any(point.z <= 0 for point in image):
        if required:
            raise ValueError('The mirror must be in front of the camera.')
        return None
    return [[min(1., max(0., p.x)), min(1., max(0., 1. - p.y))] for p in image]


def add_background(scene, projection, directory, report):
    camera_world, normalized_lens, normal, plane_distance, offset = projection
    transform = reflection(normal, plane_distance)
    origin = camera_world.translation
    virtual = transform @ origin
    target = normal.dot(origin) + offset
    fx, fy, cx, cy = normalized_lens
    x0, y0, x1, y1 = report['uv_bounds']
    vertices, coordinates = [], []
    steps = 20
    for row in range(steps + 1):
        for column in range(steps + 1):
            s, t = column / steps, row / steps
            coordinates.append((s, t))
    for s, t in coordinates:
        u, v = x0 + (x1 - x0) * s, y1 - (y1 - y0) * t
        ray = camera_world.to_3x3() @ Vector(((u - cx) / fx, (cy - v) / fy, -1.))
        outgoing = transform.to_3x3() @ ray
        travel = (target - normal.dot(virtual)) / normal.dot(outgoing)
        if travel <= 0:
            raise ValueError('The wall depth does not intersect the reflection behind the performer.')
        vertices.append(virtual + travel * outgoing)
    # Use a local origin so the artist can position the wall with normal object transforms.
    centre = sum(vertices, Vector()) / len(vertices)
    mesh = bpy.data.meshes.new('Reflected Set Projection')
    faces = []
    stride = steps + 1
    for row in range(steps):
        for column in range(steps):
            start = row * stride + column
            faces.append((start, start + 1, start + stride + 1, start + stride))
    mesh.from_pydata([point - centre for point in vertices], [], faces)
    mesh.update()
    uv = mesh.uv_layers.new(name='Reflection Plate')
    uv.uv.foreach_set('vector', [value for face in faces for index in face for value in coordinates[index]])
    obj = bpy.data.objects.new('Reflected Set Wall', mesh)
    scene.collection.objects.link(obj)
    obj.location = centre
    obj.visible_shadow = obj.visible_diffuse = obj.visible_volume_scatter = False
    material = bpy.data.materials.new('Reflected Set Plate')
    material.use_nodes = True
    material.cycles.emission_sampling = 'NONE'
    nodes = material.node_tree.nodes
    nodes.clear()
    source = nodes.new('ShaderNodeTexImage')
    source.image = bpy.data.images.load(str(directory / 'wall.png'), check_existing=False)
    source.image.name = 'Reflected Set Colour'
    source.image.pack()
    evidence = nodes.new('ShaderNodeTexImage')
    evidence.image = bpy.data.images.load(str(directory / 'observed.png'), check_existing=False)
    evidence.image.name = 'Reflected Set Observed Mask'
    evidence.label = 'Observed in source frames'
    evidence.image.pack()
    output = nodes.new('ShaderNodeOutputMaterial')
    glow = nodes.new('ShaderNodeEmission')
    material.node_tree.links.new(source.outputs['Color'], glow.inputs['Color'])
    material.node_tree.links.new(glow.outputs[0], output.inputs['Surface'])
    mesh.materials.append(material)
    report['wall_offset_behind_camera_units'] = offset
    obj['shot_reader_reflected_set'] = json.dumps(report)
    state = scene.shot_reader_room
    if state.reflected_wall:
        state.reflected_wall.hide_render = True
    state.reflected_wall = obj
    for light in scene.objects:
        if light.type != 'LIGHT':
            continue
        generated = light.get('shot_reader_preview_light') or (
            light.name == light.data.name == 'Soft Light' and light.data.type == 'AREA'
            and light.data.shape == 'DISK' and light.data.energy == 220 and light.data.size == 4
        )
        if generated:
            light.visible_glossy = False
    return obj


def finish():
    global job, status
    from . import refresh

    process, temp, scene, camera, projection = job
    if process.poll() is None:
        return .4
    try:
        directory = Path(temp.name)
        if process.returncode:
            log = (directory / 'run.log').read_text(encoding='utf-8', errors='replace').strip()
            raise ValueError(log.splitlines()[-1].removeprefix('ValueError: ') if log else 'Reflected set reconstruction failed.')
        if scene.camera != camera:
            raise ValueError('The camera changed. Run the reflected set reconstruction again.')
        report = json.loads((directory / 'reflection.json').read_text(encoding='utf-8'))
        add_background(scene, projection, directory, report)
        status = f"Reflected set · {report['observed_fraction']:.0%} observed"
    except (OSError, ValueError, RuntimeError, ReferenceError, KeyError) as error:
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
    if bpy.app.timers.is_registered(finish):
        bpy.app.timers.unregister(finish)
    status = ''


class ShotReaderBuildReflectedRoom(bpy.types.Operator):
    bl_idname = 'shot_reader.build_reflected_room'
    bl_label = 'Reconstruct Reflected Set'
    bl_description = 'Recover the locked mirror background and project it on an editable wall behind the camera'

    @classmethod
    def poll(cls, context):
        from . import reference_background

        scene = context.scene
        return bool(job is None and context.mode == 'OBJECT' and scene.camera and scene.camera.data.type == 'PERSP'
                    and scene.shot_reader.body_mirror and reference_background(scene) and fitted_rig(context))

    def execute(self, context):
        from . import ADDON, reference_background, start_worker, worker_environment

        global job, status
        scene, settings = context.scene, context.scene.shot_reader
        prefs = context.preferences.addons[ADDON].preferences
        clip = reference_background(scene).clip
        camera, mirror, rig = scene.camera, settings.body_mirror, fitted_rig(context)
        seed = settings.body_anchor_frame
        weights = Path(bpy.path.abspath(prefs.matte_weights))
        rw, rh = scene.render.resolution_x, scene.render.resolution_y
        if not (weights.is_file() and scene.frame_start <= seed <= scene.frame_end and
                scene.frame_end - scene.frame_start >= 6 and clip.size[1] and abs(clip.size[0] / clip.size[1] - rw / rh) < .005):
            self.report({'ERROR'}, 'Use a linked plate, fitted mirror performer and foreground matte checkpoint.')
            return {'CANCELLED'}
        old_frame, subframe = scene.frame_current, scene.frame_subframe
        temp = tempfile.TemporaryDirectory(prefix='shot-reader-reflected-set-')
        try:
            scene.frame_set(seed)
            fps = scene.render.fps / scene.render.fps_base
            origin = clip.frame_start - clip.frame_offset
            interval = max(1, (scene.frame_end - scene.frame_start + 17) // 17)
            frames = sorted({*range(scene.frame_start, scene.frame_end + 1, interval), seed, scene.frame_end})
            fx, fy, cx, cy = camera_intrinsics(scene, camera.data)
            front = min(point[1] for point in mirror.bound_box)
            projection = (camera.matrix_world.normalized(), [fx / rw, fy / rh, cx / rw, cy / rh],
                          *mirror_basis(mirror.matrix_world, front), scene.shot_reader_room.reflected_depth)
            request = {'clip': bpy.path.abspath(clip.filepath, library=clip.library), 'weights': str(weights),
                       'size': [rw, rh], 'times': [(frame - origin) / fps for frame in frames],
                       'views': plate_views(scene, rig, (seed - origin) / fps), 'mirror': mirror_polygon(scene, mirror)}
            (Path(temp.name) / 'request.json').write_text(json.dumps(request), encoding='utf-8')
            process = start_worker([bpy.path.abspath(prefs.body_python), str(Path(__file__).with_name('reflected_texture.py')),
                                    temp.name], worker_environment(prefs), temp)
        except (OSError, RuntimeError, ValueError) as error:
            temp.cleanup()
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        finally:
            scene.frame_set(old_frame, subframe=subframe)
        job = process, temp, scene, camera, projection
        status = 'Reconstructing reflected set…'
        bpy.app.timers.register(finish, first_interval=.4)
        return {'FINISHED'}
