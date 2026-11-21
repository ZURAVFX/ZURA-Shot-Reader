"""Capture plate colour on the editable skinned performer."""

import json
from pathlib import Path
import tempfile

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
import numpy as np

from .calibration import camera_intrinsics
from .mirror_refine import mirror_basis, reflection


MODIFIER = 'Shot Reader Plate Colour'
job = None
status = ''


def performer(context):
    from . import selected_character

    rig = selected_character(context)
    if not rig or not rig.get('shot_reader_alignment') or 'bone_113' not in rig.pose.bones:
        return None, None
    mesh = next((obj for obj in rig.children_recursive if obj.type == 'MESH' and obj.vertex_groups
                 and not obj.get('shot_reader_hair_report')), None)
    return rig, mesh


def positions(vertices, prop):
    values = np.empty(len(vertices) * 3, dtype=np.float32)
    vertices.foreach_get(prop, values)
    return values.reshape(-1, 3)


def plate_views(scene, rig, time):
    settings, camera = scene.shot_reader, scene.camera
    mirror = settings.body_mirror if rig['shot_reader_alignment'].startswith('Mirror') else None
    head = rig.matrix_world @ (rig.pose.bones['bone_113'].head + Vector((0., .01, .035)))
    if mirror:
        front = min(vertex[1] for vertex in mirror.bound_box)
        head = reflection(*mirror_basis(mirror.matrix_world, front)) @ head
    local = camera.matrix_world.inverted() @ head
    fx, fy, cx, cy = camera_intrinsics(scene, camera.data)
    if local.z >= -.01:
        raise ValueError('The fitted head is outside the plate camera.')
    pin = [(fx * local.x / -local.z + cx) / scene.render.resolution_x,
           (cy - fy * local.y / -local.z) / scene.render.resolution_y]
    x0, y0, x1, y1 = settings.body_region
    if not (x0 < pin[0] < x1 and y0 < pin[1] < y1):
        raise ValueError('Frame the fitted performer in the plate view, including the visible head.')
    captured = {'time': time, 'head': pin, 'region': [x0, y0, x1, y1]}
    if not mirror:
        return [captured]
    u, v = settings.body_direct_head
    u0, v0, u1, v1 = settings.foreground_region
    if not (u0 < u < u1 and v0 < v < v1 and settings.foreground_region_frame == scene.frame_current):
        raise ValueError('Pin the direct-view head and frame the foreground performer on the same frame.')
    return [{'time': time, 'head': [u, v], 'region': [u0, v0, u1, v1]}, captured]


def export_samples(context, rig, body, request, frames, seed, directory):
    scene, camera = context.scene, context.scene.camera
    settings = scene.shot_reader
    mirror = settings.body_mirror if rig.get('shot_reader_alignment') == 'Mirror camera estimate' else None
    if 'Mirror' in rig['shot_reader_alignment'] and not mirror:
        raise ValueError('Select the reflective surface used for this fit.')
    width, height = request['size']
    rw, rh = scene.render.resolution_x, scene.render.resolution_y
    front = min(vertex[1] for vertex in mirror.bound_box) if mirror else 0.
    count = len(body.data.vertices)
    body.data.calc_loop_triangles()
    triangles = [tuple(face.vertices) for face in body.data.loop_triangles]
    edges = np.array([tuple(edge.vertices) for edge in body.data.edges])
    projections, confidence, hair_blocks = [], [], []
    volume = rig.get('shot_reader_hair')
    hair_meshes = []
    if volume and not volume.hide_render:
        for hair in (volume, volume.get('shot_reader_draped_hair')):
            if hair and not hair.hide_render:
                hair.data.calc_loop_triangles()
                hair_meshes.append((hair, np.array([tuple(face.vertices) for face in hair.data.loop_triangles])))
    old_frame, subframe = scene.frame_current, scene.frame_subframe
    restore = bool(volume and hair_meshes and volume.hide_viewport)
    try:
        if restore:
            volume.hide_viewport = False
        scene.frame_set(seed)
        request['views'] = plate_views(scene, rig, request['seed_time'])

        # The geometry and topology are exported before the colour modifier is added.
        depsgraph = context.evaluated_depsgraph_get()
        for frame in frames:
            scene.frame_set(frame)
            mesh = body.evaluated_get(depsgraph)
            if len(mesh.data.vertices) != count:
                raise ValueError('Disable topology-changing modifiers before baking plate colour.')
            world = np.asarray(mesh.matrix_world)
            points = positions(mesh.data.vertices, 'co') @ world[:3, :3].T + world[:3, 3]
            normals = positions(mesh.data.vertices, 'normal') @ np.linalg.inv(world[:3, :3])
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-6)
            tree = BVHTree.FromPolygons(points.tolist(), triangles, all_triangles=True)
            hair_points, hair_faces, start = [], [], 0
            for hair, triangles in hair_meshes:
                evaluated = hair.evaluated_get(depsgraph)
                matrix = np.asarray(evaluated.matrix_world)
                coords = positions(evaluated.data.vertices, 'co') @ matrix[:3, :3].T + matrix[:3, 3]
                if len(coords) == len(hair.data.vertices):
                    hair_points.append(coords)
                    hair_faces.append(triangles + start)
                    start += len(coords)
            hair_tree = (BVHTree.FromPolygons(np.concatenate(hair_points).tolist(), np.concatenate(hair_faces).tolist(),
                                              all_triangles=True) if hair_points else None)
            camera_world = camera.matrix_world.normalized()
            project = np.array(camera_world.inverted())
            fx, fy, cx, cy = np.asarray(camera_intrinsics(scene, camera.data)) * [width / rw, height / rh, width / rw, height / rh]
            paths = [np.eye(4)]
            if mirror:
                paths.append(np.asarray(reflection(*mirror_basis(mirror.matrix_world, front))))
            uv, weights, blocked = [], [], []
            for path in paths:
                view = project @ path
                local = points @ view[:3, :3].T + view[:3, 3]
                depth = -local[:, 2]
                pixels = local[:, :2] / np.maximum(depth[:, None], .01) * [fx, -fy] + [cx, cy]
                observer = path[:3, :3] @ np.array(camera_world.translation) + path[:3, 3]
                direction = points - observer
                lengths = np.maximum(np.linalg.norm(direction, axis=1), 1e-6)
                direction /= lengths[:, None]
                facing = -(normals * direction).sum(axis=1)
                visible = ((pixels >= [1, 1]) & (pixels < [width - 1, height - 1])).all(axis=1)
                eligible = np.flatnonzero((facing > .22) & visible & (depth > .015))
                quality = np.zeros(count, dtype=np.float32)
                occluded = np.zeros(count, dtype=bool)
                origin = Vector(observer)
                tolerance = .004 * np.linalg.norm(world[:3, 0])
                for index in eligible:
                    ray = Vector(direction[index])
                    limit = float(lengths[index] - tolerance)
                    hit = tree.ray_cast(origin, ray, limit)[0]
                    if hit is None:
                        quality[index] = facing[index] ** 2
                        if hair_tree:
                            occluded[index] = hair_tree.ray_cast(origin, ray, limit)[0] is not None
                uv.append(pixels.astype(np.float32))
                weights.append(quality)
                blocked.append(occluded)
            projections.append(uv)
            confidence.append(weights)
            hair_blocks.append(blocked)
    finally:
        if restore:
            volume.hide_viewport = True
        scene.frame_set(old_frame, subframe=subframe)
    np.savez(directory / 'surface.npz', uv=np.asarray(projections).swapaxes(0, 1),
             weights=np.asarray(confidence).swapaxes(0, 1), edges=edges, times=request['times'],
             rest=positions(body.data.vertices, 'co'), hair_occluded=np.asarray(hair_blocks).swapaxes(0, 1))


def set_colour(body, colour, confidence):
    mesh = body.data
    srgb = np.clip(colour, 0., 1.)
    linear = np.where(srgb <= .04045, srgb / 12.92, ((srgb + .055) / 1.055) ** 2.4)
    values = np.column_stack((linear, np.ones(len(colour)))).astype(np.float32)
    attribute = mesh.color_attributes.get('Plate Colour') or mesh.color_attributes.new(name='Plate Colour', type='FLOAT_COLOR', domain='POINT')
    attribute.data.foreach_set('color', values.ravel())
    observed = mesh.attributes.get('Plate Colour Confidence') or mesh.attributes.new('Plate Colour Confidence', 'FLOAT', 'POINT')
    observed.data.foreach_set('value', confidence.astype(np.float32))
    if MODIFIER in body.modifiers:
        return

    material = bpy.data.materials.new('Plate Colour')
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    base = nodes.get('Principled BSDF')
    base.inputs['Roughness'].default_value = .8
    vertex = nodes.new('ShaderNodeVertexColor')
    vertex.layer_name = attribute.name
    light = nodes.new('ShaderNodeEmission')
    ray = nodes.new('ShaderNodeLightPath')
    seen = nodes.new('ShaderNodeMath')
    seen.operation = 'MAXIMUM'
    mix = nodes.new('ShaderNodeMixShader')
    links.new(vertex.outputs['Color'], base.inputs['Base Color'])
    links.new(vertex.outputs['Color'], light.inputs['Color'])
    links.new(ray.outputs['Is Camera Ray'], seen.inputs[0])
    links.new(ray.outputs['Is Glossy Ray'], seen.inputs[1])
    links.new(seen.outputs[0], mix.inputs[0])
    links.new(base.outputs[0], mix.inputs[1])
    links.new(light.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], nodes.get('Material Output').inputs['Surface'])
    material.cycles.emission_sampling = 'NONE'

    layer = body.modifiers.new(MODIFIER, 'NODES')
    group = bpy.data.node_groups.new(MODIFIER, 'GeometryNodeTree')
    layer.node_group = group
    for direction in ('INPUT', 'OUTPUT'):
        group.interface.new_socket(name='Geometry', in_out=direction, socket_type='NodeSocketGeometry')
    incoming = group.nodes.new('NodeGroupInput')
    assign = group.nodes.new('GeometryNodeSetMaterial')
    assign.inputs['Material'].default_value = material
    outgoing = group.nodes.new('NodeGroupOutput')
    group.links.new(incoming.outputs[0], assign.inputs['Geometry'])
    group.links.new(assign.outputs['Geometry'], outgoing.inputs[0])


def finish():
    global job, status
    from . import refresh

    process, temp, body, mesh = job
    if process.poll() is None:
        return .4
    try:
        directory = Path(temp.name)
        if process.returncode:
            log = (directory / 'run.log').read_text(encoding='utf-8', errors='replace').strip()
            raise ValueError(log.splitlines()[-1].removeprefix('ValueError: ') if log else 'Plate colour bake failed.')
        report = json.loads((directory / 'colour.json').read_text(encoding='utf-8'))
        with np.load(directory / 'colour.npz', allow_pickle=False) as result:
            if body.data != mesh or len(result['colour']) != len(body.data.vertices):
                raise ValueError('The performer mesh changed during the bake.')
            if body.data.users > 1:
                body.data = body.data.copy()
            set_colour(body, result['colour'], result['confidence'])
        body['shot_reader_colour_report'] = json.dumps(report)
        status = f"Plate colour · {report['observed_vertices']:,} observed vertices"
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
    if bpy.app.timers.is_registered(finish):
        bpy.app.timers.unregister(finish)
    status = ''


class ShotReaderBakePerformerColour(bpy.types.Operator):
    bl_idname = 'shot_reader.bake_performer_colour'
    bl_label = 'Bake Performer Colour from Plate'
    bl_description = 'Apply colours observed in the linked plate to the fitted skin, keeping the original material editable'

    @classmethod
    def poll(cls, context):
        from . import reference_background

        scene = context.scene
        return bool(job is None and context.mode == 'OBJECT' and scene.camera and scene.camera.data.type == 'PERSP'
                    and reference_background(scene) and performer(context)[1])

    def execute(self, context):
        from . import ADDON, reference_background, start_worker, worker_environment

        global job, status
        scene = context.scene
        rig, body = performer(context)
        prefs = context.preferences.addons[ADDON].preferences
        weights = Path(bpy.path.abspath(prefs.matte_weights))
        clip = reference_background(scene).clip
        rw, rh = scene.render.resolution_x, scene.render.resolution_y
        if not (weights.is_file() and clip.size[1] and abs(clip.size[0] / clip.size[1] - rw / rh) < .005
                and scene.render.pixel_aspect_x == scene.render.pixel_aspect_y):
            self.report({'ERROR'}, 'Set the foreground matte checkpoint and match the scene aspect ratio to the plate.')
            return {'CANCELLED'}
        duration = scene.frame_end - scene.frame_start + 1
        if duration < 5:
            self.report({'ERROR'}, 'Include at least five frames so the bake can check separate observations.')
            return {'CANCELLED'}
        seed = scene.shot_reader.body_anchor_frame if rig['shot_reader_alignment'].startswith('Mirror') else scene.frame_current
        if not scene.frame_start <= seed <= scene.frame_end:
            self.report({'ERROR'}, 'Set the pin or current frame inside the scene range.')
            return {'CANCELLED'}
        step = max(1, (duration + 16) // 17)
        frames = sorted({*range(scene.frame_start, scene.frame_end + 1, step), seed, scene.frame_end})
        fps = scene.render.fps / scene.render.fps_base
        origin = clip.frame_start - clip.frame_offset
        scale = min(1., 960 / max(rw, rh))
        request = {'clip': bpy.path.abspath(clip.filepath, library=clip.library), 'weights': str(weights),
                   'size': [round(rw * scale), round(rh * scale)], 'times': [(frame - origin) / fps for frame in frames],
                   'seed_time': (seed - origin) / fps}
        hair = rig.get('shot_reader_hair')
        model = Path(bpy.path.abspath(prefs.hair_weights))
        if hair and not hair.hide_render and model.is_file():
            request['hair_model'] = str(model)
        temp = tempfile.TemporaryDirectory(prefix='shot-reader-colour-')
        try:
            directory = Path(temp.name)
            export_samples(context, rig, body, request, frames, seed, directory)
            (directory / 'request.json').write_text(json.dumps(request), encoding='utf-8')
            process = start_worker([bpy.path.abspath(prefs.body_python), str(Path(__file__).with_name('performer_colour.py')),
                                    temp.name], worker_environment(prefs), temp)
        except (ValueError, RuntimeError, OSError) as error:
            temp.cleanup()
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        job = process, temp, body, body.data
        status = 'Baking performer colour…'
        bpy.app.timers.register(finish, first_interval=.4)
        return {'FINISHED'}
