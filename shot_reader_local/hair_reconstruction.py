"""Build editable hair geometry from the local plate and fitted rig."""

import json
from pathlib import Path
import tempfile

import bpy
from mathutils import Matrix
import numpy as np

from .calibration import camera_intrinsics
from .colour_bake import performer, plate_views, positions
from .mirror_refine import mirror_basis, reflection
from .reflected_room import mirror_polygon
from .silhouette_fit import body_modes


BONES = ('bone_113', 'bone_037', 'bone_035')
BODY_SHAPE = 'Shot Reader Plate Silhouette'
job = None
status = ''


def geometry(context, rig, body, mirror, seed, frames, requests, directory):
    scene, camera = context.scene, context.scene.camera
    old, subframe = scene.frame_current, scene.frame_subframe
    rw, rh = scene.render.resolution_x, scene.render.resolution_y
    width, height = requests['size']
    points, poses, cameras, lenses, apertures = [], [], [], [], []
    scratch = None
    extra = {}
    try:
        scene.frame_set(seed)
        requests['views'] = plate_views(scene, rig, requests['seed_time'])
        rel = np.asarray(rig.matrix_world.inverted() @ body.matrix_world)
        rest = positions(body.data.vertices, 'co') @ rel[:3, :3].T + rel[:3, 3]
        normals = positions(body.data.vertices, 'normal') @ np.linalg.inv(rel[:3, :3])
        normals /= np.maximum(1e-7, np.linalg.norm(normals, axis=1, keepdims=True))
        body.data.calc_loop_triangles()
        faces = np.array([tuple(tri.vertices) for tri in body.data.loop_triangles], dtype=np.int32)
        graph = context.evaluated_depsgraph_get()
        keys = body.data.shape_keys
        shaped = keys and keys.use_relative and BODY_SHAPE not in keys.key_blocks
        if shaped:
            torso, shoulder = (np.asarray(rig.data.bones[name].head_local) for name in (BONES[2], BONES[1]))
            arms = [[np.asarray(rig.data.bones[f'bone_{i:03}'].head_local) for i in pair]
                    for pair in ((39, 40), (75, 76))]
            modes = body_modes(rest, torso, shoulder, arms) @ np.linalg.inv(rel[:3, :3]).T
            scratch = body.shape_key_add(name='Shot Reader Shape Probe', from_mix=False)
            basis = positions(keys.reference_key.data, 'co')
            deltas = [[] for _ in modes]
        for frame in frames:
            scene.frame_set(frame)
            mesh = body.evaluated_get(graph)
            if len(mesh.data.vertices) != len(rest):
                raise ValueError('Pause topology-changing modifiers to reconstruct hair.')
            world = np.asarray(mesh.matrix_world)
            points.append(positions(mesh.data.vertices, 'co') @ world[:3, :3].T + world[:3, 3])
            if scratch:
                scratch.value = 1.
                for mode, target in zip(modes, deltas):
                    scratch.data.foreach_set('co', (basis + mode).astype(np.float32).ravel())
                    body.data.update()
                    context.view_layer.update()
                    posed = body.evaluated_get(graph)
                    target.append(positions(posed.data.vertices, 'co') @ world[:3, :3].T + world[:3, 3] - points[-1])
                scratch.value = 0.
                context.view_layer.update()
            poses.append([np.asarray(rig.matrix_world @ rig.pose.bones[name].matrix @ rig.data.bones[name].matrix_local.inverted())
                          for name in BONES])
            active = camera.matrix_world.inverted()
            pair = [np.asarray(active)]
            if mirror:
                front = min(point[1] for point in mirror.bound_box)
                pair.append(np.asarray(active @ reflection(*mirror_basis(mirror.matrix_world, front))))
                apertures.append(mirror_polygon(scene, mirror))
            cameras.append(pair)
            lenses.append(np.asarray(camera_intrinsics(scene, camera.data)) * [width / rw, height / rh, width / rw, height / rh])
        if scratch:
            extra = {'body_modes': np.asarray(deltas, dtype=np.float32), 'rest_modes': modes.astype(np.float32)}
    finally:
        if scratch:
            body.shape_key_remove(scratch)
        scene.frame_set(old, subframe=subframe)
    requests['mirror'] = apertures
    np.savez(directory / 'input.npz', body=points, hair_pose=poses, views=cameras, lenses=lenses, rest=rest,
             normals=normals, faces=faces, head=np.asarray(rig.data.bones[BONES[0]].head_local), **extra)


def add_body_shape(body, delta):
    basis = positions(body.data.shape_keys.reference_key.data, 'co')
    if delta.shape != basis.shape or not np.isfinite(delta).all():
        raise ValueError('The silhouette correction does not match the fitted mesh.')
    shape = body.shape_key_add(name=BODY_SHAPE, from_mix=False)
    shape.data.foreach_set('co', (basis + delta).astype(np.float32).ravel())
    shape.value = 1.
    body.data.update()


def add_hair(scene, rig, data, report):
    mesh = bpy.data.meshes.new('Plate Hair Volume')
    mesh.from_pydata(data['vertices'].tolist(), [], data['faces'].tolist())
    mesh.update()
    obj = bpy.data.objects.new('Plate Hair Volume', mesh)
    scene.collection.objects.link(obj)
    obj.parent = rig
    obj.matrix_parent_inverse = Matrix.Identity(4)
    weights = data['weights']
    for index, name in enumerate(BONES):
        group = obj.vertex_groups.new(name=name)
        for vertex, value in enumerate(weights[:, index]):
            if value > .0001:
                group.add([vertex], float(value), 'REPLACE')
    skin = obj.modifiers.new('Follow Performer', 'ARMATURE')
    skin.object = rig
    mesh.polygons.foreach_set('use_smooth', [True] * len(mesh.polygons))
    material = bpy.data.materials.new('Plate Hair Colour')
    material.use_nodes = True
    rgb = np.clip(data['colours'], 0., 1.)
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    values = np.column_stack((linear, np.ones(len(rgb)))).astype(np.float32)
    attribute = mesh.color_attributes.new(name='Observed Hair Colour', type='FLOAT_COLOR', domain='POINT')
    attribute.data.foreach_set('color', values.ravel())
    material.diffuse_color = (*np.median(linear, axis=0), 1.)
    nodes, links = material.node_tree.nodes, material.node_tree.links
    surface = nodes.new('ShaderNodeEmission')
    colour = nodes.new('ShaderNodeVertexColor')
    colour.layer_name = attribute.name
    links.new(colour.outputs['Color'], surface.inputs['Color'])
    links.new(surface.outputs[0], nodes.get('Material Output').inputs['Surface'])
    material.cycles.emission_sampling = 'NONE'
    mesh.materials.append(material)
    obj['shot_reader_hair_report'] = json.dumps(report)
    rig['shot_reader_hair'] = obj
    return obj


def feather_hair_edges(mesh, faces, source, original_material):
    """Fade only the open boundaries of the fitted hair patch."""
    edges = np.sort(faces[:, [[1, 2], [2, 0], [0, 1]]], axis=2)
    _, edge_ids, incidence = np.unique(edges.reshape(-1, 2), axis=0, return_inverse=True, return_counts=True)
    open_edge = incidence[edge_ids].reshape(-1, 3) == 1
    distance = source[faces]
    cross = np.cross(distance[:, 1] - distance[:, 0], distance[:, 2] - distance[:, 0])
    widths = np.linalg.norm(source[edges[..., 0]] - source[edges[..., 1]], axis=2)
    altitude = np.linalg.norm(cross, axis=1)[:, None] / np.maximum(widths, 1.e-6)
    for name, values in (
        ('Hair Triangle Coordinates', np.tile(np.eye(3), (len(faces), 1))),
        ('Hair Edge Scale', np.repeat(open_edge * altitude / .010, 3, axis=0)),
        ('Hair Shared Edges', np.repeat(~open_edge, 3, axis=0)),
    ):
        attribute = mesh.attributes.new(name=name, type='FLOAT_VECTOR', domain='CORNER')
        attribute.data.foreach_set('vector', values.astype(np.float32).ravel())

    material = original_material.copy()
    material.name = 'Draped Hair Colour'
    nodes, link = material.node_tree.nodes, material.node_tree.links.new
    attributes = []
    for name in ('Hair Triangle Coordinates', 'Hair Edge Scale', 'Hair Shared Edges'):
        node = nodes.new('ShaderNodeAttribute')
        node.attribute_name = name
        attributes.append(node.outputs['Vector'])
    multiply, add = (nodes.new('ShaderNodeVectorMath') for _ in range(2))
    multiply.operation, add.operation = 'MULTIPLY', 'ADD'
    link(attributes[0], multiply.inputs[0])
    link(attributes[1], multiply.inputs[1])
    link(multiply.outputs['Vector'], add.inputs[0])
    link(attributes[2], add.inputs[1])
    separate = nodes.new('ShaderNodeSeparateXYZ')
    link(add.outputs['Vector'], separate.inputs[0])
    first, second = (nodes.new('ShaderNodeMath') for _ in range(2))
    first.operation = second.operation = 'MINIMUM'
    link(separate.outputs['X'], first.inputs[0])
    link(separate.outputs['Y'], first.inputs[1])
    link(first.outputs[0], second.inputs[0])
    link(separate.outputs['Z'], second.inputs[1])
    mix = nodes.new('ShaderNodeMixShader')
    link(second.outputs[0], mix.inputs[0])
    link(nodes.new('ShaderNodeBsdfTransparent').outputs[0], mix.inputs[1])
    link(nodes.get('Emission').outputs[0], mix.inputs[2])
    link(mix.outputs[0], nodes.get('Material Output').inputs['Surface'])
    mesh.materials.append(material)


def add_draped_hair(scene, volume, body, faces, report, *, scalp=False):
    indices, compact = np.unique(faces, return_inverse=True)
    name = 'Editable Scalp Hair' if scalp else 'Draped Hair Surface'
    mesh = bpy.data.meshes.new(name)
    source = positions(body.data.vertices, 'co')
    mesh.from_pydata(source[indices].tolist(), [], compact.reshape(-1, 3).tolist())
    mesh.update()
    mesh.polygons.foreach_set('use_smooth', [True] * len(mesh.polygons))
    index = mesh.attributes.new(name='Body Vertex Index', type='INT', domain='POINT')
    index.data.foreach_set('value', indices.astype(np.int32))
    rgb = np.asarray(report['rgb'])
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    colour = mesh.color_attributes.new(name='Observed Hair Colour', type='FLOAT_COLOR', domain='POINT')
    colour.data.foreach_set('color', np.tile([*linear, 1.], len(indices)).astype(np.float32))
    if scalp:
        material = bpy.data.materials.new('Scalp Hair')
        material.diffuse_color = (*linear, 1.)
        material.use_nodes = True
        shader = material.node_tree.nodes.get('Principled BSDF')
        shader.inputs['Base Color'].default_value = material.diffuse_color
        shader.inputs['Roughness'].default_value = .68
        mesh.materials.append(material)
    else:
        feather_hair_edges(mesh, faces, source, volume.data.materials[0])
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    obj.parent = volume.parent if scalp else volume
    reference = 'shot_reader_scalp_hair' if scalp else 'shot_reader_draped_hair'
    volume[reference] = obj
    obj[reference] = True

    group = bpy.data.node_groups.new('Follow Performer Surface', 'GeometryNodeTree')
    modifier = obj.modifiers.new('Follow Performer Surface', 'NODES')
    modifier.node_group = group
    for direction in ('INPUT', 'OUTPUT'):
        group.interface.new_socket(name='Geometry', in_out=direction, socket_type='NodeSocketGeometry')
    nodes, link = group.nodes, group.links.new
    incoming = nodes.new('NodeGroupInput')
    outgoing = nodes.new('NodeGroupOutput')
    info = nodes.new('GeometryNodeObjectInfo')
    info.transform_space = 'RELATIVE'
    info.inputs['Object'].default_value = body
    index_node = nodes.new('GeometryNodeInputNamedAttribute')
    index_node.data_type = 'INT'
    index_node.inputs['Name'].default_value = index.name
    position = nodes.new('GeometryNodeSetPosition')
    link(incoming.outputs['Geometry'], position.inputs['Geometry'])
    for field, socket in (('Position', 'Position'), ('Normal', 'Offset')):
        sample = nodes.new('GeometryNodeSampleIndex')
        sample.data_type = 'FLOAT_VECTOR'
        link(info.outputs['Geometry'], sample.inputs['Geometry'])
        link(index_node.outputs['Attribute'], sample.inputs['Index'])
        link(nodes.new(f'GeometryNodeInput{field}').outputs[field], sample.inputs['Value'])
        source_output = sample.outputs['Value']
        if field == 'Normal':
            offset = nodes.new('ShaderNodeVectorMath')
            offset.operation = 'SCALE'
            offset.inputs['Scale'].default_value = .011 if scalp else .003
            link(source_output, offset.inputs[0])
            source_output = offset.outputs['Vector']
        link(source_output, position.inputs[socket])
    link(position.outputs['Geometry'], outgoing.inputs['Geometry'])
    if scalp:
        edge_smooth = obj.modifiers.new('Smooth Scalp Hairline', 'SUBSURF')
        edge_smooth.levels = 2
        edge_smooth.render_levels = 2
        thickness = obj.modifiers.new('Scalp Thickness', 'SOLIDIFY')
        thickness.thickness = .011
        obj['shot_reader_hair_coverage'] = 'Estimated crown and nape following the captured head'
        return obj
    for prop in ('hide_render', 'hide_viewport'):
        driver = obj.driver_add(prop).driver
        driver.type = 'AVERAGE'
        target = driver.variables.new().targets[0]
        target.id = volume
        target.data_path = prop


def add_scalp_hair(scene, rig, body, volume, report):
    if existing := volume.get('shot_reader_scalp_hair'):
        return existing
    rel = np.asarray(rig.matrix_world.inverted() @ body.matrix_world)
    rest = positions(body.data.vertices, 'co') @ rel[:3, :3].T + rel[:3, 3]
    head = np.asarray(rig.data.bones[BONES[0]].head_local)
    x, y, z = (rest - head).T
    # Put the hairline above the temples and forehead, tapering down the nape.
    hairline = np.interp(y, [-.09, -.048, .0, .035, .072], [.160, .136, .121, .075, .024])
    skin = (abs(x) < .095) & (z > hairline)
    body.data.calc_loop_triangles()
    faces = np.array([tuple(tri.vertices) for tri in body.data.loop_triangles], dtype=np.int32)
    faces = faces[skin[faces].all(axis=1)]
    if not len(faces):
        return None
    scalp = add_draped_hair(scene, volume, body, faces, report, scalp=True)
    volume.hide_viewport = True
    scalp.hide_render = True
    return scalp


class ShotReaderToggleViewportHair(bpy.types.Operator):
    bl_idname = 'shot_reader.toggle_viewport_hair'
    bl_label = 'Switch Viewport Hair'
    bl_description = 'Switch between the scalp-following estimate and the plate-fitted reconstruction in the viewport'
    bl_options = {'UNDO'}

    def execute(self, context):
        rig, body = performer(context)
        volume = rig.get('shot_reader_hair') if rig else None
        if not volume or not body:
            return {'CANCELLED'}
        scalp = volume.get('shot_reader_scalp_hair')
        if not scalp:
            scalp = add_scalp_hair(context.scene, rig, body, volume, json.loads(volume['shot_reader_hair_report']))
            return {'FINISHED'} if scalp else {'CANCELLED'}
        volume.hide_viewport = not volume.hide_viewport
        scalp.hide_viewport = not volume.hide_viewport
        return {'FINISHED'}


def add_hair_motion(obj, vertices, modes, path, frames, labels):
    if (modes.shape != (len(labels), len(vertices), 3) or path.shape != (len(frames), len(labels))
            or not np.isfinite(path).all() or not np.isfinite(modes).all()):
        raise ValueError('The fitted hair animation does not match its geometry and frame range.')
    obj.shape_key_add(name='Basis', from_mix=False)
    for index, label in enumerate(labels):
        key = obj.shape_key_add(name=label, from_mix=False)
        key.slider_min = -1.
        key.slider_max = 1.
        key.data.foreach_set('co', (vertices + modes[index]).astype(np.float32).ravel())
        for frame, coefficient in zip(frames, path[:, index]):
            key.value = float(coefficient)
            key.keyframe_insert(data_path='value', frame=frame, group='Plate Hair Motion')
    animation = obj.data.shape_keys.animation_data
    for layer in animation.action.layers:
        for strip in layer.strips:
            channelbag = strip.channelbag(animation.action_slot)
            if channelbag:
                for curve in channelbag.fcurves:
                    for point in curve.keyframe_points:
                        point.interpolation = 'LINEAR'


def finish():
    global job, status
    from . import refresh

    process, temp, scene, camera, rig, body, frames = job
    if process.poll() is None:
        return .4
    try:
        folder = Path(temp.name)
        if process.returncode:
            log = (folder / 'run.log').read_text(encoding='utf-8', errors='replace').strip()
            raise ValueError(log.splitlines()[-1].removeprefix('ValueError: ') if log else 'Local hair reconstruction failed.')
        if scene.camera != camera:
            raise ValueError('The camera changed during hair reconstruction.')
        report = json.loads((folder / 'hair.json').read_text(encoding='utf-8'))
        with np.load(folder / 'hair.npz', allow_pickle=False) as data:
            volume = add_hair(scene, rig, data, report)
            if 'hair_modes' in data:
                add_hair_motion(volume, data['vertices'], data['hair_modes'], data['hair_path'], frames,
                                report['hair_motion']['controls'])
            if 'body_delta' in data:
                add_body_shape(body, data['body_delta'])
            if 'shell_faces' in data:
                add_draped_hair(scene, volume, body, data['shell_faces'], report)
            add_scalp_hair(scene, rig, body, volume, report)
        scene.frame_set(scene.frame_current, subframe=scene.frame_subframe)
        status = f"Held-out silhouette · {report['silhouette_before_iou']:.1%} → {report['silhouette_after_iou']:.1%}"
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


class ShotReaderReconstructHair(bpy.types.Operator):
    bl_idname = 'shot_reader.reconstruct_hair'
    bl_label = 'Refine Silhouette and Hair from Plate'
    bl_description = 'Fit the visible torso and shoulders, and reconstruct hair on the fitted rig'

    @classmethod
    def poll(cls, context):
        from . import reference_background

        rig, mesh = performer(context)
        return bool(job is None and context.mode == 'OBJECT' and reference_background(context.scene) and mesh
                    and not rig.get('shot_reader_hair'))

    def execute(self, context):
        from . import ADDON, reference_background, start_worker, worker_environment

        global job, status
        scene = context.scene
        rig, body = performer(context)
        clip = reference_background(scene).clip
        prefs = context.preferences.addons[ADDON].preferences
        model = Path(bpy.path.abspath(prefs.hair_weights))
        mirror = scene.shot_reader.body_mirror if rig['shot_reader_alignment'].startswith('Mirror') else None
        duration = scene.frame_end - scene.frame_start + 1
        if not model.is_file() or duration < 8 or (rig['shot_reader_alignment'].startswith('Mirror') and not mirror):
            self.report({'ERROR'}, 'Set a local hair segmentation model and use at least eight fitted frames.')
            return {'CANCELLED'}
        seed = scene.shot_reader.body_anchor_frame if mirror else scene.frame_current
        fps = scene.render.fps / scene.render.fps_base
        origin = clip.frame_start - clip.frame_offset
        step = max(1, (duration + 16) // 17)
        frames = sorted({*range(scene.frame_start, scene.frame_end + 1, step), scene.frame_end})
        rw, rh = scene.render.resolution_x, scene.render.resolution_y
        scale = min(1., 768 / max(rw, rh))
        request = {'clip': bpy.path.abspath(clip.filepath, library=clip.library), 'model': str(model),
                   'size': [round(rw * scale), round(rh * scale)], 'times': [(frame - origin) / fps for frame in frames],
                   'seed_time': (seed - origin) / fps}
        temp = tempfile.TemporaryDirectory(prefix='shot-reader-hair-')
        try:
            path = Path(temp.name)
            geometry(context, rig, body, mirror, seed, frames, request, path)
            (path / 'request.json').write_text(json.dumps(request), encoding='utf-8')
            command = [bpy.path.abspath(prefs.body_python), str(Path(__file__).with_name('hair_volume.py')), temp.name]
            process = start_worker(command, worker_environment(prefs), temp)
        except (ValueError, RuntimeError, OSError, KeyError) as error:
            temp.cleanup()
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        job = process, temp, scene, scene.camera, rig, body, frames
        status = 'Reconstructing hair locally…'
        bpy.app.timers.register(finish, first_interval=.4)
        return {'FINISHED'}
