"""Reconstruct an editable prop when the plate supports a consistent handheld track."""

import json
from math import cos, pi, sin
from pathlib import Path
import tempfile

import bpy
from mathutils import Matrix, Vector
import numpy as np

from .arm_alignment import ARMS, projection
from .colour_bake import performer
from .handheld_grip import CONSTRAINT, fit_grip
from .reflected_room import mirror_polygon


job = None
status = ''
NO_ITEM = 'No consistent bright elongated handheld item was detected'
DEFAULT_RADII = [(-.50, .80), (-.46, 1.), (.17, .89), (.31, .72), (.35, .58),
                 (.44, .53), (.49, .44), (.50, .31)]
DEFAULT_PROFILE = [[length, -.5 * radius, .5 * radius] for length, radius in DEFAULT_RADII]


def capture_priors(context, rig, mirror, frames, width, height):
    scene = context.scene
    scale = np.array([width / scene.render.resolution_x, height / scene.render.resolution_y])
    priors, cameras, apertures = [], [], []
    frame, subframe = scene.frame_current, scene.frame_subframe
    graph = context.evaluated_depsgraph_get()
    try:
        for number in frames:
            scene.frame_set(number)
            record, transforms = [], []
            pose = rig.evaluated_get(graph).pose.bones
            for surface in ((None, mirror) if mirror else (None,)):
                view, lens = projection(scene, rig, surface)
                lens = np.asarray(lens) * np.tile(scale, 2)
                fx, fy, cx, cy = lens
                hands = {}
                for name, indexes in ARMS.items():
                    palm = sum((pose[f'bone_{i:03}'].head for i in indexes[3:]), Vector()) * .5
                    x, y, z = view @ palm
                    hands[name] = {'xy': [x * fx / max(-z, .01) + cx, cy - y * fy / max(-z, .01)], 'depth': -z}
                record.append(hands)
                transforms.append(((view @ rig.matrix_world.inverted()).inverted(), lens))
            priors.append(record)
            cameras.append(transforms)
            if mirror:
                apertures.append(mirror_polygon(scene, mirror))
    finally:
        scene.frame_set(frame, subframe=subframe)
    return priors, cameras, apertures


def create_prop(scene, report, frames, priors, cameras, directory):
    view, hand = report['view'], report['hand']
    transforms, lengths, diameters = [], [], []
    for index, (x, y, angle, length, diameter) in enumerate(report['motion']):
        inverse, (fx, fy, cx, cy) = cameras[index][view]
        depth = priors[index][view][hand]['depth']
        centre = inverse @ Vector(((x - cx) * depth / fx, (cy - y) * depth / fy, -depth))
        direction = inverse.to_3x3() @ Vector((cos(angle) / fx, -sin(angle) / fy, 0.))
        normal = inverse.to_3x3() @ Vector((0., 0., 1.))
        z_axis = direction.normalized()
        y_axis = normal.normalized()
        x_axis = y_axis.cross(z_axis).normalized()
        transforms.append((centre, Matrix((x_axis, y_axis, z_axis)).transposed().to_quaternion()))
        lengths.append(length * depth * direction.length)
        diameters.append(diameter * depth * ((sin(angle) / fx) ** 2 + (cos(angle) / fy) ** 2) ** .5)

    length, diameter = np.median(lengths), np.median(diameters)
    # The plate determines the silhouette and its motion. A tapered circular section supplies editable depth.
    rings = report.get('profile', DEFAULT_PROFILE)
    sign = 1. if view else -1.
    count = 24
    vertices = []
    for station, low, high in rings:
        radius = max(.015, (high - low) * .5) * diameter
        midpoint = (low + high) * .5 * diameter * sign
        vertices.extend((midpoint + radius * cos(2 * pi * i / count), radius * sin(2 * pi * i / count),
                         station * length) for i in range(count))
    faces = [tuple(reversed(range(count)))]
    for row in range(len(rings) - 1):
        for col in range(count):
            a, b = row * count + col, row * count + (col + 1) % count
            faces.append((a, b, b + count, a + count))
    faces.append(tuple(range((len(rings) - 1) * count, len(vertices))))
    mesh = bpy.data.meshes.new('Tracked Handheld Item')
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new('Tracked Handheld Item', mesh)
    scene.collection.objects.link(obj)
    obj.rotation_mode = 'QUATERNION'
    previous = transforms[0][1]
    for index, (frame, (centre, orientation)) in enumerate(zip(frames, transforms)):
        orientation.make_compatible(previous)
        obj.location, obj.rotation_quaternion = centre, orientation
        obj.scale.z = float(np.clip(lengths[index] / length, .82, 1.18))
        for channel in ('location', 'rotation_quaternion', 'scale'):
            obj.keyframe_insert(data_path=channel, frame=frame, group='Handheld Plate Track')
        previous = orientation

    material = bpy.data.materials.new('Handheld Plate Colour')
    material.use_nodes = True
    rgb = np.asarray(report['colour'])
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    material.diffuse_color = (*linear, 1.)
    nodes = material.node_tree.nodes
    nodes.clear()
    glow = nodes.new('ShaderNodeEmission')
    glow.inputs['Color'].default_value = (*linear, 1.)
    material.node_tree.links.new(glow.outputs[0], nodes.new('ShaderNodeOutputMaterial').inputs['Surface'])
    if 'appearance' in report:
        uv = [(station + .5, .5 - .5 * sign * cos(2 * pi * i / count))
              for station, _, _ in rings for i in range(count)]
        layer = mesh.uv_layers.new(name='Observed Tube Surface')
        layer.uv.foreach_set('vector', [value for face in faces for index in face for value in uv[index]])
        visible = mesh.attributes.new(name='Observed Tube Side', type='FLOAT', domain='POINT')
        visible.data.foreach_set('value', [max(0., min(1., 3. * sin(2 * pi * i / count)))
                                           for _ in rings for i in range(count)])
        image = nodes.new('ShaderNodeTexImage')
        image.image = bpy.data.images.load(str(directory / 'handheld-colour.png'), check_existing=False)
        image.image.name = 'Observed Tube Colour'
        image.image.pack()
        image.extension = 'EXTEND'
        mask = nodes.new('ShaderNodeAttribute')
        mask.attribute_name = visible.name
        blend = nodes.new('ShaderNodeMixRGB')
        blend.inputs[1].default_value = (*linear, 1.)
        links = material.node_tree.links
        links.new(mask.outputs['Fac'], blend.inputs[0])
        links.new(image.outputs['Color'], blend.inputs[2])
        links.new(blend.outputs['Color'], glow.inputs['Color'])
    material.cycles.emission_sampling = 'NONE'
    mesh.materials.append(material)
    obj['shot_reader_handheld_report'] = json.dumps({key: value for key, value in report.items() if key != 'motion'})
    scene.frame_set(scene.frame_current, subframe=scene.frame_subframe)
    return obj


def finish():
    global job, status
    from . import refresh

    process, temp, scene, camera, frames, priors, cameras, scale, rig, window = job
    if process.poll() is None:
        return .4
    try:
        if process.returncode:
            raise ValueError('The local handheld tracking pass failed.')
        if scene.camera != camera:
            raise ValueError('The active camera changed during handheld tracking.')
        report = json.loads((Path(temp.name) / 'handheld.json').read_text(encoding='utf-8'))
        status = NO_ITEM
        if report:
            report['median_projected_error_px'] /= scale
            prop = create_prop(scene, report, frames, priors, cameras, Path(temp.name))
            status = 'Handheld mesh added from the plate'
            with bpy.context.temp_override(window=window, scene=scene, view_layer=scene.view_layers[0]):
                grip = fit_grip(bpy.context, rig, prop, report['hand'], frames)
            if grip:
                status = 'Handheld mesh and inferred grip added'
    except (ValueError, RuntimeError, OSError, KeyError, ReferenceError) as error:
        status = str(error)
    finally:
        temp.cleanup()
        job = None
        refresh()


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


class ShotReaderTrackHandheld(bpy.types.Operator):
    bl_idname = 'shot_reader.track_handheld'
    bl_label = 'Track Handheld Item from Plate'
    bl_description = 'Fit and animate a bright elongated object supported by the linked footage'

    @classmethod
    def poll(cls, context):
        from . import reference_background

        return bool(job is None and context.mode == 'OBJECT' and reference_background(context.scene)
                    and performer(context)[0] and context.scene.frame_end - context.scene.frame_start >= 12
                    and not any(obj.get('shot_reader_handheld_report') for obj in context.scene.objects))

    def execute(self, context):
        from . import ADDON, reference_background, start_worker, worker_environment

        global job, status
        scene = context.scene
        rig, _ = performer(context)
        mirror = scene.shot_reader.body_mirror if rig['shot_reader_alignment'].startswith('Mirror') else None
        clip = reference_background(scene).clip
        prefs = context.preferences.addons[ADDON].preferences
        scale = min(1., 768 / max(scene.render.resolution_x, scene.render.resolution_y))
        width, height = round(scene.render.resolution_x * scale), round(scene.render.resolution_y * scale)
        step = max(1, (scene.frame_end - scene.frame_start) // 160)
        frames = sorted({*range(scene.frame_start, scene.frame_end + 1, step), scene.frame_end})
        fps = scene.render.fps / scene.render.fps_base
        origin = clip.frame_start - clip.frame_offset
        temp = tempfile.TemporaryDirectory(prefix='shot-reader-handheld-')
        try:
            priors, cameras, mirror_view = capture_priors(context, rig, mirror, frames, width, height)
            data = {'clip': bpy.path.abspath(clip.filepath, library=clip.library), 'size': [width, height],
                    'native_size': [scene.render.resolution_x, scene.render.resolution_y],
                    'times': [(frame - origin) / fps for frame in frames], 'hands': priors, 'mirror': mirror_view,
                    'default_profile': DEFAULT_PROFILE}
            (Path(temp.name) / 'request.json').write_text(json.dumps(data), encoding='utf-8')
            command = [bpy.path.abspath(prefs.python), str(Path(__file__).with_name('handheld_observations.py')), temp.name]
            process = start_worker(command, worker_environment(prefs), temp)
        except (ValueError, RuntimeError, OSError, KeyError) as error:
            temp.cleanup()
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        job = process, temp, scene, scene.camera, frames, priors, cameras, scale, rig, context.window
        status = 'Checking the plate for a handheld item…'
        bpy.app.timers.register(finish, first_interval=.4)
        return {'FINISHED'}


def grip_constraints(rig):
    return [constraint for bone in rig.pose.bones
            if (constraint := bone.constraints.get(CONSTRAINT))]


class ShotReaderToggleHandheldGrip(bpy.types.Operator):
    bl_idname = 'shot_reader.toggle_handheld_grip'
    bl_label = 'Toggle Handheld Grip'
    bl_options = {'UNDO'}

    @classmethod
    def poll(cls, context):
        rig, _ = performer(context)
        return bool(rig and grip_constraints(rig))

    def execute(self, context):
        rig, _ = performer(context)
        constraints = grip_constraints(rig)
        enabled = not any(item.enabled for item in constraints)
        for item in constraints:
            item.enabled = enabled
        return {'FINISHED'}
