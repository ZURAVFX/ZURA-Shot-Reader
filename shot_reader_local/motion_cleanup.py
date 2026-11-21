"""Keep the raw capture while reducing short rotation spikes in the arm chains."""

import json

import bpy
import numpy as np

from .colour_bake import performer


ARM_BONES = set(range(39, 65)) | set(range(75, 101))
RAW = 'shot_reader_raw_arm_action'
SMOOTH = 'shot_reader_smoothed_arm_action'


def arm_curves(action, slot):
    groups = {}
    for layer in action.layers:
        for strip in layer.strips:
            bag = strip.channelbag(slot)
            if not bag:
                continue
            for curve in bag.fcurves:
                path = curve.data_path
                if path.startswith('pose.bones["bone_') and path.endswith('"].rotation_quaternion'):
                    bone = path.split('"')[1][5:]
                    if bone.isdecimal() and int(bone) in ARM_BONES:
                        groups.setdefault(path, {})[curve.array_index] = curve
    return groups


def smooth_quaternions(values):
    aligned = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1.e-8)
    for i in range(1, len(aligned)):
        if aligned[i - 1] @ aligned[i] < 0.:
            aligned[i] *= -1.
    padded = np.pad(aligned, ((2, 2), (0, 0)), mode='edge')
    neighbours = np.stack([padded[i:i + len(aligned)] for i in range(5)], axis=1)
    dots = np.einsum('fqc,fc->fq', neighbours, aligned)
    separation = 2. * np.arccos(np.clip(dots, -1., 1.))
    closer = neighbours * np.where(dots[..., None] < 0., -1., 1.)
    weights = np.exp(-((separation / .19) ** 2)) * [1., 2., 3., 2., 1.]
    result = np.einsum('fq,fqc->fc', weights, closer)
    result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1.e-8)
    return aligned, result


def step_noise(values):
    return float(np.mean(np.linalg.norm(np.diff(values, n=2, axis=0), axis=1)))


def set_action(rig, action):
    animation = rig.animation_data
    identifier = animation.action_slot.identifier
    animation.action = action
    slot = next((item for item in action.slots if item.identifier == identifier), None)
    if slot:
        animation.action_slot = slot


def stabilize(rig):
    animation = rig.animation_data
    if not animation or not animation.action or not animation.action_slot or rig.get(RAW):
        return None
    original = animation.action
    source_slot = animation.action_slot
    variant = original.copy()
    variant.name = f'{original.name} · Stabilized Arms'
    slot = next(item for item in variant.slots if item.identifier == source_slot.identifier)
    stats = []
    for group in arm_curves(variant, slot).values():
        if len(group) != 4:
            continue
        curves = [group[i] for i in range(4)]
        frames = [[key.co.x for key in curve.keyframe_points] for curve in curves]
        if len(frames[0]) < 6 or any(sample != frames[0] for sample in frames[1:]):
            continue
        q = np.array([[key.co.y for key in curve.keyframe_points] for curve in curves]).T
        raw, clean = smooth_quaternions(q)
        if step_noise(clean) >= step_noise(raw):
            continue
        offset = np.rad2deg(2 * np.arccos(np.clip(abs(np.einsum('ij,ij->i', raw, clean)), 0., 1.)))
        if np.percentile(offset, 95) > 5.:
            continue
        for curve, values in zip(curves, clean.T):
            for key, value in zip(curve.keyframe_points, values):
                key.co.y = float(value)
                key.interpolation = 'LINEAR'
            curve.update()
        stats.append((step_noise(raw), step_noise(clean), float(np.median(offset))))
    if not stats:
        bpy.data.actions.remove(variant)
        return None
    before, after, offset = np.mean(stats, axis=0)
    report = {'filtered_arm_channels': len(stats), 'rotation_noise_before': float(before),
              'rotation_noise_after': float(after), 'mean_bone_median_change_degrees': float(offset)}
    rig[RAW], rig[SMOOTH] = original, variant
    set_action(rig, variant)
    rig['shot_reader_arm_stabilization'] = json.dumps(report)
    return report


class ShotReaderStabilizeArmMotion(bpy.types.Operator):
    bl_idname = 'shot_reader.stabilize_arm_motion'
    bl_label = 'Stabilize Arm Motion'
    bl_options = {'UNDO'}

    @classmethod
    def poll(cls, context):
        rig, _ = performer(context)
        return bool(rig and rig.animation_data and rig.animation_data.action)

    def execute(self, context):
        rig, _ = performer(context)
        if original := rig.get(RAW):
            set_action(rig, original if rig.animation_data.action == rig.get(SMOOTH) else rig[SMOOTH])
        elif not stabilize(rig):
            self.report({'INFO'}, 'No compatible arm rotation keys needed stabilization.')
            return {'CANCELLED'}
        scene = context.scene
        scene.frame_set(scene.frame_current, subframe=scene.frame_subframe)
        return {'FINISHED'}
