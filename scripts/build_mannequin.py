"""Build ZURA's original poseable mannequin with Blender's native constraints."""

from pathlib import Path
import math

import bpy
from mathutils import Matrix, Vector


bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
asset = bpy.data.collections.new("Shot Reader Character")
scene.collection.children.link(asset)
armature = bpy.data.armatures.new("ZURA Mannequin")
rig = bpy.data.objects.new("ZURA Control Rig", armature)
asset.objects.link(rig)
bpy.context.view_layer.objects.active = rig
rig.select_set(True)
rig.show_in_front = True
rig["shot_reader_height_m"] = 1.78
rig["shot_reader_floor_m"] = 0.0
rig["shot_reader_rig_name"] = "ZURA Control Rig"
rig["shot_reader_forward"] = 1
bpy.ops.object.mode_set(mode="EDIT")


def bone(name, head, tail, parent=None, control=False):
    result = armature.edit_bones.new(name)
    result.head, result.tail = head, tail
    if parent:
        result.parent = armature.edit_bones[parent]
        result.use_connect = (result.head - result.parent.tail).length < 1e-5 and not control
    result.use_deform = not control
    return result


bone("Ctrl_Master", (0, 0, 0), (0, 0, .18), control=True)
bone("Ctrl_Hips", (0, 0, .92), (0, 0, 1.08), "Ctrl_Master", True)
bone("Ctrl_Chest", (0, 0, 1.08), (0, 0, 1.46), "Ctrl_Hips", True)
bone("Ctrl_Head", (0, 0, 1.46), (0, 0, 1.73), "Ctrl_Chest", True)
limbs = []
for side, sign in (("Left", -1), ("Right", 1)):
    for part, base, hinge, end, pole, parent, prefix in (
        ("Hand", (.19, 0, 1.43), (.28, -.025, 1.17), (.30, .065, .94),
         (.5, -.38, 1.20), "Ctrl_Chest", "Arm"),
        ("Foot", (.095, 0, .93), (.105, .06, .54), (.11, 0, .12),
         (.14, .55, .52), "Ctrl_Hips", "Leg"),
    ):
        base, hinge, end, pole = [(sign * x, y, z) for x, y, z in (base, hinge, end, pole)]
        upper, lower = f"{prefix}_Upper_{side}", f"{prefix}_Lower_{side}"
        target, joint = f"Ctrl_{part}_IK_{side}", f"Ctrl_{prefix}Pole_IK_{side}"
        bone(upper, base, hinge, parent)
        bone(lower, hinge, end, upper)
        bone(target, end, (end[0], end[1] + .13, end[2]), "Ctrl_Master", True)
        bone(joint, pole, (pole[0], pole[1], pole[2] + .08), "Ctrl_Master", True)
        limbs.append((upper, lower, target, joint, hinge, part))
bpy.ops.object.mode_set(mode="OBJECT")

controls = armature.collections.new("Artist Controls")
internals = armature.collections.new("IK Chains")
for item in armature.bones:
    is_control = item.name.startswith("Ctrl_")
    (controls if is_control else internals).assign(item)
    if is_control:
        item.color.palette = "THEME04" if item.name.endswith("Left") else (
            "THEME03" if item.name.endswith("Right") else "THEME02"
        )
internals.is_visible = False


def widget(name, points):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(points, [(i, (i + 1) % len(points)) for i in range(len(points))], [])
    return bpy.data.objects.new(name, mesh)


ring = widget("Control Ring", [(math.cos(a), 0, math.sin(a)) for a in (i * math.tau / 24 for i in range(24))])
diamond = widget("Pole Target", [(0, 0, 1), (1, 0, 0), (0, 0, -1), (-1, 0, 0)])
square = widget("IK Target", [(-1, 0, -1), (1, 0, -1), (1, 0, 1), (-1, 0, 1)])
for control in (b for b in rig.pose.bones if b.name.startswith("Ctrl_")):
    shape, size = (diamond, .045) if "Pole" in control.name else (
        (square, .075) if "_IK_" in control.name else (ring, .17)
    )
    if control.name == "Ctrl_Master":
        size = .32
    control.custom_shape = shape
    control.use_custom_shape_bone_size = False
    control.custom_shape_scale_xyz = (size,) * 3

for upper, lower, target, joint, hinge, _part in limbs:
    constraint = rig.pose.bones[lower].constraints.new("IK")
    constraint.target = constraint.pole_target = rig
    constraint.subtarget, constraint.pole_subtarget = target, joint
    constraint.chain_count = 2
    constraint.use_stretch = False
    best = (float("inf"), 0)
    for angle in (math.radians(n) for n in range(-180, 180, 10)):
        constraint.pole_angle = angle
        bpy.context.view_layer.update()
        best = min(best, ((rig.pose.bones[lower].head - Vector(hinge)).length, angle))
    constraint.pole_angle = best[1]
bpy.context.view_layer.update()

material = bpy.data.materials.new("Mannequin Clay")
material.diffuse_color = (.23, .42, .54, 1)
material.use_nodes = True
shader = material.node_tree.nodes.get("Principled BSDF")
shader.inputs["Base Color"].default_value = material.diffuse_color
shader.inputs["Roughness"].default_value = .63


def piece(name, parent, position, radii, axis=(0, 0, 1)):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=12)
    obj = bpy.context.object
    obj.name = name
    obj.data.transform(Matrix.Diagonal((*radii, 1)))
    for face in obj.data.polygons:
        face.use_smooth = True
    obj.data.materials.append(material)
    asset.objects.link(obj)
    for collection in tuple(obj.users_collection):
        if collection != asset:
            collection.objects.unlink(obj)
    obj.parent, obj.parent_type, obj.parent_bone = rig, "BONE", parent
    bpy.context.view_layer.update()
    rotation = Vector((0, 0, 1)).rotation_difference(Vector(axis)).to_matrix().to_4x4()
    obj.matrix_world = Matrix.Translation(position) @ rotation


piece("Pelvis", "Ctrl_Hips", (0, 0, .99), (.157, .105, .13))
piece("Torso", "Ctrl_Chest", (0, 0, 1.27), (.206, .11, .235))
piece("Neck", "Ctrl_Head", (0, 0, 1.49), (.062, .062, .095))
piece("Head", "Ctrl_Head", (0, .012, 1.65), (.102, .097, .127))
for upper, lower, target, _joint, _hinge, part in limbs:
    for segment, radius in ((upper, .069), (lower, .054)):
        limb = rig.pose.bones[segment]
        direction = limb.tail - limb.head
        piece(segment, segment, (limb.head + limb.tail) / 2,
              (radius, radius, direction.length / 2 + .035), direction)
    tip = rig.pose.bones[target].head
    if part == "Foot":
        piece(target.removeprefix("Ctrl_"), target, tip + Vector((0, .062, -.057)), (.061, .123, .063))
    else:
        piece(target.removeprefix("Ctrl_"), target, tip + Vector((0, .02, -.046)), (.048, .033, .077))

output = Path(__file__).resolve().parents[1] / "shot_reader_local" / "character.blend"
bpy.data.libraries.write(str(output), {asset}, compress=True)
