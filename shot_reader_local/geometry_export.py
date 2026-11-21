"""Render editable reconstruction geometry through the matched camera."""

from contextlib import ExitStack, contextmanager
from pathlib import Path
import tempfile

import bpy

from .colour_bake import MODIFIER, performer


def change(stack, owner, attribute, value):
    original = getattr(owner, attribute)
    if original != value:
        setattr(owner, attribute, value)
        stack.callback(setattr, owner, attribute, original)


def clay_without_slots(stack, obj, material):
    group = bpy.data.node_groups.new('Shot Reader Geometry Material', 'GeometryNodeTree')
    stack.callback(bpy.data.node_groups.remove, group)
    for direction in ('INPUT', 'OUTPUT'):
        group.interface.new_socket(name='Geometry', in_out=direction, socket_type='NodeSocketGeometry')
    incoming = group.nodes.new('NodeGroupInput')
    outgoing = group.nodes.new('NodeGroupOutput')
    assign = group.nodes.new('GeometryNodeSetMaterial')
    assign.inputs['Material'].default_value = material
    group.links.new(incoming.outputs['Geometry'], assign.inputs['Geometry'])
    group.links.new(assign.outputs['Geometry'], outgoing.inputs['Geometry'])
    modifier = obj.modifiers.new(group.name, 'NODES')
    stack.callback(obj.modifiers.remove, modifier)
    modifier.node_group = group


def show_geometry(stack, scene, body, rig):
    room = scene.shot_reader_room.room
    if body and (colour := body.modifiers.get(MODIFIER)):
        change(stack, colour, 'show_render', False)
    material = bpy.data.materials.new('Shot Reader Geometry Clay')
    stack.callback(bpy.data.materials.remove, material)
    material.use_nodes = True
    material.diffuse_color = (.46, .49, .54, 1.)
    clay = material.node_tree.nodes.get('Principled BSDF')
    clay.inputs['Base Color'].default_value = material.diffuse_color
    clay.inputs['Roughness'].default_value = .68
    if rig and (hair := rig.get('shot_reader_hair')) and not hair.hide_render:
        # The fitted hair supplies the camera silhouette even when the compact scalp is shown in the viewport.
        if scalp := hair.get('shot_reader_scalp_hair'):
            change(stack, scalp, 'hide_render', True)
    if room:
        simplified = any(child.get('shot_reader_fixture_proxy') for child in room.children)
        change(stack, room, 'hide_render', simplified)
        for child in room.children:
            if child.get('shot_reader_fixture_proxy'):
                change(stack, child, 'hide_render', False)
            if child.get('shot_reader_fixture_surface') in {'counter', 'faucet'}:
                drivers = child.animation_data.drivers if child.animation_data else ()
                for driver in drivers:
                    if driver.data_path == 'hide_render':
                        change(stack, driver, 'mute', True)
                change(stack, child, 'hide_render', False)
                change(stack, child, 'visible_glossy', True)
                if shell := child.modifiers.get('Fixture Thickness'):
                    change(stack, shell, 'show_render', True)
    mirror = scene.shot_reader.body_mirror
    reflected_wall = scene.shot_reader_room.reflected_wall
    for obj in scene.objects:
        if obj == reflected_wall:
            change(stack, obj, 'visible_camera', False)
        if (obj.get('shot_reader_preview_fixture_viewport') is not None
                or (obj.get('shot_reader_reflected_set') and obj != reflected_wall)
                or obj.get('shot_reader_estimated_projection') is True or obj.get('shot_reader_observed_projection') is True):
            change(stack, obj, 'hide_render', True)
        if rays := obj.get('shot_reader_preview_wall_rays'):
            change(stack, obj, 'visible_camera', bool(rays[0]))
            change(stack, obj, 'visible_glossy', bool(rays[1]))
        is_mirror = obj == mirror or str(obj.get('shot_reader_class', '')).casefold().split(':')[-1] == 'mirror'
        if is_mirror or obj.hide_render or obj.type not in {'MESH', 'CURVE'}:
            continue
        if not obj.material_slots:
            clay_without_slots(stack, obj, material)
            continue
        for slot in obj.material_slots:
            original_link, original_material = slot.link, slot.material

            def restore(slot=slot, link=original_link, material=original_material):
                slot.material = material
                slot.link = link

            slot.link = 'OBJECT'
            stack.callback(restore)
            slot.material = material


@contextmanager
def geometry_render(context, directory, resolution):
    scene = context.scene
    rig, body = performer(context)
    render = scene.render
    frame, subframe = scene.frame_current, scene.frame_subframe
    try:
        with ExitStack() as stack:
            change(stack, render, 'engine', 'CYCLES')
            change(stack, render, 'resolution_percentage', resolution)
            change(stack, render, 'use_compositing', False)
            change(stack, render, 'use_sequencer', False)
            change(stack, render, 'film_transparent', False)
            change(stack, render, 'use_file_extension', True)
            image = render.image_settings
            for attribute in ('color_depth', 'color_mode', 'file_format'):
                stack.callback(setattr, image, attribute, getattr(image, attribute))
            image.file_format, image.color_mode, image.color_depth = 'PNG', 'RGB', '8'
            change(stack, scene.cycles, 'samples', 16)
            change(stack, render, 'filepath', str(directory / 'geometry-'))
            change(stack, scene, 'frame_step', 1)
            show_geometry(stack, scene, body, rig)
            scene.frame_set(frame, subframe=subframe)
            yield scene
    finally:
        scene.frame_set(frame, subframe=subframe)


class ShotReaderExportGeometry(bpy.types.Operator):
    bl_idname = 'shot_reader.export_geometry'
    bl_label = 'Export Geometry for AI Render'
    bl_description = 'Render clean PNGs through the matched camera with the physical mirror and editable geometry'

    directory: bpy.props.StringProperty(name='Output Folder', subtype='DIR_PATH')
    resolution: bpy.props.IntProperty(name='Resolution', subtype='PERCENTAGE', default=50, min=10, max=100)
    sequence: bpy.props.BoolProperty(name='Whole Shot', default=False)

    @classmethod
    def poll(cls, context):
        from . import reference_background

        return bool(context.scene.camera and reference_background(context.scene))

    def invoke(self, context, _event):
        self.directory = bpy.path.abspath('//') if bpy.data.filepath else bpy.app.tempdir
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, _context):
        self.layout.prop(self, 'directory')
        self.layout.prop(self, 'resolution')
        self.layout.prop(self, 'sequence')

    def execute(self, context):
        if not self.directory:
            self.report({'ERROR'}, 'Choose an output folder.')
            return {'CANCELLED'}
        try:
            parent = Path(bpy.path.abspath(self.directory))
            parent.mkdir(parents=True, exist_ok=True)
            folder = Path(tempfile.mkdtemp(prefix='Shot Reader Geometry ', dir=parent))
            with geometry_render(context, folder, self.resolution) as scene:
                if not self.sequence:
                    scene.render.filepath = str(folder / f'geometry-{scene.frame_current:04d}.png')
                result = bpy.ops.render.render(animation=self.sequence, write_still=not self.sequence, scene=scene.name)
                if result != {'FINISHED'} or not any(folder.glob('geometry-*.png')):
                    raise RuntimeError('Blender did not produce a geometry image.')
        except (OSError, RuntimeError, ValueError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        self.report({'INFO'}, f'Geometry saved to {folder}')
        return {'FINISHED'}
