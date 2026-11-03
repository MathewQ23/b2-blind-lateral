#!/usr/bin/env python3
"""Apply the original silver/blue studio look to an exact B2 training scene."""
from pathlib import Path
import argparse
import json
import os
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

PROJECT = Path(__file__).resolve().parents[1]


def build(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    root = ET.parse(source).getroot()
    root.set('model', 'Silver blue B2 - original series studio with physical stairs')
    compiler = root.find('compiler')
    meshdir = source.parent / compiler.get('meshdir', 'assets')
    compiler.set('meshdir', os.path.relpath(meshdir.resolve(), destination.parent))
    asset, world = root.find('asset'), root.find('worldbody')
    for texture in asset.findall('texture'):
        if texture.get('type') == 'skybox':
            texture.set('rgb1', '.62 .70 .82')
            texture.set('rgb2', '.10 .12 .17')
    ET.SubElement(asset, 'texture', name='studio_floor', type='2d', builtin='checker', mark='edge',
                  rgb1='.30 .33 .38', rgb2='.27 .30 .35', markrgb='.90 .52 .22', width='512', height='512')
    ET.SubElement(asset, 'material', name='studio_floor', texture='studio_floor', texuniform='true',
                  texrepeat='5 5', reflectance='0', specular='.1', shininess='.2')
    colors = dict(shell='.72 .75 .80 1', blue='.16 .42 .95 1', graphite='.17 .18 .21 1')
    for name, rgba in colors.items():
        ET.SubElement(asset, 'material', name='studio_' + name, rgba=rgba, specular='.4', shininess='.55')
    for geom in world.findall('geom'):
        if geom.get('type') in ('box', 'plane'):
            geom.set('material', 'studio_floor')
            geom.set('rgba', '1 1 1 1')
            if geom.get('type') == 'box' and geom.get('name', '').startswith(('up_', 'down_')):
                p = np.fromstring(geom.get('pos'), sep=' ')
                s = np.fromstring(geom.get('size'), sep=' ')
                # Nosing is visual only; the physical edge remains the original box.
                x = p[0] - s[0] if geom.get('name').startswith('up_') else p[0] + s[0]
                ET.SubElement(world, 'geom', name='studio_edge_' + geom.get('name'), type='box',
                              pos=f'{x} {p[1]} {p[2]+s[2]+.001}', size=f'.004 {s[1]} .001',
                              rgba='1 .54 .17 1', contype='0', conaffinity='0', mass='0', group='2')
    body = world.find('body')
    for geom in body.iter('geom'):
        mesh = geom.get('mesh', '')
        if not mesh:
            continue
        if mesh == 'base_link' or mesh.endswith(('_thigh', '_thigh_protect')):
            material = 'shell'
        elif mesh.endswith('_hip'):
            material = 'blue'
        elif mesh.endswith(('_calf', '_foot')) or mesh in ('fake_head_Link', 'fake_tail_link', 'unitree_ladar'):
            material = 'graphite'
        else:
            continue
        geom.set('material', 'studio_' + material)
        geom.set('rgba', colors[material])
    for light in world.findall('light'):
        world.remove(light)
    ET.SubElement(world, 'light', mode='trackcom', pos='1.2 -1.6 6', dir='-1.2 1.6 -4.8',
                  diffuse='.8 .8 .8', specular='.25 .25 .25', cutoff='60')
    visual = root.find('visual')
    ET.SubElement(visual, 'rgba', haze='.42 .48 .58 1')
    ET.indent(root)
    ET.ElementTree(root).write(destination, encoding='unicode')
    original, styled = (mujoco.MjModel.from_xml_path(str(p)) for p in (source, destination))
    for field in ('nq', 'nv', 'nu'):
        assert getattr(original, field) == getattr(styled, field)
    for field in ('body_mass', 'body_inertia', 'body_ipos', 'jnt_range', 'actuator_ctrlrange'):
        np.testing.assert_array_equal(getattr(original, field), getattr(styled, field))
    for i in range(original.ngeom):
        if not (original.geom_contype[i] or original.geom_conaffinity[i]):
            continue
        name = original.geom(i).name
        body_id = original.geom_bodyid[i]
        j = (styled.geom(name).id if name else
             styled.body_geomadr[body_id] + i - original.body_geomadr[body_id])
        for field in ('geom_type', 'geom_pos', 'geom_size', 'geom_quat', 'geom_friction', 'geom_contype', 'geom_conaffinity'):
            np.testing.assert_array_equal(getattr(original, field)[i], getattr(styled, field)[j])
    report = dict(source=str(source), scene=str(destination), physics_identical=True,
                  look='Original series silver shell, blue hips, graphite calves; grey checker and orange grid',
                  extra_collision_geometry=False)
    destination.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(build(a.source, a.output), ensure_ascii=False, indent=2))
