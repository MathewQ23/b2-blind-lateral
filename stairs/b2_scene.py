"""B2 + fixed cargo + the shared Taishan route. No Go1 geometry is reused."""
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import numpy as np
from stairs.taishan import PROJECT_ROOT,TaishanConfig,StairRoute,FLOOR_Z

SOURCE=PROJECT_ROOT/'vendor/unitree_b2'
SETTINGS=json.loads((PROJECT_ROOT/'configs/b2_payload_training.json').read_text())
STAGES=tuple(tuple(s) for s in SETTINGS['stages'])
FEET=('FR','FL','RR','RL')


def build_b2_scene(directory,stage=0,seed=20261028,height=None,payload=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    height=STAGES[stage][0] if height is None else float(height)
    payload=STAGES[stage][1] if payload is None else float(payload)
    if not 0<=payload<=35:raise ValueError('Payload must be within the 0..35 kg simulation envelope')
    config=TaishanConfig(name='b2_taishan_payload',seed=seed+stage,rise_m=height,
        path_width_m=SETTINGS['path_width_m'],
        tread_m=.32-.08*(height-.02)/.20,rise_jitter_m=min(.015,height*.1),
        landing_length_m=1.2,summit_length_m=2.2,approach_length_m=2.8,exit_length_m=2.8,
        first_step_x_m=1.2,scenery=False)
    route=StairRoute(config)
    root=ET.parse(SOURCE/'b2.xml').getroot()
    root.set('model','B2 Taishan fixed payload')
    ET.SubElement(root,'option',timestep='.004',integrator='implicitfast',iterations='8',ls_iterations='10')
    visual=ET.SubElement(root,'visual')
    ET.SubElement(visual,'global',offwidth='1600',offheight='1000')
    ET.SubElement(visual,'headlight',ambient='.35 .35 .35',diffuse='.7 .7 .7',specular='.1 .1 .1')
    ET.SubElement(visual,'quality',shadowsize='2048')
    world=root.find('worldbody');body=world.find("body[@name='base_link']")
    assets=root.find('asset');sensors=root.find('sensor')
    base_mass=sum(float(x.get('mass')) for x in root.findall('.//inertial'))
    collision_default=root.find("./default/default/default[@class='collision']/geom")
    collision_default.set('contype','2');collision_default.set('conaffinity','1');collision_default.set('condim','3')
    torso_geoms=[]
    for i,g in enumerate(body.findall('geom')):
        if g.get('class')=='collision':
            g.set('name',f'torso_{i}');torso_geoms.append(g.get('name'))
    for foot in FEET:
        calf=body.find(f".//body[@name='{foot}_calf']")
        toe=next(g for g in calf.findall('geom') if g.get('class')=='collision' and g.get('type')=='sphere')
        toe.set('name',foot);toe.set('friction','.8 .005 .0001')
        ET.SubElement(calf,'site',name=foot,pos=toe.get('pos'),size='.005')
        ET.SubElement(sensors,'framelinvel',name=f'{foot}_velocity',objtype='site',objname=foot)
        ET.SubElement(sensors,'contact',name=f'{foot}_contact',geom1=foot,reduce='mindist',num='1',data='found')
    for g in torso_geoms:
        ET.SubElement(sensors,'contact',name=f'{g}_contact',geom1=g,reduce='mindist',num='1',data='found')
    # Native joint order differs from actuator/SDK order. The environment maps by joint names.
    actuator_order=[a.get('joint') for a in root.find('actuator')]
    ET.SubElement(sensors,'framezaxis',name='upvector',objtype='site',objname='imu')
    ET.SubElement(sensors,'framelinvel',name='global_linvel',objtype='site',objname='imu')
    # Local velocity is calculated directly from global velocity and body rotation in B2Env.
    for s in sensors.findall('jointactuatorfrc'):s.attrib.pop('noise',None)
    ET.SubElement(assets,'texture',name='sky',type='skybox',builtin='gradient',rgb1='.5 .63 .73',rgb2='.9 .93 .93',width='512',height='512')
    ET.SubElement(world,'light',directional='true',pos='0 -4 12',dir='.1 .3 -1',diffuse='.7 .7 .7')
    ET.SubElement(world,'geom',name='floor',type='plane',pos=f'0 0 {FLOOR_Z}',size='0 0 .1',
        contype='1',conaffinity='0',condim='3',rgba='.2 .26 .23 1',friction='.8 .005 .0001')
    for i,s in enumerate(route.segments):
        ET.SubElement(world,'geom',name=s['name'],type='box',
            pos=f"{(s['x0']+s['x1'])/2} 0 {(s['z']-.3)/2}",
            size=f"{(s['x1']-s['x0'])/2} {config.path_width_m/2} {(s['z']+.3)/2}",
            contype='1',conaffinity='0',condim='3',priority='1',friction='.8 .005 .0001',
            rgba=f'{.57+(i%3)*.035} {.56+(i%3)*.035} {.50+(i%3)*.035} 1')
    cargo=[]
    if payload>0:
        fraction=payload/SETTINGS['target_payload_kg']
        pieces=[('rack',SETTINGS['rack_dimensions_m'],SETTINGS['rack_center_body_m'],3.*fraction,'.15 .16 .18 1')]
        for i,center in enumerate(SETTINGS['case_centers_body_m']):
            pieces.append((f'water_case_{i}',SETTINGS['case_dimensions_m'],center,13.5*fraction,'.08 .43 .78 1'))
        for name,dims,pos,mass,color in pieces:
            b=ET.SubElement(body,'body',name=name,pos=' '.join(map(str,pos)))
            inertia=[mass*(dims[(i+1)%3]**2+dims[(i+2)%3]**2)/12 for i in range(3)]
            ET.SubElement(b,'inertial',mass=str(mass),pos='0 0 0',diaginertia=' '.join(map(str,inertia)))
            ET.SubElement(b,'geom',name=name,type='box',size=' '.join(str(x/2) for x in dims),
                contype='2',conaffinity='1',condim='3',mass='0',rgba=color,group='2')
            ET.SubElement(sensors,'contact',name=f'{name}_contact',geom1=name,reduce='mindist',num='1',data='found')
            torso_geoms.append(name)
            cargo.append(dict(name=name,mass_kg=mass,body_position_m=pos,dimensions_m=dims,inertia_kg_m2=inertia))
    else:
        # Keep contact-sensor schema constant across stages, without creating fictitious payload mass.
        for name in ('rack','water_case_0','water_case_1'):
            ET.SubElement(body,'geom',name=name,type='sphere',size='.001',pos='0 0 0',
                contype='0',conaffinity='0',mass='0',rgba='0 0 0 0',group='5')
            ET.SubElement(sensors,'contact',name=f'{name}_contact',geom1=name,reduce='mindist',num='1',data='found')
            torso_geoms.append(name)
    keyframes=root.find('keyframe');keyframes.clear()
    native_joint_order=[j.get('name') for j in body.findall('.//joint') if j.get('type')!='free']
    pose=np.tile([0.,.8,-1.6],4)
    for name,(x,z,yaw) in route.spawn_positions().items():
        # Wider summit supports B2's entire footprint at a downhill start.
        if name=='downhill_start':x=route.segment('summit')['x1']-.9
        if name=='reverse_downhill_start':x=route.segment('summit')['x0']+.9
        q=np.r_[x,0,z+.55,math.cos(yaw/2),0,0,math.sin(yaw/2),pose]
        ET.SubElement(keyframes,'key',name=name,qpos=' '.join(map(str,q)),ctrl=' '.join(['0']*12))
    dest=directory/'assets';dest.mkdir(exist_ok=True)
    for mesh in assets.findall('mesh'):
        path=dest/mesh.get('file')
        if not path.exists():shutil.copy2(SOURCE/'assets'/mesh.get('file'),path)
    shutil.copy2(SOURCE/'LICENSE',dest/'LICENSE')
    ET.indent(root);scene=directory/'scene.xml';ET.ElementTree(root).write(scene,encoding='unicode')
    meta=dict(robot='Unitree B2',task_version='b2_payload_v1',source=json.loads((SOURCE/'source.json').read_text())['commit'],
        config=asdict(config),height=height,payload_kg=payload,official_model_mass_kg=base_mass,
        total_model_mass_kg=base_mass+payload,advertised_robot_mass_kg=60.,
        mass_discrepancy='Official MJCF sums to ~83.5 kg versus product-page ~60 kg. Source inertias retained; requires calibration before hardware transfer.',
        cargo=cargo,payload_assumptions=SETTINGS,foot_order=FEET,actuator_joint_order=actuator_order,
        native_joint_order=native_joint_order,keyframes=['home','downhill_start','reverse_uphill_start','reverse_downhill_start'],
        foot_contact_sensors={f:[f'{f}_contact'] for f in FEET},
        trunk_contact_sensors=[f'{g}_contact' for g in torso_geoms],
        segments=route.segments,steps_up=24,steps_down=24,summit_height_m=route.segment('summit')['z'],
        route_length_m=route.segments[-1]['x1']-route.segments[0]['x0'],ctrl_dt=.02,sim_dt=.004,
        torque_limits_source='Official MJCF actuator ctrlrange: hips/thighs +/-200 Nm, calves +/-300 Nm',
        control='Software PD recalculated at each 250 Hz physics substep; policy at 50 Hz',
        caveats=['Rigid payload approximation','Unmeasured case/rack dimensions','No thermal, battery-voltage, strap compliance or liquid slosh model'])
    (directory/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    return scene,meta,route
