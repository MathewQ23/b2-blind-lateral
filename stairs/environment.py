"""Terrain-aware Go1 environment based on DeepMind's Apache-2.0 Playground.

Uses the upstream robot, PD actuators, sensor observations and locomotion costs.
The six physical steps, terrain observations and progress/success logic are local.
"""
from pathlib import Path
import json
import xml.etree.ElementTree as ET

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go1 import base, go1_constants, joystick
from mujoco import mjx


class StairsEnv(joystick.Joystick):
    def __init__(self, directory, height=0.0, steps=6, tread=0.30, impl="jax", top_controller="policy"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.height = float(height)
        self.top_controller = top_controller
        self.steps = int(steps)
        self.tread = float(tread)
        self.start_x = 0.8
        self.goal_x = self.start_x + self.steps * self.tread + 0.45
        self.episode_length = 750
        assets = base.get_assets()
        source = go1_constants.FULL_COLLISIONS_FLAT_TERRAIN_XML.read_text()
        root = ET.fromstring(source)

        def expand(element):
            for child in list(element):
                if child.tag == "include":
                    included = ET.fromstring(assets[child.attrib["file"]])
                    expand(included)
                    at = list(element).index(child)
                    element.remove(child)
                    for node in list(included):
                        element.insert(at, node)
                        at += 1
                else:
                    expand(child)
        expand(root)
        root.find("compiler").set("meshdir", "assets")
        # Keep all body/leg-to-terrain collisions; avoid robot self-contact pairs.
        robot_default = root.find("./default/default[@class='go1']/geom")
        robot_default.set("contype", "2")
        robot_default.set("conaffinity", "1")
        root.find("option").set("iterations", "4")
        world = root.find("worldbody")
        for i in range(self.steps):
            left = self.start_x + i * self.tread
            right = left + self.tread
            z = max((i + 1) * self.height, 0.0002)
            ET.SubElement(world, "geom", {
                "name": f"stair_{i}", "type": "box",
                "pos": f"{(left+right)/2} 0 {z/2}",
                "size": f"{(right-left)/2} 0.9 {z/2}",
                "contype": "1", "conaffinity": "0", "condim": "3",
                "priority": "1", "friction": "0.8 0.005 0.0001",
                "rgba": f"{0.28+i*0.055} {0.38+i*0.045} 0.65 1",
            })
        platform_left = self.start_x + self.steps*self.tread
        platform_top = max(self.steps*self.height, 0.0002)
        ET.SubElement(world,"geom",dict(name="top_platform",type="box",
            pos=f"{platform_left+1.0} 0 {platform_top/2}",
            size=f"1.0 0.9 {platform_top/2}",contype="1",conaffinity="0",
            condim="3",priority="1",friction="0.8 0.005 0.0001",rgba="0.65 0.65 0.72 1"))
        sensors = root.find("sensor")
        # MJX-JAX supports geom matching but not body-matching contact sensors.
        terrain_names = ["floor"] + [f"stair_{i}" for i in range(self.steps)] + ["top_platform"]
        self.foot_contact_names = []
        for foot in go1_constants.FEET_GEOMS:
            names = [f"{foot}_floor_found"]
            for terrain_name in terrain_names[1:]:
                name = f"{foot}_{terrain_name}_found"
                ET.SubElement(sensors, "contact", dict(name=name, geom1=foot,
                    geom2=terrain_name, reduce="mindist", num="1", data="found"))
                names.append(name)
            self.foot_contact_names.append(names)
        self.trunk_contact_names = []
        trunk = root.find(".//body[@name='trunk']")
        for number, geom in enumerate(trunk.findall("geom")):
            if geom.get("class") != "collision":
                continue
            geom_name = f"torso_collision_{number}"
            geom.set("name", geom_name)
            for terrain_name in terrain_names:
                name = f"{geom_name}_{terrain_name}_found"
                ET.SubElement(sensors, "contact", dict(name=name, geom1=geom_name,
                    geom2=terrain_name, reduce="mindist", num="1", data="found"))
                self.trunk_contact_names.append(name)
        xml_path = self.directory / "scene.xml"
        ET.indent(root)
        xml_path.write_text(ET.tostring(root, encoding="unicode"))
        target_assets = self.directory / "assets"
        target_assets.mkdir(exist_ok=True)
        for name, content in assets.items():
            if name.endswith((".stl", ".obj", ".png")):
                dest = target_assets / name
                if not dest.exists():
                    dest.write_bytes(content)
        config = joystick.default_config()
        config.impl = impl
        config.njmax = 512
        config.episode_length = self.episode_length
        config.action_scale = 0.45
        config.reward_config.scales.pose = 0.2
        config.reward_config.scales.feet_height = -0.05
        config.reward_config.scales.feet_clearance = -0.5
        config.reward_config.max_foot_height = 0.07 + self.height * 0.6
        config.noise_config.level = 0.25
        # Use the base initializer to load our physical terrain and full collisions.
        base.Go1Env.__init__(self, str(xml_path), config)
        self._post_init()
        self._trunk_contact = jnp.array([int(self.mj_model.sensor_adr[self.mj_model.sensor(n).id])
                                       for n in self.trunk_contact_names])
        self._all_foot_contacts = jnp.array([[int(self.mj_model.sensor_adr[self.mj_model.sensor(n).id])
                                             for n in names] for names in self.foot_contact_names])
        self._scan_x = jnp.repeat(jnp.array([-0.3, 0.0, 0.3, 0.6, 0.9]), 3)
        self._scan_y = jnp.tile(jnp.array([-0.2, 0.0, 0.2]), 5)
        self.reward_keys = list(config.reward_config.scales) + [
            "progress", "height", "heading", "lateral"]
        (self.directory / "metadata.json").write_text(json.dumps(dict(
            height=self.height,steps=self.steps,tread=self.tread,goal_x=self.goal_x,
            ctrl_dt=self.dt,sim_dt=self.sim_dt,backend=impl,
            top_controller=top_controller,
            reward_parts=self.reward_keys,body_terrain_collisions=True,self_collisions=False),indent=2))

    def terrain(self, x):
        level = jnp.clip(jnp.floor((x-self.start_x)/self.tread)+1, 0, self.steps)
        return level * self.height

    def _get_obs(self, data, info):
        obs = super()._get_obs(data, info)
        # The actor receives simulated local terrain heights (not camera images).
        w, x, y, z = data.qpos[3:7]
        yaw = jnp.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        sx = data.qpos[0] + jnp.cos(yaw)*self._scan_x - jnp.sin(yaw)*self._scan_y
        scan = jnp.clip(self.terrain(sx) - data.qpos[2], -0.6, 0.6)
        extra = jnp.concatenate((scan, jnp.array([jnp.sin(yaw), jnp.cos(yaw)])))
        return {k: jnp.concatenate((v, extra)) for k, v in obs.items()}

    def reset(self, key):
        state = super().reset(key)
        key, a, b, c = jax.random.split(key, 4)
        qpos = self._init_q.at[0].set(jax.random.uniform(a, (), minval=-0.2, maxval=0.2))
        yaw = jax.random.uniform(b, (), minval=-0.08, maxval=0.08)
        qpos = qpos.at[3:7].set(jnp.array([jnp.cos(yaw/2), 0., 0., jnp.sin(yaw/2)]))
        data = mjx_env.make_data(self.mj_model, qpos=qpos,
            qvel=jnp.zeros(self.mj_model.nv), ctrl=self._default_pose,
            impl=self._config.impl, naconmax=self._config.naconmax, njmax=self._config.njmax)
        data = mjx.forward(self.mjx_model, data)
        info = dict(state.info)
        info.update(rng=key, command=jnp.array([jax.random.uniform(c, (), minval=.3, maxval=.55),0.,0.]),
            episode_steps=jnp.array(0, jnp.int32), success=jnp.array(False),
            failure=jnp.array(False), timeout=jnp.array(False),
            episode_return=jnp.array(0.), episode_id=jnp.array(0, jnp.int32),
            on_platform=jnp.array(False),platform_hold=jnp.array(0,jnp.int32))
        metrics = {name: jnp.array(0.) for name in self.reward_keys}
        return state.replace(data=data, info=info, metrics=metrics, obs=self._get_obs(data, info))

    def _cost_feet_clearance(self, data):
        pos = data.site_xpos[self._feet_site_id]
        relative = pos[:,2]-self.terrain(pos[:,0])
        vel = data.sensordata[self._foot_linvel_sensor_adr]
        return jnp.sum(jnp.abs(relative-self._config.reward_config.max_foot_height)
                       *jnp.linalg.norm(vel[:,:2], axis=-1))

    def step(self, state, action):
        info = dict(state.info)
        control_action=action
        if self.top_controller=="pd_hold":
            control_action=jnp.where(info["on_platform"],jnp.zeros_like(action),action)
        targets = jnp.clip(self._default_pose + control_action*self._config.action_scale,
                          self._lowers, self._uppers)
        data = mjx_env.step(self.mjx_model, state.data, targets, self.n_substeps)
        contact = jnp.any(data.sensordata[self._all_foot_contacts] > 0, axis=-1)
        first_contact = (info["feet_air_time"] > 0) & (contact | info["last_contact"])
        info["feet_air_time"] = info["feet_air_time"] + self.dt
        feet = data.site_xpos[self._feet_site_id]
        relative_feet = feet[:,2]-self.terrain(feet[:,0])
        info["swing_peak"] = jnp.maximum(info["swing_peak"], relative_feet)
        relative_body = data.qpos[2] - jnp.mean(self.terrain(feet[:,0]))
        invalid = ~jnp.all(jnp.isfinite(data.qpos)) | ~jnp.all(jnp.isfinite(data.qvel))
        failure = ((self.get_upvector(data)[2] < .35) | (relative_body < .13)
                   | (jnp.abs(data.qpos[1]) > .75) | jnp.any(data.sensordata[self._trunk_contact] > 0)
                   | invalid)
        stop_x = self.goal_x + .20
        info["on_platform"] = ((info["on_platform"] & (data.qpos[0]>stop_x-.05))
                                | (data.qpos[0]>stop_x))
        forward_target=jnp.where(info["command"][0]>.01,info["command"][0],.4)
        info["command"] = jnp.array([jnp.where(info["on_platform"],0.,forward_target),0.,0.])
        top = self.steps*self.height
        landed = jnp.sum(contact & (feet[:,0] > self.start_x+self.steps*self.tread)) >= 2
        stable = ((data.qpos[0] > self.goal_x) & landed & (self.get_upvector(data)[2]>.8)
                  & (data.qpos[2]>top+.18) & (data.qpos[2]<top+.45) & ~failure)
        info["platform_hold"] = jnp.where(stable,info["platform_hold"]+1,0)
        success = (info["platform_hold"]>=50) & ~failure
        info["episode_steps"] += 1
        timeout = info["episode_steps"] >= self.episode_length
        done = failure | success | timeout
        rewards = self._get_reward(data, action, info, state.metrics,
                                   failure, first_contact, contact)
        rewards = {k: v*self._config.reward_config.scales[k] for k,v in rewards.items()}
        vx = self.get_global_linvel(data)[0]
        w,x,y,z = data.qpos[3:7]
        yaw = jnp.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        rewards["pose"] += 2*jnp.exp(-jnp.sum(jnp.square(self.get_local_linvel(data)[:2]))/.04)*stable*info["on_platform"]
        rewards.update(progress=2*jnp.clip(vx,-1,1)*(~info["on_platform"]),
            height=-2*jnp.square(relative_body-.28),
            heading=-.5*jnp.square(yaw), lateral=-.5*jnp.square(data.qpos[1]))
        reward = sum(rewards.values())*self.dt + 2.*success - .5*failure
        reward = jnp.where(invalid, -.5, reward)
        info.update(last_last_act=info["last_act"], last_act=action,
            feet_air_time=info["feet_air_time"]*(~contact), last_contact=contact,
            swing_peak=info["swing_peak"]*(~contact), success=success, failure=failure,
            timeout=timeout, episode_return=info["episode_return"]+reward)
        obs = self._get_obs(data, info)
        return state.replace(data=data, info=info, metrics=rewards, obs=obs,
                             reward=reward, done=done.astype(jnp.float32))
