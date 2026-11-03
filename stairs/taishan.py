"""Parametric, bidirectional stone stairs, inspired by Mount Tai.

Scene construction only: the legacy StairsEnv/PPO task is deliberately separate.
All dimensions are simulation design choices, not a survey of Mount Tai.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import shutil
import xml.etree.ElementTree as ET

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAYGROUND = PROJECT_ROOT / "vendor/mujoco_playground/mujoco_playground"
XML_ROOT = PLAYGROUND / "_src/locomotion/go1/xmls"
ROBOT_ASSETS = PLAYGROUND / "external_deps/mujoco_menagerie/unitree_go1/assets"
FEET = ("FR", "FL", "RR", "RL")
FLOOR_Z = -0.35
SOURCES = [
    {"url": "https://tsgw.taian.gov.cn/art/2021/5/26/art_363479_10320188.html",
     "title": "泰山管委：红门游览线（景点介绍）",
     "use": "Steep stone stairs, multiple sections, cliffs and summit gateway; no measured step dimensions."},
    {"url": "https://shandong-chorography.org/database/b9/section/1/article/142/",
     "title": "《泰山志》十八盘（公开归档镜像）",
     "use": "Historical context; the quoted 70–80 degrees describes rock strata, not a stair design angle."},
]


@dataclass(frozen=True)
class TaishanConfig:
    name: str = "taishan_inspired"
    seed: int = 20261028
    flights: int = 3
    steps_per_flight: int = 8
    rise_m: float = 0.22
    tread_m: float = 0.24
    rise_jitter_m: float = 0.015
    tread_jitter_m: float = 0.02
    path_width_m: float = 1.2
    landing_length_m: float = 0.9
    summit_length_m: float = 1.6
    approach_length_m: float = 2.2
    exit_length_m: float = 2.2
    first_step_x_m: float = 0.9
    friction: float = 0.8
    scenery: bool = True

    def __post_init__(self):
        for name in ("flights", "steps_per_flight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("rise_m", "tread_m", "path_width_m", "landing_length_m",
                     "summit_length_m", "approach_length_m", "exit_length_m", "friction"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for prefix in ("rise", "tread"):
            jitter = getattr(self, f"{prefix}_jitter_m")
            if not 0 <= jitter < getattr(self, f"{prefix}_m"):
                raise ValueError(f"Invalid {prefix} jitter")
        if not math.isfinite(self.first_step_x_m):
            raise ValueError("first_step_x_m must be finite")
        # Room for the complete Go1 footprint at independent start positions.
        if min(self.approach_length_m, self.exit_length_m, self.summit_length_m) < 1.2:
            raise ValueError("Start, summit and finish platforms must be at least 1.2 m long")
        if self.path_width_m < 0.6:
            raise ValueError("Path width must be at least 0.6 m for Go1")


class StairRoute:
    """One source of truth for physical surfaces, height scans and spawn poses."""

    def __init__(self, config: TaishanConfig):
        self.config = config
        rng = np.random.default_rng(config.seed)
        count = config.flights * config.steps_per_flight

        def varied(nominal, jitter):
            delta = rng.uniform(-1, 1, count)
            delta -= delta.mean()
            delta /= max(1.0, float(np.max(np.abs(delta))))
            return nominal + jitter * delta

        self.rises = varied(config.rise_m, config.rise_jitter_m)
        self.treads = varied(config.tread_m, config.tread_jitter_m)
        self.segments = []
        x = config.first_step_x_m - config.approach_length_m
        z = 0.0

        def add(name, length, kind, direction=0):
            nonlocal x
            self.segments.append(dict(name=name, x0=float(x), x1=float(x + length),
                                      z=float(z), kind=kind, direction=direction))
            x += length

        add("approach", config.approach_length_m, "platform")
        for i, (rise, tread) in enumerate(zip(self.rises, self.treads)):
            z += rise
            add(f"up_{i:03d}", tread, "step", 1)
            if (i + 1) % config.steps_per_flight == 0 and i + 1 < count:
                add(f"up_landing_{(i + 1) // config.steps_per_flight}", config.landing_length_m, "landing", 1)
        add("summit", config.summit_length_m, "platform")
        for i, (rise, tread) in enumerate(zip(self.rises[::-1], self.treads[::-1])):
            z -= rise
            if i == count - 1:
                z = 0.0
            add(f"down_{i:03d}", tread, "step", -1)
            if (i + 1) % config.steps_per_flight == 0 and i + 1 < count:
                add(f"down_landing_{(i + 1) // config.steps_per_flight}", config.landing_length_m, "landing", -1)
        add("finish", config.exit_length_m, "platform")

    def segment(self, name):
        return next(s for s in self.segments if s["name"] == name)

    def height_at(self, x, y=0.0, array_module=np):
        """Finite-width walking surface; pass jax.numpy for a JIT-able scan.

        Outside the path, return the catch-floor height, not an infinite stair.
        Decorative rocks and railings are not collision surfaces.
        """
        xp = array_module
        x, y = xp.asarray(x), xp.asarray(y)
        ends = xp.asarray([s["x1"] for s in self.segments])
        heights = xp.asarray([s["z"] for s in self.segments])
        idx = xp.clip(xp.searchsorted(ends, x, side="right"), 0, len(self.segments) - 1)
        inside = ((x >= self.segments[0]["x0"]) & (x < ends[-1])
                  & (xp.abs(y) < self.config.path_width_m / 2))
        return xp.where(inside, heights[idx], FLOOR_Z)

    def spawn_positions(self):
        approach, summit, finish = [self.segment(n) for n in ("approach", "summit", "finish")]
        return {
            "home": ((approach["x0"] + approach["x1"]) / 2, 0.0, 0.0),
            "downhill_start": (summit["x1"] - 0.65, summit["z"], 0.0),
            "reverse_uphill_start": ((finish["x0"] + finish["x1"]) / 2, 0.0, math.pi),
            "reverse_downhill_start": (summit["x0"] + 0.65, summit["z"], math.pi),
        }


def _expanded_robot():
    root = ET.parse(XML_ROOT / "scene_mjx_fullcollisions_flat_terrain.xml").getroot()

    def expand(element):
        for child in list(element):
            if child.tag == "include":
                included = ET.parse(XML_ROOT / child.attrib["file"]).getroot()
                expand(included)
                position = list(element).index(child)
                element.remove(child)
                for offset, node in enumerate(list(included)):
                    element.insert(position + offset, node)
            else:
                expand(child)

    expand(root)
    # Includes introduce repeated top-level sections. Merge for predictable edits.
    for tag in ("asset", "worldbody", "sensor", "keyframe", "custom"):
        sections = root.findall(tag)
        for section in sections[1:]:
            sections[0].extend(list(section))
            root.remove(section)
    return root


def build_scene(directory, config: TaishanConfig):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    route = StairRoute(config)
    root = _expanded_robot()
    root.set("model", config.name)
    root.find("compiler").set("meshdir", "assets")
    root.find("option").set("iterations", "8")
    robot_default = root.find("./default/default[@class='go1']/geom")
    robot_default.set("contype", "2")
    robot_default.set("conaffinity", "1")
    world, assets, sensors = (root.find(n) for n in ("worldbody", "asset", "sensor"))
    for light in list(world.findall("light")):
        world.remove(light)
    floor = world.find("geom[@name='floor']")
    floor.set("pos", f"0 0 {FLOOR_Z}")
    floor.set("rgba", ".19 .23 .20 1")
    floor.attrib.pop("material", None)
    floor.set("friction", f"{config.friction} .005 .0001")
    for texture in list(assets.findall("texture")):
        if texture.get("type") == "skybox":
            texture.set("rgb1", ".52 .64 .70")
            texture.set("rgb2", ".90 .92 .88")
    ET.SubElement(assets, "texture", name="stone_noise", type="2d", builtin="flat",
                  rgb1=".62 .60 .54", rgb2=".42 .43 .40", mark="random",
                  markrgb=".46 .45 .41", random=".12", width="512", height="512")
    ET.SubElement(assets, "material", name="stone", texture="stone_noise",
                  texuniform="true", texrepeat="3 3", roughness=".95", specular=".08")
    ET.SubElement(world, "light", directional="true", pos="0 -5 12",
                  dir=".15 .4 -1", diffuse=".75 .72 .65", castshadow="true")
    ET.SubElement(world, "light", directional="true", pos="0 5 8",
                  dir="0 -.5 -1", diffuse=".3 .35 .4", castshadow="false")
    root.find("visual/global").set("offwidth", "1600")
    root.find("visual/global").set("offheight", "1000")
    root.find("visual/quality").set("shadowsize", "4096")
    root.find("visual/map").set("zfar", "20")
    summit = route.segment("summit")
    length = route.segments[-1]["x1"] - route.segments[0]["x0"]
    center = (route.segments[-1]["x1"] + route.segments[0]["x0"]) / 2
    root.find("statistic").set("center", f"{center} 0 {summit['z']/2}")
    root.find("statistic").set("extent", str(length * .6))
    width = config.path_width_m
    colors = np.random.default_rng(config.seed + 1).uniform(.85, 1.0, len(route.segments))
    terrain_names = ["floor"]
    for s, shade in zip(route.segments, colors):
        zbottom = -0.30
        ET.SubElement(world, "geom", name=s["name"], type="box",
                      pos=f"{(s['x0']+s['x1'])/2} 0 {(s['z']+zbottom)/2}",
                      size=f"{(s['x1']-s['x0'])/2} {width/2} {(s['z']-zbottom)/2}",
                      material="stone", rgba=f"{shade} {shade} {shade} 1",
                      contype="1", conaffinity="0", condim="3", priority="1",
                      friction=f"{config.friction} .005 .0001", group="0")
        terrain_names.append(s["name"])
    if config.scenery:
        _add_scenery(world, assets, route)

    foot_contacts = {}
    for foot in FEET:
        names = [f"{foot}_floor_found"]
        for terrain in terrain_names[1:]:
            name = f"{foot}_{terrain}_found"
            ET.SubElement(sensors, "contact", name=name, geom1=foot, geom2=terrain,
                          reduce="mindist", num="1", data="found")
            names.append(name)
        foot_contacts[foot] = names
    trunk_contacts = []
    for i, geom in enumerate(world.find("body[@name='trunk']").findall("geom")):
        if geom.get("class") == "collision":
            geom.set("name", f"torso_collision_{i}")
            for terrain in terrain_names:
                name = f"torso_{i}_{terrain}_found"
                ET.SubElement(sensors, "contact", name=name, geom1=geom.get("name"),
                              geom2=terrain, reduce="mindist", num="1", data="found")
                trunk_contacts.append(name)

    keyframes = root.find("keyframe")
    default_qpos = np.fromstring(keyframes.find("key[@name='home']").get("qpos"), sep=" ")
    keyframes.clear()
    for name, (x, z, yaw) in route.spawn_positions().items():
        q = default_qpos.copy()
        q[:3] = [x, 0, z + default_qpos[2]]
        q[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        ET.SubElement(keyframes, "key", name=name, qpos=" ".join(map(str, q)),
                      ctrl=" ".join(map(str, default_qpos[7:])))
    target_assets = directory / "assets"
    target_assets.mkdir(exist_ok=True)
    for mesh in assets.findall("mesh"):
        filename = mesh.get("file")
        if filename:
            shutil.copy2(ROBOT_ASSETS / filename, target_assets / filename)
    license_path = ROBOT_ASSETS.parent / "LICENSE"
    if license_path.exists():
        shutil.copy2(license_path, target_assets / "LICENSE")
    ET.indent(root)
    scene_path = directory / "scene.xml"
    ET.ElementTree(root).write(scene_path, encoding="unicode")
    metadata = dict(
        schema_version=1, config=asdict(config), sources=SOURCES,
        scope="Taishan-inspired synthetic training geometry; dimensions are not field measurements.",
        task_integration="Scene only. Legacy stairs.train and scripts/evaluate.py do not consume this scene.",
        route_direction="+x: ascent -> summit -> descent -> finish; reverse keyframes also provided",
        steps_up=len(route.rises), steps_down=len(route.rises), summit_height_m=summit["z"],
        route_length_m=length, nominal_flight_slope_deg=math.degrees(math.atan2(config.rise_m, config.tread_m)),
        rise_range_m=[float(route.rises.min()), float(route.rises.max())],
        tread_range_m=[float(route.treads.min()), float(route.treads.max())],
        segments=route.segments, keyframes=list(route.spawn_positions()),
        suggested_tasks={
            "ascent": {"start": "home", "goal_x": (summit["x0"]+summit["x1"])/2},
            "descent": {"start": "downhill_start", "goal_x": route.spawn_positions()["reverse_uphill_start"][0]},
            "traverse": {"start": "home", "goal_x": route.spawn_positions()["reverse_uphill_start"][0]},
        },
        terrain_geoms=terrain_names, foot_contact_sensors=foot_contacts,
        trunk_contact_sensors=trunk_contacts, control_dt=.02, simulation_dt=.004,
        body_terrain_collisions=True, self_collisions=False,
        decorative_geometry="group 1; no collisions, can be hidden without changing physics",
        suggested_path_limit_abs_y_m=width/2-.1, catch_floor_z_m=FLOOR_Z,
        integration_notes=[
            "Use this route's finite-width height_at for observations and relative body/foot heights.",
            "Finish after descending to the exit platform; do not reuse the old summit success condition.",
            "Terminate lateral exits; the decorative railings do not constrain the robot.",
            "Include all foot contacts and gait metrics in training/evaluation archives.",
        ],
    )
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    return scene_path, metadata


def _add_scenery(world, assets, route):
    """Cheap visual stone cheeks, railings and rocks outside the walking surface."""
    width = route.config.path_width_m
    rng = np.random.default_rng(route.config.seed + 2)

    def visual(**kwargs):
        return ET.SubElement(world, "geom", contype="0", conaffinity="0", group="1", **kwargs)

    for i, s in enumerate(route.segments):
        for side in (-1, 1):
            y = side * (width / 2 + .12)
            visual(name=f"cheek_{i}_{side}", type="box",
                    pos=f"{(s['x0']+s['x1'])/2} {y} {(s['z']-.30)/2}",
                    size=f"{(s['x1']-s['x0'])/2} .12 {(s['z']+.30)/2}",
                    material="stone", rgba=".70 .72 .67 1")
            if s["kind"] != "step" or i % 4 == 0:
                x = (s["x0"] + s["x1"]) / 2
                visual(name=f"post_{i}_{side}", type="box", pos=f"{x} {y} {s['z']+.40}",
                        size=".045 .045 .40", rgba=".35 .36 .33 1")
            # Rail follows actual steps and landings; all decoration stays out of the path.
            next_z = route.segments[min(i+1,len(route.segments)-1)]["z"]
            visual(name=f"rail_{i}_{side}", type="capsule", size=".025",
                    fromto=f"{s['x0']} {y} {s['z']+.8} {s['x1']} {y} {next_z+.8}",
                    rgba=".22 .25 .24 1")
            if i % 3 == 0:
                x = (s["x0"] + s["x1"]) / 2
                # Low rocky shoulders preserve visibility of the route from cameras.
                # Faceted stone instead of smooth ellipsoids; generated mesh is visual only.
                phi = (1 + math.sqrt(5)) / 2
                vertices = np.array([(0,a,b*phi) for a in (-1,1) for b in (-1,1)]
                                    + [(a,b*phi,0) for a in (-1,1) for b in (-1,1)]
                                    + [(b*phi,0,a) for a in (-1,1) for b in (-1,1)], dtype=float)
                vertices /= np.linalg.norm(vertices,axis=1,keepdims=True)
                vertices *= rng.uniform(.8,1.2,(12,1))
                vertices *= [rng.uniform(.65,.95),rng.uniform(.55,.8),rng.uniform(.55,.8)]
                mesh_name=f"rock_mesh_{i}_{side}"
                ET.SubElement(assets,"mesh",name=mesh_name,vertex=" ".join(map(str,vertices.ravel())))
                visual(name=f"rock_{i}_{side}", type="mesh", mesh=mesh_name,
                        pos=f"{x} {side*(width/2+1.15)} {s['z']-.55}",
                        euler=f"0 {rng.uniform(-.4,.4)} {rng.uniform(-.3,.3)}",
                        material="stone", rgba=".58 .63 .56 1")
    summit = route.segment("summit")
    x, z = (summit["x0"]+summit["x1"])/2, summit["z"]
    for side in (-1, 1):
        visual(name=f"summit_gate_{side}", type="box", pos=f"{x} {side*(width/2+.22)} {z+.95}",
                size=".12 .12 .95", material="stone", rgba=".70 .64 .55 1")
    visual(name="summit_gate_beam", type="box", pos=f"{x} 0 {z+1.9}",
            size=f".17 {width/2+.42} .13", rgba=".45 .30 .22 1")
