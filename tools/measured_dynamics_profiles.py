"""Freeze named simulation configurations without changing global robot settings."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

PROJECT = Path(__file__).resolve().parents[1] / 'sckd_sim'
CURRENT_CONFIGS = {'scene_path': 'models/generated/x4_950_scene.mjcf',
                   'motors_path': 'models/config/motors_x4_950.json',
                   'robot_variant_path': 'models/config/x4_950.json',
                   'chassis_path': 'models/config/chassis_x4_950.json'}
PROFILE_FILES = {'scene_path': 'scene.mjcf', 'motors_path': 'motors.json',
                 'robot_variant_path': 'robot.json', 'chassis_path': 'chassis.json'}


def snapshot_profile(name, destination, *, project=None, current_paths=None):
    if name not in ('current', 'baseline', 'fitted'):
        raise ValueError(f'Unknown parameter profile: {name}')
    project = Path(project) if project is not None else PROJECT
    source = project / 'models/config/measured_dynamics' / name
    if name == 'current':
        paths = ({k: Path(v) for k,v in current_paths.items()} if current_paths is not None
                 else {k: project/v for k,v in CURRENT_CONFIGS.items()})
    else:
        paths = {k: source/v for k,v in PROFILE_FILES.items()}
    # Read all sources before creating the destination, so incomplete profiles fail early.
    configs = {k: json.loads(p.read_text()) for k,p in paths.items() if k != 'scene_path'}
    tree = ET.parse(paths['scene_path'])
    compiler = tree.find('compiler')
    meshdir = (paths['scene_path'].parent / compiler.get('meshdir', '')).resolve()
    compiler.set('meshdir', str(meshdir))
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    resolved = {k: str(destination/v) for k,v in PROFILE_FILES.items()}
    configs['robot_variant_path']['motor_config'] = resolved['motors_path']
    for key, config in configs.items():
        Path(resolved[key]).write_text(json.dumps(config, indent=2, ensure_ascii=False)+'\n')
    tree.write(resolved['scene_path'], encoding='utf-8', xml_declaration=True)
    return resolved


def command_identity(payload):
    """Hash the complete experiment input, excluding only configuration locations."""
    data = {k:v for k,v in payload.items() if k not in {'project_root', *CURRENT_CONFIGS}}
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def replay_scene(output):
    manifest = json.loads((Path(output)/'manifest.json').read_text())
    if 'config_paths' not in manifest:
        # Legacy results predate model snapshots.
        return PROJECT / CURRENT_CONFIGS['scene_path']
    scene = Path(manifest['config_paths']['scene_path'])
    if not scene.is_absolute():
        scene = Path(output) / scene
    if not scene.is_file():
        raise FileNotFoundError(f'Recorded model snapshot is missing: {scene}')
    return scene
