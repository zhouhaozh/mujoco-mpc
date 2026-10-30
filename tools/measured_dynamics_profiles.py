"""Validate and freeze directory-based replay configurations.

``snapshot_profile`` remains as an internal compatibility path for legacy fitting
tools; replay entry points should use self-contained configuration directories.
"""
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from measured_replay_initialization import _load_settings

PROJECT = Path(__file__).resolve().parents[1] / 'sckd_sim'
CURRENT_CONFIGS = {'scene_path': 'models/generated/x4_950_scene.mjcf',
                   'motors_path': 'models/config/motors_x4_950.json',
                   'robot_variant_path': 'models/config/x4_950.json',
                   'chassis_path': 'models/config/chassis_x4_950.json'}
PROFILE_FILES = {'scene_path': 'scene.mjcf', 'motors_path': 'motors.json',
                 'robot_variant_path': 'robot.json', 'chassis_path': 'chassis.json'}
CONFIG_DIRECTORY_FILES = (*PROFILE_FILES.values(), 'replay_initialization.json')


def _relative_path(value, *, field):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'{field} must be a local path without ..: {value}')
    return path


def validate_config_dir(source):
    """Validate and return an independent measured-dynamics configuration directory."""
    source = Path(source).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f'Configuration directory does not exist: {source}')
    required = [source / name for name in CONFIG_DIRECTORY_FILES]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f'Configuration directory is missing required files: {", ".join(missing)}')
    meshes = source / 'meshes'
    if not meshes.is_dir():
        raise FileNotFoundError('Configuration directory is missing required directory: meshes')
    symlinks = [path.relative_to(source) for path in source.rglob('*') if path.is_symlink()]
    if symlinks:
        raise ValueError(f'Configuration directory cannot contain symlinks: {symlinks[0]}')

    configs = {name: json.loads((source / name).read_text())
               for name in ('robot.json', 'motors.json', 'chassis.json')}
    invalid = [name for name, config in configs.items() if not isinstance(config, dict)]
    if invalid:
        raise ValueError(f'Configuration JSON must contain an object: {", ".join(invalid)}')
    motor_config = _relative_path(configs['robot.json'].get('motor_config', ''),
                                  field='robot.json motor_config')
    if motor_config != Path('motors.json'):
        raise ValueError('robot.json motor_config must be motors.json')
    _load_settings(source / 'replay_initialization.json')

    try:
        tree = ET.parse(source / 'scene.mjcf')
    except ET.ParseError as error:
        raise ValueError(f'scene.mjcf is not valid XML: {error}') from error
    if any(True for _ in tree.getroot().iter('include')):
        raise ValueError('scene.mjcf include elements are not supported in configuration directories')
    compiler = tree.find('compiler')
    directories = {'mesh': 'meshdir', 'texture': 'texturedir'}
    assetdir = _relative_path(compiler.get('assetdir', '.') if compiler is not None else '.',
                              field='scene.mjcf assetdir')
    for element in tree.iter():
        value = element.get('file')
        if value is None:
            continue
        directory_name = directories.get(element.tag)
        directory = (compiler.get(directory_name, str(assetdir))
                     if compiler is not None and directory_name else str(assetdir))
        relative = (_relative_path(directory, field=f'scene.mjcf {directory_name or "assetdir"}') /
                    _relative_path(value, field=f'scene.mjcf {element.tag} file'))
        asset = (source / relative).resolve()
        try:
            asset.relative_to(source)
        except ValueError as error:
            raise ValueError(f'scene.mjcf asset escapes configuration directory: {relative}') from error
        if not asset.is_file():
            raise FileNotFoundError(f'scene.mjcf asset is missing: {relative}')
    return source


def snapshot_config_dir(source, destination):
    """Copy one validated, self-contained configuration directory for a replay."""
    source = validate_config_dir(source)
    destination = Path(destination).expanduser().resolve()
    if destination == source or source in destination.parents:
        raise ValueError('Snapshot destination must be outside the source configuration directory')
    shutil.copytree(source, destination)
    validate_config_dir(destination)
    return {key: str(destination / name) for key, name in PROFILE_FILES.items()}


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
