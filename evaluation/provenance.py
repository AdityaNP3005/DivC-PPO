"""Saved artifact identity and source provenance; no config/Isaac/torch imports."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile

# Paper method IDs -> (arm id in checkpoint metadata, privileged critic, per-axis credit);
# checkpoint_contract re-exports this table.
METHODS = {"ppo": (1, False, False), "divc-ppo": (3, False, True)}


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_identity(root=None):
    if root is None:
        from rl_env.project_config import PROJECT_ROOT
        root = PROJECT_ROOT
    root = Path(root).resolve()
    folders = ('config', 'rl_env', 'training', 'evaluation', 'tools')
    files = [file for folder in folders for file in (root / folder).rglob('*.py')]
    files += list(root.glob('*.py')) + list(root.glob('*.sh'))
    hashes = {str(file.relative_to(root)): _digest(file) for file in sorted(files)}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    command = ['git', '--no-optional-locks', '-c', f'safe.directory={root}', '-C', str(root)]
    try:
        commit = subprocess.check_output([*command, 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output([*command, 'status', '--porcelain', '--untracked-files=all', '--',
                                          *folders, ':(top,glob)*.py', ':(top,glob)*.sh'],
                                         text=True, stderr=subprocess.DEVNULL)
        dirty = bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {'sha256': digest, 'files': hashes, 'git_commit': commit, 'git_source_dirty': dirty}


def checkpoint_identity(checkpoint):
    """Read saved model step/seed and policy family without unpickling a model.

    Family identifies saved architecture; comparison_protocol and strict matched
    checkpoint gates remain separate. Missing legacy fields are explicitly unknown.
    """
    checkpoint = Path(checkpoint)
    with zipfile.ZipFile(checkpoint) as archive:
        data = json.loads(archive.read('data'))
    kwargs = data.get('policy_kwargs') or {}
    credit = kwargs.get('decomposed_config') or {}
    shape = (data.get('observation_space') or {}).get('_shape')
    decomposed = bool(credit.get('decomposed_advantage', False))
    try:
        asymmetric = 0 < int(kwargs.get('actor_obs_dim') or 0) < int(shape[0])
    except (TypeError, ValueError, IndexError):
        asymmetric = False
    method = next((name for name, (_, asym, decomp) in METHODS.items()
                   if (asym, decomp) == (asymmetric, decomposed)), None) if shape else None
    params = checkpoint.parent / 'params'
    result = {'run_id': None, 'method_id': method, 'model_seed': data.get('seed'), 'training_seed': data.get('seed'),
            'training_seed_source': 'saved_model', 'training_manifest_verified': False,
            'model_step': data.get('num_timesteps'), 'observation_shape': shape,
            'saved_actor_obs_dim': kwargs.get('actor_obs_dim'),
            'config_sha256': {name: _digest(params / name) for name in ('agent.yaml', 'env.yaml')
                              if (params / name).is_file()}}
    # A final manifest can name the current run's seed even on resume. Periodic
    # models must never inherit unrelated final-model provenance.
    manifest_path = checkpoint.parent / 'training_result.json'
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        manifest = None
    if isinstance(manifest, dict) and manifest.get('checkpoint_sha256') == _digest(checkpoint):
        training_source = manifest.get('source')
        result.update(training_seed=manifest.get('training_seed', manifest.get('seed', data.get('seed'))),
                      training_seed_source='verified_training_manifest', training_manifest_verified=True,
                      run_id=manifest.get('run_id'), training_manifest_sha256=_digest(manifest_path),
                      training_source_sha256=training_source.get('sha256') if isinstance(training_source, dict) else None)
    return result


def runtime_assets(detector=None):
    """Capture already-frozen object and resolved local detector identity."""
    from config import task
    from config.perception import YOLO_DEFAULTS
    detector = detector or os.environ.get('YOLO_MODEL_PATH') or YOLO_DEFAULTS['model_path']
    usd = task.OBJECT_USD
    return {'object': {'usd': usd, 'usd_sha256': _digest(Path(usd)) if usd and Path(usd).is_file() else None,
                       'scale': task.OBJECT_USD_SCALE, 'height_m': task.OBJECT_CYL_HEIGHT_M,
                       'radius_m': task.OBJECT_CYL_RADIUS_M, 'floor_z_m': task.CYLINDER_FLOOR_Z_M},
            'detector': {'path': str(Path(detector).resolve()),
                         'sha256': _digest(Path(detector)) if Path(detector).is_file() else None}}
