"""Release an Isaac application on unhandled runtime errors, including setup errors."""
import traceback


def _application_in_frame(frame):
    app = frame.f_locals.get('simulation_app') or frame.f_globals.get('simulation_app')
    if app is not None:
        return app
    launcher = frame.f_locals.get('app_launcher') or frame.f_globals.get('app_launcher')
    # Constructor/extension failure can precede both assignments in the caller.
    # Recover only an actual AppLauncher owner from that constructor's traceback.
    launcher_type = frame.f_globals.get('AppLauncher')
    owner = frame.f_locals.get('self')
    if isinstance(launcher_type, type) and isinstance(owner, launcher_type):
        launcher = owner
    try:
        return getattr(launcher, 'app', None)
    except (AttributeError, RuntimeError):
        return None  # AppLauncher has not created a SimulationApp yet.


def close_failed_app(error):
    applications = []
    environments = []
    for frame, _ in traceback.walk_tb(error.__traceback__):
        for variable, filename, complete_key in (
            ('out_dir', 'validation_status.json', 'validation_complete'),
            ('video_dir', 'play_status.json', 'play_complete'),
        ):
            output = frame.f_locals.get(variable) or frame.f_globals.get(variable)
            if not output:
                continue
            from pathlib import Path
            import json
            from evaluation.validation_contract import atomic_json
            path = Path(output) / filename
            try:
                status = json.loads(path.read_text())
                if status.get('reason') == 'pending':
                    status.update({complete_key: False, 'reason': 'failed', 'error': str(error), 'exit_code': 2})
                    atomic_json(path, status)
            except (OSError, ValueError):
                pass
        app = _application_in_frame(frame)
        if app is not None and all(app is not item for item in applications):
            applications.append(app)
        env = frame.f_locals.get('env') or frame.f_globals.get('env')
        if env is not None and hasattr(env, 'close'):
            unwrapped = getattr(env, 'unwrapped', env)
            if all(unwrapped is not getattr(item, 'unwrapped', item) for item in environments):
                environments.append(env)
    # Release cameras/render products before tearing down Kit. Keeping a live
    # environment during app.close can hang native shutdown after a setup error.
    for env in environments:
        try:
            env.close()
        except Exception as cleanup_error:
            print(f'[ERROR] Environment cleanup failed: {cleanup_error}', flush=True)
    for app in applications:
        try:
            if app.is_running():
                app.app.post_uncancellable_quit(2)
                app.close(wait_for_replicator=False)
        except Exception as cleanup_error:
            print(f'[ERROR] Isaac cleanup failed: {cleanup_error}', flush=True)
