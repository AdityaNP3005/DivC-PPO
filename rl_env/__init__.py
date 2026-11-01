"""Register this project's training and evaluation environments."""
from rl_env.project_config import TRAIN_ENV_ID, PLAY_ENV_ID

try:
    import gymnasium as gym
except ImportError:
    gym = None

if gym is not None:
    for task, configuration in ((TRAIN_ENV_ID, "UR10eEnvCfg"),
                                (PLAY_ENV_ID, "UR10eEnvCfg_PLAY")):
        gym.register(
            id=task,
            entry_point="rl_env.env.episode_env:UR10eEpisodeEnv",
            disable_env_checker=True,
            kwargs={"env_cfg_entry_point": f"rl_env.env.assembly:{configuration}"},
        )
