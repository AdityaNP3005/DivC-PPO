"""Register this project's training and evaluation environments."""
from rl_env.project_config import TRAIN_ENV_ID

try:
    import gymnasium as gym
except ImportError:
    gym = None

if gym is not None:
    gym.register(
        id=TRAIN_ENV_ID,
        entry_point="rl_env.env.episode_env:UR10eEpisodeEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": "rl_env.env.assembly:UR10eEnvCfg"},
    )
