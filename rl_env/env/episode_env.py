"""Project-owned episode boundary integration; shared IsaacLab stays untouched.

UR10eEpisodeEnv captures the pre-reset policy observation; Sb3VecEnvWrapper hands it to
SB3/VecNormalize as terminal_observation (training and validation).
"""

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.sb3 import Sb3VecEnvWrapper as IsaacSb3VecEnvWrapper


def _clone_observation(value):
    if isinstance(value, dict):
        return {key: _clone_observation(item) for key, item in value.items()}
    return value.detach().clone()


class UR10eEpisodeEnv(ManagerBasedRLEnv):
    """Capture policy/critic input before auto-reset and invalidate perception."""

    def step(self, action):
        self._terminal_policy_observation = None
        self._capturing_step_terminal = True
        try:
            return super().step(action)
        finally:
            self._capturing_step_terminal = False

    def _reset_idx(self, env_ids):
        in_step = getattr(self, "_capturing_step_terminal", False)
        if in_step and len(env_ids) > 0:
            # Reward/termination have already been computed, but assets, last
            # action, reset-relative EE origin and observation terms are intact.
            # Current project layouts have no observation history. Do not append
            # an extra frame to continuing environments' histories.
            policy = self.observation_manager.compute_group("policy", update_history=False)
            self._terminal_policy_observation = _clone_observation(policy)
        elif not in_step:
            self._terminal_policy_observation = None

        super()._reset_idx(env_ids)
        # Reset can randomize the object in the SAME common_step_counter used
        # by terminal rewards. The next observation must read the reset camera.
        self._yolo_detection_cache = {}
        if in_step:
            # The stock step reset path does not forward kinematics as explicit
            # reset does. Refresh transforms before its configured RTX rerender.
            self.scene.write_data_to_sim()
            self.sim.forward()


class Sb3VecEnvWrapper(IsaacSb3VecEnvWrapper):
    def _process_extras(self, obs, terminated, truncated, extras, reset_ids):
        infos = super()._process_extras(obs, terminated, truncated, extras, reset_ids)
        if len(reset_ids) == 0:
            return infos
        terminal = getattr(self.unwrapped, "_terminal_policy_observation", None)
        if terminal is None:
            raise RuntimeError("Missing pre-reset observation: use UR10eEpisodeEnv for SB3.")
        terminal = self._process_obs({"policy": terminal})
        for idx in reset_ids:
            if isinstance(terminal, dict):
                value = {key: item[idx].copy() for key, item in terminal.items()}
            else:
                value = terminal[idx].copy()
            infos[idx]["terminal_observation"] = value
        return infos
