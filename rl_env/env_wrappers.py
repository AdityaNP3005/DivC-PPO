"""Ordered wrapper traversal without Isaac, SB3, policy or control dependencies."""


def find_isaac_env(obj, *, required_attrs, visited=None, envs_types=None):
    """Find the first matching owner; preserve each caller's envs-container policy.

    Callbacks historically accept only list/tuple envs; collectors also traverse
    other iterables. Attribute order and the shared cycle guard are significant.
    """
    visited = set() if visited is None else visited
    if obj is None or id(obj) in visited:
        return None
    visited.add(id(obj))
    if all(hasattr(obj, name) for name in required_attrs):
        return obj
    for attr in ('venv', 'env', 'unwrapped', '_env', '_wrapped_env'):
        found = find_isaac_env(getattr(obj, attr, None), required_attrs=required_attrs,
                              visited=visited, envs_types=envs_types)
        if found is not None:
            return found
    envs = getattr(obj, 'envs', None)
    if envs_types is not None:
        envs = envs if isinstance(envs, envs_types) else ()
    else:
        envs = envs or ()
    for sub in envs:
        found = find_isaac_env(sub, required_attrs=required_attrs,
                              visited=visited, envs_types=envs_types)
        if found is not None:
            return found
    return None


def training_scene_env(training_env):
    """Scene owner behind an SB3 training env (callbacks: list/tuple envs traversal only)."""
    env = find_isaac_env(training_env, required_attrs=("scene", "num_envs"), envs_types=(list, tuple))
    if env is None:
        raise RuntimeError("IsaacLab env not found in the training_env wrapper chain")
    return env
