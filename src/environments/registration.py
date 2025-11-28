from . import gridworld
from util.wrappers import NavixGymnaxWrapper

def make(env_id: str, **env_kwargs):
    # Dynamically get the class from gridworld module by name
    if hasattr(gridworld, env_id):
        env_class = getattr(gridworld, env_id)
        env = env_class(**env_kwargs)
        return env, env.default_params
    elif env_id.startswith("Navix-"):
        env = NavixGymnaxWrapper(env_id, **env_kwargs)
        return env, None
    else:
        raise ValueError(f"Environment ID '{env_id}' is not associated with a registered Env.")