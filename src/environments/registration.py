from . import gridworld

def make(env_id: str, **env_kwargs):
    # Dynamically get the class from gridworld module by name
    if hasattr(gridworld, env_id):
        env_class = getattr(gridworld, env_id)
        env = env_class(**env_kwargs)
    else:
        raise ValueError(f"Environment ID '{env_id}' is not associated with a registered Single Task Env.")

    return env, env.default_params