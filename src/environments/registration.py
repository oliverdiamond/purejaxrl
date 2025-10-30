from .gridworld import MazeRGB, TwoRoomsRGB, MazeOneHot, TwoRoomsOneHot

def make(env_id: str, **env_kwargs):
    if env_id == "MazeRGB":
        env = MazeRGB(**env_kwargs)
    elif env_id == "TwoRoomsRGB":
        env = TwoRoomsRGB(**env_kwargs)
    elif env_id == "MazeOneHot":
        env = MazeOneHot(**env_kwargs)
    elif env_id == "TwoRoomsOneHot":
        env = TwoRoomsOneHot(**env_kwargs)
    else:
        raise ValueError("Environment ID is not assosiated with a registered Single Task Env.")

    return env, env.default_params