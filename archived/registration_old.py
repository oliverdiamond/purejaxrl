from ..src.environments.gridworld import Maze, TwoRooms

def make(env_id: str, **env_kwargs):
    if env_id == "Maze":
        env = Maze(**env_kwargs)
    elif env_id == "TwoRooms":
        env = TwoRooms(**env_kwargs)
    else:
        raise ValueError("Environment ID is not assosiated with a registered Single Task Env.")

    return env, env.default_params

def make_multitask(env_id: str, **env_kwargs):
    if env_id == "TwoRoomsMT":
        env = TwoRoomsMT(**env_kwargs)
    elif env_id == "TwoRoomsMT5":
        env = TwoRoomsMT5(**env_kwargs)
    elif env_id == "TwoRoomsMT15":
        env = TwoRoomsMT15(**env_kwargs)
    elif env_id == "TwoRoomsMTFixedHallway5":
        env = TwoRoomsMTFixedHallway5(**env_kwargs)
    elif env_id == "TwoRoomsMTFixedStart5":
        env = TwoRoomsMTFixedStart5(**env_kwargs)
    elif env_id == "TwoRoomsMTHallwayAsTask":
        env = TwoRoomsMTHallwayAsTask(**env_kwargs)
    elif env_id == "TwoRoomsMTHallwayAsTask5":
        env = TwoRoomsMTHallwayAsTask5(**env_kwargs)
    elif env_id == "TwoRoomsMTHallwayAsTask15":
        env = TwoRoomsMTHallwayAsTask15(**env_kwargs)
    elif env_id == "TwoRoomsMTHallwayAsTaskRandomStart5":
        env = TwoRoomsMTHallwayAsTaskRandomStart5(**env_kwargs)
    else:
        raise ValueError("Environment ID is not assosiated with a registered Multitask Env.")

    return env, env.default_params