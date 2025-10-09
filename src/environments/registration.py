from .rooms import TwoRooms, TwoRooms15, TwoRooms5
from .maze import Maze
from .rooms_multitask import TwoRoomsMT, TwoRoomsMT5, TwoRoomsMT15, TwoRoomsMTFixedHallway5, TwoRoomsMTFixedStart5, TwoRoomsMTHallwayAsTask, TwoRoomsMTHallwayAsTask5, TwoRoomsMTHallwayAsTask15, TwoRoomsMTHallwayAsTaskRandomStart5

def make(env_id: str, **env_kwargs):
    if env_id == "Maze":
        return Maze(**env_kwargs)
    elif env_id == "TwoRooms":
        env = TwoRooms(**env_kwargs)
    elif env_id == "TwoRooms5":
        env = TwoRooms5(**env_kwargs)
    elif env_id == "TwoRooms15":
        env = TwoRooms15(**env_kwargs)
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