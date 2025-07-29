from .rooms import TwoRooms, TwoRooms15, TwoRooms5
from .rooms_multitask import TwoRoomsMultiTask, TwoRoomsMultiTask5, TwoRoomsMultiTask15, TwoRoomsMultiTaskEasy, TwoRoomsMultiTaskEasy5, TwoRoomsMultiTaskEasy15

def make(env_id: str, **env_kwargs):
    if env_id == "TwoRooms":
        env = TwoRooms(**env_kwargs)
    elif env_id == "TwoRooms5":
        env = TwoRooms5(**env_kwargs)
    elif env_id == "TwoRooms15":
        env = TwoRooms15(**env_kwargs)
    else:
        raise ValueError("Environment ID is not assosiated with a registered Single Task Env.")

    return env, env.default_params

def make_multitask(env_id: str, **env_kwargs):
    if env_id == "TwoRoomsMultiTask":
        env = TwoRoomsMultiTask(**env_kwargs)
    elif env_id == "TwoRoomsMultiTask5":
        env = TwoRoomsMultiTask5(**env_kwargs)
    elif env_id == "TwoRoomsMultiTask15":
        env = TwoRoomsMultiTask15(**env_kwargs)
    elif env_id == "TwoRoomsMultiTaskEasy":
        env = TwoRoomsMultiTaskEasy(**env_kwargs)
    elif env_id == "TwoRoomsMultiTaskEasy5":
        env = TwoRoomsMultiTaskEasy5(**env_kwargs)
    elif env_id == "TwoRoomsMultiTaskEasy15":
        env = TwoRoomsMultiTaskEasy15(**env_kwargs)
    else:
        raise ValueError("Environment ID is not assosiated with a registered Multitask Env.")

    return env, env.default_params