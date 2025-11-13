from collections.abc import Iterable, Mapping
import os
import pickle
import filelock as fl
import src.experiment.experiment_model as experiment


def write_finished_indices(exp, fpath, unfinished_indices):
    # Compute the experiment indices that have been already completed
    perms = set((perm for perm in range(exp.numPermutations())))
    completed_indices = perms.difference(set(unfinished_indices))

    # Create the data to save
    finished_idxs = set()
    params_dict: dict[int, Mapping] = {}
    for ind in completed_indices:
        params_dict[ind] = exp.getPermutation(ind)
        finished_idxs.add(ind)

    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(fpath), exist_ok=True)

    # Save completed indices and params
    lock = fl.FileLock(fpath + ".lock")
    with lock:
        with open(fpath, "wb") as finished_jobs:
            pickle.dump((finished_idxs, params_dict), finished_jobs)


def load_unfinished_indices(exp, fpath):
    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    
    lock = fl.FileLock(fpath + ".lock")
    # Open and load `finished_indices.pkl`
    indices_completed = set()
    if os.path.exists(fpath):
        with lock:
            with open(fpath, "rb") as finished_jobs:
                indices_completed, _ = pickle.load(finished_jobs)

    indices_completed = indices_completed
    perms = set((perm for perm in range(exp.numPermutations())))
    indices = perms.difference(indices_completed)
    return sorted(list(indices))


def gather_missing(paths: Iterable[str], loader=experiment.load, all=False, base="./"):
    path_to_indices: dict[str, list[int]] = {}

    for path in paths:
        exp = loader(path)
        size = exp.numPermutations()
        context = exp.buildSaveContext(0)
        fpath = os.path.join(context.resolve(), "finished_indices.pkl")

        indices: list[int]
        if all:
            indices = list((perm for perm in range(exp.numPermutations())))
        else:
            indices = load_unfinished_indices(exp, fpath)

            # Write 'finished_indices.pkl' if it doesn't exist so that we can
            # use it next time
            if not os.path.exists(fpath):
                write_finished_indices(exp, fpath, indices)
        # else:
        #     indices = list(detectMissingIndices(exp, 1, base=base))

        #     # Write 'finished_indices.pkl' if it doesn't exist so that we can
        #     # use it next time
        #     if not context.exists(fname):
        #         write_finished_indices(exp, fname, indices)

        path_to_indices[path] = sorted(indices)
        print(f"\N{ESC}[33;1m{path}: \N{ESC}[30;0m{len(indices)} / {size}")

    return path_to_indices