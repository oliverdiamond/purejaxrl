#!/usr/bin/env python3

import os
import pickle
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from typing import IO, Optional, Union

import filelock as fl

sys.path.insert(0, ".")

import argparse
import dataclasses
import math
import time
from functools import partial

import PyExpUtils.runner.Slurm as Slurm
#from PyExpUtils.runner.utils import approximate_cost
from PyExpUtils.utils.cmdline import flagString
from PyExpUtils.utils.generator import group

import src.experiment.experiment_model as experiment
from util import gather_missing

_default_history_file = f"{os.path.expanduser('~')}/.slurm_history"

parser = argparse.ArgumentParser()
parser.add_argument(
    "-a", "--all", action="store_true", help=("force run all indices")
)
parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    help=(
        "run the script specified by the '--entry' argument with "
        + "the '--verbose' flag"
    ),
)
parser.add_argument(
    "--cpu-venv",
    action="store_true",
    default=False,
    help="If local-venv is false, use the virtual environment located in ./venv_cpu.tar.xz " 
    +"otherwise use the virtual environment with jax[cuda12] located in ./venv.tar.xz",
)
parser.add_argument(
    "--history-file",
    type=str,
    required=False,
    default=_default_history_file,
    help="the history file to write history to",
)
parser.add_argument(
    "-r",
    "--run-local",
    action="store_true",
    help="immediately run the job locally instead of scheduling",
)
parser.add_argument(
    "-L",
    "--local-venv",
    action="store_true",
    help="use the local virtual environment located in ./.venv/",
)
parser.add_argument(
    "--cluster", type=str, required=True, help="config for slurm job options"
)
parser.add_argument(
    "-e", type=str, nargs="+", required=True, help="experiment config file"
)
parser.add_argument(
    "--entry",
    type=str,
    required=True,
    help="experiment file to run",
)
parser.add_argument(
    "--mode",
    type=str,
    choices=["vec", "seq", "par"],
    default="seq",
    help=(
        "controls how experiment permutations are scheduled for each provided config file: "
        + "'vec' - all permutations vectorized in a single job, "
        + "'seq' - permutations are split into batches of size `--perms_per_group` and run sequentially in separate jobs, "
        + "'par' - permutations are split into batches of size slurm.sequential*slurm.cores and then run in seperate jobs. "
        + "        For each job, perms are run in parallel on separate cpu cores using nvidia MPS."
    ),
)
parser.add_argument(
    "--perms-per-group",
    type=int,
    default=1,
    help="number of permutations to run sequentially in each job when --run-mode=seq is used",
)

parser.add_argument(
    "--results",
    type=str,
    default="./",
    help="parent directory to save results in",
)
parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="output jobscript and information, but do not schedule",
)
parser.add_argument(
    "--fix_cores",
    action="store_true",
    default=False,
    help="When mode=par, fix the number of cores requested to be the same for all jobs,"
    + "as specified in the cluster config."
    + "Otherwise number of cores will be reduced depending on group size.",
)
parser.add_argument(
    "-f",
    "--file",
    type=str,
    default="",
    help="use the specified file name for the generated job script; "
    + "if unset, generates a temporary file for the jobscript",
)
parser.add_argument(
    "-y", "--yes", action="store_true", help="automatic yes to prompts"
)
# parser.add_argument(
#     "--checkpoint-every",
#     type=float,
#     default=-1.0,
#     help=(
#         "minimum number of minutes to wait between checkpoints, set < 0 "
#         + "to turn off checkpointing; checkpointing only occurs between "
#         + "jitted rollout steps, determined by the 'rollout_steps' key "
#         + "in the relevant configuration file"
#     ),
# )
parser.add_argument(
    "-g",
    "--gpu",
    type=int,
    required=True,
    help="gpus per node",
)
parser.add_argument(
    "--gpu-mem",
    type=int,
    choices=[10, 20, 40, 80],
    required=False,
    default=80,
    help="(fir cluster only) Specify total gpu memory to use in GB, must be one of [10, 20, 40, 80]."+
         "Smaller values result in faster scheduling and more efficient usage." + 
         "If not specified, will use the full gpu memory.",
)
parser.add_argument(
    "--preallocate",
    action=argparse.BooleanOptionalAction,
    default=None,
    help=(
        "controls if jax pre-allocates by setting "
        + "`XLA_PYTHON_CLIENT_PREALLOCATE`, if `false`, then "
        + "`XLA_PYTHON_CLIENT_MEM_FRACTION` is set based on "
        + "the `SLURM_GPUS_PER_NODE` and `SLURM_NTASKS_PER_NODE` environment "
        + "variables"
    ),
)
parser.add_argument(
    "--dependency",
    type=str,
    required=False,
    default="",
    help="the job dependency, specified like sbatch",
)

cmdline = parser.parse_args()
if cmdline.gpu > 0 and cmdline.preallocate is None:
    parser.error("-g/--gpu requires --preallocate/--no-preallocate")

SlurmNodeOptions = Union[Slurm.SingleNodeOptions, Slurm.MultiNodeOptions]

ANNUAL_ALLOCATION = 724


####################################################################
# Generate scheduling bash script
####################################################################
cwd = os.getcwd()
project_name = os.path.basename(cwd)

# The directory that holds the virtual environment
venv: str

# The original virtual environment to use, either the directory which held the
# venv or a tar'd version which is extracted to the temporary slurm directory
venv_origin: str

if not cmdline.local_venv:
    if cmdline.cpu_venv:
        venv_origin = f"{cwd}/venv_cpu.tar.xz"
    else:
        venv_origin = f"{cwd}/venv.tar.xz"
    venv = "$SLURM_TMPDIR"
else:
    venv = "."
    venv_origin = ".venv"


def get_xla_python_client_preallocate(options: argparse.Namespace) -> str:
    using_gpu = options.gpu

    if using_gpu and not options.preallocate:
        return "XLA_PYTHON_CLIENT_PREALLOCATE=false"

    return ""


def get_xla_python_client_mem_fraction(
    options: argparse.Namespace, 
    sub: SlurmNodeOptions, 
    p: float = 0.10
) -> str:
    gpus_per_node = options.gpu
    using_gpu = gpus_per_node > 0

    if using_gpu and options.preallocate:
        # preallocate (1 - p)% of the GPU, spread across jax processes
        cores = sub.cores
        thresh = p * int(gpus_per_node)
        gpu_percent_per_core = (gpus_per_node - thresh) / cores
        return f"XLA_PYTHON_CLIENT_MEM_FRACTION={gpu_percent_per_core:.5f}"

    return ""


def get_cuda_visible_devices(options: argparse.Namespace) -> str:
    using_gpu = options.gpu

    if using_gpu:
        return "CUDA_VISIBLE_DEVICES=0"

    return 'CUDA_VISIBLE_DEVICES=""'


def get_jax_platforms(options: argparse.Namespace) -> str:
    using_gpu = options.gpu

    if using_gpu:
        return "JAX_PLATFORMS=cuda,cpu"

    return "JAX_PLATFORMS=cpu"


def get_env(options: argparse.Namespace, sub: SlurmNodeOptions) -> tuple[str, list[str]]:
    xla_python_client_preallocate = get_xla_python_client_preallocate(options)
    xla_python_client_mem_fraction = get_xla_python_client_mem_fraction(
        options,
        sub
    )
    cuda_visible_devices = get_cuda_visible_devices(options)
    jax_platforms = get_jax_platforms(options)

    env_vars = [
        xla_python_client_mem_fraction,
        xla_python_client_preallocate,
        cuda_visible_devices,
        jax_platforms,
    ]

    return " ".join(env_vars), env_vars


def run_nvidia_smi(options: argparse.Namespace) -> str:
    using_gpu = options.gpu
    if using_gpu:
        return "\n".join(
            [
                "# Log nvidia-smi and CPU info every minute for the first 20 minutes, then every 30 minutes thereafter",
                "echo \"timestamp,name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,power.limit,gpu_uuid,cpu_mem_used_gb,cpu_mem_total_gb,cpu_mem_percent,cpu_load_avg_1min,cpu_cores_available,cpu_cores_requested\" > \"$SCRATCH/nvidia_smi_$SLURM_JOB_ID.csv\"",
                "{",
                "    # First 20 minutes - log every minute",
                "    for i in {1..20}; do",
                "        TIMESTAMP=$(date '+%Y/%m/%d %H:%M:%S.%3N')",
                "        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,power.limit,gpu_uuid --format=csv,noheader,nounits)",
                "        CPU_MEM_USED_GB=$(free -m | awk 'NR==2{printf \"%.2f\", $3/1024}')",
                "        CPU_MEM_TOTAL_GB=$(free -m | awk 'NR==2{printf \"%.2f\", $2/1024}')",
                "        CPU_MEM_PERCENT=$(free | awk 'NR==2{printf \"%.1f\", $3*100/$2}')",
                "        CPU_LOAD=$(uptime | awk -F'load average:' '{print $2}' | awk -F',' '{print $1}' | xargs)",
                "        CPU_CORES_AVAILABLE=$(nproc)",
                "        CPU_CORES_REQUESTED=${SLURM_CPUS_PER_TASK:-1}",
                "        echo \"$TIMESTAMP,$GPU_INFO,$CPU_MEM_USED_GB,$CPU_MEM_TOTAL_GB,$CPU_MEM_PERCENT,$CPU_LOAD,$CPU_CORES_AVAILABLE,$CPU_CORES_REQUESTED\" >> \"$SCRATCH/nvidia_smi_$SLURM_JOB_ID.csv\"",
                "        sleep 60",
                "    done",
                "    # After 20 minutes - log every 30 minutes",
                "    while true; do",
                "        TIMESTAMP=$(date '+%Y/%m/%d %H:%M:%S.%3N')",
                "        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,power.limit,gpu_uuid --format=csv,noheader,nounits)",
                "        CPU_MEM_USED_GB=$(free -m | awk 'NR==2{printf \"%.2f\", $3/1024}')",
                "        CPU_MEM_TOTAL_GB=$(free -m | awk 'NR==2{printf \"%.2f\", $2/1024}')",
                "        CPU_MEM_PERCENT=$(free | awk 'NR==2{printf \"%.1f\", $3*100/$2}')",
                "        CPU_LOAD=$(uptime | awk -F'load average:' '{print $2}' | awk -F',' '{print $1}' | xargs)",
                "        CPU_CORES_AVAILABLE=$(nproc)",
                "        CPU_CORES_REQUESTED=${SLURM_CPUS_PER_TASK:-1}",
                "        echo \"$TIMESTAMP,$GPU_INFO,$CPU_MEM_USED_GB,$CPU_MEM_TOTAL_GB,$CPU_MEM_PERCENT,$CPU_LOAD,$CPU_CORES_AVAILABLE,$CPU_CORES_REQUESTED\" >> \"$SCRATCH/nvidia_smi_$SLURM_JOB_ID.csv\"",
                "        sleep 1800  # 30 minutes",
                "    done",
                "} &",
                "echo",
            ]
        )
    else:
        return "\n".join(
            [
                "# Log CPU info every minute for the first 20 minutes, then every 30 minutes thereafter",
                "echo \"timestamp,cpu_mem_used_gb,cpu_mem_total_gb,cpu_mem_percent,cpu_load_avg_1min,cpu_cores_available,cpu_cores_requested\" > \"$SCRATCH/cpu_$SLURM_JOB_ID.csv\"",
                "{",
                "    # First 20 minutes - log every minute",
                "    for i in {1..20}; do",
                "        TIMESTAMP=$(date '+%Y/%m/%d %H:%M:%S.%3N')",
                "        CPU_MEM_USED_GB=$(free -m | awk 'NR==2{printf \"%.2f\", $3/1024}')",
                "        CPU_MEM_TOTAL_GB=$(free -m | awk 'NR==2{printf \"%.2f\", $2/1024}')",
                "        CPU_MEM_PERCENT=$(free | awk 'NR==2{printf \"%.1f\", $3*100/$2}')",
                "        CPU_LOAD=$(uptime | awk -F'load average:' '{print $2}' | awk -F',' '{print $1}' | xargs)",
                "        CPU_CORES_AVAILABLE=$(nproc)",
                "        CPU_CORES_REQUESTED=$(SLURM_CPUS_PER_TASK)",
                "        echo \"$TIMESTAMP,$CPU_MEM_USED_GB,$CPU_MEM_TOTAL_GB,$CPU_MEM_PERCENT,$CPU_LOAD,$CPU_CORES_AVAILABLE,$CPU_CORES_REQUESTED\" >> \"$SCRATCH/cpu_$SLURM_JOB_ID.csv\"",
                "        sleep 60",
                "    done",
                "    # After 20 minutes - log every 30 minutes",
                "    while true; do",
                "        TIMESTAMP=$(date '+%Y/%m/%d %H:%M:%S.%3N')",
                "        CPU_MEM_USED_GB=$(free -m | awk 'NR==2{printf \"%.2f\", $3/1024}')",
                "        CPU_MEM_TOTAL_GB=$(free -m | awk 'NR==2{printf \"%.2f\", $2/1024}')",
                "        CPU_MEM_PERCENT=$(free | awk 'NR==2{printf \"%.1f\", $3*100/$2}')",
                "        CPU_LOAD=$(uptime | awk -F'load average:' '{print $2}' | awk -F',' '{print $1}' | xargs)",
                "        CPU_CORES_AVAILABLE=$(nproc)",
                "        CPU_CORES_REQUESTED=$(SLURM_CPUS_PER_TASK)",
                "        echo \"$TIMESTAMP,$CPU_MEM_USED_GB,$CPU_MEM_TOTAL_GB,$CPU_MEM_PERCENT,$CPU_LOAD,$CPU_CORES_AVAILABLE,$CPU_CORES_REQUESTED\" >> \"$SCRATCH/cpu_$SLURM_JOB_ID.csv\"",
                "        sleep 1800  # 30 minutes",
                "    done",
                "} &",
                "echo",
            ]
        )

def check_mps_enabled(options: argparse.Namespace) -> str:
    using_gpu = options.gpu
    if using_gpu:
        # Print nvidia-smi output at start of job
        return "\n".join(
            [
                "# Check if MPS is running",
                "ps -ef | grep mps",
                "echo",
            ]
        )
    return ""


def mps_script(
    runner: str,
    li: Sequence[int],
    options: argparse.Namespace,
    sub: SlurmNodeOptions,
) -> str:
    using_gpu = options.gpu

    nvidia_smi = run_nvidia_smi(options)
    mps_running = check_mps_enabled(options)

    env, env_vars = get_env(options, sub)
    env_str = "\n".join(env_vars)

    # Choose execution command based on mode
    if options.mode == "par":
        exec_cmd = f"parallel '{env} {runner} {{}}' ::: {' '.join([str(i) for i in li])}"
    else:
        exec_cmd = f"{env} {runner} {' '.join([str(i) for i in li])}"

    return f"""
echo
echo "Device: {"cpu" if not using_gpu else "gpu"}"
echo

echo
echo "=== === === === === === === === === "
echo "Setting the following env variables:"
echo "=== === === === === === === === === "
echo '{env_str}'

echo
echo "=== === === === === === === === === "
echo "MPS Running? (applicable: {"yes" if mps_running else "no"})"
echo "=== === === === === === === === === "
{mps_running}

{nvidia_smi}
{exec_cmd}
"""  # noqa: E501


def get_gpu_strings(options: argparse.Namespace):
    gpu = ""
    gpu_modules = ""
    gpu_mps = ""

    if options.gpu:
        if CLUSTER_NAME == "clusters/fir":
            if options.gpu_mem == 80:
                mig = "hbm3:1"
            elif options.gpu_mem == 40:
                mig = "hbm3_3g.40gb:1"
            elif options.gpu_mem == 20:
                mig = "hbm3_2g.20gb:1"
            elif options.gpu_mem == 10:
                mig = "hbm3_1g.10gb:1"
            gpu = f"#SBATCH --gpus-per-node=nvidia_h100_80gb_{mig}"
        else:
            gpu = f"#SBATCH --gpus-per-node={options.gpu}" 

        gpu_modules = """module load StdEnv/2023
module load python/3.11
module load gcc/12.3
module load cudacore/.12.2.2
module load cudnn/9.2.1.18
module load cuda/12.2
module load mujoco/3.1.6
        """

        gpu_mps = """mkdir -p $HOME/tmp
export CUDA_MPS_LOG_DIRECTORY=$HOME/tmp/nvidia-log
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
nvidia-cuda-mps-control -d
        """

    return gpu, gpu_modules, gpu_mps


def get_extract_env_cmd(options: argparse.Namespace):
    if not options.local_venv:
        return " ".join(
            [
                "srun --ntasks=$SLURM_NNODES --ntasks-per-node=1",
                f"tar -xf {venv_origin} -C {venv}",
            ]
        )
    return ""


def get_job_script(parallel: str, options: argparse.Namespace):
    gpu, gpu_modules, gpu_mps = get_gpu_strings(options)
    extract_venv = get_extract_env_cmd(options)

    # the contents of the string below will be the bash script that is
    # scheduled on compute Canada change the script accordingly (e.g. add the
    # necessary `module load X` commands)
    return f"""#!/bin/bash
#SBATCH --signal=B:SIGTERM@180
{gpu}

echo
echo "=== === === === === === === === === "
echo "Loading modules"
echo "=== === === === === === === === === "
echo '{gpu_modules}'
{gpu_modules}
module load mujoco

echo
echo "=== === === === === === === === === "
echo "MPS Options (if applicable)"
echo "=== === === === === === === === === "
echo '{"Not applicable" if not gpu_mps else gpu_mps}'
{gpu_mps}
echo "=== === === === === === === === === "

echo
echo "=== === === === === === === === === "
echo "Working directory: {cwd}"
echo "=== === === === === === === === === "
cd {cwd}

echo
echo "=== === === === === === === === === "
echo "Extracting directory: {"No, using local venv" if not extract_venv else extract_venv}"
echo "=== === === === === === === === === "
{extract_venv}

export MPLBACKEND=TKAgg
export OMP_NUM_THREADS=1

echo
echo "=== === === === === === === === === "
echo "Starting job"
echo "=== === === === === === === === === "
{parallel}
    """  # noqa: E501


####################################################################


####################################################################
# Environment check
####################################################################
if not cmdline.debug and not os.path.exists(venv_origin):
    if not cmdline.local_venv:
        print("WARNING: zipped virtual environment not found at:", venv_origin)
        print("Trying to make one now")
        print("Make sure to run `scripts/setup_cc.sh` first.")
        code = os.system("tar -caf venv.tar.xz env")
        if code:
            raise Exception("Failed to make virtual env")
    else:
        print("WARNING: virtual environment not found at:", venv_origin)
        print(
            "Make sure to run `scripts/setup_cc.sh` or otherwise setup the "
            "venv first."
        )
####################################################################

####################################################################
# Scheduling logic
####################################################################
slurm = Slurm.fromFile(cmdline.cluster)
CLUSTER_NAME = cmdline.cluster.rstrip('.json')

threads = (
    slurm.threads_per_task if isinstance(slurm, Slurm.SingleNodeOptions) else 1
)

# compute how many "tasks" to clump into each job
group_size = int(slurm.cores / threads) * slurm.sequential

# compute how much time the jobs are going to take
hours, minutes, seconds = slurm.time.split(":")
total_hours = int(hours) + (int(minutes) / 60) + (int(seconds) / 3600)
####################################################################


missing = gather_missing(cmdline.e, loader=experiment.load, all=cmdline.all)
####################################################################

####################################################################
# compute cost
####################################################################
memory = Slurm.memory_in_mb(slurm.mem_per_core)
# compute_cost = partial(
#     approximate_cost,
#     cores_per_job=slurm.cores,
#     mem_per_core=memory,
#     hours=total_hours,
# )
# cost = sum(
#     compute_cost(math.ceil(len(job_list) / group_size))
#     for job_list in missing.values()
# )
# perc = (cost / ANNUAL_ALLOCATION) * 100

# print(
#     f"Expected to use {cost:.2f} core years, which is {perc:.4f}% "
#     + "of our annual allocation"
# )
if not cmdline.debug:
    input("Press Enter to confirm or ctrl+c to exit")
####################################################################


####################################################################
# log to history file
####################################################################
def log_history(history_file):
    lines = []
    if os.path.isfile(history_file):
        with open(history_file, "r") as infile:
            lines = infile.readlines()

    with open(history_file, "w") as outfile:
        if len(lines) > 5000:
            lines = lines[1:]
        cmd = os.getcwd() + " ".join(sys.argv)[1:]
        lines.append(f"$ {cmd}\n")
        outfile.writelines(lines)


log_history(cmdline.history_file)
####################################################################


####################################################################
# helper functions for scheduling
####################################################################
def to_cmdline_flags(options):
    Slurm.validate(options)
    args = [
        ("--account", options.account),
        ("--time", options.time),
        ("--mem-per-cpu", options.mem_per_core),
        ("--output", options.log_path),
    ]
    args += [
        ("--ntasks", 1),
        ("--nodes", 1),
        ("--cpus-per-task", options.cores),
    ]

    return flagString(args)


def schedule(
    runner: str,
    l: Sequence[int],  # noqa: E741
    sub: SlurmNodeOptions,
    options: argparse.Namespace,
) -> None:
    # generate the bash script which will be scheduled
    using_gpu = options.gpu
    parallel = mps_script(runner, l, options, sub)
    script = get_job_script(parallel, options)

    if options.debug:
        print(to_cmdline_flags(sub))
        print(script)
        print("############################################################\n"
              "############################################################\n")
        return

    __schedule(
        script,
        sub,
        options,
        prefix="auto_mps_" if using_gpu else "auto_cpu_",
    )
    # DO NOT REMOVE. This will prevent you from overburdening the slurm
    # scheduler. Be a good citizen.
    time.sleep(2)


def __schedule(
    script: str,
    opts: Optional[SlurmNodeOptions],
    options: argparse.Namespace,
    prefix: str = "",
):
    def __inner_schedule(jobscript: IO[str], local: bool):
        jobscript.write(script)
        jobscript.close()

        if local:
            os.system(f"bash ./{jobscript.name}")
            return

        cmd_args = ""
        if opts is not None:
            cmd_args = to_cmdline_flags(opts)

        # Set job dependency
        dep = ""
        if options.dependency:
            dep = "--dependency {dependency}"

        fname = jobscript.name
        jobname = f"-J {fname}"
        os.system(f"sbatch {jobname} {dep} {cmd_args} {fname}")
        return

    if options.file:
        with open(options.file, "w") as jobscript:
            __inner_schedule(jobscript, cmdline.run_local)
    else:
        fname = ""
        with tempfile.NamedTemporaryFile(
            mode="w", prefix=prefix, delete=False, suffix=".sh"
        ) as jobscript:
            __inner_schedule(jobscript, cmdline.run_local)
            fname = jobscript.name

        # On Python versions <= 3.12, the temp file will not be cleaned up, so
        # do that manually here, if the file exists
        if fname and os.path.isfile(fname):
            os.remove(fname)


def schedule_indicies(path, indicies, sub):
    print("scheduling:", path, indicies)
    # build the executable string instead of activating the venv every
    # time, just use its python directly
    runner = " ".join(
        [
            f"{venv}/.venv/bin/python",
            f"{cmdline.entry}",
            "--gpu" if cmdline.gpu else "--no-gpu",
            f"-e {path}",
            "-v" if cmdline.verbose else "",
            "-i",
        ]
    )

    schedule(runner, indicies, sub, cmdline)
    

####################################################################


####################################################################
# start scheduling
####################################################################


if cmdline.mode == "vec":
    assert "vec" in cmdline.entry, "Vectorized entry point required for vectorized scheduling"
    sub = dataclasses.replace(slurm, cores=1)
    for path, indicies in missing.items():
        schedule_indicies(path, indicies, sub)
elif cmdline.mode == "seq":
    assert "vec" not in cmdline.entry, "Non-vectorized entry point required for sequential scheduling"
    sub = dataclasses.replace(slurm, cores=1)
    for path in missing:
        for g in group(missing[path], cmdline.perms_per_group):
            schedule_indicies(path, list(g), sub)
elif cmdline.mode == "par":
    #TODO In the future it might be nice to to vmap over hypers but split sweeps for different envs across CPU cores, but probably not much more efficient than this approach.
    assert "vec" not in cmdline.entry, "Non-vectorized entry point required for splitting jobs across CPU cores"
    for path in missing:
        for g in group(missing[path], group_size):
            l = list(g)
            # make sure to only request the number of CPU cores necessary
            tasks = group_size if cmdline.fix_cores else min([group_size, len(l)])
            par_tasks = max(int(tasks // slurm.sequential), 1)
            cores = par_tasks * threads
            sub = dataclasses.replace(slurm, cores=cores)
            schedule_indicies(path, l, sub)