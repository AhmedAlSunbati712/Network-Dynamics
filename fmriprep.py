import os
import clustrix
import subprocess

from fmriprep_runner import run_fmriprep  
clustrix.configure(
    cluster_type="slurm",
    cluster_host="ndoli.dartmouth.edu",
    username="f006w08",
    password="<Password>",
    remote_work_dir="/dartfs-hpc/rc/home/8/f006w08/clustrix",
    python_executable="python3",
    use_1password=False,
    default_cores=16,
    default_memory="64GB",
    default_time="01:00:00",
    default_partition="standard",
    module_loads=["python"],
    environment_variables={"OMP_NUM_THREADS": "1"},
    pre_execution_commands=[
        "export PATH=$HOME/.local/bin:/usr/bin:$PATH",
        "which python3 || echo 'Python3 not found in PATH'",
        "module list"
    ]
)

base_dir = os.path.abspath('./BIDSdffr2')
output_path = os.path.abspath('./preprocessed')
work_path = os.path.abspath('./work')
license_path = os.path.expanduser('~/Dropbox/Dartbrains/License/license.txt')

subjects = sorted([
    d for d in os.listdir(base_dir)
    if d.startswith('sub-') and os.path.isdir(os.path.join(base_dir, d))
])

run_fmriprep(subjects[0], base_dir, output_path, work_path, license_path)
