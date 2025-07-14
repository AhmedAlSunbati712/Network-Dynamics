import os
import subprocess

base_dir = os.path.abspath('./data/BIDSdffr')
output_path = os.path.abspath('./data/preprocessed')
work_path = os.path.abspath('./data/work')

subjects = sorted([d for d in os.listdir(base_dir) if d.startswith('sub-') and os.path.isdir(os.path.join(base_dir, d))])

for sub in subjects:
    sub_label = sub.replace('sub-', '')

    cmd = [
        'fmriprep-docker',
        base_dir,
        output_path,
        'participant',
        '--participant-label', sub,
        '--write-graph',
        '--fs-no-reconall',
        '--notrack',
        '--fs-license-file', os.path.expanduser('~/Dropbox/Dartbrains/License/license.txt'),
        '--work-dir', work_path
    ]

    print(f"Running fmriprep for {sub}...")
    try:
        subprocess.run(cmd, check=True)
        print(f"Finished processing {sub}\n")
    except subprocess.CalledProcessError as e:
        print(f"Error running fmriprep for {sub}: {e}")
