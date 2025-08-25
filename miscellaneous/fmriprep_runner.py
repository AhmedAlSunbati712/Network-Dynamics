import clustrix

@clustrix.cluster(cores=16, memory='64GB', time='02:00:00')
def run_fmriprep(sub, base_dir, output_path, work_path, license_path):
    import subprocess
    cmd = [
        'fmriprep-docker',
        base_dir,
        output_path,
        'participant',
        '--participant-label', sub.replace('sub-', ''),
        '--write-graph',
        '--fs-no-reconall',
        '--notrack',
        '--fs-license-file', license_path,
        '--work-dir', work_path
    ]
    try:
        print(f"Running fmriprep for {sub}...")
        subprocess.run(cmd, check=True)
        print(f"Finished processing {sub}\n")
    except subprocess.CalledProcessError as e:
        print(f"Error running fmriprep for {sub}: {e}")
        raise
