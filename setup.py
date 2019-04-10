#!/usr/bin/env python3

import os, subprocess
from pathlib import Path
container_dir = Path("/StepDetectionProject")
host_dir = Path(os.getcwd())

BUILD = False
COMMAND = "bash"
subprocess.call("clear")

if BUILD:
    # Building the image
    subprocess.call("docker build -t step-detection:latest .", shell=True)
    print(f"The Image has been created/pulled!\n")

# Running the container from the image
print(f"Running the container with this image...\n")
subprocess.call(f"docker run -it --rm --name app "
                f"-p 5000-5001:5000-5001 "
                f"-v /.:{container_dir} "
                f"step-detection:latest "
                f"{COMMAND}", shell=True)

print(f"\nThe container has been exited!\n")

