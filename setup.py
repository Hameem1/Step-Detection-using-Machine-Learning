#!/usr/bin/env python3

import os
from pathlib import Path
container_dir = Path("/StepDetectionProject")
host_dir = Path(os.getcwd())

# Building the image
os.system(f"docker build -t step-detection:latest .")
print(f"The Image has been created/pulled!\n")

# Running the container from the image
print(f"Running the container with this image...\n")
os.system(f"docker run -it --name app "
          f"-p 5000-5001:5000-5001 "
          f"-v /.:{container_dir} "
          f"step-detection:latest "
          f"bash")
print(f"The container has been exited!\n")

