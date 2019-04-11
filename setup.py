#!/usr/bin/env python3

import os, subprocess
from pathlib import Path
container_dir = Path("/StepDetectionProject")

subprocess.call("clear")

subprocess.call("docker pull hameem/step-detection:dev", shell=True)
# print(f"The Image has been created/pulled!\n")

# Running the container from the image
print(f"Running the container with this image...\n")
subprocess.call("docker-compose run --rm --service-ports application", shell=True)

print(f"\nThe container has been exited!\n")

