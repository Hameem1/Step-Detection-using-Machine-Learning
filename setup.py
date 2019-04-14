#!/usr/bin/env python3

# This script pulls the image from dockerhub & runs the container with the image, using docker-compose.
# The container is removed upon exiting from it.

import os, subprocess

IMAGE_NAME = "hameem/step-detection:dev"

subprocess.call("clear")

# Pulling the image from dockerhub
subprocess.call(f"docker pull {IMAGE_NAME}", shell=True)
print(f"The Image has been pulled!\n")

# Running a container from the image
print(f"Running the container with image : {IMAGE_NAME}\n")
subprocess.call("docker-compose run --rm --service-ports application", shell=True)

print(f"\nThe container has been exited!\n")
