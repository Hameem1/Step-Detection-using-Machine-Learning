# Step Detection using ML

The problem of step detection in human gait patterns requires mathematical models which can automatically assess whether a step has been taken or not. 
Performing strictly rule based analysis on the data has proven to be effective yet cumbersome. 
This implementation constructs a machine learning pipeline to accomplish the same task by using carefully selected features from a large dataset of labeled gait patterns. 
In addition, the importance of feature selection is analyzed and is used to supplement model performance by neglecting any features which do not prove to strongly correlate with the target variable. 

The results determine that a random forest classifier trained on a set of selected features performs higher than 90% on all of the test metrics used in the evaluation, i.e. accuracy, precision, recall and f1 score.


## Overview

* This project is developed using Python 3.7.
* All directories with lowercase names are custom Python packages.
* The function docstrings follow the "numpydoc" format (https://numpydoc.readthedocs.io/en/latest/format.html).
* Two config files are used to modify overall program behavior:
    * config.py
    * model_config.py
* The Flask web framework is used to expose two endpoints for data visualization:
    * http://localhost:5000 (Raw Data)
    * http://localhost:5001 (Feature Data)
* To mitigate the "works on my machine?" problem, the project runs in a Docker container.
* The image can be built from scratch using the Dockerfile and docker-compose.yml files, however, 
a pre-built image has been made available on dockerhub.
* Running the 'setup.py' file automates the docker environment creation.
* The project directory is used as a shared volume between the host system and the container. This allows for easy development using the docker environment.
* This implementation is compatible with both Windows and Linux systems.
* The data used in this implementation comes from : http://www.am.sanken.osaka-u.ac.jp/BiometricDB/SimilarActionsInertialDB.html

## Installation:

**1)** Clone this repository to your local machine.

**2)** Install Docker and perform necessary configurations (explained in the next section).

**3)** Copy the dataset folder named **"OU-InertGaitAction_wStepAnnotation"** into the **Data Sets** directory of the cloned project.

**4)** Open "Git Bash" (or similar) inside the project directory. 

**5)** Run `python setup.py`. This will setup the Docker environment and land you inside the container's terminal.

**6)** Run `ls` to verify that all the project files are available inside the container.

**7)** Run `python app.py` inside the container. This will perform the pre-processing and visualization steps on the original dataset.


## Setting up Docker

### For Windows

**1)** Make sure any VMM software (e.g. VirtualBox) is uninstalled/disabled, as their settings for the windows 'Hyper-V' 
hypervisor conflict with docker.

**2)** Download and install "Docker Desktop for Windows" (set it to use linux based containers):
https://hub.docker.com/editions/community/docker-ce-desktop-windows

**3)** Docker requires the network connection to be private. To make sure this is the case, open windows powershell as administrator 
and execute the following command:
 
 `Set-NetConnectionProfile -interfacealias "vEthernet (DockerNAT)" -NetworkCategory Private`

**4)** In the notification tray, right-click docker's whale icon and go to "Settings".

**5)** If using a VPN client (e.g. Cisco Anyconnect), docker's network address needs to be configured so that it doesn't 
clash with the VPN client. For this, go to the "Network" tab and set the "Subnet Address" to `192.168.100.0`, or an address which is 
allowed by the VPN route settings. This can found in the vpn client's route details.

**6)** Go to the General tab and check the box that says - `Expose daemon on tcp://localhost:2375 without TLS`. This allows 
connections from development tools like PyCharm. Warning: This step may require turning off any 3rd party Anti-virus software.

Optionally: Uncheck the box which says `Start Docker Desktop when you log in`, as this program may slow down startup.

**7)** Go to the "Shared Drives" tab and check the box for whichever drive the project resides in (this is to allow volume mounting).

**8)** Restart the computer for good measure.

**9)** Make sure Docker is running in the system tray.

**10)** Run this command to ensure a functioning install:
`docker run hello-world`

### For Linux

Getting Docker to work on Linux is pretty straightforward. Just follow the steps in the official documentation:

https://docs.docker.com/install/linux/docker-ce/ubuntu/


## IDE Tips

From personal experience; PyCharm works best for Dockerized Python applications. 
VS Code is also great, but as of April 2019, it lacks the ability to use the container's Python interpreter. To get 
around this problem, a virtual environment can be made using the provided requirements.txt file. 
This environment's interpreter can then aid in development, however, this is just an easy workaround and will be 
rendered unnecessary with upcoming updates to VS Code. A terminal (like GitBash) can also be used rather easily, since 
the project uses a docker-compose file to automate all the configurations.


## Recent Changes

- Configured the project to run in a container environment using Docker.
- Fixed minor issues with the visualization dashboards.
- Automated project installation by creating setup.py.


## What the code does

- Cleans up the default data set as it requires pre-processing.
- Creates a new Subject object (this sets the stage for accessing all the data of that subject).
- Implements web-based dashboards for data visualization.
- Generates 51 time-domain features using a sliding window.
- Applies super-sampling to counter class imbalance. 
- Applies recursive feature elimination with cross validation to select the best features.
- Trains a Random Forest Classifier and Data Normalization Model.
- Exports the trained models for deployment.

---

### Note:

- Read the in-code documentation to follow along with what is being done.
