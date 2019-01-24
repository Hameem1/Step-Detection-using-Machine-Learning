# Step Analyzer

## Steps:

**1)** Copy the data set folder named **"OU-InertGaitAction_wStepAnnotation"** into this project directory.

**2)** Packages required : given in the **"requirements.txt"** file

**3)** Run **app.py** to run everything automatically.

## Recent Changes:

- Implemented a window to go over every feature calculation
- Created an entire module for generating features from the data
- Added functionality for creating a .csv file with Filename, Id, Gender, Age
- Deprecated the use of the end_tab_remover function
- Made major changes to the overall structure of the code
- Upgraded the graphing capabilities to be web based (using Dash)

## What the code does:

- Cleans up the default data set as it requires some normalization
- Creates a new Subject object (this sets the stage for accessing everything about that subject)
- Demonstrates the use of the data_plot function to view a web based graphing interface
- Generates a list of features currently available in the features.py module
- Displays the generated features

---

### Note:

- Read the in-code documentation to follow along with what is being done.
