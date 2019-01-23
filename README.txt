TODO: Turn this into a README.md file

Steps:
------
1) Copy the data set folder named "OU-InertGaitAction_wStepAnnotation" into this project directory.

2) Packages required : given in the "requirements.txt" file

3) Run app.py to run everything automatically.


What the code does:
-------------------
- Cleans up the default data set as it requires some normalization
- Creates a new Subject object (this sets the stage for accessing everything about that subject)
- Demonstrates the use of the data_plot function to plot different sections of the data set
- Advanced plotting capabilities using the plotly library
- Web based interface and interactive component selection using Dash

Changes:
--------
- Added functionality for creating a .csv file with Filename, Id, Gender, Age
- Fixed the dataset filtering based on labels (added the null_data label)
- Deprecated the use of the end_tab_remover function
- Made major changes to the overall structure of the code
- Allowed access to the entire library as a whole and as well as separate modules
- Fixed the step marker problem. Preferred use with "valid" mode
- Optimized the plotly code
- Converted the program to be web based using Dash

Note:
-----
Read the documentation to follow along with what is being done.
