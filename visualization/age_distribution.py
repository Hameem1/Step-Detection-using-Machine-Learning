import os
import plotly.offline as pyo
import plotly.graph_objs as go
from config import data_files_path
from data_generator.dataset_generator import read_csv


def gen_age_histogram(open_plot=True):
    """
    Generates a histogram representing the distribution of Subject Age by Gender.

    Parameters
    ----------
    open_plot : bool, optional
        Show the plot after generation (default = True)

    """

    df = read_csv(f'{data_files_path}\\subject_data')
    male = go.Histogram(x=df[df['Gender'] == 'Male']['Age'],
                        xbins=dict(start=min(df['Age']),
                                   end=max(df['Age'])),
                        name='Male',
                        opacity=0.75)

    female = go.Histogram(x=df[df['Gender'] == 'Female']['Age'],
                          xbins=dict(start=min(df['Age']),
                                     end=max(df['Age'])),
                          name='Female',
                          opacity=0.75)

    data = [male, female]
    layout = go.Layout(title='Distribution of Subjects by Age',
                       barmode='overlay',
                       font=dict(family='arial', size=18, color='#000000'))

    fig = go.Figure(data=data, layout=layout)
    filename = 'age_distribution_by_gender'
    if not os.path.exists(data_files_path):
        print(f'\nWARNING: The path does not exist. Creating new directory...\n{data_files_path}\n')
        os.mkdir(data_files_path)
    pyo.plot(fig, filename=f'{data_files_path}\\{filename}.html', auto_open=open_plot)
    print(f'\nAge Distribution Histogram generated.\nLocation: "{data_files_path}\\{filename}.html"\n')
