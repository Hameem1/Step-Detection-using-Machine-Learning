import plotly.offline as pyo
import plotly.graph_objs as go
from dataset.dataset_manipulator import read_csv

df = read_csv('..\\subject_data')

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
pyo.plot(fig, filename='age_distribution_by_gender.html')


if __name__ == '__main__':
    pass
    # print(f'\nThis module runs from within "dataset_generator.py"\n')
else:
    print(f'Module Imported : {__name__}')
    # main()

