from model_config import *
from config import data_files_path, Path
import plotly.graph_objs as go
import plotly.offline as pyo
from sklearn.linear_model import LinearRegression

data = pd.read_csv(f'{data_files_path}/Computation times.csv', sep='\t', index_col=0)
n = list(data['Samples used'].unique())

X = [data['Features used'][data['Samples used'] == i].values[:, np.newaxis] for i in n]
y = [data['Time Taken'][data['Samples used'] == i].round(2).values for i in n]
model = [LinearRegression().fit(X[i], y[i]) for i in range(data['Samples used'].nunique())]

traces = [go.Scatter(x=data['Features used'][data['Samples used'] == i],
                     y=data['Time Taken'][data['Samples used'] == i].round(2),
                     name=str(i / 10 ** 6) + ' M',
                     mode='markers',
                     opacity=0.75)
          for i in n]

best_fit = [go.Scatter(x=data['Features used'][data['Samples used'] == i],
                       y=model[j].predict(X[j]),
                       name=str(i / 10 ** 6) + ' M_best_fit',
                       mode='lines')
            for i, j in zip(n, range(data['Samples used'].nunique()))]

traces.extend(best_fit)

layout = go.Layout(title='Computation Times for "n" Training Samples',
                   xaxis={'title': "No. of Features Used"},
                   yaxis={'title': "Time Taken (s)"},
                   font=dict(family='arial', size=18, color='#000000'))

fig = go.Figure(data=traces, layout=layout)
pyo.plot(fig, filename=f'{data_files_path}/Computation Times Plot.html')