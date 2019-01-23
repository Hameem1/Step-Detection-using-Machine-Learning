"""This module implements the plotting functionality"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np

# Global variables
axes = {"x": {"acc": "Ax", "gyr": "Gx"},
        "y": {"acc": "Ay", "gyr": "Gy"},
        "z": {"acc": "Az", "gyr": "Gz"}}

sub = None
axis = None

app = dash.Dash(__name__)
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

# Main Div (1st level)
app.layout = html.Div([

    # Sub-Div (2nd level)
    # Heading (Title)
    html.Div([html.H1(children="Project Dashboard",
                      className='twelve columns',
                      style={'text-align': 'center',
                             'margin': '2% 0% 2% 0%',
                             'letter-spacing': 2})], className='row'),
    # Sub-Div (2nd level)
    # sensor and position selector
    html.Div([

        # Sub-Div (3rd level)
        # DropDown
        html.Div([dcc.Dropdown(id='motion-dropdown',
                               multi=False,
                               options=[{'label': 'Valid', 'value': 'valid'},
                                        {'label': 'Complete', 'value': 'complete'},
                                        {'label': 'Level', 'value': 'level'},
                                        {'label': 'Upstairs', 'value': 'upstairs'},
                                        {'label': 'Downstairs', 'value': 'downstairs'},
                                        {'label': 'Incline', 'value': 'incline'},
                                        {'label': 'Decline', 'value': 'decline'}],
                               value='valid',
                               placeholder="Select motion type",
                               style={'height': '40px',
                                      'fontSize': 20,
                                      'margin': '2% 0% 2% 0%',
                                      'textAlign': 'center'})
                  ], style={'text-align': 'center', 'float': 'left'}, className='three columns'),

        # Sub-Div (3rd level)
        # DropDown
        html.Div([dcc.Dropdown(id='sensor-dropdown',
                               multi=False,
                               options=[{'label': 'Accelerometer', 'value': 'acc'},
                                        {'label': 'Gyroscope', 'value': 'gyr'}],
                               value='acc',
                               placeholder="Select sensor type",
                               style={'height': '40px',
                                      'fontSize': 20,
                                      'margin': '2% 0% 2% 0%',
                                      'textAlign': 'center'})
                  ], style={'text-align': 'center', 'float': 'left'}, className='three columns offset-by-one'),

        # Sub-Div (3rd level)
        # DropDown
        html.Div([dcc.Dropdown(id='position-dropdown',
                               multi=False,
                               options=[{'label': 'Center', 'value': 'center'},
                                        {'label': 'Left', 'value': 'left'},
                                        {'label': 'Right', 'value': 'right'}],
                               value='center',
                               placeholder="Select sensor position",
                               style={'height': '40px',
                                      'fontSize': 20,
                                      'margin': '2% 0% 2% 0%',
                                      'textAlign': 'center'})
                  ], style={'text-align': 'center', 'float': 'left'}, className='three columns offset-by-one')

    ], style={'margin': '4% 0% 4% 14%', 'float': 'center'}, className='row'),

    # Sub-Div (2nd level)
    # Stocks Graph
    html.Div([dcc.Graph(id='data-plot')], className='row')

], className='ten columns offset-by-one')


@app.callback(Output('data-plot', 'figure'),
              [Input('sensor-dropdown', 'value'),
               Input('position-dropdown', 'value'),
               Input('motion-dropdown', 'value')])
def graph_callback(sensor, position, motion):
    """This function plots a graph for the given parameters

        :param sensor: str(selected from the dropdown menu)
        :param position: str(selected from the dropdown menu)
        :param motion: str(selected from the dropdown menu)
        :return figure: a figure property
        """

    # Graph variables
    fs = sub.sensor_pos[position].fs
    t = np.array([num for num in range(0, len(sub.sensor_pos[position].label[motion]))]) / fs
    visible = "x" if axis == "all" else axis
    xlabel = 'time (s)'
    ylabel = 'Acceleration (g)' if sensor == "acc" else 'Angular Velocity (rad/s)'
    title = 'Plot of ' + '"' + ('Acceleration' if sensor == "acc" else 'Angular Velocity') + '"' + ' vs Time'

    # Generating plot traces
    traces = [go.Scatter(x=t,
                         y=sub.sensor_pos[position].label[motion][axes[ax][sensor]],
                         mode="lines",
                         name=axes[ax][sensor]) for ax in axes]

    # Keeping only one trace visible by default
    for trace in traces:
        if trace.name is not axes[visible][sensor]:
            trace.visible = "legendonly"

    # Generating x and y step coordinates for all axes
    steps_x, steps_y, step_count = step_marker(t, sub, position, sensor, motion)

    # Generating step traces
    step_traces = [go.Scatter(x=np.array(steps_x[ax]) / fs,
                              y=steps_y[ax],
                              mode="markers",
                              name=axes[ax][sensor]) for ax in axes]

    # Keeping only one matching step trace visible by default
    for step_trace in step_traces:
        if step_trace.name is not str(axes[visible][sensor]):
            step_trace.visible = "legendonly"

    # Combining the traces with step traces
    traces.extend(step_traces)

    # Defining the layout for the plot
    layout = go.Layout(title=title,
                       xaxis=dict(title=xlabel),
                       yaxis=dict(title=ylabel),
                       font=dict(family='arial', size=18, color='#000000'))

    # Plotting the figure
    fig = dict(data=traces, layout=layout)
    return fig


def step_marker(t, subject, pos, sensor, motion_type):
    """This function returns the x and y coordinates for steps detected in the given subject data vs t

        :param t: numpy array of the time axis
        :param subject: instance of the Subject class
        :param pos: str ('center', 'left', 'right')
        :param sensor: str ('acc', 'gyr')
        :param motion_type: str('complete', 'valid', etc)
        :returns steps_x, steps_y, step_count: dict(steps_x, steps_y), int(step_count)
        """

    steps_x = {'x': [], 'y': [], 'z': []}
    steps_y = {'x': [], 'y': [], 'z': []}

    for ax in axes:
        data = subject.sensor_pos[pos].label[motion_type]

        step_count = 0
        for i in range(1, len(t)):
            if data.loc[i, 'StepLabel'] > (data.loc[i - 1, 'StepLabel']):
                steps_x[ax].append(i)
                steps_y[ax].append(float("{0:.3f}".format(data.loc[i, axes[ax][sensor]])))
                step_count += 1
        print(f'\nStep Count for {axes[ax][sensor]} = {step_count}\n')

    return steps_x, steps_y, step_count


def data_plot(subject, sensor_axis="all"):
    """This function accepts the data values from a function call and makes them global

    :param subject: A Subject class object
    :param sensor_axis: str ('x', 'y', 'z', 'all')
    """

    global sub, axis
    sub = subject
    axis = sensor_axis
    app.run_server(debug=True)


if __name__ == '__main__':
    # app.run_server(debug=True)
    print(f"In __main__ of graphing.py")
