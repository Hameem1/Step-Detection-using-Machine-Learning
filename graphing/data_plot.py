"""
This module implements the plotting functionality for the base data.

"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
from config import USED_CLASS_LABEL, SENSOR, Fs

# Global variables
axes = {"x": {"acc": "Ax", "gyr": "Gx"},
        "y": {"acc": "Ay", "gyr": "Gy"},
        "z": {"acc": "Az", "gyr": "Gz"}}

sub = None
axis = None
STEP_POSITIONS = []

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
    # motion type, sensor type and position selector
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
                               value=USED_CLASS_LABEL,
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
                               value=SENSOR,
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
    # Data Graph
    html.Div([dcc.Graph(id='data-plot')], className='row')

], className='ten columns offset-by-one')


@app.callback(Output('data-plot', 'figure'),
              [Input('sensor-dropdown', 'value'),
               Input('position-dropdown', 'value'),
               Input('motion-dropdown', 'value')])
def graph_callback(sensor, position, motion):
    """
    This function plots a graph for the given parameters.

    Parameters
    ----------
    sensor : str
        selected from the dropdown menu
    position : str
        selected from the dropdown menu
    motion : str
        selected from the dropdown menu

    Returns
    -------
    figure : dict
        A figure property

    """

    # Graph variables
    t = np.array([num for num in range(0, len(sub.sensor_pos[position].label[motion]))]) / Fs
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
    steps_y = step_marker(position)

    # Generating step traces
    step_traces = [go.Scatter(x=np.array(STEP_POSITIONS) / Fs,
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


def step_marker(pos, sensor=SENSOR, motion_type=USED_CLASS_LABEL):
    """
    This function returns the y-coordinates for steps detected in the given subject data vs t.

    Parameters
    ----------
    pos : {'center', 'left', 'right'}
    sensor : {'acc', 'gyr'}
    motion_type : str
        'complete', 'valid', etc.

    Returns
    -------
    steps_y : dict
        Y-coordinates for the steps

    """

    steps_y = {'x': [], 'y': [], 'z': []}
    data = sub.sensor_pos[pos].label[motion_type]

    for ax in axes:
        for i in STEP_POSITIONS:
            steps_y[ax].append(float("{0:.3f}".format(data.loc[i, axes[ax][sensor]])))

    print(f'\nStep Count for Subject - {sub.subject_id[:-4]} = {len(STEP_POSITIONS)}\n')
    return steps_y


def data_plot(subject, actual_step_positions, sensor_axis="all"):
    """
    This function accepts the data values from a function call and makes them global.
    It also acts as the feature plotting endpoint and starts the Dash server.

    Parameters
    ----------
    subject : Subject
        A Subject class object
    actual_step_positions : list
        containing x_axis values for steps (from the original dataset)
    sensor_axis : {'x', 'y', 'z', 'all'}

    """

    global sub, axis, STEP_POSITIONS
    sub = subject
    axis = sensor_axis
    STEP_POSITIONS = actual_step_positions
    app.run_server(debug=False, port=5000)


if __name__ == '__main__':
    print(f"In __main__ of data_plot.py")
else:
    print(f"\nModule imported : {__name__}\n")
