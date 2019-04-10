"""
This module implements the plotting functionality for the base data.

"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
from config import USED_CLASS_LABEL, SENSOR, Fs, DEBUGGER

# Global variables
axes = {"x": {"acc": "Ax", "gyr": "Gx"},
        "y": {"acc": "Ay", "gyr": "Gy"},
        "z": {"acc": "Az", "gyr": "Gz"}}

SUB = None
axis = None
STEP_POSITIONS = {}

app = dash.Dash(__name__)
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

# Main Div (1st level)
app.layout = html.Div([

    # Sub-Div (2nd level)
    # Heading (Title)
    html.Div([html.H1(children="Raw Data Dashboard",
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
    This function plots a graph for the given parameters in the dropdown menus.

    Parameters
    ----------
    sensor : str
        Accelerometer, Gyroscope
    position : str
        Center, Left, Right
    motion : str
        Valid, Complete, Level, Upstairs, Downstairs, Incline, Decline

    Returns
    -------
    figure : dict
        A figure property

    """

    # Graph variables
    # Generating a time series (in secs) for the selected graphing parameters
    t = np.array([num for num in range(0, len(SUB.sensor_pos[position].label[motion]))]) / Fs
    # Setting one axis to keep visible by default
    visible = "x" if axis == "all" else axis
    # Setting Graph Labels
    xlabel = 'time (s)'
    ylabel = 'Acceleration (g)' if sensor == "acc" else 'Angular Velocity (rad/s)'
    title = 'Plot of ' + '"' + ('Acceleration' if sensor == "acc" else 'Angular Velocity') + '"' + ' vs Time'

    # Generating plot traces
    traces = [go.Scatter(x=t,
                         y=SUB.sensor_pos[position].label[motion][axes[ax][sensor]],
                         mode="lines",
                         name=axes[ax][sensor]) for ax in axes]

    # Keeping only one trace visible by default
    for trace in traces:
        if trace.name is not axes[visible][sensor]:
            trace.visible = "legendonly"

    # Generating y-axis step coordinates for all sensor axes
    steps_y = step_marker(position, sensor, motion)

    # Generating step traces
    step_traces = [go.Scatter(x=np.array(STEP_POSITIONS[motion]) / Fs,
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


def step_marker(pos, sensor, motion_type):
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
    data = SUB.sensor_pos[pos].label[motion_type]

    for ax in axes:
        for i in STEP_POSITIONS[motion_type]:
            steps_y[ax].append(float("{0:.3f}".format(data.loc[i, axes[ax][sensor]])))

    print(f'\nStep Count for Subject - {SUB.subject_id[:-4]} ({motion_type}) = '
          f'{len(STEP_POSITIONS[motion_type])}\n')
    return steps_y


def data_plot(sub, actual_step_positions, sensor_axis="all"):
    """
    This function accepts the data values from a function call and makes them global.
    It also acts as the data plotting endpoint and starts the Dash server.

    Parameters
    ----------
    sub : Subject
        A Subject class object
    actual_step_positions : list
        containing x_axis values for steps (from the original dataset_operations)
    sensor_axis : {'x', 'y', 'z', 'all'}

    """

    global SUB, axis, STEP_POSITIONS
    SUB = sub
    axis = sensor_axis
    STEP_POSITIONS = actual_step_positions
    app.run_server(debug=DEBUGGER, host="0.0.0.0", port=5000)


if __name__ == '__main__':
    print(f"In __main__ of data_plot.py")
else:
    print(f"\nModule imported : {__name__}\n")
