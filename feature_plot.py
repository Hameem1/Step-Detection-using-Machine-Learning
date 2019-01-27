"""This module implements the plotting functionality for the feature data generated from the base data"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
from data_structs import fs

# Global variables
SENSOR_TYPE = ''
FEATURES_LIST = {}
FEATURES = {}

# Configuring Dash app
app = dash.Dash(__name__)
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True


def serve_layout():
    # Main Div (1st level)
    layout = html.Div([

        # Sub-Div (2nd level)
        # Heading (Title)
        html.Div([html.H1(children="Project Dashboard",
                          className='twelve columns',
                          style={'text-align': 'center',
                                 'margin': '2% 0% 2% 0%',
                                 'letter-spacing': 2})], className='row'),
        # Sub-Div (2nd level)
        # axis and feature selector
        html.Div([

            # Sub-Div (3rd level)
            # DropDown
            html.Div([dcc.Dropdown(id='axis-dropdown',
                                   multi=False,
                                   options=[{'label': axis, 'value': axis} for axis in FEATURES_LIST],
                                   value='Ax',
                                   placeholder="Select axis",
                                   style={'height': '40px',
                                          'fontSize': 20,
                                          'margin': '2% 0% 2% 0%',
                                          'textAlign': 'center'})
                      ], style={'text-align': 'center', 'float': 'left'}, className='three columns offset-by-two'),

            # Sub-Div (3rd level)
            # DropDown
            html.Div([dcc.Dropdown(id='features-dropdown',
                                   multi=False,
                                   value='mean',
                                   placeholder="Select feature",
                                   style={'height': '40px',
                                          'fontSize': 20,
                                          'margin': '2% 0% 2% 0%',
                                          'textAlign': 'center'})
                      ], style={'text-align': 'center', 'float': 'left'}, className='three columns offset-by-one')

        ], style={'margin': '4% 0% 4% 14%', 'float': 'center'}, className='row'),

        # Sub-Div (2nd level)
        # Features Graph
        html.Div([dcc.Graph(id='feature-plot')], className='row')

    ], className='ten columns offset-by-one')
    return layout


app.layout = serve_layout


@app.callback(Output('features-dropdown', 'options'),
              [Input('axis-dropdown', 'value')])
def features_dropdown_callback(value):
    options_list = [{'label': feature,
                     'value': feature} for feature in FEATURES_LIST[value]]

    return options_list


# @app.callback(Output('feature-plot', 'figure'),
#               [Input('axis-dropdown', 'value'),
#                Input('features-dropdown', 'value')])
def graph_callback(axis, feature):
    """This function plots a graph for the given parameters

        :param axis: str(selected from the dropdown menu)
        :param feature: str(selected from the dropdown menu)
        :return figure: a figure property
        """

    # Graph variables
    t = np.array([num for num in range(0, len(FEATURES[axis][feature]))]) / fs
    xlabel = 'time (s)'
    ylabel = 'Feature Value'
    title = f'{feature.capitalize}  vs Time'

    # Generating plot traces
    traces = [go.Scatter(x=t,
                         y=FEATURES[axis][feature],
                         mode="lines",
                         name=feature)]

    # # Generating x and y step coordinates for all axes
    # steps_x, steps_y, step_count = step_marker(t, sub, position, sensor, motion)
    #
    # # Generating step traces
    # step_traces = [go.Scatter(x=np.array(steps_x[ax]) / fs,
    #                           y=steps_y[ax],
    #                           mode="markers",
    #                           name=axes[ax][sensor]) for ax in axes]
    #
    # # Keeping only one matching step trace visible by default
    # for step_trace in step_traces:
    #     if step_trace.name is not str(axes[visible][sensor]):
    #         step_trace.visible = "legendonly"
    #
    # # Combining the traces with step traces
    # traces.extend(step_traces)

    # Defining the layout for the plot
    layout = go.Layout(title=title,
                       xaxis=dict(title=xlabel),
                       yaxis=dict(title=ylabel),
                       font=dict(family='arial', size=18, color='#000000'))

    # Plotting the figure
    fig = dict(data=traces, layout=layout)
    return fig


# def step_marker(t, subject, pos, sensor, motion_type):
#     """This function returns the x and y coordinates for steps detected in the given subject data vs t
#
#         :param t: numpy array of the time axis
#         :param subject: instance of the Subject class
#         :param pos: str ('center', 'left', 'right')
#         :param sensor: str ('acc', 'gyr')
#         :param motion_type: str('complete', 'valid', etc)
#         :returns steps_x, steps_y, step_count: dict(steps_x, steps_y), int(step_count)
#         """
#
#     steps_x = {'x': [], 'y': [], 'z': []}
#     steps_y = {'x': [], 'y': [], 'z': []}
#
#     for ax in axes:
#         data = subject.sensor_pos[pos].label[motion_type]
#
#         step_count = 0
#         for i in range(1, len(t)):
#             if data.loc[i, 'StepLabel'] > (data.loc[i - 1, 'StepLabel']):
#                 steps_x[ax].append(i)
#                 steps_y[ax].append(float("{0:.3f}".format(data.loc[i, axes[ax][sensor]])))
#                 step_count += 1
#         print(f'\nStep Count for {axes[ax][sensor]} = {step_count}\n')
#
#     return steps_x, steps_y, step_count


def feature_plot(features_list, features, sensor_type='acc'):
    """This function accepts the data values from a function call and makes them global

    :param features_list: list(features_list)
    :param features: dict(features)
    :param sensor_type: optional('acc' or 'gyr')
    """

    global FEATURES_LIST, FEATURES, SENSOR_TYPE
    FEATURES_LIST = features_list
    FEATURES = features
    SENSOR_TYPE = sensor_type
    app.run_server(debug=False, port=5001)


if __name__ == '__main__':
    print(f"In __main__ of graphing.py")
else:
    print(f"\nModule imported : {__name__}\n")
