"""This module implements the plotting functionality for the feature data generated from the base data"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np

# Global variables
SUB = None
FEATURES_LIST = {}
FEATURES = {}
STEP_POSITIONS = []
SENSOR_TYPE = ''
WINDOW_TYPE = ''
fs = 100

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


@app.callback(Output('feature-plot', 'figure'),
              [Input('axis-dropdown', 'value'),
               Input('features-dropdown', 'value')])
def graph_callback(axis, feature):
    """This function plots a graph for the given parameters

        :param axis: str(selected from the dropdown menu)
        :param feature: str(selected from the dropdown menu)
        :return figure: a figure property
        """

    # Graph variables
    try:
        t = np.array([num for num in range(len(FEATURES[axis][feature]))]) / fs
        xlabel = 'time (s)'
        ylabel = 'Feature Value'
        title = f'{feature.capitalize()}  vs Time'

        # Generating plot traces
        traces = [go.Scatter(x=t,
                             y=FEATURES[axis][feature],
                             mode="lines",
                             name=feature)]

        # Generating x and y step coordinates for x and y axis
        steps_x, steps_y, step_count = feature_step_marker(dict(feature=FEATURES[axis][feature]), STEP_POSITIONS[axis])

        # Generating step traces
        step_trace = [go.Scatter(x=np.array(steps_x) / fs,
                                 y=steps_y,
                                 mode="markers",
                                 name="steps")]

        # Combining the trace with step trace
        traces.extend(step_trace)

        # Defining the layout for the plot
        layout = go.Layout(title=title,
                           xaxis=dict(title=xlabel),
                           yaxis=dict(title=ylabel),
                           font=dict(family='arial', size=18, color='#000000'))

        # Plotting the figure
        fig = dict(data=traces, layout=layout)
        return fig

    except KeyError as error:
        print(f'Error Occurred : {error} - Invalid axis/feature combination selected')


def feature_step_marker(feature_data, step_pos):
    """This function returns the x and y coordinates for steps detected in the given feature data vs t

        :param feature_data: dict(feature=[data])
        :param step_pos: list containing x_axis values for steps
        :returns steps_x, steps_y, step_count: dict(steps_x, steps_y), int(step_count)
        """

    steps_x = []
    steps_y = []

    step_count = 0

    for i in step_pos:
        steps_x.append(i)
        steps_y.append("{0:.5f}".format(float(next(iter(feature_data.values()))[i])))
        step_count += 1

    print(f'\nStep Count for {next(iter(feature_data.keys()))} = {step_count}\n')

    return steps_x, steps_y, step_count


def feature_plot(sub, features_list, features, step_positions, window_type, sensor_type='acc'):
    """This function accepts the data values from a function call and makes them global

    :param step_positions: dict of lists containing x_axis values for steps
    :param window_type: str('sliding' or 'hopping')
    :param sub: Subject(the subject for which the features have been provided)
    :param features_list: list(features_list)
    :param features: dict(features)
    :param sensor_type: optional('acc' or 'gyr')
    """

    global FEATURES_LIST, FEATURES, SENSOR_TYPE, SUB, WINDOW_TYPE, STEP_POSITIONS
    SUB = sub
    WINDOW_TYPE = window_type
    FEATURES_LIST = features_list
    FEATURES = features
    STEP_POSITIONS = step_positions
    SENSOR_TYPE = sensor_type
    app.run_server(debug=False, port=5001)


if __name__ == '__main__':
    print(f"In __main__ of feature_plot.py")
else:
    print(f"\nModule imported : {__name__}\n")
