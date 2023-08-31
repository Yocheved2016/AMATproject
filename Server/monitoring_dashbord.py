import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import numpy as np
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
from Server.getRunData import create_dict

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SOLAR])

classes = create_dict()

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Dashboard for ML Analysis", className="text-primary"), width={'size': 6, 'offset': 3}),
    ]),

    dbc.Row([
        dbc.Col(
            dcc.Dropdown(
                id='class-dropdown',
                options=[{'label': cls, 'value': cls} for cls in classes.keys()],
                value=list(classes.keys())[0],
                multi=False,
                className="bg-light text-dark"  # Use Bootstrap classes for dropdown
            ),
            width={'size': 3}
        ),
    ], style={'margin-bottom': '20px'}),

    dbc.Row([
        dbc.Col([
            dcc.Graph(id='difference-graph'),
        ], width={'size': 4}),
        dbc.Col([
            dcc.Graph(id='entropy-difference-graph'),
        ], width={'size': 4}),
        dbc.Col([
            dcc.Graph(id='confidence-difference-graph'),
        ], width={'size': 4}),
    ]),

], style={'padding': '20px'})


@app.callback(
    [Output('difference-graph', 'figure'),
     Output('entropy-difference-graph', 'figure'),
     Output('confidence-difference-graph', 'figure')],
    Input('class-dropdown', 'value')
)
def update_graph(selected_class):
    class_data = classes[selected_class]
    initial_histogram = class_data["initial_histogram"]
    histogram_measurements = class_data["histogram_measurements"]
    initial_entropy = class_data["initial_entropy"]
    entropy_measurements = class_data["entropy_measurements"]
    initial_confidence = class_data["initial_confidence"]
    confidence_measurements = class_data["confidence_measurements"]

    # Calculate differences from initial value
    hist_differences = [measurement - initial_histogram for measurement in histogram_measurements]
    entropy_differences = [measurement - initial_entropy for measurement in entropy_measurements]
    confidence_differences = [measurement - initial_confidence for measurement in confidence_measurements]
    # Calculate entropy differences from initial entropy
    # entropy_values = [initial_entropy] + histogram_measurements  # Add initial entropy as first value
    # entropy_differences = [entropy_value - initial_entropy for entropy_value in entropy_values]

    # Create the graph figure for differences
    graph_figure = {
        'data': [
            go.Scatter(x=list(range(len(histogram_measurements))), y=hist_differences, mode='lines+markers',
                       name='Differences')
        ],
        'layout': {
            'title': f'Difference from average histogram  {selected_class}',
            'xaxis': {'title': 'Time'},
            'yaxis': {'title': 'Difference'}
        }
    }

    # Create the graph figure for entropy differences
    entropy_diff_figure = {
        'data': [
            go.Scatter(x=list(range(len(entropy_measurements))), y=entropy_differences, mode='lines+markers',
                       name='Entropy Differences')
        ],
        'layout': {
            'title': f'Entropy Differences for {selected_class}',
            'xaxis': {'title': 'Time'},
            'yaxis': {'title': 'Entropy Difference'}
        }
    }
    confidence_diff_figure = {
        'data': [
            go.Scatter(x=list(range(len(confidence_measurements))), y=confidence_differences, mode='lines+markers',
                       name='Confidence Differences')
        ],
        'layout': {
            'title': f'Confidence Differences for {selected_class}',
            'xaxis': {'title': 'Time'},
            'yaxis': {'title': 'Confidence Difference'}
        }
    }

    return graph_figure, confidence_diff_figure, entropy_diff_figure


if __name__ == '__main__':
    app.run_server(debug=True)
