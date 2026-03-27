# Copyright (c) 2025
#           Thomas Bömer (thomas.bömer@tu-dortmund.de)
#           Nico Koltermann (nico.koltermann@tu-dortmund.de) 
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os


import plotly.express as px
import plotly.graph_objects as go

import pandas as pd


import dash
from dash import dcc, html

from visualizePromptFrequency import plot_evaluation_time
from visualizeResults import load_and_plot_objective_ranges

from visualization_helper import insert_line_breaks, load_all_data, get_by_id


def violin_plot(input: list[list[float]]):

    # Prepare the data for the violin plot
    data = []
    for i, values in enumerate(input):
        data.extend({"Population": i, "Fitness Value": value} for value in values)

    # Convert to a DataFrame
    df = pd.DataFrame(data)

    # Create the violin plot
    return px.violin(df, x="Population", y="Fitness Value", box=True, points="all",
                    title="Violin Plot of Fitness Values")


def heatmap(input: list[list[float]]):
    # Prepare the data for the heatmap
    data = []
    for i, values in enumerate(input):
        for value in values:
            data.append({"Population": i, "Fitness Value": value})

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Create the heatmap
    return px.density_heatmap(df, x="Population", y="Fitness Value", title="Heatmap of Fitness Value Distribution")


def best_area_plot(input: list[list[float]]):

    data = []
    for i, values in enumerate(input):
        for value in values:
            data.append({"Population": i, "Fitness Value": value})

    df = pd.DataFrame(data)
    # Group by Population to create area traces
    fig = go.Figure()
    for pop, group in df.groupby("Population"):
        fig.add_trace(
            go.Scatter(
                x=group["Population"],
                y=group["Fitness Value"],
                mode='lines',
                fill='tozeroy',  # Fill area under the curve
                name=f"Population {pop}"
            )
        )

    # Add layout and show combined figure
    fig.update_layout(
        title="Combined Plot with Area Plot",
        xaxis_title="Population",
        yaxis_title="Fitness Value"
    )

    return fig


def line_plot(input: list[list[float]]):

    populations = list(range(len(input)))
    means = [sum(pop) / len(pop) for pop in input]
    medians = [sorted(pop)[len(pop) // 2] for pop in input]

    # Create the line plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=populations, y=means, mode='lines+markers', name='Mean'))
    fig.add_trace(go.Scatter(x=populations, y=medians, mode='lines+markers', name='Median'))

    # Customize layout
    return fig.update_layout(title="Trends of Summary Statistics Across Populations",
                      xaxis_title="Population",
                      yaxis_title="Statistic Value")


def get_distributions(folder_path):

    all_data_path = os.path.join(folder_path, "all_programs")
    data_folder_path = os.path.join(folder_path, "pops")

    pops = load_all_data(data_folder_path, True)
    all_programs = load_all_data(all_data_path)

    population_distribution = []

    for population in pops:

        current_pop_dist = []

        json_content = population['programs']

        offsping_ids = [x['offspring_id'] for x in json_content]

        for id in offsping_ids:
            program = get_by_id(all_programs, id)
            current_pop_dist += program['programs']['details']['detailed_fitness']

        population_distribution.append(current_pop_dist)

    return population_distribution


def plot_all_distributions(dir):

    experiments = [os.path.join(dir, x) for x in os.listdir(dir) if os.path.isdir(os.path.join(dir, x))]


    input_data = {}
    for e in experiments:
        input_data[e] = get_distributions(e)

    # Process input data into a DataFrame
    data = []
    for path, distributions in input_data.items():
        for i, distribution in enumerate(distributions):
            for value in distribution:
                data.append({"Path": path, "Population": i, "Fitness Value": value})

    df = pd.DataFrame(data)

    # Initialize the Dash app

    figs = {}
    for path in input_data.keys():
        fig = go.Figure()
        group = df[df['Path'] == path]
        fig.add_trace(go.Scatter(x=group["Population"], y=group["Fitness Value"], mode='lines', fill='tozeroy', name=path))
        figs[path] = fig

    # Initialize the Dash app
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1("EoH Summary", style={'textAlign': 'center'}),

        # Dropdown to select path for all plots
        dcc.Dropdown(
            id='path-dropdown',
            options=[{'label': path, 'value': path} for path in input_data.keys()],
            value=list(input_data.keys())[0],  # Default value
            style={'width': '50%', 'margin': 'auto'}
        ),

        # Layout for the multiple plots
        html.Div([
            dcc.Graph(id='violin-plot', style={'width': '48%', 'display': 'inline-block'}),
            dcc.Graph(id='line-plot', style={'width': '48%', 'display': 'inline-block'})
        ]),

        html.Div([
            dcc.Graph(id='heatmap-plot', style={'width': '48%', 'display': 'inline-block'}),
            dcc.Graph(id='area-plot', style={'width': '48%', 'display': 'inline-block'})
        ])

    ])

    # Callback to update the graphs based on dropdown selection
    @app.callback(
        [
            dash.dependencies.Output('violin-plot', 'figure'),
            dash.dependencies.Output('heatmap-plot', 'figure'),
            dash.dependencies.Output('area-plot', 'figure'),
            dash.dependencies.Output('line-plot', 'figure')
        ],
        [dash.dependencies.Input('path-dropdown', 'value')]
    )
    def update_graph(selected_path):
        filtered_data = input_data[selected_path]

        # Return all plots with the updated data for the selected path
        return (
            violin_plot(filtered_data),  # Violin Plot
            heatmap(filtered_data),  # Heatmap
            plot_evaluation_time(selected_path, False),  # Area Plot
            load_and_plot_objective_ranges(selected_path, save_file=False)  # Line Plot
        )


    app.run_server(debug=True)

if __name__ == "__main__":

    experiment = 'results_20241127_005340_multibay_reshuffle_gemma227b_ueFalse'

    plot_all_distributions(
        r'/home/nico/lfo/final_experiments/save_server')
