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
import json
import plotly.graph_objects as go
import re

from solver.ceoh.utils.visualization_helper import insert_line_breaks

def load_and_plot_objective_ranges(folder_path, output_file="interactive_result_plot.html", save_file=True):
    min_objectives = []
    max_objectives = []
    avg_objectives = []
    all_objectives = []
    all_filenames = []
    filenames = []
    min_tooltips = []
    max_tooltips = []
    avg_tooltips = []
    all_tooltips = []

    data_folder_path = os.path.join(folder_path, "pops")
    output_folder_path = os.path.join(folder_path, "visualization")

    if save_file:
        os.makedirs(output_folder_path, exist_ok=True)

    json_files = sorted(
        [f for f in os.listdir(data_folder_path) if f.endswith('.json')],
        key=lambda x: int(re.search(r'\d+', x).group())
    )

    for filename in json_files:
        file_path = os.path.join(data_folder_path, filename)
        with open(file_path, 'r') as file:
            json_content = json.load(file)

            objective_values = []

            for entry in json_content:
                objective = entry.get("objective")
                offspring_id = entry.get("offspring_id")

                if objective is None:
                    continue

                # ─────────────────────────────────────────────
                # OLD FORMAT: algorithm + code at top level
                # ─────────────────────────────────────────────
                if "algorithm" in entry and entry.get("algorithm") is not None:
                    objective_values.append(
                        (
                            objective,
                            offspring_id,
                            entry.get("algorithm"),
                            entry.get("code", "")
                        )
                    )

                # ─────────────────────────────────────────────
                # NEW FORMAT: multiple sub_offsprings
                # ─────────────────────────────────────────────
                elif "sub_offsprings" in entry and isinstance(entry["sub_offsprings"], list):
                    for sub in entry["sub_offsprings"]:
                        algorithm = sub.get("algorithm")
                        code = sub.get("code", "")
                        subproblem = sub.get("subproblem", "unknown")

                        if algorithm is None:
                            continue

                        # Optional: include subproblem in algorithm text
                        algorithm_with_context = (
                            f"Subproblem: {subproblem}\n{algorithm}"
                        )

                        objective_values.append(
                            (
                                objective,
                                offspring_id,
                                algorithm_with_context,
                                code
                            )
                        )

            for obj_value, offspring_id, algorithm, code in objective_values:
                algorithm_text = "".join(algorithm) if isinstance(algorithm, list) else algorithm
                code_text = "".join(code) if isinstance(code, list) else code

                algorithm_formatted = insert_line_breaks(algorithm_text)
                code_formatted = code_text.replace('\n', '<br>')

                all_objectives.append(obj_value)
                all_filenames.append(filename)
                all_tooltips.append(
                    f"Offspring ID: {offspring_id}<br>Objective: {obj_value}<br>Algorithm:<br>{algorithm_formatted}<br><br>Code:<br>{code_formatted}"
                )

            # Find the minimum, maximum, and average objective values for the current file
            if objective_values:
                min_obj, min_offspring_id, min_algorithm, min_code = min(objective_values, key=lambda x: x[0])
                max_obj, max_offspring_id, max_algorithm, max_code = max(objective_values, key=lambda x: x[0])
                avg_obj = sum(obj[0] for obj in objective_values) / len(objective_values)

                min_algorithm_formatted = insert_line_breaks(
                    "".join(min_algorithm) if isinstance(min_algorithm, list) else min_algorithm)
                max_algorithm_formatted = insert_line_breaks(
                    "".join(max_algorithm) if isinstance(max_algorithm, list) else max_algorithm)

                # Replace new lines in code with `<br>` for tooltip display
                min_code_formatted = min_code.replace('\n', '<br>')
                max_code_formatted = max_code.replace('\n', '<br>')

                # Create tooltip text with objective value, algorithm, and code for min and max objectives
                min_tooltips.append(
                    f"Offspring ID: {min_offspring_id}<br>Objective: {min_obj}<br>Algorithm:<br>{min_algorithm_formatted}<br><br>Code:<br>{min_code_formatted}"
                )
                max_tooltips.append(
                    f"Offspring ID: {max_offspring_id}<br>Objective: {max_obj}<br>Algorithm:<br>{max_algorithm_formatted}<br><br>Code:<br>{max_code_formatted}"
                )
                avg_tooltips.append(f"Average Objective: {avg_obj:.2f}")

                min_objectives.append(min_obj)
                max_objectives.append(max_obj)
                avg_objectives.append(avg_obj)
                filenames.append(filename)

    # Create Plotly traces for minimum, maximum, average objectives, and all objectives as extra dots
    min_trace = go.Scatter(
        x=filenames,
        y=min_objectives,
        mode='markers+lines',
        name='Minimum Objective',
        text=min_tooltips,
        hoverinfo='text',
        line=dict(color='green')
    )

    max_trace = go.Scatter(
        x=filenames,
        y=max_objectives,
        mode='markers+lines',
        name='Maximum Objective',
        text=max_tooltips,
        hoverinfo='text',
        line=dict(color='red')
    )

    avg_trace = go.Scatter(
        x=filenames,
        y=avg_objectives,
        mode='markers+lines',
        name='Average Objective',
        text=avg_tooltips,
        hoverinfo='text',
        line=dict(color='blue')
    )

    all_points_trace = go.Scatter(
        x=all_filenames,
        y=all_objectives,
        mode='markers',
        name='All Heuristics',
        text=all_tooltips,
        hoverinfo='text',
        marker=dict(color='grey', opacity=0.5),
        visible=False
    )

    fig = go.Figure()
    fig.add_trace(min_trace)
    fig.add_trace(max_trace)
    fig.add_trace(avg_trace)
    fig.add_trace(all_points_trace)

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                buttons=[
                    dict(
                        label="Show All Heuristics",
                        method="update",
                        args=[
                            {"visible": [True, True, True, True]},  # Show all traces including 'all_points_trace'
                            {"title": "Objective Value Ranges Across JSON Files (All Heuristics Shown)"},
                        ],
                    ),
                    dict(
                        label="Hide All Heuristics",
                        method="update",
                        args=[
                            {"visible": [True, True, True, False]},  # Hide 'all_points_trace'
                            {"title": "Objective Value Ranges Across JSON Files"},
                        ],
                    )
                ],
                showactive=True,
                x=1,
                xanchor="right",
                y=1.15,
                yanchor="top"
            )
        ]
    )

    # Update layout for titles and labels
    fig.update_layout(
        title="Objective Value Ranges Across JSON Files",
        xaxis_title="JSON Files",
        yaxis_title="Objective Values",
        xaxis=dict(tickangle=45)
    )


    if save_file:
        # Save as an interactive HTML file in the specified output folder
        output_file_path = os.path.join(output_folder_path, output_file)
        fig.write_html(output_file_path)

        print(f"Interactive plot saved as {output_file_path}")

    return fig


if __name__ == "__main__":
    load_and_plot_objective_ranges(
        r'C:\Users\dq0913\PycharmProjects\multibay-reshuffling\eoh_results\results_20251219_112628_lns_multi_robot_upmp_coupled_qwen25coder32b_ueFalse_uiFalse_uitypeNone')
