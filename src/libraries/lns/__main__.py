import os
import click

from .base_class import LNS_Base
from util.vrp_visualizer import plot_routes_from_distance_matrix

@click.group()
@click.pass_context
def main(ctx):
    pass

@click.group()
@click.pass_context
def main(ctx):
    pass


@main.command()
@click.pass_context
def help(ctx):
    """
    Show help for a command. Usage: `cli help [command]`
    """
    print(
"""
Usage: lns run <problem_name>

How to use this LNS framework:

1. Create your problem file in the `wrapper/` folder.
     Example: `wrapper/vehicle_routing.py`

2. Your problem file MUST define the following functions and objects:

    - `repair(current, rng)`      --> Repair operator
    - `destroy(current, rng)`     --> Destroy operator
    - `load_data()`               --> (Optional) Load problem-specific data
    - `initial_state()`           --> Returns initial ProblemState instance
    - `stop()`                    --> Returns ALNS stopping condition (e.g. MaxRuntime)
    - `accept()`                  --> Returns ALNS acceptance method (e.g. HillClimbing)
    - `select()`                  --> Returns ALNS selection strategy (e.g. RandomSelect)

3. Run your problem with:

       lns run <problem_name>

    Example:

       lns run vehicle_routing

- This will load the file: `wrapper/vehicle_routing.py`

Make sure your problem file has all required components, otherwise you'll get an import error.
"""
    )


@main.command()
@click.argument("problem")
def run(problem):
    run_alns(problem)

def list_lns_problems():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return [f.split(".py")[0] for f in os.listdir(os.path.join(dir_path, "wrapper")) if f.endswith(".py")]

def run_alns(problem):

    if problem not in list_lns_problems():
        raise NotImplementedError(f"The Problem {problem} cannot be loaded. Make sure there is a implementation! \n"
                                  f"Please run ```lns help``` for more information!")

    lns = LNS_Base(problem)

    lns.init_lns()

    lns.run()

    print(f"Finished LNS with Problem: {problem}")

    return lns
