import ast
import json
import os
import click
import logging

import solver.ceoh.eoh as eoh
from solver.ceoh.utils.getParas import Paras
from solver.ceoh.llm.models import get_model_info
from solver.ceoh.utils.createFolders import check_instances, create_idea_folders, save_paras

import warnings
warnings.filterwarnings("ignore")


def _normalize_coupled_problems(coupled_problems: str | None):
    """
    Returns:
      - None if coupled_problems is None/"None"/"" or resolves to an empty list
      - list[str] otherwise (trimmed, empty entries removed)
    """
    if coupled_problems is None:
        return None

    # Click will pass default="None" literally as the string "None"
    cp = str(coupled_problems).strip()
    if cp == "" or cp.lower() == "none":
        return None

    # Split, strip whitespace, remove empties (handles trailing commas)
    items = [x.strip() for x in cp.split(",")]
    items = [x for x in items if x]  # remove ""

    return items if items else None


@click.group()
@click.pass_context
def main(ctx):
    pass


@main.command()
@click.argument("problem")
@click.option('--model_name', default="llama3.1:70b", help='LLM model')
@click.option('--use_example', default=1, help='Use Example for LMM request')
@click.option('--ideas_iteration_start', default=-1, help='Use Idea start for LMM request. If -1 its disabled')
@click.option('--continue_target_folder', default="None", help='Folder for continue experiemnts')
@click.option('--continue_target_pop', default=-1, help='Population to continue.')
@click.option('--injection_loops', default=-1, help='Number of extraction loops.')
@click.option('--ec_operators', default="e1,e2,m1,m2", help='Prompt strategies')
@click.option('--ec_n_pop', default="20", help='Number of populations')
@click.option('--ec_m', default="5", help='Number of parents in prompts.')
@click.option('--ec_pop_size', default="20", help='Number of times each active strategy is called.')
@click.option('--exp_n_proc', default="1", help='Number of parallel processes.')
@click.option('--ui_experiment', default="None", help='Experiment variation indicator.')
@click.option('--eoh_experiment_file', default="exp_1.json", help='File with experiment config.')
@click.option('--use_math_model', default=0, help='Use math model for LMM request')
@click.option('--idea_number', default=-1, help='Number of ideas use in P1. If -1 its disabled')
@click.option('--method', default="eoh", help='Select what method to use. Default is eoh.')
@click.option('--lns_paras', default="None", help='Parameters for the lns. Default is None.')
@click.option('--lns_paras_second', default="None", help='Parameters for the second lns. Default is None.')
@click.option('--alns_cycles_iters', default="0", help='Parameters for the alns_cycles_iters. Default is 0.')
@click.option('--alns_cycles_timeout', default="0", help='Parameters for the alns_cycles_timeout. Default is 0.')
@click.option('--coupled_problems', default="None", help='Comma-separated coupled prompt problems, or None.')
@click.option('--coupled_iterations', default="1", help='Number of outer coupled iterations. Default is 1.')
@click.option('--continue_target_loop', default=-1, help='Loop number to continue')
@click.option('--ec_initial_operator', default="i1", help='Prompt operator for initialization.')
def run(problem, model_name, use_example,
        ideas_iteration_start, continue_target_folder, continue_target_pop, injection_loops,
        ec_operators, ec_n_pop, ec_m, ec_pop_size, ui_experiment, exp_n_proc, eoh_experiment_file, use_math_model,
        idea_number, method, lns_paras, lns_paras_second, alns_cycles_iters, alns_cycles_timeout, coupled_problems,
        coupled_iterations, continue_target_loop, ec_initial_operator):

    if continue_target_folder == "None":
        exp_continue_pop_nr = 20
        exp_continue_folder = "experiment"
        set_init_pop = False
    else:
        exp_continue_pop_nr = continue_target_pop
        exp_continue_folder = continue_target_folder
        set_init_pop = True

    paras = Paras()

    if "lns" in problem and lns_paras == "None":
        print("ERROR: lns_paras not defined for an lns problem!")
        print("Expected: ", paras.lns_paras)
        raise KeyError
    elif lns_paras != "None":
        lns_paras = ast.literal_eval(lns_paras)
        print("LNS PARAS", lns_paras)
    else:
        lns_paras = None

    if "lns" in problem and "cycle" in problem and lns_paras_second == "None":
        print("ERROR: lns_paras_second not defined for an cycle lns problem!")
        print("Expected: ", paras.lns_paras_second)
        raise KeyError
    elif lns_paras_second != "None":
        lns_paras_second = ast.literal_eval(lns_paras_second)
        print("LNS PARAS SECOND", lns_paras_second)
    else:
        lns_paras_second = None
        alns_cycles_iters = 0
        alns_cycles_timeout = 0

    if coupled_problems == "None":
        coupled_list = [str(problem)]
    else:
        coupled_list = _normalize_coupled_problems(coupled_problems)
    print("Coupled problems normalized:", coupled_list)

    llm_api_endpoint, add_url_info, api_key, llm_use_local = get_model_info(model_name)

    paras.set_paras(
        eoh_experiment_file=eoh_experiment_file,
        llm_api_key=api_key,
        llm_api_endpoint=llm_api_endpoint,
        llm_use_local=llm_use_local,
        llm_url_info=add_url_info,
        method=method,
        problem=problem,
        llm_model=model_name,
        llm_local_url=llm_api_endpoint,
        ec_pop_size=int(ec_pop_size),
        ec_operators=ec_operators.split(','),
        ec_use_example=bool(use_example),
        ec_use_idea=ideas_iteration_start > 0 or 'p1' in ec_operators,
        ec_n_pop=int(ec_n_pop),
        exp_n_proc=int(exp_n_proc),
        ec_m=int(ec_m),
        exp_debug_mode=False,
        eva_numba_decorator=False,
        exp_continue_pop_nr=int(exp_continue_pop_nr),
        exp_continue_folder=exp_continue_folder,
        exp_use_continue=int(set_init_pop),
        llm_temperature=None,
        injection_loops=int(injection_loops),
        ideas_iteration_start=int(ideas_iteration_start),
        idea_number=int(idea_number),
        idea_type=ui_experiment,
        ec_use_math_model=bool(use_math_model),
        lns_paras=lns_paras,
        lns_paras_second=lns_paras_second,
        alns_cycles_iters=int(alns_cycles_iters),
        alns_cycles_timeout=int(alns_cycles_timeout),

        coupled_problems=coupled_list,
        coupled_iterations=int(coupled_iterations),
        exp_continue_loop_nr=int(continue_target_loop),
        ec_initial_operator=str(ec_initial_operator),
    )

    evolution = eoh.EVOL(paras)

    current_folder = os.environ["CURRENT_EXPERIMENT"]

    logging.basicConfig(
        level=logging.DEBUG,  # Set the logging level to DEBUG or INFO as needed
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=os.path.join(current_folder, 'parallel_log.log'),  # Specify a log file
        filemode='w'  # Use 'w' to overwrite the log file each time, or 'a' to append
    )

    save_paras(paras)
    create_idea_folders(paras)
    check_instances(paras)

    evolution.run()
