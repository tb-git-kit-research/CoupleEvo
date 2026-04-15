import os
import time
import subprocess
from dotenv import load_dotenv

#####################################################################################
# Environment Setup
#####################################################################################

load_dotenv()

BASE_PATH = os.getenv("BASE_PATH")

if not BASE_PATH:
    print("BASE_PATH for evaluation not set.")
    print("Please create a .env file and add this variable.")
    print("For more information, please look into the readme.")
    time.sleep(10)
    exit(1)

BASE_PATH = os.path.abspath(BASE_PATH)

OUTPUT_PATH = os.path.join(BASE_PATH, "eoh_results")
INSTANCES_PATH = os.path.join(
    BASE_PATH,
    "data",
    "mr_experiment_instances",
    "instances",
)

os.environ.update({
    "BASE_PATH": BASE_PATH,
    "OUTPUT_PATH": OUTPUT_PATH,
    "INSTANCES_PATH": INSTANCES_PATH,
})

print(f"BASE_PATH: {BASE_PATH}")
print(f"OUTPUT_PATH: {OUTPUT_PATH}")
print(f"INSTANCES_PATH: {INSTANCES_PATH}")

os.makedirs(OUTPUT_PATH, exist_ok=True)

#####################################################################################
# General Settings
#####################################################################################

os.environ.update({
    "PYTHONUNBUFFERED": "False",
    "TIMEOUT": "120",
    "LOGGING": "False",
    "DETAILED_OUTPUT": "False",
})

#####################################################################################
# LLM Configuration
#####################################################################################

print("########################## OLLAMA HOSTNAME ##########################")
os.environ["LLM_LOCAL_URL"] = "YOUR URL HERE"  # e.g., "http://localhost:11434"
print(os.environ["LLM_LOCAL_URL"])

# Install dependencies quietly
subprocess.run(
    ["pip", "install", "."],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    check=False,
)

print("########################## LLM MODELS AVAILABLE ##########################")

MODEL_NAME = "qwen3-coder:30b"
os.environ["MODEL_NAME"] = MODEL_NAME

#####################################################################################
# Experiment Configuration
#####################################################################################

print("########################## Experiment ##########################")

EOH_PROBLEM = "lns_IRP_coupled" #  "lns_IRP_coupled" "lns_multi_robot_upmp_coupled"
os.environ["EOH_PROBLEM"] = EOH_PROBLEM

if EOH_PROBLEM == "lns_IRP_coupled":
    eoh_experiment_file = "U_C_N25_Q18_T?"
elif EOH_PROBLEM == "lns_multi_robot_upmp_coupled":
    eoh_experiment_file = (
        "exp_bay_5_wh_2_fill_0.8_seed_0-9_priority_5_robots_2_ad_1010.json"
    )
else:
    raise ValueError(f"Unsupported EOH_PROBLEM: {EOH_PROBLEM}")

coordination_strategy = "sequential"  # "integrated", "sequential", "iterate"

#####################################################################################
# Strategy Configuration
#####################################################################################

if coordination_strategy == "integrated":
    config = {
        "ec_operators": "e1coupled,e2coupled,m1coupled,m1coupled",
        "ec_n_pop": "40",
        "ec_pop_size": "20",
        "ec_m": "5",
        "coupled_iterations": "1",
        "ec_initial_operator": "i1coupled",
    }
elif coordination_strategy == "sequential":
    config = {
        "ec_operators": "e1,e2,m1,m2",
        "ec_n_pop": "20",
        "ec_pop_size": "20",
        "ec_m": "5",
        "coupled_iterations": "1",
        "ec_initial_operator": "i1",
    }
elif coordination_strategy == "iterate":
    config = {
        "ec_operators": "e1,e2,m1,m2",
        "ec_n_pop": "5",
        "ec_pop_size": "20",
        "ec_m": "5",
        "coupled_iterations": "4",
        "ec_initial_operator": "i1",
    }
else:
    raise ValueError(f"Unsupported coordination_strategy: {coordination_strategy}")

#####################################################################################
# LNS Parameters
#####################################################################################

if EOH_PROBLEM == "lns_IRP_coupled":
    lns_paras = {
        "num_iters": 1000,
        "runtime_limit": 10,
        "no_improvement_time": 10,
        "no_improvement_iters": 1000,
    }

    lns_paras_second = lns_paras.copy()

    coupled_problems = (
        "None"
        if coordination_strategy == "integrated"
        else "lns_IRP_inventory,lns_IRP_vrp"
    )

elif EOH_PROBLEM == "lns_multi_robot_upmp_coupled":
    lns_paras = {
        "num_iters": 10,
        "runtime_limit": 10,
        "no_improvement_time": 10,
        "no_improvement_iters": 10,
    }

    lns_paras_second = {
        "num_iters": 90,
        "runtime_limit": 10,
        "no_improvement_time": 10,
        "no_improvement_iters": 90,
    }

#####################################################################################
# Execution
#####################################################################################

def run_experiment():
    base_cmd = [
        "ceoh", "run", EOH_PROBLEM,
        "--model_name", MODEL_NAME,
        "--ec_operators", config["ec_operators"],
        "--ec_n_pop", config["ec_n_pop"],
        "--ec_pop_size", config["ec_pop_size"],
        "--ec_m", config["ec_m"],
        "--use_example", "0",
        "--eoh_experiment_file", eoh_experiment_file,
        "--lns_paras", str(lns_paras),
        "--lns_paras_second", str(lns_paras_second),
        "--alns_cycles_iters", "1000",
        "--coupled_iterations", config["coupled_iterations"],
    ]

    if coordination_strategy == "integrated":
        base_cmd.extend([
            "--alns_cycles_timeout", "60",
            "--ec_initial_operator", config["ec_initial_operator"],
        ])
    else:
        base_cmd.extend([
            "--alns_cycles_timeout", "60",
            "--coupled_problems", coupled_problems,
        ])

    subprocess.run(base_cmd, check=False)


N = 3
for i in range(N):
    run_experiment()