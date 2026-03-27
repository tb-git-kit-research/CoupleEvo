import os
import time
import subprocess
from dotenv import load_dotenv
import glob

# Load environment variables from .env
load_dotenv()

# Get required environment variables
BASE_PATH = os.getenv("BASE_PATH")

if not BASE_PATH:
    print("BASE_PATH for evaluation not set.")
    print("Please create a .env file and add this variable")
    print("For more information, please look into the readme")
    time.sleep(10)
    exit(1)

#####################################################################################
#################                 General Settings                  #################
#####################################################################################

os.environ["PYTHONUNBUFFERED"] = "False"
os.environ["TIMEOUT"] = "120"  # Timeout for every request thread in seconds
os.environ["LOGGING"] = "False"
os.environ["DETAILED_OUTPUT"] = "False"  # Show all code and explanations

#####################################################################################
#################              Build Process and Execute            #################
#####################################################################################

print("########################## OLLAMA HOSTNAME ##########################")
os.environ["LLM_LOCAL_URL"] = "https://api.imi-services.imi.kit.edu/api/generate" # "https://api.imi-services.imi.kit.edu" # "https://api3.imi-services.imi.kit.edu"
print(os.environ["LLM_LOCAL_URL"])

# Install dependencies
subprocess.run(["pip", "install", "."], stdout=subprocess.DEVNULL)

# Check available LLM models
print("########################## LLM MODELS AVAILABLE ##########################")

EOH_PROBLEM = "lns_multi_robot_upmp_coupled"
os.environ["EOH_PROBLEM"] = EOH_PROBLEM


MODEL_NAME = 'qwen3-coder:30b' # "gemma3:27b" 'llama3.1:70b', 'gemma2:27b', 'nemotron:latest', 'qwen2.5-coder:32b', 'gpt-4o-2024-08-06'
os.environ["MODEL_NAME"] = MODEL_NAME

IDEA_MODEL_NAME = "o3-mini-2025-01-31"
os.environ["IDEA_MODEL_NAME"] = IDEA_MODEL_NAME

IDEA_EXPERIMENT_TYPE = "NONE"  # strategy or injection or "NONE", strategy_e2
print(f"IDEA_EXPERIMENT_TYPE: {IDEA_EXPERIMENT_TYPE}")

if IDEA_EXPERIMENT_TYPE == "NONE":
    N = 3
    for i in range(N):  # Run the strategy N times
        USE_EXAMPLE = 0
        CONTINUE_TARGET_POP = -1
        CONTINUE_TARGET_LOOP = -1

        lns_paras = {
            "num_iters" : 10,
            "runtime_limit" : 10,
            "no_improvement_time" : 10,
            "no_improvement_iters" : 10,
        }

        lns_paras_second = {
            "num_iters": 90,
            "runtime_limit": 10,
            "no_improvement_time": 10,
            "no_improvement_iters": 90,
        }

        subprocess.run([
                "ceoh", "run", EOH_PROBLEM,
                "--model_name", MODEL_NAME,
                "--ec_operators", "e1,e2,m1,m2",
                "--ec_n_pop", "5",
                "--ec_pop_size", "20",
                "--ec_m", "5",
                "--use_example", str(USE_EXAMPLE),
                "--continue_target_pop", str(CONTINUE_TARGET_POP),
                "--eoh_experiment_file", "exp_bay_5_wh_2_fill_0.8_seed_0-9_priority_5_robots_2_ad_1010.json",
                "--lns_paras", str(lns_paras),
                "--lns_paras_second", str(lns_paras_second),
                #"--continue_target_folder", r"C:\Users\dq0913\PycharmProjects\multibay-reshuffling\eoh_results\results_20251215_123745_lns_multi_robot_upmp_coupled_qwen25coder32b_ueFalse_uiFalse_uitypeNone",
                "--alns_cycles_iters", str(1000),
                "--alns_cycles_timeout", str(60),
                "--coupled_problems", "lns_multi_robot_upmp_move_search," # problem with defined prompts
                                      "lns_multi_robot_upmp_vrp",
                "--coupled_iterations", str(4),
                #"--continue_target_loop", str(CONTINUE_TARGET_LOOP),
            ])
