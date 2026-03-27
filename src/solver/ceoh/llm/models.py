

import os

###################################################################
# LLM Models for configuration of local or openrouter requests

OPENROUTER_MODELS = [
    'meta-llama/llama-3.2-3b-instruct:free'
    'deepseek/deepseek-r1',
    'anthropic/claude-3.5-sonnet',
    'deepseek/deepseek-chat',
    'minimax/minimax-01',
    'google/gemma-2-27b-it',
    'google/gemma-3-27b-it:free',
    'openai/gpt-oss-20b:free',
    'qwen/qwen3-coder:free',
    'openai/gpt-4.1-nano',
]

DEEPSEEK_MODELS = [
    'deepseek-chat',
    'deepseek-reasoner',
]

OPENAI_MODELS = [
    'gpt-4',
    'gpt-4o',
    'gpt-4o-2024-08-06',
    'gpt-4o-2024-11-20',
    'gpt-4o-mini-2024-07-18',
    'gpt-4.1-nano-2025-04-14',
    'gpt-5-mini-2025-08-07',
    'gpt-5-nano-2025-08-07',
]

LOCAL_MODELS = [
    'llama3.1:70b',
    'gemma2:27b',
    'gemma3:12b',
    'gemma3:27b',
    'nemotron:latest',
    'qwen2.5-coder:1.5b',
    'qwen2.5-coder:3b',
    'qwen2.5-coder:7b',
    'qwen2.5-coder:14b',
    'qwen2.5-coder:32b',
    'codestral:22b',
    'deepseek-r1:32b',
    'qwen3-coder:30b',
    'qwen3:4b',
    'qwen3:8b',
    'qwen3:14b',
    'qwen3:32b',
]

def get_model_info(model_name):

    llm_use_local = False
    add_url_info = False

    if model_name in LOCAL_MODELS:
        print("--- USE Local Model")
        print(f"--- > {model_name}")

        # Enter full url of your model
        # get environment variable from .env file LLM_LOCAL_URL
        llm_api_endpoint = os.getenv('LLM_LOCAL_URL', "https://api3.imi-ki03.imi.kit.edu/api/generate")
        # llm_api_endpoint = os.getenv('LLM_LOCAL_URL', "https://imi-ki03.imi.kit.edu/api/generate")
        llm_use_local = True
        api_key = ""

    elif model_name in OPENROUTER_MODELS:
        print("--- USE OPENROUTER Model")
        print(f"--- > {model_name}")

        api_key = os.getenv('OPENROUTER_API_KEY')
        llm_api_endpoint = "openrouter.ai"
        add_url_info="/api/v1/chat/completions"

    elif model_name in DEEPSEEK_MODELS:
        print("--- USE DEEPSEEK Model")
        print(f"--- > {model_name}")

        api_key = os.getenv('DEEPSEEK_API_KEY')
        llm_api_endpoint = "api.deepseek.com"
        add_url_info="/chat/completions"

    else:
        print("--- USE REST Model")
        print(f"--- > {model_name}")

        api_key = os.getenv('OPENAI_API_KEY')
        llm_api_endpoint = "api.openai.com"
        add_url_info="/v1/chat/completions"

    if api_key == None:
        print('##########################################')
        print(' -- No LLM API Key is set. -- ')
        print(' -- Have a look into the .env file! -- ')
        print('##########################################')
        exit(0)

    return llm_api_endpoint, add_url_info, api_key, llm_use_local