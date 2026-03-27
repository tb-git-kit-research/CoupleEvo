
import os

from solver.ceoh.llm.api_general import InterfaceAPI
from solver.ceoh.llm.api_local_llm import InterfaceLocalLLM


class InterfaceLLM:
    def __init__(self, paras):

        self.api_endpoint = paras.llm_api_endpoint
        self.api_key = paras.llm_api_key
        self.model_LLM = paras.llm_model
        self.debug_mode = paras.exp_debug_mode
        self.llm_use_local = paras.llm_use_local
        self.llm_local_url = paras.llm_local_url
        self.llm_temperature = paras.llm_temperature
        self.llm_url_info = paras.llm_url_info

        print("- check LLM API")

        if self.llm_use_local:
            print('local llm delopyment is used ...')
            
            if self.llm_local_url == None or self.llm_local_url == 'xxx' :
                print(">> Stop with empty url for local llm !")
                exit()

            self.interface_llm = InterfaceLocalLLM(
                self.llm_local_url,
                self.model_LLM,
                self.llm_temperature,
            )

        else:
            print('remote llm api is used ...')
            print(self.model_LLM)


            if self.api_key is None or self.api_endpoint is None or self.api_key == 'xxx' or self.api_endpoint == 'xxx':
                print(">> Stop with wrong API setting: Set api_endpoint (e.g., api.chat...) and api_key (e.g., kx-...) !")
                exit()

            self.interface_llm = InterfaceAPI(
                self.api_endpoint,
                self.api_key,
                self.model_LLM,
                self.debug_mode,
                self.llm_temperature,
                self.llm_url_info
            )

            
        res = self.interface_llm.get_response("1+1=?")

        if res is None:
            print(">> Error in LLM API, wrong endpoint, key, model or local deployment!")
            exit()


    def get_response(self, prompt_content):
        response, json_data = self.interface_llm.get_response(prompt_content)

        if response is None:
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            print("[API ERROR]: Response is None")
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            return None, None

        if response == "":
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            print("[API ERROR]: Response is Empty")
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

        return response, json_data

