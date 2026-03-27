
import os
import re

from solver.ceoh.llm.models import get_model_info

from solver.ceoh.utils.getParas import Paras
from solver.ceoh.problems import problems
from solver.ceoh.llm.interface_LLM import InterfaceLLM
from solver.ceoh.utils.get_coupled_heuristics import extract_heuristics


class Evolution():

    def __init__(self, paras, prompts, **kwargs):

        # set prompt interface
        self.paras = paras
        self.prompts = prompts

        self.llm_temperature = paras.llm_temperature
        self.prompt_task = (
            "".join(prompts.get_task())
            if isinstance(prompts.get_task(), list)
            else prompts.get_task()
        )
        self.prompt_func_name = prompts.get_func_name()
        self.prompt_func_inputs = prompts.get_func_inputs()
        self.prompt_func_outputs = prompts.get_func_outputs()
        self.prompt_inout_inf = prompts.get_inout_inf()
        self.prompt_other_inf = prompts.get_other_inf()
        self.prompt_example = prompts.get_examples()


        self.use_example = paras.ec_use_example
        self.use_idea = paras.ec_use_idea
        self.use_math_model = paras.ec_use_math_model
        if self.use_math_model:
            self.prompt_math_model = prompts.get_math_model()
        else:
            self.prompt_math_model = ""

        self.llm_url_info = paras.llm_url_info

        # Handle possibly coupled (list-of-lists) prompt inputs
        if isinstance(self.prompt_func_inputs[0], list):
            # Coupled case: build per-subproblem joined strings
            self.joined_inputs = [
                ", ".join(f"'{s}'" for s in sub_inputs)
                for sub_inputs in self.prompt_func_inputs
            ]
            self.joined_outputs = [
                ", ".join(f"'{s}'" for s in sub_outputs)
                for sub_outputs in self.prompt_func_outputs
            ]
        else:
            # Single-problem case (legacy)
            if len(self.prompt_func_inputs) > 1:
                self.joined_inputs = ", ".join(f"'{s}'" for s in self.prompt_func_inputs)
            else:
                self.joined_inputs = f"'{self.prompt_func_inputs[0]}'"

            if len(self.prompt_func_outputs) > 1:
                self.joined_outputs = ", ".join("'" + s + "'" for s in self.prompt_func_outputs)
            else:
                self.joined_outputs = "'" + self.prompt_func_outputs[0] + "'"

        # set LLMs
        self.api_endpoint = paras.llm_api_endpoint
        self.api_key = paras.llm_api_key
        self.model_LLM = paras.llm_model
        self.debug_mode = paras.exp_debug_mode  # close prompt checking

        self.interface_llm = InterfaceLLM(paras)

    def get_prompt_ideas(self, ideas):
        ideas_content = ""
        for i, indiv in enumerate(ideas):
            ideas_content += (
                "\n "
                f"No. {i + 1} idea details are: \n"
                f"Idea Name: {indiv['Idea Name']}\n"
                f"Logic: {indiv['Logic']}\n"
                f"Implementation: {indiv['Implementation']}\n"
            )
        return ideas_content

    def get_prompt_i1(self):

        prompt_content = self.prompt_task

        if self.use_example:
            prompt_content += f"\n" + self.prompt_example

        if self.use_math_model:
            prompt_content += "\n " + self.prompt_math_model

        if self.use_example:
            prompt_content += (f"\n"
                               f"First, understand the provided problem description and input and output examples. "
                               f"Then extract the main constraints of the problem. "
                               f"Second, think about how these constraints affect the requested heuristic. "
                               )
            prompt_content += f"Third, "
        else:
            prompt_content += (f"\n"
                               f"First, ")

        prompt_content += (f"describe your new algorithm and main steps in one sentence. "
                           f"The description must be inside a brace. "
                           f"\n "
                           f"Next, implement it in Python as a function named {self.prompt_func_name}. "
                           f"This function should accept {str(len(self.prompt_func_inputs))} input(s): {self.joined_inputs }. "
                           f"The function should return {str(len(self.prompt_func_outputs))} output(s): {self.joined_outputs}. "
                           f"{self.prompt_inout_inf} {self.prompt_other_inf} \n"
                           f"Do not give additional explanations. "
                           f"Remember: Describe your new algorithm and main steps in one sentence inside a brace and implement the algorithm in Python as a function."
                           )
        return prompt_content


    def get_prompt_e1(self, indivs):
        prompt_indiv = ""
        for i, indiv in enumerate(indivs):
            prompt_indiv += (
                f"No. {i + 1} algorithm and the corresponding code are: \n"
                f"{indiv['algorithm']} \n{indiv['code']} \n"
            )

        prompt_content = self.prompt_task

        if self.use_example:
            prompt_content += "\n" + self.prompt_example

        if self.use_math_model:
            prompt_content += "\n " + self.prompt_math_model

        prompt_content += (
            f"\nI have {len(indivs)} existing algorithms with their codes as follows: \n{prompt_indiv}"
            f"Please help me create a new algorithm that has a totally different form from the given ones. ")

        if self.use_example:
            prompt_content += (f"\n"
                               f"First, understand the provided problem description, input, and output examples. "
                               f"Then extract the main constraints of the problem. "
                               f"Second, think about how these constraints affect the requested heuristic. "
                               )
            prompt_content += f"Third, "
        else:
            prompt_content += (f"\n"
                               f"First, ")
        prompt_content += (
            f"describe your new algorithm and main steps in one sentence. The description must be inside a brace. "
            f"\n "
            f"Next, implement it in Python as a function named {self.prompt_func_name}. "
            f"This function should accept {len(self.prompt_func_inputs)} input(s): {self.joined_inputs}. "
            f"The function should return {len(self.prompt_func_outputs)} output(s): {self.joined_outputs}. "
            f"{self.prompt_inout_inf} {self.prompt_other_inf}\n"
            f"Do not give additional explanations. "
            f"Remember: Describe your new algorithm and main steps in one sentence inside a brace and implement the algorithm in Python as a function."

        )
        return prompt_content

    def get_prompt_e2(self, indivs):
        prompt_indiv = "".join(
            f"No. {i + 1} algorithm and the corresponding code are: \n{indiv['algorithm']} \n{indiv['code']} \n"
            for i, indiv in enumerate(indivs)
        )

        prompt_content = self.prompt_task

        if self.use_example:
            prompt_content += (f"\n{self.prompt_example}")

        if self.use_math_model:
            prompt_content += "\n " + self.prompt_math_model

        prompt_content += (
            f"\n I have {len(indivs)} existing algorithms with their codes as follows: \n{prompt_indiv}"
            f"Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them. ")

        if self.use_example:
            prompt_content += (f"\n"
                               f"First, understand the provided problem description, input, and output examples. "
                               f"Then extract the main constraints of the problem. "
                               f"Second, think about how these constraints affect the requested heuristic. "
                               )
            prompt_content += (f"Third, "
                               f"identify the common backbone idea in the provided algorithms. "
                               f"Fourth, ")
        else:
            prompt_content += (f"\n"
                               f"First, "
                               f"identify the common backbone idea in the provided algorithms. "
                               f"Second, ")

        prompt_content += (
            f"based on the backbone idea, describe your new algorithm in one sentence. "
            f"The description must be inside a brace. "
            f"\n "
            f"Next, implement it in Python as a function named {self.prompt_func_name}. "
            f"This function should accept {len(self.prompt_func_inputs)} input(s): {self.joined_inputs}. "
            f"The function should return {len(self.prompt_func_outputs)} output(s): {self.joined_outputs}. "
            f"{self.prompt_inout_inf} {self.prompt_other_inf}\n"
            f"Do not give additional explanations. "
            f"Remember: Describe your new algorithm and main steps in one sentence inside a brace and implement the algorithm in Python as a function."

        )
        return prompt_content

    def get_prompt_m1(self, indiv1):
        prompt_content = self.prompt_task

        if self.use_example:
            prompt_content += f"\n{self.prompt_example}"

        if self.use_math_model:
            prompt_content += "\n " + self.prompt_math_model

        prompt_content += (
            f"\nI have one algorithm with its code as follows:\n"
            f"Algorithm description: {indiv1['algorithm']}\n"
            f"Code:\n{indiv1['code']}\n"
            f"Please assist me in creating a new algorithm that has a different form but can be a modified version of the algorithm provided.\n"
        )

        if self.use_example:
            prompt_content += (f"\n"
                               f"First, understand the provided problem description, input, and output examples. "
                               f"Then extract the main constraints of the problem. "
                               f"Second, think about how these constraints affect the requested heuristic. "
                               )
            prompt_content += (f"Third, "
                               )
        else:
            prompt_content += (f"\n"
                               f"First, ")

        prompt_content += (
            f"describe your new algorithm and main steps in one sentence. The description must be inside a brace. "
            f"\n "
            f"Next, implement it in Python as a function named {self.prompt_func_name}. "
            f"This function should accept {len(self.prompt_func_inputs)} input(s): {self.joined_inputs}. "
            f"The function should return {len(self.prompt_func_outputs)} output(s): {self.joined_outputs}. "
            f"{self.prompt_inout_inf} {self.prompt_other_inf}\n"
            f"Do not give additional explanations. "
            f"Remember: Describe your new algorithm and main steps in one sentence inside a brace and implement the algorithm in Python as a function."
        )
        return prompt_content

    def get_prompt_m2(self, indiv1):
        prompt_content = self.prompt_task

        if self.use_example:
            prompt_content += f"\n{self.prompt_example}"

        if self.use_math_model:
            prompt_content += "\n " + self.prompt_math_model

        prompt_content += (
            f"\nI have one algorithm with its code as follows:\n"
            f"Algorithm description: {indiv1['algorithm']}\n"
            f"Code:\n{indiv1['code']}\n"
            f"Please identify the main algorithm parameters and assist me in creating a new algorithm that has different parameter settings for the score function provided.\n")

        if self.use_example:
            prompt_content += (f"\n"
                               f"First, understand the provided problem description, input, and output examples. "
                               f"Then extract the main constraints of the problem. "
                               f"Second, think about how these constraints affect the requested heuristic. "
                               )
            prompt_content += (f"Third, "
                               )
        else:
            prompt_content += (f"\n"
                               f"First, ")

        prompt_content += (
            f"describe your new algorithm and main steps in one sentence. The description must be inside a brace. "
            f"\n "
            f"Next, implement it in Python as a function named {self.prompt_func_name}. "
            f"This function should accept {len(self.prompt_func_inputs)} input(s): {self.joined_inputs}. "
            f"The function should return {len(self.prompt_func_outputs)} output(s): {self.joined_outputs}. "
            f"{self.prompt_inout_inf} {self.prompt_other_inf}\n"
            f"Do not give additional explanations. "
            f"Remember: Describe your new algorithm and main steps in one sentence inside a brace and implement the algorithm in Python as a function."

        )
        return prompt_content

    def get_prompt_m3(self, indiv1):
        prompt_content = self.prompt_task

        if self.use_example:
            prompt_content += f"\n{self.prompt_example}"

        if self.use_math_model:
            prompt_content += "\n " + self.prompt_math_model

        if self.use_example:
            prompt_content += ("\n"
                               "First, understand the provided problem description, input, and output examples. "
                               "Then extract the main constraints of the problem. "
                               "Second, think about how these constraints affect the requested heuristic. "
                               )
            prompt_content += ("Third, "
                               )
        else:
            prompt_content += (f"\n"
                               f"First, ")

        prompt_content += (
            f"you need to identify the main components in the function below. "
            f"Next, analyze whether any of these components can be overfit to the in-distribution instances. "
            f"Then, based on your analysis, simplify the components to enhance the generalization to potential out-of-distribution instances. "
            f"Finally, provide the revised code, keeping the function name, inputs, and outputs unchanged.\n"
            f"{indiv1['code']}\n"
            f"{self.prompt_inout_inf}\n"
            f"Do not give additional explanations. "
            f"Remember: Describe your new algorithm and main steps in one sentence inside a brace and implement the algorithm in Python as a function."
        )
        return prompt_content

    def get_prompt_p0(self, ideas):

        prompt_content = self.prompt_task

        if self.use_example:
            prompt_content += (f"\n{self.prompt_example}")

        prompt_content += "I have some ideas to design a new heuristic: \n"
        prompt_content += self.get_prompt_ideas(ideas)
        prompt_content += ""

        prompt_content += "Please help me create a new algorithm. "
        prompt_content += "The new algorithm should also be inspired by the provided ideas. "

        if self.use_example:
            prompt_content += (f"\n"
                               f'First, understand the provided problem description, input, and output examples. '
                               f"Then extract the main constraints of the problem. "
                               f"Second, think about how these constraints affect the requested heuristic. "
                               )
            prompt_content += (f"Third, "
                               f"identify the common backbone idea in the provided algorithms. "
                               f"Fourth, ")
        else:
            prompt_content += (f"\n"
                               f"First, "
                               f"identify the common backbone idea in the provided algorithms. "
                               f"Second, ")

        prompt_content += (
                f"based on the backbone idea, describe your new algorithm in one sentence. "
                f"The description must be inside a brace. "
                f"\n "
                f"Next, implement it in Python as a function named {self.prompt_func_name}. "
                f"This function should accept {len(self.prompt_func_inputs)} input(s): {self.joined_inputs}. "
                f"The function should return {len(self.prompt_func_outputs)} output(s): {self.joined_outputs}. "
                f"{self.prompt_inout_inf} {self.prompt_other_inf}\n"
                f"Do not give additional explanations. "
                f"Remember: Describe your new algorithm and main steps in one sentence inside a brace and implement the algorithm in Python as a function."

        )
        return prompt_content

    def get_prompt_p1(self, indivs, ideas):
        prompt_indiv = "".join(
            f"No. {i + 1} algorithm and the corresponding code are: \n{indiv['algorithm']} \n{indiv['code']} \n"
            for i, indiv in enumerate(indivs)
        )

        prompt_content = self.prompt_task

        if self.use_example:
            prompt_content += (f"\n{self.prompt_example}")

        prompt_content += "Here are some ideas to design a new heuristic: \n"
        prompt_content += self.get_prompt_ideas(ideas)
        prompt_content += ""

        prompt_content += (
                f"\n I have {len(indivs)} existing algorithms with their codes as follows: \n{prompt_indiv}")

        prompt_content += "Please help me create a new algorithm that enriches the given algorithms with the new ideas provided. "

        if self.use_example:
            prompt_content += (f"\n"
                               f'First, understand the provided problem description, input, and output examples. '
                               f"Than extract the main constraints of the problem. "
                               f"Second, think about how these constraints affect the requested heuristic. "
                               )
            prompt_content += (f"Third, "
                               f"identify the common backbone idea in the provided algorithms. "
                               f"Fourth, ")
        else:
            prompt_content += (f"\n"
                               f"First, "
                               f"identify the common backbone idea in the provided algorithms. "
                               f"Second, ")

        prompt_content += (
                f"based on the backbone ideas, describe your new algorithm in one sentence. "
                f"The description must be inside a brace. "
                f"\n "
                f"Next, implement it in Python as a function named {self.prompt_func_name}. "
                f"This function should accept {len(self.prompt_func_inputs)} input(s): {self.joined_inputs}. "
                f"The function should return {len(self.prompt_func_outputs)} output(s): {self.joined_outputs}. "
                f"{self.prompt_inout_inf} {self.prompt_other_inf}\n"
                f"Do not give additional explanations. "
                f"Remember: Describe your new algorithm and main steps in one sentence inside a brace and implement the algorithm in Python as a function."
        )
        return prompt_content

    def get_prompt_r1(self, programs):
        # Build list of existing program descriptions + code
        prompt_programs = ""
        for i, prog in enumerate(programs):
            prompt_programs += (
                f"No. {i + 1} heuristic program and its code are:\n"
                f"{prog['algorithm']}\n"
                f"{prog['code']}\n"
            )

        prompt_content = self.prompt_task

        if self.use_example:
            prompt_content += f"\n{self.prompt_example}"

        if self.use_math_model:
            prompt_content += "\n " + self.prompt_math_model

        prompt_content += (
            f"\nI have {len(programs)} heuristic programs with their codes as follows:\n"
            f"{prompt_programs}"
            f"Please help me design a new heuristic algorithm that complements and adds value to the existing set of heuristics. "
            f"The new heuristic should contribute a different perspective or mechanism relative to the listed programs."
        )

        if self.use_example:
            prompt_content += (
                f"\n"
                f"First, understand the provided problem description, input, and output examples. "
                f"Then extract the main constraints of the problem. "
                f"Second, think about how these constraints affect the requested heuristic. "
            )
            prompt_content += "Third, "
        else:
            prompt_content += "\nFirst, "

        prompt_content += (
            f"describe your new heuristic algorithm and main steps in one sentence. "
            f"The description must be inside a brace. "
            f"\n "
            f"Next, implement it in Python as a function named {self.prompt_func_name}. "
            f"This function should accept {len(self.prompt_func_inputs)} input(s): {self.joined_inputs}. "
            f"The function should return {len(self.prompt_func_outputs)} output(s): {self.joined_outputs}. "
            f"{self.prompt_inout_inf} {self.prompt_other_inf}\n"
            f"Do not give additional explanations. "
            f"Remember: Describe your new algorithm and main steps in one sentence inside a brace and implement the algorithm in Python as a function."
        )

        return prompt_content

    def get_prompt_i1coupled(self):

        # --------------------------------------------------
        # Global coupled problem description
        # --------------------------------------------------
        n_subproblems = len(self.prompts.get_task())

        prompt_content = ""

        prompt_content += (
            f"\nI am solving a coupled optimization problem consisting of "
            f"{n_subproblems} interacting subproblems.\n"
            f"I will fist describe the coupled problem and than each subproblem.\n\n"
        )

        try:
            prompt_content += self.prompts.get_coupled_task()
        except Exception:
            pass

        # --------------------------------------------------
        # Per-subproblem sections
        # --------------------------------------------------
        for i in range(n_subproblems):

            inputs_str = ", ".join(self.prompt_func_inputs[i])
            outputs_str = ", ".join(self.prompt_func_outputs[i])

            prompt_content += (
                f"\n==============================\n"
                f"### Subproblem {i}\n\n"
                f"{self.prompt_task[i]}\n\n"
                f"Function name: {self.prompt_func_name[i]}\n"
                f"Inputs: {inputs_str}\n"
                f"Outputs: {outputs_str}\n\n"
                f"{self.prompt_inout_inf[i]}\n"
                f"{self.prompt_other_inf[i]}\n\n"
            )

        # --------------------------------------------------
        # Prompt strategy specification
        # --------------------------------------------------

        prompt_content += (
            f"\nPlease help me create a new heuristic. \n")

        # --------------------------------------------------
        # Output specification (strict and machine-readable)
        # --------------------------------------------------
        prompt_content += (
            f"Please design ONE heuristic for EACH subproblem above.\n\n"
            f"For each subproblem:\n"
            f"- Briefly explain your core ideas behind the code.\n"
            f"- Provide a complete Python function implementation.\n\n"
            f"Return your answer using EXACTLY the following structure for EACH subproblem:\n\n"
            f"SUBPROBLEM <index>\n"
            f"THOUGHTS: <short explanation>\n"
            f"CODE:\n"
            f"```python\n"
            f"<complete python function>\n"
            f"```\n"
            f"---\n\n"
            f"Rules:\n"
            f"- Provide exactly {n_subproblems} subproblems.\n"
            f"- <index> must range from 0 to {n_subproblems - 1}.\n"
            f"- Each index must appear exactly once.\n"
            f"- The code MUST be inside a Python markdown code block.\n"
            f"- Use the exact headers: SUBPROBLEM, THOUGHTS, CODE.\n"
            f"- Separate entries using '---'.\n"
            f"- Do NOT include any additional text.\n"
        )

        return prompt_content

    def get_prompt_e1coupled(self, parents):

        """
        Construct a coupled prompt using per-subproblem prompt components.

        parents: list of lists
          parents[p][i] is the heuristic dict for subproblem i of parent p
        """

        if not parents or not parents[0]:
            raise ValueError("parents must be a non-empty list of subproblem heuristics")

        n_subproblems = len(self.prompts.get_task())
        print("n_subproblems", n_subproblems)


        # --------------------------------------------------
        # Global coupled problem description
        # --------------------------------------------------

        prompt_content = ""

        prompt_content += (
            f"\nI am solving a coupled optimization problem consisting of "
            f"{len(self.prompts.get_task())} interacting subproblems.\n"
            f"I will fist describe the coupled problem and than each subproblem.\n\n"
        )

        try:
            prompt_content += self.prompts.get_coupled_task()
        except Exception:
            pass

        # --------------------------------------------------
        # Per-subproblem sections
        # --------------------------------------------------
        for i in range(n_subproblems):
            print("i", i)

            inputs_str = ", ".join(self.prompt_func_inputs[i])
            outputs_str = ", ".join(self.prompt_func_outputs[i])

            prompt_content += (
                f"\n==============================\n"
                f"### Subproblem {i}\n\n"
                f"{self.prompt_task[i]}\n\n"
                f"Function name: {self.prompt_func_name[i]}\n"
                f"Inputs: {inputs_str}\n"
                f"Outputs: {outputs_str}\n\n"
                f"{self.prompt_inout_inf[i]}\n"
                f"{self.prompt_other_inf[i]}\n\n"
            )

            # List all parent heuristics for this subproblem
            prompt_content += (
                f"Existing heuristics for this subproblem:\n"
            )

            print("parents", parents)
            for p_idx, parent in enumerate(parents):
                sp = parent["sub_offsprings"][i]

                prompt_content += (
                    f"\n\nParent {p_idx + 1}:\n"
                    f"Subproblem Index: {i}\n"
                    f"Subproblem Name: {sp['subproblem']}\n"
                    f"Algorithm:\n{sp['algorithm']}\n"
                    f"Code:\n{sp['code']}\n"
                )

        prompt_content += (
            f"\n==============================\n")

        # --------------------------------------------------
        # Prompt strategy specification
        # --------------------------------------------------

        prompt_content += (
            f"\nPlease help me create new heuristics that are totally different form from the given ones. \n")

        # --------------------------------------------------
        # Output specification (strict and machine-readable)
        # --------------------------------------------------
        prompt_content += (
            f"Please design ONE heuristic for EACH subproblem above.\n\n"
            f"For each subproblem:\n"
            f"- Briefly explain your core ideas behind the code.\n"
            f"- Provide a complete Python function implementation.\n\n"
            f"Return your answer using EXACTLY the following structure for EACH subproblem:\n\n"
            f"SUBPROBLEM <index>\n"
            f"THOUGHTS: <short explanation>\n"
            f"CODE:\n"
            f"```python\n"
            f"<complete python function>\n"
            f"```\n"
            f"---\n\n"
            f"Rules:\n"
            f"- Provide exactly {n_subproblems} subproblems.\n"
            f"- <index> must range from 0 to {n_subproblems - 1}.\n"
            f"- Each index must appear exactly once.\n"
            f"- The code MUST be inside a Python markdown code block.\n"
            f"- Use the exact headers: SUBPROBLEM, THOUGHTS, CODE.\n"
            f"- Separate entries using '---'.\n"
            f"- Do NOT include any additional text.\n"
        )
        print("prompt_content", prompt_content)

        return prompt_content

    def get_prompt_e2coupled(self, parents):

        """
        Construct a coupled prompt using per-subproblem prompt components.

        parents: list of lists
          parents[p][i] is the heuristic dict for subproblem i of parent p
        """

        if not parents or not parents[0]:
            raise ValueError("parents must be a non-empty list of subproblem heuristics")

        n_subproblems = len(self.prompts.get_task())
        print("n_subproblems", n_subproblems)


        # --------------------------------------------------
        # Global coupled problem description
        # --------------------------------------------------

        prompt_content = ""

        prompt_content += (
            f"\nI am solving a coupled optimization problem consisting of "
            f"{len(self.prompts.get_task())} interacting subproblems.\n"
            f"I will fist describe the coupled problem and than each subproblem.\n\n"
        )

        try:
            prompt_content += self.prompts.get_coupled_task()
        except Exception:
            pass

        # --------------------------------------------------
        # Per-subproblem sections
        # --------------------------------------------------
        for i in range(n_subproblems):
            print("i", i)

            inputs_str = ", ".join(self.prompt_func_inputs[i])
            outputs_str = ", ".join(self.prompt_func_outputs[i])

            prompt_content += (
                f"\n==============================\n"
                f"### Subproblem {i}\n\n"
                f"{self.prompt_task[i]}\n\n"
                f"Function name: {self.prompt_func_name[i]}\n"
                f"Inputs: {inputs_str}\n"
                f"Outputs: {outputs_str}\n\n"
                f"{self.prompt_inout_inf[i]}\n"
                f"{self.prompt_other_inf[i]}\n\n"
            )

            # List all parent heuristics for this subproblem
            prompt_content += (
                f"Existing heuristics for this subproblem:\n"
            )

            print("parents", parents)
            for p_idx, parent in enumerate(parents):
                sp = parent["sub_offsprings"][i]

                prompt_content += (
                    f"\n\nParent {p_idx + 1}:\n"
                    f"Subproblem Index: {i}\n"
                    f"Subproblem Name: {sp['subproblem']}\n"
                    f"Algorithm:\n{sp['algorithm']}\n"
                    f"Code:\n{sp['code']}\n"
                )

        prompt_content += (
            f"\n==============================\n")

        # --------------------------------------------------
        # Prompt strategy specification
        # --------------------------------------------------

        prompt_content += (
            f"\n Please help me create new heuristics that are totally different form from the given ones but can be motivated from them.\n"
        )

        # --------------------------------------------------
        # Output specification (strict and machine-readable)
        # --------------------------------------------------
        prompt_content += (
            f"Please design ONE heuristic for EACH subproblem above.\n\n"
            f"For each subproblem:\n"
            f"- Briefly explain your core ideas behind the code.\n"
            f"- Provide a complete Python function implementation.\n\n"
            f"Return your answer using EXACTLY the following structure for EACH subproblem:\n\n"
            f"SUBPROBLEM <index>\n"
            f"THOUGHTS: <short explanation>\n"
            f"CODE:\n"
            f"```python\n"
            f"<complete python function>\n"
            f"```\n"
            f"---\n\n"
            f"Rules:\n"
            f"- Provide exactly {n_subproblems} subproblems.\n"
            f"- <index> must range from 0 to {n_subproblems - 1}.\n"
            f"- Each index must appear exactly once.\n"
            f"- The code MUST be inside a Python markdown code block.\n"
            f"- Use the exact headers: SUBPROBLEM, THOUGHTS, CODE.\n"
            f"- Separate entries using '---'.\n"
            f"- Do NOT include any additional text.\n"
        )
        print("prompt_content", prompt_content)

        return prompt_content

    def get_prompt_m1coupled(self, parents):

        """
        Construct a coupled prompt using per-subproblem prompt components.

        parents: list of lists
          parents[p][i] is the heuristic dict for subproblem i of parent p
        """

        if not parents or not parents[0]:
            raise ValueError("parents must be a non-empty list of subproblem heuristics")

        n_subproblems = len(self.prompts.get_task())
        print("n_subproblems", n_subproblems)


        # --------------------------------------------------
        # Global coupled problem description
        # --------------------------------------------------

        prompt_content = ""

        prompt_content += (
            f"\nI am solving a coupled optimization problem consisting of "
            f"{len(self.prompts.get_task())} interacting subproblems.\n"
            f"I will fist describe the coupled problem and than each subproblem.\n\n"
        )

        try:
            prompt_content += self.prompts.get_coupled_task()
        except Exception:
            pass

        # --------------------------------------------------
        # Per-subproblem sections
        # --------------------------------------------------
        for i in range(n_subproblems):
            print("i", i)

            inputs_str = ", ".join(self.prompt_func_inputs[i])
            outputs_str = ", ".join(self.prompt_func_outputs[i])

            prompt_content += (
                f"\n==============================\n"
                f"### Subproblem {i}\n\n"
                f"{self.prompt_task[i]}\n\n"
                f"Function name: {self.prompt_func_name[i]}\n"
                f"Inputs: {inputs_str}\n"
                f"Outputs: {outputs_str}\n\n"
                f"{self.prompt_inout_inf[i]}\n"
                f"{self.prompt_other_inf[i]}\n\n"
            )

            # List all parent heuristics for this subproblem
            prompt_content += (
                f"Existing heuristics for this subproblem:\n"
            )

            print("parents", parents)
            for p_idx, parent in enumerate(parents):
                sp = parent["sub_offsprings"][i]

                prompt_content += (
                    f"\n\nParent {p_idx + 1}:\n"
                    f"Subproblem Index: {i}\n"
                    f"Subproblem Name: {sp['subproblem']}\n"
                    f"Algorithm:\n{sp['algorithm']}\n"
                    f"Code:\n{sp['code']}\n"
                )

        prompt_content += (
            f"\n==============================\n")

        # --------------------------------------------------
        # Prompt strategy specification
        # --------------------------------------------------

        prompt_content += (
            f"\n Please assist me in creating new heuristics that are a different form but can be a modified version of the heuristics provided.\n"
        )

        # --------------------------------------------------
        # Output specification (strict and machine-readable)
        # --------------------------------------------------
        prompt_content += (
            f"Please design ONE heuristic for EACH subproblem above.\n\n"
            f"For each subproblem:\n"
            f"- Briefly explain your core ideas behind the code.\n"
            f"- Provide a complete Python function implementation.\n\n"
            f"Return your answer using EXACTLY the following structure for EACH subproblem:\n\n"
            f"SUBPROBLEM <index>\n"
            f"THOUGHTS: <short explanation>\n"
            f"CODE:\n"
            f"```python\n"
            f"<complete python function>\n"
            f"```\n"
            f"---\n\n"
            f"Rules:\n"
            f"- Provide exactly {n_subproblems} subproblems.\n"
            f"- <index> must range from 0 to {n_subproblems - 1}.\n"
            f"- Each index must appear exactly once.\n"
            f"- The code MUST be inside a Python markdown code block.\n"
            f"- Use the exact headers: SUBPROBLEM, THOUGHTS, CODE.\n"
            f"- Separate entries using '---'.\n"
            f"- Do NOT include any additional text.\n"
        )
        print("prompt_content", prompt_content)

        return prompt_content

    def get_prompt_m2coupled(self, parents):

        """
        Construct a coupled prompt using per-subproblem prompt components.

        parents: list of lists
          parents[p][i] is the heuristic dict for subproblem i of parent p
        """

        if not parents or not parents[0]:
            raise ValueError("parents must be a non-empty list of subproblem heuristics")

        n_subproblems = len(self.prompts.get_task())

        # --------------------------------------------------
        # Global coupled problem description
        # --------------------------------------------------

        prompt_content = ""

        prompt_content += (
            f"\nI am solving a coupled optimization problem consisting of "
            f"{len(self.prompts.get_task())} interacting subproblems.\n"
            f"I will fist describe the coupled problem and than each subproblem.\n\n"
        )

        try:
            prompt_content += self.prompts.get_coupled_task()
        except Exception:
            pass

        # --------------------------------------------------
        # Per-subproblem sections
        # --------------------------------------------------
        for i in range(n_subproblems):
            print("i", i)

            inputs_str = ", ".join(self.prompt_func_inputs[i])
            outputs_str = ", ".join(self.prompt_func_outputs[i])

            prompt_content += (
                f"\n==============================\n"
                f"### Subproblem {i}\n\n"
                f"{self.prompt_task[i]}\n\n"
                f"Function name: {self.prompt_func_name[i]}\n"
                f"Inputs: {inputs_str}\n"
                f"Outputs: {outputs_str}\n\n"
                f"{self.prompt_inout_inf[i]}\n"
                f"{self.prompt_other_inf[i]}\n\n"
            )

            # List all parent heuristics for this subproblem
            prompt_content += (
                f"Existing heuristics for this subproblem:\n"
            )

            print("parents", parents)
            for p_idx, parent in enumerate(parents):
                sp = parent["sub_offsprings"][i]

                prompt_content += (
                    f"\n\nParent {p_idx + 1}:\n"
                    f"Subproblem Index: {i}\n"
                    f"Subproblem Name: {sp['subproblem']}\n"
                    f"Algorithm:\n{sp['algorithm']}\n"
                    f"Code:\n{sp['code']}\n"
                )

        prompt_content += (
            f"\n==============================\n")

        # --------------------------------------------------
        # Prompt strategy specification
        # --------------------------------------------------

        prompt_content += (
            f"\n Please identify the main heuristic parameters and assist me in creating new heuristics that has different parameter settings for the score heuristics provided.\n"
        )

        # --------------------------------------------------
        # Output specification (strict and machine-readable)
        # --------------------------------------------------
        prompt_content += (
            f"Please design ONE heuristic for EACH subproblem above.\n\n"
            f"For each subproblem:\n"
            f"- Briefly explain your core ideas behind the code.\n"
            f"- Provide a complete Python function implementation.\n\n"
            f"Return your answer using EXACTLY the following structure for EACH subproblem:\n\n"
            f"SUBPROBLEM <index>\n"
            f"THOUGHTS: <short explanation>\n"
            f"CODE:\n"
            f"```python\n"
            f"<complete python function>\n"
            f"```\n"
            f"---\n\n"
            f"Rules:\n"
            f"- Provide exactly {n_subproblems} subproblems.\n"
            f"- <index> must range from 0 to {n_subproblems - 1}.\n"
            f"- Each index must appear exactly once.\n"
            f"- The code MUST be inside a Python markdown code block.\n"
            f"- Use the exact headers: SUBPROBLEM, THOUGHTS, CODE.\n"
            f"- Separate entries using '---'.\n"
            f"- Do NOT include any additional text.\n"
        )
        print("prompt_content", prompt_content)

        return prompt_content

    def _extract_alg_and_code_from_response(self, response):
        algorithm = re.findall(r"\{(.*?)\}", response, re.DOTALL)
        if len(algorithm) > 0:
            algorithm = [algorithm[0]]

        if len(algorithm) == 0:
            if 'python' in response:
                algorithm = re.findall(r'^.*?(?=python)', response, re.DOTALL)
            elif 'import' in response:
                algorithm = re.findall(r'^.*?(?=import)', response, re.DOTALL)
            else:
                algorithm = re.findall(r'^.*?(?=def)', response, re.DOTALL)

        code = re.findall(r"import.*return", response, re.DOTALL)
        if len(code) == 0:
            code = re.findall(r"def.*return", response, re.DOTALL)

        return algorithm, code

    def _get_alg(self, prompt_content):

        response, full_response = self.interface_llm.get_response(prompt_content)

        algorithm, code = self._extract_alg_and_code_from_response(response)

        n_retry = 1
        while (len(algorithm) == 0 or len(code) == 0):
            if self.debug_mode:
                print("Error: algorithm or code not identified, wait 1 seconds and retrying ... ")

            response, full_response = self.interface_llm.get_response(prompt_content)

            algorithm, code = self._extract_alg_and_code_from_response(response)

            if n_retry > 3:
                break
            n_retry += 1

        algorithm = algorithm[0]
        code = code[0]

        # Grab all return values instead of just the first
        matches = re.findall(r'return\s+(.*?)(?:\n|$)', response)

        if matches:
            match_ret_val = matches[-1]  # last return variable
        else:
            matches = re.findall(r'return\s+(.+)', response)
            match_ret_val = matches[-1] if matches else None

        if match_ret_val:
            code_all = code + " " + match_ret_val
        else:
            code_all = code

        return [code_all, algorithm], full_response

    def i1(self):

        prompt_content = self.get_prompt_i1()

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ i1 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm], full_response = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm], prompt_content, full_response

    def e1(self, parents):

        prompt_content = self.get_prompt_e1(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e1 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm], full_response = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm], prompt_content, full_response

    def e2(self, parents):

        prompt_content = self.get_prompt_e2(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e2 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm], full_response = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm], prompt_content, full_response

    def m1(self, parents):

        prompt_content = self.get_prompt_m1(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m1 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm], full_response = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm], prompt_content, full_response

    def m2(self, parents):

        prompt_content = self.get_prompt_m2(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m2 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm], full_response = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm], prompt_content, full_response

    def m3(self, parents):

        prompt_content = self.get_prompt_m3(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m3 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm], full_response = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm], prompt_content, full_response

    def p1(self, parents, ideas):

        prompt_content = self.get_prompt_p1(parents,ideas)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ p1 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm], full_response = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm], prompt_content, full_response

    def p0(self, ideas):

        prompt_content = self.get_prompt_p0(ideas)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ p0 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm], full_response = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm], prompt_content, full_response

    def i1coupled(self):
        """
        Initial coupled heuristic generation.
        Generates one heuristic per subproblem in a single LLM call.
        """

        prompt_content = self.get_prompt_i1coupled()

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ i1coupled ] :\n")
            print(prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        # --------------------------------------------------
        # LLM call
        # --------------------------------------------------
        response, full_response = self.interface_llm.get_response(prompt_content)

        # --------------------------------------------------
        # Parse LLM response
        # --------------------------------------------------
        try:
            heuristics = extract_heuristics(response)
        except Exception as e:
            raise ValueError("Failed to parse heuristics from i1coupled response") from e

        if not heuristics:
            raise ValueError("No heuristics returned in i1coupled response")

        # --------------------------------------------------
        # Build offspring list
        # --------------------------------------------------
        offspring = {
            'objective': None,
            'sub_offsprings': None,
        }

        sub_offsprings = []

        subproblems = self.prompts.get_subproblems()

        for h in heuristics:
            idx = h.get("subproblem_index")
            if idx is None:
                raise ValueError("Missing subproblem_index in heuristic entry")

            sub_offspring = {
                'subproblem': subproblems[int(idx)],
                'algorithm': h.get("thoughts"),
                'code': h.get("code"),
            }

            sub_offsprings.append(sub_offspring)

        offspring["sub_offsprings"] = sub_offsprings

        return offspring, prompt_content, full_response

    def e1coupled(self, parents):
        prompt_content = self.get_prompt_e1coupled(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e1coupled ] :\n")
            print(prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        # --------------------------------------------------
        # LLM call
        # --------------------------------------------------
        response, full_response = self.interface_llm.get_response(prompt_content)

        # --------------------------------------------------
        # Parse LLM response
        # --------------------------------------------------
        try:
            heuristics = extract_heuristics(response)
        except Exception as e:
            raise ValueError("Failed to parse heuristics from e1coupled response") from e

        if not heuristics:
            raise ValueError("No heuristics returned in e1coupled response")

        # --------------------------------------------------
        # Build offspring list
        # --------------------------------------------------
        offspring = {
            'objective': None,
            'sub_offsprings': None,
        }

        sub_offsprings = []

        subproblems = self.prompts.get_subproblems()

        for h in heuristics:
            idx = h.get("subproblem_index")
            if idx is None:
                raise ValueError("Missing subproblem_index in heuristic entry")

            sub_offspring = {
                'subproblem': subproblems[int(idx)],
                'algorithm': h.get("thoughts"),
                'code': h.get("code"),
            }

            sub_offsprings.append(sub_offspring)

        offspring["sub_offsprings"] = sub_offsprings

        return offspring, prompt_content, full_response

    def e2coupled(self, parents):
        prompt_content = self.get_prompt_e2coupled(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e2coupled ] :\n")
            print(prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        # --------------------------------------------------
        # LLM call
        # --------------------------------------------------
        response, full_response = self.interface_llm.get_response(prompt_content)

        # --------------------------------------------------
        # Parse LLM response
        # --------------------------------------------------
        try:
            heuristics = extract_heuristics(response)
        except Exception as e:
            raise ValueError("Failed to parse heuristics from e2coupled response") from e

        if not heuristics:
            raise ValueError("No heuristics returned in e2coupled response")

        # --------------------------------------------------
        # Build offspring list
        # --------------------------------------------------
        offspring = {
            'objective': None,
            'sub_offsprings': None,
        }

        sub_offsprings = []

        subproblems = self.prompts.get_subproblems()

        for h in heuristics:
            idx = h.get("subproblem_index")
            if idx is None:
                raise ValueError("Missing subproblem_index in heuristic entry")

            sub_offspring = {
                'subproblem': subproblems[int(idx)],
                'algorithm': h.get("thoughts"),
                'code': h.get("code"),
            }

            sub_offsprings.append(sub_offspring)

        offspring["sub_offsprings"] = sub_offsprings

        return offspring, prompt_content, full_response

    def m1coupled(self, parents):
        prompt_content = self.get_prompt_m1coupled(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m1coupled ] :\n")
            print(prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        # --------------------------------------------------
        # LLM call
        # --------------------------------------------------
        response, full_response = self.interface_llm.get_response(prompt_content)

        # --------------------------------------------------
        # Parse LLM response
        # --------------------------------------------------
        try:
            heuristics = extract_heuristics(response)
        except Exception as e:
            raise ValueError("Failed to parse heuristics from m1coupled response") from e

        if not heuristics:
            raise ValueError("No heuristics returned in m1coupled response")

        # --------------------------------------------------
        # Build offspring list
        # --------------------------------------------------
        offspring = {
            'objective': None,
            'sub_offsprings': None,
        }

        sub_offsprings = []

        subproblems = self.prompts.get_subproblems()

        for h in heuristics:
            idx = h.get("subproblem_index")
            if idx is None:
                raise ValueError("Missing subproblem_index in heuristic entry")

            sub_offspring = {
                'subproblem': subproblems[int(idx)],
                'algorithm': h.get("thoughts"),
                'code': h.get("code"),
            }

            sub_offsprings.append(sub_offspring)

        offspring["sub_offsprings"] = sub_offsprings

        return offspring, prompt_content, full_response

    def m2coupled(self, parents):
        prompt_content = self.get_prompt_m2coupled(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m2coupled ] :\n")
            print(prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        # --------------------------------------------------
        # LLM call
        # --------------------------------------------------
        response, full_response = self.interface_llm.get_response(prompt_content)

        # --------------------------------------------------
        # Parse LLM response
        # --------------------------------------------------
        try:
            heuristics = extract_heuristics(response)
        except Exception as e:
            raise ValueError("Failed to parse heuristics from m2coupled response") from e

        if not heuristics:
            raise ValueError("No heuristics returned in m2coupled response")

        # --------------------------------------------------
        # Build offspring list
        # --------------------------------------------------
        offspring = {
            'objective': None,
            'sub_offsprings': None,
        }

        sub_offsprings = []

        subproblems = self.prompts.get_subproblems()

        for h in heuristics:
            idx = h.get("subproblem_index")
            if idx is None:
                raise ValueError("Missing subproblem_index in heuristic entry")

            sub_offspring = {
                'subproblem': subproblems[int(idx)],
                'algorithm': h.get("thoughts"),
                'code': h.get("code"),
            }

            sub_offsprings.append(sub_offspring)

        offspring["sub_offsprings"] = sub_offsprings

        return offspring, prompt_content, full_response


if __name__ == "__main__":
    eoh_experiment_file = (
            "exp_bay_5_wh_2_fill_0.8_seed_0-9_priority_5_robots_2_ad_1010.json"
        )

    os.environ["CURRENT_EXPERIMENT"] =r"C:\Users\dq0913\PycharmProjects\multibay-reshuffling\eoh_results\results_20251218_114155_lns_multi_robot_upmp_coupled_qwen25coder32b_ueFalse_uiFalse_uitypeNone"
    os.environ["OUTPUT_PATH"] = r"C:\Users\dq0913\PycharmProjects\multibay-reshuffling\eoh_results\results_20251218_114155_lns_multi_robot_upmp_coupled_qwen25coder32b_ueFalse_uiFalse_uitypeNone"
    os.environ["LLM_LOCAL_URL"] = "https://api.imi-services.imi.kit.edu/api/generate"
    os.environ["MODEL_NAME"] = 'qwen2.5-coder:32b'  # "gemma3:27b" 'llama3.1:70b', 'gemma2:27b', 'nemotron:latest', 'qwen2.5-coder:32b', 'gpt-4o-2024-08-06'

    llm_api_endpoint, add_url_info, api_key, llm_use_local = get_model_info('qwen2.5-coder:32b')

    # --------------------------------------------------
    # Shared base parameters
    # --------------------------------------------------
    base_paras = dict(
            eoh_experiment_file=eoh_experiment_file,
            lns_paras={
                "num_iters": 10,
                "runtime_limit": 10,
                "no_improvement_time": 10,
                "no_improvement_iters": 10,
            },
            lns_paras_second={
                "num_iters": 90,
                "runtime_limit": 10,
                "no_improvement_time": 10,
                "no_improvement_iters": 90,
            },
            alns_cycles_iters=5,
            alns_cycles_timeout=60,
            coupled_problems= ["lns_multi_robot_upmp_coupled"],
    )

    paras = Paras()

    paras.set_paras(
            **base_paras,
            method = 'ceoh',
            ec_operators= ['m1coupled'],
            ec_use_math_model =False,
            problem='lns_multi_robot_upmp_coupled',
            target_subproblem='lns_multi_robot_upmp_coupled',
            llm_api_endpoint=llm_api_endpoint,
            llm_api_key="",
            llm_model='qwen2.5-coder:32b',
            llm_use_local=True,
            llm_local_url=llm_api_endpoint,
            llm_temperature=None,
            llm_url_info=add_url_info,
    )
    api_response = (
        "{The new heuristic scores each move based on its criticality, diversity of access points, "
        "dependency impact on unequal priority moves, and randomness to enhance exploration and "
        "diversification with adjusted weights.}\n\n"
        "```python\n"
        "import numpy as np\n\n"
        "def destroy_scoring(current_solution, data, rng):\n"
        "    scores = []\n"
        "    total_moves = sum(len(robot_moves) for robot_moves in current_solution)\n"
        "    k = max(1, int(total_moves * 0.2))  # Remove 20% of the moves\n"
        "    \n"
        "    # Determine critical access points based on their importance\n"
        "    critical_access_points = {ap: len(data.ap_distance[ap]) for ap in range(len(data.ap_distance))}\n"
        "    \n"
        "    # Calculate dependency impact scores for each move (unequal priority)\n"
        "    dependency_impact_uneq = {}\n"
        "    all_dependencies_uneq = data.move_dependencies['unequal_prio']['end-end'] + \\\n"
        "                            data.move_dependencies['unequal_prio']['start-end'] + \\\n"
        "                            data.move_dependencies['unequal_prio']['end-start'] + \\\n"
        "                            data.move_dependencies['unequal_prio']['start-start']\n"
        "    \n"
        "    for dep in all_dependencies_uneq:\n"
        "        if dep[0] not in dependency_impact_uneq:\n"
        "            dependency_impact_uneq[dep[0]] = 1\n"
        "        else:\n"
        "            dependency_impact_uneq[dep[0]] += 1\n"
        "    \n"
        "    # Score each move\n"
        "    for robot_moves in current_solution:\n"
        "        for move in robot_moves:\n"
        "            if move['id'] == 0:\n"
        "                continue\n"
        "            \n"
        "            pickup_ap, dropoff_ap = move['move']\n"
        "            \n"
        "            # Dependency impact score (unequal priority)\n"
        "            dep_impact_uneq_score = dependency_impact_uneq.get(move['id'], 0)\n"
        "            \n"
        "            # Access point diversity score\n"
        "            access_point_diversity = critical_access_points[pickup_ap] + critical_access_points[dropoff_ap]\n"
        "            \n"
        "            # Random factor to introduce variability\n"
        "            random_factor = rng.uniform(0.85, 1.15)\n"
        "            \n"
        "            # Score calculation considering dependency impact (unequal priority), access point diversity, and randomness\n"
        "            score = (dep_impact_uneq_score * 4 + access_point_diversity * 3) * random_factor\n"
        "            \n"
        "            scores.append((move['id'], score))\n"
        "    \n"
        "    # Sort moves by score in descending order\n"
        "    scores.sort(key=lambda x: x[1], reverse=True)\n"
        "    \n"
        "    return scores, k\n"
        "```\n"
    )

    subProblemGenerator = problems.Probs(paras)
    problem = subProblemGenerator.get_problem()  # problem evaluation function
    evo = Evolution(paras, problem.prompts)
    algorithm, code = evo._extract_alg_and_code_from_response(api_response)
    print(algorithm)
    print(code)