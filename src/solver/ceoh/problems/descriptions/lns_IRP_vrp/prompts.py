


from solver.ceoh.problems.descriptions.lns_IRP_coupled import prompts
PROBLEM_INDEX = 1

class GetPrompts:

    def __init__(self):
        base = prompts.GetPrompts()
        self.coupled_task = base.get_coupled_task()
        self.prompt_task = base.get_task()[PROBLEM_INDEX]
        self.prompt_func_name = base.get_func_name()[PROBLEM_INDEX]
        self.prompt_func_inputs = base.get_func_inputs()[PROBLEM_INDEX]
        self.prompt_func_outputs = base.get_func_outputs()[PROBLEM_INDEX]
        self.prompt_inout_inf = base.get_inout_inf()[PROBLEM_INDEX]
        self.prompt_other_inf = base.get_other_inf()[PROBLEM_INDEX]

        self.prompt_example = (
            ""
        )

        self.prompt_example = (
            ""
        )

    def get_subproblems(self):
        return self.sub_problems

    def get_coupled_task(self):
        return self.coupled_task

    def get_task(self):
        return self.prompt_task

    def get_func_name(self):
        return self.prompt_func_name

    def get_func_inputs(self):
        return self.prompt_func_inputs

    def get_func_outputs(self):
        return self.prompt_func_outputs

    def get_inout_inf(self):
        return self.prompt_inout_inf

    def get_other_inf(self):
        return self.prompt_other_inf

    def get_examples(self):
        return self.prompt_example

if __name__ == "__main__":
    gp = GetPrompts()

    print("=== Coupled Task ===")
    print(gp.get_coupled_task())

    print("\n=== Task ===")
    print(gp.get_task())

    print("\n=== Function Name ===")
    print(gp.get_func_name())

    print("\n=== Function Inputs ===")
    print(gp.get_func_inputs())

    print("\n=== Function Outputs ===")
    print(gp.get_func_outputs())

    print("\n=== Input / Output Info ===")
    print(gp.get_inout_inf())

    print("\n=== Other Info ===")
    print(gp.get_other_inf())

    print("\n=== Examples ===")
    print(gp.get_examples())