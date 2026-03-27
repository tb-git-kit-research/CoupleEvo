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

import inspect

class GetPrompts():
    def __init__(self):
        import solver.ceoh.problems.descriptions.upmp.multibay_reshuffle_astar_loaded_move_time.run as run

        self.prompt_func_name = "score_state"

        self.prompt_func_context = (f"The heuristic {self.prompt_func_name} will be used in the following A* procedure: \n"
                                    f"{inspect.getsource(run.astar_multibay_premarshalling)} \n"
                                    f"{inspect.getsource(run.WarehouseNode.is_goal)} \n"
                                    f"{inspect.getsource(run.WarehouseNode.get_neighbors)} \n"
                                    f"{inspect.getsource(run.WarehouseNode.get_objective_value)} \n"
                                    f"{inspect.getsource(run.WarehouseNode.reconstruct_path)} \n"
                                    f"{inspect.getsource(run.WarehouseNode.get_loaded_move_distance)} \n"
                                    "Analyze how the heuristic will be used in the tree search and what the objective is. "
                                    )

        self.prompt_task = ("Act as a professional algorithm designer. "
                            "Design a heuristic to guide the tree search. "
                            "\n"
                            f"{self.prompt_func_context}"
                            )

        self.prompt_func_inputs = ["state", "ap_distance", "move_sequence"]
        self.prompt_func_outputs = ["score"]
        self.prompt_inout_inf = ("'state' is the configuration of lanes after a move. "
                                 "'ap_distance' is the distance between two access points. "
                                 "'move_sequence' is the sequence of moves performed so far. "
                                 "The output named 'score' is the score for the warehouse state. ")
        self.prompt_other_inf = ("Note that 'state' is a two levels nested list with integers in the second level sublist. "
                                 "Note that 'ap_distance' is a two levels nested list with integers in the second level sublist. "
                                 "Note that 'move_sequence' is a two levels nested list with integers in the second level sublist. "
                                 "'score' must be an integer or float. "
                                 "Avoid utilizing the random component, and it is crucial to maintain self-consistency. "
                                 "Do not use libraries. "
                                 "Do not use 'while' loops. "
                                 "Do not give additional explanations. "
                                 "Don't create additional methods and please avoid nesting methods. "
                                 )
        self.prompt_example = ("'state' is represented by a two levels deep nested list."
                               "The second level list represents a lane of unit loads as a list of integers. "
                               "The first list index (index 0) is the outermost slot in the lane. "
                               "The highest list index is the innermost slot in the lane. "
                               "Lanes are accessed from the first index to the highest index. "
                               "Each integer represents a unit load and its priority class. "
                               "Unit load of the same priority class are equal. "
                               "A 1 represents the highest priority class. "
                               "A 5 represents the lowest priority class. "
                               "A 3 represents a priority class lower than 1 but higher than 5. "
                               "A 4 represents a priority class lower than 3 but higher than 5. "
                               "A 0 represents an empty slot. "
                               "Each lane must have all 0s (empty slots) grouped at the start or have no 0s at all, "
                               "ensuring that if any non-zero elements appear in a lane, all subsequent slots must also be non-zero. "
                               "Therefore, impossible configurations are: "
                               "[1, 1, 0, 0] or [2,0,2], "
                               "while possible configurations are:"
                               " [0, 0, 1, 2] or [1, 2, 3, 3]. "
                               "\n "
                               "Examples for blocking unit loads: "
                               "In the lane [0, 4, 1] the 4 blocks access to 1. "
                               "In the lane [3, 3, 2] the two 3s block access to the 2. "
                               "In the lane [0, 5, 1, 5, 2] the two 5s block access to the 2 and 1. "
                               "In the lane [0, 4, 4, 3] the two 4s block access to the 3. "
                               "\n "
                               "First example for 'state': "
                               "[[0, 2, 3], [0, 5, 5], [5, 1, 1]] "
                               "Second example for 'state': "
                               "[[2, 2, 3, 5], [0, 3, 5, 4], [0, 0, 2, 2]]"
                               "\n "
                               "'ap_distance' is represented by a two levels deep nested list. "
                               "It describes the distance to travel between the access points in front of lanes. "
                               "The first level index in 'state' corresponds to the index of the lane's access point in 'ap_distance'. "
                               "Example for 'ap_distance': "
                               "First example for 'ap_distance': "
                               "[[0,1,2], [1,0,1], [2,1,0]] "
                               "Second example for 'ap_distance': "
                               "[[0, 1, 2, 3], [1, 0, 1, 2], [3, 2, 1, 0]] "
                               "An ap_distance[0, 2] value of 3 indicates a distance of 3 between lanes 0 and 2. "
                               "\n "
                               "'move_sequence' is represented by a two levels deep nested list. "
                               "It describes the sequence of moves performed to reach the current 'state' configuration. "
                               "Each sublist represents a move. "
                               "The first index of each sublist indicates the lane index from which the unit load was removed. "
                               "The second index of each sublist indicates the lane index in which the unit load was placed. "
                               "First example for 'move_sequence': "
                               "[[0,1], [0,1], [2,1]] "
                               "Second example for 'move_sequence': "
                               "[[2, 3], [4, 5], [5, 6]] "
                               "A sublist of [2, 3] indicates a move from the lanes with index 2 to the lane with index 3. "
                               "\n "
                               "First example for 'score': "
                               "6"
                               "Second example for 'score': "
                               "0.5"
                               )

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

    def get_astar_function(self):
        return self.prompt_func_context

if __name__ == "__main__":
    getprompts = GetPrompts()
    s = getprompts.get_task()
    s += getprompts.get_func_name()
    print(s)

