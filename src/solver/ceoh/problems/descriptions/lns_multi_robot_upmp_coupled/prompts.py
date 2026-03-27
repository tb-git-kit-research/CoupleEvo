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


class GetPrompts:

    def __init__(self):
        from solver.ceoh.problems.descriptions.lns_multi_robot_upmp_coupled.run import SUBPROBLEMS
        self.sub_problems = SUBPROBLEMS

        self.coupled_task = (
            # General setup
            "Act as a professional algorithm designer. "
            "Design heuristics to be used inside a **Large Neighborhood Search (LNS)** framework"
            "for the Multi Robot Multibay Unit-Load Premarshalling Problem. "
            f"The system consists of two coupled LNS optimizers "
            f"(move sequence search and vehicle routing) that alternate. "
            "The solution of the move sequence search is passed to the vehicle routing. "
            "The vehicle routing solution is passed to the move sequence search. "
            "\n"
            # objectives
            "The overall objective is to find a sequence of moves to sort a warehouse and to minimize the makespan "
            "operating these moves with multiple robots. "
            "The move sequence search part optimizes for the fewest moves and minimal loaded move distance. "
            "The vehicle routing part minimizes for the makespan. "
            "\n"
        )

        self.prompt_task = [(
            "Your task is to design a heuristic that decides which parts to remove from the current solution in the sequence search part. "
            "The current solution is represented by a list of moves. "
            "Each move consists of an pick-up access point and an drop-off access point. "
            "Your task is to design a heuristic that scores each element (access-point) of the current solution in the sequence search part. "
            "The highest scored element(s) will be removed from the current solution. "
            "The removed elements will be replaced with None. "
            "Hence the complete move can be replaced with [None, None] or only parts of it: [None, drop-off access point], [pick-up access point ,None]"
            "The repair step will be executed in an A* algorithm. The A* will branch potential moves for the None entries. "
            "Hence the removed elements must be carefully selected to keep the runtime small."
            "Further the heuristic should decide how many elements should be removed from the current solution. "
            "\n"
        ),
            (
            "Your task is to design a heuristic that decides which parts to remove from the current solution in the vehicle routing part. "
            "The current solution is represented by a list of lists. "
            "Each sublist represents the moves operated by an robot. "
            "The sublist has dictionaries in it. "
            "Each dictionary represents one move. "
            "Each dictionary has the keys 'id', 'move', 'start_time', 'end_time', and 'robot'. "
            "'id' (integer) that identifies the move. "
            "'move' (list) consists of an pick-up access point and an drop-off access point. " 
            "'start_time' (integer) identifies the time the move begins. "
            "'end_time' (integer) identifies the time the move ends. "
            "'robot' (integer) identifies the robot performing the move. "
            "Your task is to design a heuristic that scores each move id of the current solution in the vehicle routing part. "
            "The highest scored move(s) ids will be removed from the current solution. "
            "The moves with id = 0 should be skipped. "
            "This describe the starting position of each robot and cannot be removed. "
            "The solution will be repaired in the order starting with highest ranked to the lowest scored move. "
            "Further the heuristic should decide how many moves should be removed from the current solution. "
            "\n"
        )
        ]

        self.prompt_func_name = ["destroy_scoring",
                                 "destroy_scoring",]
        self.prompt_func_inputs = [["current_solution", "data", "rng"],
                                   ["current_solution", "data", "rng"]]
        self.prompt_func_outputs = [["scores", "k"],
                                    ["scores", "k"]]

        self.prompt_inout_inf = [(
            "The input 'current_solution' is the current sequence of moves. "
            "The input 'data' is information about the moves in the sequence. "
            "The input 'rng' can be used to create randomness. "
            "The output 'scores' is the score for each element of the current solution. "
            "The output 'k' is the number of elements to remove. "
        ),
        (
            "The input 'current_solution' is the current sequence of moves of each robot. "
            "The input 'data' is information about the moves in the sequence. "
            "The input 'rng' can be used to create randomness. "
            "The output 'scores' is the score for each move id of the current solution. "
            "The output 'k' is the number of moves to remove. "
        )
        ]

        self.prompt_other_inf = [
        (
            "'current_solution' is a two-level nested list with integers in the second-level sublist. "
            "The first level list describes a move. "
            "A move is described by an pick-up access point and the drop-off access. "
            "The second level list describes the pick-up access point and the drop-off access point of each move."
            "'data' is an object. "
            "'data.ap_distance' is a distance matrix between access points. "
            "'data.virtual_lanes' represents the initial state as a list of objects. "
            "'data.virtual_lanes'[index].stacks represents the lane of the initial state warehouse. "
            "'data.move_dependencies' is a dictionary of dictionaries. "
            "'data.move_dependencies['equal_prio'] is the dict of dependencies in which the dependent moves affect unit loads of equal priority class. "
            "'data.move_dependencies['equal_prio']['end-end']' is the list of move indexes that share the same drop-off access point and transport a unit load of the same priority class. "
            "'data.move_dependencies['equal_prio']['start-end']' is the list of move indexes where the first move's pick-up access point is the same as the drop-off access point of the second move. "
            "'data.move_dependencies['equal_prio']['end-start']' is the list of move indexes where the first move's drop-off access point is the same as the pick-up access point of the second move. "
            "'data.move_dependencies['equal_prio']['start-start']' is the list of move indexes where the first move's pick-up access point is the same as the pick-up access point of the second move. "
            "'data.move_dependencies['unequal_prio'] is the dict of dependencies in which the dependent moves affect unit loads of unequal priority class. It includes the same subdicts as 'equal_prio'. "
            "A dependency is encoded as a list of two indexes. The index refer to the position in the current_solution. "
            "'rng' is a NumPy random number generator (type: numpy.random.Generator). "
            "'scores' must be a list of lists. Each sublist should include a score for the pick-up access point and drop-off access point. \n\n"
            "Do not use 'while' loops. "
            "Do not give additional explanations. "
            "Don't create additional methods and please avoid nesting methods. "
        ),
        (
            "'current_solution' is a two-level nested list with dictionaries in the second-level sublist. "
            "Each sublist represents the moves operated by an robot. "
            "The sublist has dictionaries in it. "
            "Each dictionary represents one move. "
            "current_solution[index]['id'] identifies the move. "
            "current_solution[index]['move'] is a list of an pick-up access point and an drop-off access point. " 
            "current_solution[index]['start_time'] is the time the move begins. "
            "current_solution[index]['end_time'] is the time the move ends. "
            "current_solution[index]['robot'] identifies the robot performing the move. "
            
            "'data' is an object. "
            "'data.ap_distance' is two-level nested list describing the distance matrix between access points. "
            "'data.virtual_lanes' represents the initial state as a list of objects. "
            "'data.virtual_lanes'[index].stacks represents the lane of the initial state warehouse. "
            "'data.move_dependencies' is a dictionary of dictionaries. "
            "'data.move_dependencies['equal_prio'] is the dict of dependencies in which the dependent moves affect unit loads of equal priority class. "
            "'data.move_dependencies['equal_prio']['end-end']' is a list of dependency pairs. "
            "Each pair is a list of two move ids [id1, id2] that share the same drop-off access point. "
            "'data.move_dependencies['equal_prio']['start-end']' is a list of dependency pairs. "
            "Each pair is [id1, id2], where the pick-up point of id1 is the same as the drop-off point of id2. "
            "'data.move_dependencies['equal_prio']['end-start']' is a list of dependency pairs. "
            "Each pair is [id1, id2], where the drop-off point of id1 is the same as the pick-up point of id2. "
            "'data.move_dependencies['equal_prio']['start-start']' is a list of dependency pairs. "
            "Each pair is [id1, id2], where both moves share the same pick-up access point. "
            "'data.move_dependencies['unequal_prio'] is the dict of dependencies in which the dependent moves affect unit loads of unequal priority class. It includes the same subdicts as 'equal_prio'. "
            "A dependency is encoded as a list of two ids. The index refer to the position in the current_solution. "
            "'rng' is a NumPy random number generator (type: numpy.random.Generator). "
            "'scores' must be a list of tuples. Each tuple should include each move's id and the scoring. \n\n"
            "Do not use 'while' loops. "
            "Do not give additional explanations. "
            "Don't create additional methods and please avoid nesting methods. "
        )
        ]

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
