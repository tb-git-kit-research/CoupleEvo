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
        from solver.ceoh.problems.descriptions.lns_IRP_coupled.run import SUBPROBLEMS
        self.sub_problems = SUBPROBLEMS

        self.coupled_task = (
            # General setup
            "Act as a professional algorithm designer. "
            "Design heuristics to be used inside a **Large Neighborhood Search (LNS)** framework"
            "for the Inventory Routing Problem. "
            f"The system consists of two coupled LNS optimizers "
            f"(inventory planning and vehicle routing) that alternate. "
            "The solution of the inventory planning is passed to the vehicle routing. "
            "The vehicle routing solution is passed to the inventory planning. "
            "\n"
            # objectives
            "The overall objective is to jointly determine optimal inventory replenishment decisions "
            "and distribution routes over the planning horizon while minimizing sum of inventory and routing costs. "
            "\n"
        )

        self.prompt_task = [(
            "Your task is to design a heuristic that decides which shipments to remove from the current solution in the inventory planning part. "
            "The current solution is represented by a 3D shipment tensor `shipments[t, v, i]` "
            "indicating the quantity delivered to customer `i` by vehicle `v` in period `t`. "
            "Your task is to design a heuristic that scores each relevant shipment element of the current solution. "
            "The highest scored element(s) will be removed from the current solution. "
            "The highest-scoring shipment elements will be removed by setting their corresponding entries in "
            "`shipments[t, v, i]` to zero. "
            "After destruction, a repair procedure will reinsert shipments in order to eliminate customer shortages "
            "and restore feasibility while respecting vehicle capacity and depot inventory constraints. "
            "Therefore, the heuristic must carefully select which shipment elements to remove so that the repair process "
            "remains efficient and the resulting neighborhood is meaningful but not excessively large. "
            "Further the heuristic should decide how many elements should be removed from the current solution. "
            "\n"
        ),
            (
            "Your task is to design a heuristic that decides which parts of the current solution to remove in the vehicle routing part. "
            "The current solution consists of explicit vehicle routes stored as `day_routes[v] = [(customer_i, quantity_q), ...]` "
            "indicating the quantity delivered to customer `i` by vehicle `v`. "
            "The framework iterates each day and passes day_routes for the respective day to the heuristic. "
            "Your task is to design a heuristic that scores each customer of the current solution in the vehicle routing part. "
            "The highest scored customer(s) will be removed from the current solution. "
            "Each customer i must receive exactly one score. You must not assign scores per (v, i) for this task."
            "After destruction, a repair procedure based on cost-guided backtracking will attempt to reinsert "
            "each unassigned delivery `(i, q)` into the routes while respecting vehicle capacity constraints "
            "and minimizing additional travel cost. "
            "Therefore, the heuristic must carefully select which customers to remove and how many of them "
            "to destroy in order to generate a meaningful but computationally manageable neighborhood. "           
            "The solution will be repaired in the order starting with highest ranked to the lowest scored customer. "
            "Further the heuristic should decide how many moves elements be removed from the current solution. "
            "\n"
        )
        ]

        self.prompt_func_name = ["destroy_scoring",
                                 "destroy_scoring",]
        self.prompt_func_inputs = [["solution", "data", "rng"],
                                   ["day_routes", "data", "rng"]]
        self.prompt_func_outputs = [["scores", "k"],
                                    ["scores", "k"]]

        self.prompt_inout_inf = [(
            "The input `solution` contains the current solution, including the shipments and routes. "
            "The input 'data' is information about the about the problem instance. "
            "The input 'rng' can be used to create randomness. "
            "The output 'scores' is the score to remove an element from the current solution. "
            "The output 'k' is the number of elements to remove. "
        ),
        (
            "The input `day_routes` contains routes for the respective day. "
            "The input 'data' is information about the about the problem instance. "
            "The input 'rng' can be used to create randomness. "
            "The output 'scores' is the score to remove an element from the current solution. "
            "The output 'k' is the number of elements to remove. "
        )
        ]

        self.prompt_other_inf = [
        (
            "'solution' is a dictionary with two main components:\n"
            "'shipments': a 3D NumPy array of shape (T, V, N) where `shipments[t, v, i]` is the quantity delivered "
            "to customer `i` by vehicle `v` in period `t`.\n"
            "'routes': a nested dictionary such that `routes[t][v] = [(customer_i, quantity_q), ...]` defines "
            "the visit sequence and delivered quantities for vehicle `v` in period `t`. "
            "The `routes` structure must be consistent with the nonzero entries in `shipments`.\n\n"

            "'data' is a dictionary containing all static problem parameters of the Inventory Routing Problem instance:\n"
            " 'T': number of planning periods.\n"
            " 'V': number of vehicles.\n"
            " 'N': number of customers.\n"
            " 'veh_cap': capacity of each vehicle.\n"
            " 'cost_matrix': 2D array of travel costs between the depot (index 0) and customers (indices 1..N).\n"
            " 'depot_I0': initial inventory level at the depot.\n"
            " 'depot_prod': array of length T giving depot production in each period.\n"
            " 'depot_inv_cost': inventory holding cost per unit at the depot.\n"
            " 'cust_I0': array of length N with initial inventory for each customer.\n"
            " 'cust_inv_cost': array of length N with inventory holding cost for each customer.\n"
            " 'cust_Lmax': array of length N with maximum inventory capacity for each customer.\n"
            " 'cust_demand': 2D array of shape (N, T) with shipment demand of each customer in each period.\n"

            "'rng' is a NumPy random number generator (type: numpy.random.Generator). "
            "scores' must be a 3-level nested list scores[t][v][i], "
            "where each entry is a real-valued priority assigned to the shipment element shipments[t, v, i] "
            "A value of 0 indicates that the shipment element is inactive and should not be destroyed."
            "score assigned by the destroy heuristic. "
            "'k' must be an integer. "
            "Do not use 'while' loops. "
            "Do not give additional explanations. "
            #"Don't create additional methods and please avoid nesting methods. "
        ),
        (
            "'day_routes' is a nested dictionary of lists of tuples representing routing decisions."
            "day_routes[v] = [(customer_i, quantity_q), ...] defines the visit sequence and delivered quantities for vehicle `v`. \n"
            
            "'data' is a dictionary containing all static problem parameters of the Inventory Routing Problem instance:\n"
            " 'T': number of planning periods.\n"
            " 'V': number of vehicles.\n"
            " 'N': number of customers.\n"
            " 'veh_cap': capacity of each vehicle.\n"
            " 'cost_matrix': 2D array of travel costs between the depot (index 0) and customers (indices 1..N).\n"
            " 'depot_I0': initial inventory level at the depot.\n"
            " 'depot_prod': array of length T giving depot production in each period.\n"
            " 'depot_inv_cost': inventory holding cost per unit at the depot.\n"
            " 'cust_I0': array of length N with initial inventory for each customer.\n"
            " 'cust_inv_cost': array of length N with inventory holding cost for each customer.\n"
            " 'cust_Lmax': array of length N with maximum inventory capacity for each customer.\n"
            " 'cust_demand': 2D array of shape (N, T) with shipment demand of each customer in each period.\n"

            "'rng' is a NumPy random number generator (type: numpy.random.Generator). "
            "'scores' must be a list of tuples. "
            "of the form (i, s), where i is a customer index (0 ≤ i < N), s is a real-valued priority "
            "score assigned by the destroy heuristic. "
            "'k' must be an integer. "
            "Do not use 'while' loops. "
            "Do not give additional explanations. "
            #"Don't create additional methods and please avoid nesting methods. "
        ),
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
