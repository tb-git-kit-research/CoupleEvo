# Copyright (c) 2024 - Fei Liu (fliu36-c@my.cityu.edu.hk)
#
# Modified by: 
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

class Probs():
    def __init__(self,paras):

        if not isinstance(paras.problem, str):
            self.prob = paras.problem
            print("- Prob local loaded ")
        elif paras.problem == "tsp_construct":
            from solver.ceoh.problems.descriptions.base_problems.tsp_greedy import run
            self.prob = run.TSPCONST()
            print("- Prob "+paras.problem+" loaded ")
        elif paras.problem == "bp_online":
            from solver.ceoh.problems.descriptions.base_problems.bp_online import run
            self.prob = run.BPONLINE_CONST()
            print("- Prob "+paras.problem+" loaded ")
        elif paras.problem == "cap_set":

            if paras.ec_use_example:
                raise Exception("CAP SET CANNOT BE EXCECUTED WITH EXAMPLE!")

            from solver.ceoh.problems.descriptions.base_problems.cap_set import run
            self.prob = run.CAP_SET_CONST()
            print("- Prob " + paras.problem + " loaded ")
        elif paras.problem == "multibay_reshuffle":
            from solver.ceoh.problems.descriptions.upmp.multibay_reshuffeling import run
            self.prob = run.MULTIBAY_RESHUFFLECONST(paras.eoh_experiment_file)
            print("- Prob " + paras.problem + " loaded ")
        elif paras.problem == "multibay_reshuffle_astar":
            from solver.ceoh.problems.descriptions.upmp.multibay_reshuffle_astar import run
            self.prob = run.MULTIBAY_RESHUFFLECONST_ASTAR(paras.eoh_experiment_file)
            print("- Prob " + paras.problem + " loaded ")
        elif paras.problem == "multibay_reshuffle_astar_version":
            from solver.ceoh.problems.descriptions.upmp.multibay_reshuffle_astar import run
            self.prob = run.MULTIBAY_RESHUFFLECONST_ASTAR(paras.eoh_experiment_file)
            print("- Prob " + paras.problem + " loaded ")
        elif paras.problem == "multibay_reshuffle_uninformed_astar":
            from solver.ceoh.problems.descriptions.upmp.multibay_reshuffle_uninformed_astar import run
            self.prob = run.MULTIBAY_RESHUFFLECONST_ASTAR(paras.eoh_experiment_file)
            print("- Prob " + paras.problem + " loaded ")
        elif paras.problem == "multibay_reshuffle_uninformed_astar_version":
            from solver.ceoh.problems.descriptions.upmp.multibay_reshuffle_uninformed_astar_version import run
            self.prob = run.MULTIBAY_RESHUFFLECONST_ASTAR(paras.eoh_experiment_file)
            print("- Prob " + paras.problem + " loaded ")
        elif paras.problem == "multibay_reshuffle_astar_loaded_move_time":
            from solver.ceoh.problems.descriptions.upmp.multibay_reshuffle_astar_loaded_move_time import run
            self.prob = run.MULTIBAY_RESHUFFLECONST_ASTAR(paras.eoh_experiment_file)
            print("- Prob " + paras.problem + " loaded ")

        elif paras.problem == "cvrp":
            from solver.ceoh.problems.descriptions.vrp.cvrp import run
            self.prob = run.CVRP_CONSTANT(code_string=None, paras=paras)
            print("- Prob " + paras.problem + " loaded ")
        elif paras.problem == "vrptw":
            from .descriptions.vrp.vrptw import run
            self.prob = run.VRPTW_CONSTANT(code_string=None, paras=paras)
            print("- Prob " + paras.problem + " loaded ")
        elif paras.problem == "ovrpmbltw":
            from .descriptions.vrp.ovrpmbltw import run
            self.prob = run.OVRPMBLTW_CONSTANT(code_string=None, paras=paras)
            print("- Prob " + paras.problem + " loaded ")

        elif paras.problem == "puzzle_greedy":
            from solver.ceoh.problems.descriptions.puzzle.puzzle_greedy import run
            self.prob = run.PUZZLE_GREEDY_CONST(code_string=None, paras=paras)
            print("- Prob " + paras.problem + " loaded ")
        elif paras.problem == "puzzle_astar_korf":
            from solver.ceoh.problems.descriptions.puzzle.puzzle_astar_korf import run
            self.prob = run.PUZZLE_ASTAR(code_string=None, paras=paras)
            print("- Prob " + paras.problem + " loaded ")
        elif paras.problem == "puzzle_astar_edu":
            from solver.ceoh.problems.descriptions.puzzle.puzzle_astar_edu import run
            self.prob = run.PUZZLE_ASTAR(code_string=None, paras=paras)
            print("- Prob " + paras.problem + " loaded ")
        elif paras.problem == "puzzle_astar_edu_uninformed_astar":
            from solver.ceoh.problems.descriptions.puzzle.puzzle_astar_edu_uninformed_astar import run
            self.prob = run.PUZZLE_ASTAR(code_string=None, paras=paras)
            print("- Prob " + paras.problem + " loaded ")

        elif paras.problem == "lns_cvrp":
            from .descriptions.lns_cvrp import run
            self.prob = run.LNS_CVRP_CONSTANT(code_string=None, paras=paras)
            print("- Prob " + paras.problem + " loaded ")
        elif paras.problem == "lns_cvrp_pool":
            from .descriptions.lns_cvrp_pool import run
            self.prob = run.LNS_CVRP_CONSTANT(code_string=None, paras=paras)
            print("- Prob " + paras.problem + " loaded ")

        elif paras.problem == "lns_vrptw":
            from .descriptions.lns_vrptw import run
            self.prob = run.LNS_CVRP_CONSTANT(code_string=None, paras=paras)
            print("- Prob " + paras.problem + " loaded ")
        elif paras.problem == "lns_vrpl":
            from .descriptions.lns_vrpl import run
            self.prob = run.LNS_CVRP_CONSTANT(code_string=None, paras=paras)
            print("- Prob " + paras.problem + " loaded ")
        elif paras.problem == "lns_vrpltw":
            from .descriptions.lns_vrpltw import run
            self.prob = run.LNS_CVRP_CONSTANT(code_string=None, paras=paras)
            print("- Prob " + paras.problem + " loaded ")
        elif paras.problem == "lns_vrpbltw":
            from .descriptions.lns_vrpbltw import run
            self.prob = run.LNS_CVRP_CONSTANT(code_string=None, paras=paras)
            print("- Prob " + paras.problem + " loaded ")

        elif paras.problem == "lns_multi_robot_upmp_move_search":
            from .descriptions.lns_multi_robot_upmp_move_search import run
            self.prob = run.LNS_MOVE_ROUTE_CONSTANT(code_string=None, paras=paras)
            print("- Prob " + paras.problem + " loaded ")
        elif paras.problem == "lns_multi_robot_upmp_vrp":
            from .descriptions.lns_multi_robot_upmp_vrp import run
            self.prob = run.LNS_MOVE_ROUTE_CONSTANT(code_string=None, paras=paras)
            print("- Prob " + paras.problem + " loaded ")

        elif paras.problem == "lns_multi_robot_upmp_coupled":
            from .descriptions.lns_multi_robot_upmp_coupled import run
            self.prob = run.LNS_MOVE_ROUTE_CONSTANT(code_string=None, paras=paras)
            print("- Prob " + paras.problem + " loaded ")

        elif paras.problem == "lns_IRP_coupled":
            from .descriptions.lns_IRP_coupled import run
            self.prob = run.LNS_IRP_CONSTANT(code_string=None, paras=paras)
            print("- Prob " + paras.problem + " loaded ")
        elif paras.problem == "lns_IRP_inventory":
            from .descriptions.lns_IRP_inventory import run
            self.prob = run.LNS_IRP_CONSTANT(code_string=None, paras=paras)
            print("- Prob " + paras.problem + " loaded ")
        elif paras.problem == "lns_IRP_vrp":
            from .descriptions.lns_IRP_vrp import run
            self.prob = run.LNS_IRP_CONSTANT(code_string=None, paras=paras)
            print("- Prob " + paras.problem + " loaded ")

        elif paras.problem == "lns_LRP_coupled":
            from .descriptions.lns_LRP_coupled import run
            self.prob = run.LNS_LRP_CONSTANT(code_string=None, paras=paras)
            print("- Prob " + paras.problem + " loaded ")
        elif paras.problem == "lns_LRP_location":
            from .descriptions.lns_LRP_location import run
            self.prob = run.LNS_LRP_CONSTANT(code_string=None, paras=paras)
            print("- Prob " + paras.problem + " loaded ")
        elif paras.problem == "lns_LRP_vrp":
            from .descriptions.lns_LRP_vrp import run
            self.prob = run.LNS_LRP_CONSTANT(code_string=None, paras=paras)
            print("- Prob " + paras.problem + " loaded ")

        elif paras.problem == "inventory_greedy":
            from .descriptions.inventory_greedy import run
            self.prob = run.INVENTORY_CONST(code_string=None, paras=paras)
            print("- Prob " + paras.problem + " loaded ")

        else:
            print("problem "+paras.problem+" not found!")


    def get_problem(self):

        return self.prob
