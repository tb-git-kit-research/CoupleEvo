

class Probs():
    def __init__(self,paras):

        if not isinstance(paras.problem, str):
            self.prob = paras.problem
            print("- Prob local loaded ")
            print("- Prob " + paras.problem + " loaded ")
        elif paras.problem == "multibay_reshuffle_astar_loaded_move_time":
            from solver.ceoh.problems.descriptions.upmp.multibay_reshuffle_astar_loaded_move_time import run
            self.prob = run.MULTIBAY_RESHUFFLECONST_ASTAR(paras.eoh_experiment_file)
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


        else:
            print("problem "+paras.problem+" not found!")


    def get_problem(self):

        return self.prob
