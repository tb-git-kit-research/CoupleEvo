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

import random

from .utils import createFolders
from .methods import methods
from .problems import problems

# main class for AEL
class EVOL:

    # initilization
    def __init__(self, paras, prob=None, **kwargs):

        print("----------------------------------------- ")
        print("---              Start EoH            ---")
        print("-----------------------------------------")
        # Create folder #
        createFolders.create_folders(paras)
        print("- output folder created -")

        self.paras = paras

        print("-  parameters loaded -")

        self.prob = prob

        # Set a random seed
        random.seed(2024)

        
    # run methods
    def run(self):

        problemGenerator = problems.Probs(self.paras)

        problem = problemGenerator.get_problem() #problem evaluation function

        if self.paras.selection == None:
            self.paras.selection = "prob_rank"

        if self.paras.management == None:
            self.paras.management = "pop_greedy"

        methodGenerator = methods.Methods(self.paras, problem) # parent_selection, pop_manager

        method = methodGenerator.get_method() # eoh

        method.run()

        print("> End of Evolution! ")
        print("----------------------------------------- ")
        print("---     EoH successfully finished !   ---")
        print("-----------------------------------------")

        exit(1)
