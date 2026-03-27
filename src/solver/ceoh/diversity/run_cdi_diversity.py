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

from solver.ceoh.diversity.cdi_calculation import run_cdi

if __name__ == "__main__":
    # Base directories (adjust paths as needed)
    base_dir = r"C:\Users\smthboem\PycharmProjects\multibay-reshuffling\eoh_results"
    output_dir = r"C:\Users\smthboem\PycharmProjects\multibay-reshuffling\eoh_analysis"
    filter_model_folders = ["gpt4o20240806_ueTrue_uiTrue"]

    # Run the CDI computation workflow
    diversity_df_overall = run_cdi(base_dir, output_dir, filter_model_folders)
    if diversity_df_overall is not None:
        print(diversity_df_overall)
