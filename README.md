# CoupleEvo: Evolving Heuristics for Coupled Optimization Problems Using Large Language Models

This repository contains complementary material for the paper "CoupleEvo: Evolving Heuristics for Coupled Optimization Problems Using Large Language Models".

The repository includes:

 - [Code of the best performing heuristic tuples (click here)](docs/best_heuristics.md)
 - [Prompt templates (click here)](docs/prompt_overview.md)
 - Full source code

<p align="center">
  <img src="docs/figures/coupled_problem_eoh_loop_integrated.drawio_breit.pdf.drawio.png" width="100%"/>
</p>
<p align="left">
  <em>Figure: CoupleEvo evolves multiple heuristics for coupled optimization problems.</em>
</p>

The main contributions of this work are summarized as follows:

- **CoupleEvo:** We formulate the first LLM-driven framework for automated heuristic design in coupled optimization problems, extending existing single-problem approaches to settings with multiple interdependent NP-hard subproblems.

- **Evolutionary Strategies:** 
We propose three evolutionary coordination strategies - *Sequential Evolution*,*Iterative Evolution*,and *Integrated Evolution*.
These strategies govern how heuristics for interacting subproblems are generated within the evolutionary process.

- **Comprehensive Evaluation:** We conduct extensive experimental evaluations on two representative coupled optimization domains:
Inventory Routing Problem (IRP)
Multi-robot Multi-bay Unit-Load Pre-Marshalling Problem (MR-MUPMP)  

For details, we refer to the paper.

### Run the project
- unzip data.zip to here
- create an .env with BASE_PATH and your API Keys eg. OPENAI_API_KEY, OPENROUTER_API_KEY, DEEPSEEK_API_KEY
- configure run_experiments.py
- run run_experiments.py