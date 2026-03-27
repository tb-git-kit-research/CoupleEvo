I am solving a coupled optimization problem consisting of 2 interacting subproblems.  
I will fist describe the coupled problem and than each subproblem.  

Act as a professional algorithm designer. Design heuristics to be used inside a **Large Neighborhood Search (LNS)** frameworkfor the Inventory Routing Problem.  
The system consists of two coupled LNS optimizers (inventory planning and vehicle routing) that alternate.  
The solution of the inventory planning is passed to the vehicle routing.  
The vehicle routing solution is passed to the inventory planning.  

The overall objective is to jointly determine optimal inventory replenishment decisions and distribution routes over the planning horizon while minimizing sum of inventory and routing costs.  

==============================  

### Subproblem 0

Function name: destroy_scoring  
Inputs: solution, data, rng  
Outputs: scores, k  

The input `solution` contains the current solution, including the shipments and routes.  
The input 'data' is information about the about the problem instance.  
The input 'rng' can be used to create randomness.  
The output 'scores' is the score to remove an element from the current solution.  
The output 'k' is the number of elements to remove.  

'solution' is a dictionary with two main components:  

'shipments': a 3D NumPy array of shape (T, V, N) where `shipments[t, v, i]` is the quantity delivered to customer `i` by vehicle `v` in period `t`.  

'routes': a nested dictionary such that `routes[t][v] = [(customer_i, quantity_q), ...]` defines the visit sequence and delivered quantities for vehicle `v` in period `t`.  
The `routes` structure must be consistent with the nonzero entries in `shipments`.  

'data' is a dictionary containing all static problem parameters of the Inventory Routing Problem instance:  

 'T': number of planning periods.<br>
 'V': number of vehicles.<br>
 'N': number of customers.<br>
 'veh_cap': capacity of each vehicle.<br>
 'cost_matrix': 2D array of travel costs between the depot (index 0) and customers (indices 1..N).<br>
 'depot_I0': initial inventory level at the depot.<br>
 'depot_prod': array of length T giving depot production in each period.<br>
 'depot_inv_cost': inventory holding cost per unit at the depot.<br>
 'cust_I0': array of length N with initial inventory for each customer.<br>
 'cust_inv_cost': array of length N with inventory holding cost for each customer.<br>
 'cust_Lmax': array of length N with maximum inventory capacity for each customer.<br>
 'cust_demand': 2D array of shape (N, T) with shipment demand of each customer in each period.<br>

'rng' is a NumPy random number generator (type: numpy.random.Generator).  
scores' must be a 3-level nested list scores[t][v][i], where each entry is a real-valued priority assigned to the shipment element shipments[t, v, i]  
A value of 0 indicates that the shipment element is inactive and should not be destroyed.score assigned by the destroy heuristic.  

'k' must be an integer.  
Do not use 'while' loops.  
Do not give additional explanations.  

Existing heuristics for this subproblem:

Parent 1:<br>
Subproblem Index: 0<br>
Subproblem Name: lns_multi_robot_upmp_move_search<br>
Algorithm:<br>
[description omitted]<br>
Code:<br>
[code omitted]<br>

{more parents}

==============================  

### Subproblem 1   

Function name: destroy_scoring  
Inputs: day_routes, data, rng  
Outputs: scores, k  

The input `day_routes` contains routes for the respective day.  
The input 'data' is information about the about the problem instance.  
The input 'rng' can be used to create randomness.  
The output 'scores' is the score to remove an element from the current solution.  
The output 'k' is the number of elements to remove.  

'day_routes' is a nested dictionary of lists of tuples representing routing decisions.  
day_routes[v] = [(customer_i, quantity_q), ...] defines the visit sequence and delivered quantities for vehicle `v`.  

'data' is a dictionary containing all static problem parameters of the Inventory Routing Problem instance:  

 'T': number of planning periods.<br>
 'V': number of vehicles.<br>
 'N': number of customers.<br>
 'veh_cap': capacity of each vehicle.<br>
 'cost_matrix': 2D array of travel costs between the depot (index 0) and customers (indices 1..N).<br>
 'depot_I0': initial inventory level at the depot.<br>
 'depot_prod': array of length T giving depot production in each period.<br>
 'depot_inv_cost': inventory holding cost per unit at the depot.<br>
 'cust_I0': array of length N with initial inventory for each customer.<br>
 'cust_inv_cost': array of length N with inventory holding cost for each customer.<br>
 'cust_Lmax': array of length N with maximum inventory capacity for each customer.<br>
 'cust_demand': 2D array of shape (N, T) with shipment demand of each customer in each period.<br>

'rng' is a NumPy random number generator (type: numpy.random.Generator).  
'scores' must be a list of tuples. of the form (i, s), where i is a customer index (0 ≤ i < N), s is a real-valued priority score assigned by the destroy heuristic.  

'k' must be an integer.  
Do not use 'while' loops.  
Do not give additional explanations.  

Existing heuristics for this subproblem:

Parent 1:<br>
Subproblem Index: 1<br>
Subproblem Name: lns_multi_robot_upmp_vrp<br>
Algorithm:<br>
[description omitted]<br>
Code:<br>
[code omitted]<br>

{more parents}

==============================  

Please help me create new heuristics that are totally different form from the given ones.  
Please design ONE heuristic for EACH subproblem above.  

For each subproblem:  
- Briefly explain your core ideas behind the code.  
- Provide a complete Python function implementation.  

Return your answer using EXACTLY the following structure for EACH subproblem:  

SUBPROBLEM \<index\>  
THOUGHTS: \<short explanation\>  
CODE: \```python < complete python function >\```


Rules:
- Provide exactly 2 subproblems.
- <index> must range from 0 to 1.
- Each index must appear exactly once.
- The code MUST be inside a Python markdown code block.
- Use the exact headers: SUBPROBLEM, THOUGHTS, CODE.
- Separate entries using '---'.
- Do NOT include any additional text.