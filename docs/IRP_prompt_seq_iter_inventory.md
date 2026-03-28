I am solving a coupled optimization problem consisting of 2 interacting subproblems.  
I will fist describe the coupled problem and than each subproblem.  

Act as a professional algorithm designer. Design heuristics to be used inside a **Large Neighborhood Search (LNS)** frameworkfor the Inventory Routing Problem.  
The system consists of two coupled LNS optimizers (inventory planning and vehicle routing) that alternate.  
The solution of the inventory planning is passed to the vehicle routing.  
The vehicle routing solution is passed to the inventory planning.  

The overall objective is to jointly determine optimal inventory replenishment decisions and distribution routes over the planning horizon while minimizing sum of inventory and routing costs.

Your task is to design a heuristic that decides which shipments to remove from the current solution in the inventory planning part.
The current solution is represented by a 3D shipment tensor `shipments[t, v, i]` 
indicating the quantity delivered to customer `i` by vehicle `v` in period `t`. 
Your task is to design a heuristic that scores each relevant shipment element of the current solution. 
The highest scored element(s) will be removed from the current solution.
The highest-scoring shipment elements will be removed by setting their corresponding entries in `shipments[t, v, i]` to zero. 
After destruction, a repair procedure will reinsert shipments in order to eliminate customer shortages 
and restore feasibility while respecting vehicle capacity and depot inventory constraints. 
Therefore, the heuristic must carefully select which shipment elements to remove so that the repair process 
remains efficient and the resulting neighborhood is meaningful but not excessively large. 
Further the heuristic should decide how many elements should be removed from the current solution. 

I have 5 existing algorithms with their codes as follows: 
No. 1 algorithm and the corresponding code are: 

This algorithm scores customers based on the inverse of their demand variance multiplied by a route disruption factor that considers both the number of routes they appear in and their position within those routes, selecting customers that are both stable and centrally located to maximize solution exploration while maintaining structural integrity.
 

I have 5 existing algorithms with their codes as follows:

No. 1 algorithm and the corresponding code are:

{algorithm}
{code}

No. 2 algorithm and the corresponding code are:
{next parent heuristic}

Please help me create a new algorithm that has a totally different form from the given ones.

First, describe your new algorithm and main steps in one sentence. The description must be inside a brace.

Next, implement it in Python as a function named destroy_scoring. This function should accept 3 input(s): 'solution', 'data', 'rng'. The function should return 2 output(s): 'scores', 'k'. The input day_routes contains routes for the respective day. The input 'data' is information about the about the problem instance. The input 'rng' can be used to create randomness. The output 'scores' is the score to remove an element from the current solution. The output 'k' is the number of elements to remove.
'solution' is a dictionary with two main components:
'shipments': a 3D NumPy array of shape (T, V, N) where `shipments[t, v, i]` is the quantity delivered 
to customer `i` by vehicle `v` in period `t`.
'routes': a nested dictionary such that `routes[t][v] = [(customer_i, quantity_q), ...]` defines 
the visit sequence and delivered quantities for vehicle `v` in period `t`. 
The `routes` structure must be consistent with the nonzero entries in `shipments`.

'data' is a dictionary containing all static problem parameters of the Inventory Routing Problem instance:<br>
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
A value of 0 indicates that the shipment element is inactive and should not be destroyed.
score assigned by the destroy heuristic.
'k' must be an integer.

Do not use 'while' loops.
Do not give additional explanations.

Remember: Describe your new algorithm and main steps in one sentence inside a brace and implement the algorithm in Python as a function.