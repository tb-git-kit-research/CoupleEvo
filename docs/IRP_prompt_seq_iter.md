Your task is to design a heuristic that decides which parts of the current solution to remove in the vehicle routing part. The current solution consists of explicit vehicle routes stored as `day_routes[v] = [(customer_i, quantity_q), ...]` indicating the quantity delivered to customer `i` by vehicle `v`. The framework iterates each day and passes day_routes for the respective day to the heuristic. Your task is to design a heuristic that scores each customer of the current solution in the vehicle routing part. The highest scored customer(s) will be removed from the current solution. Each customer i must receive exactly one score. You must not assign scores per (v, i) for this task.After destruction, a repair procedure based on cost-guided backtracking will attempt to reinsert each unassigned delivery `(i, q)` into the routes while respecting vehicle capacity constraints and minimizing additional travel cost. Therefore, the heuristic must carefully select which customers to remove and how many of them to destroy in order to generate a meaningful but computationally manageable neighborhood. The solution will be repaired in the order starting with highest ranked to the lowest scored customer. Further the heuristic should decide how many moves elements be removed from the current solution. 

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

Next, implement it in Python as a function named destroy_scoring. This function should accept 3 input(s): 'day_routes', 'data', 'rng'. The function should return 2 output(s): 'scores', 'k'. The input day_routes contains routes for the respective day. The input 'data' is information about the about the problem instance. The input 'rng' can be used to create randomness. The output 'scores' is the score to remove an element from the current solution. The output 'k' is the number of elements to remove. 'day_routes' is a nested dictionary of lists of tuples representing routing decisions.day_routes[v] = [(customer_i, quantity_q), ...] defines the visit sequence and delivered quantities for vehicle v.

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

'scores' must be a list of tuples of the form (i, s), where i is a customer index (0 ≤ i < N), s is a real-valued priority score assigned by the destroy heuristic.
'k' must be an integer.

Do not use 'while' loops.
Do not give additional explanations.

Remember: Describe your new algorithm and main steps in one sentence inside a brace and implement the algorithm in Python as a function.