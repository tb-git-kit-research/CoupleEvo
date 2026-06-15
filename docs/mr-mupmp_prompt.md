I am solving a coupled optimization problem consisting of 2 interacting subproblems.  
I will first describe the coupled problem and then each subproblem.  

Act as a professional algorithm designer. Design heuristics to be used inside a **Large Neighborhood Search (LNS)** framework for the Multi-Robot Multibay Unit-Load Premarshalling Problem.  
The system consists of two coupled LNS optimizers (move sequence search and vehicle routing) that alternate.  
The solution of the move sequence search is passed to the vehicle routing.  
The vehicle routing solution is passed to the move sequence search.  

The overall objective is to find a sequence of moves to sort a warehouse and to minimize the makespan operating these moves with multiple robots.  
The move sequence search part optimizes for the fewest moves and minimal loaded move distance.  
The vehicle routing part minimizes the makespan.  

==============================  

### Subproblem 0  

Function name: destroy_scoring  
Inputs: current_solution, data, rng  
Outputs: scores, k  

The input 'current_solution' is the current sequence of moves.  
The input 'data' is information about the moves in the sequence.  
The input 'rng' can be used to create randomness.  

The output 'scores' is the score for each element of the current solution.  
The output 'k' is the number of elements to remove.  

'current_solution' is a two-level nested list with integers in the second-level sublist.  
The first level list describes a move.  
A move is described by an pick-up access point and the drop-off access.  
The second level list describes the pick-up access point and the drop-off access point of each move.  

'data' is an object.  

'data.ap_distance' is a distance matrix between access points.  
'data.virtual_lanes' represents the initial state as a list of objects.  
'data.virtual_lanes'[index].stacks represents the lane of the initial state warehouse.  

'data.move_dependencies' is a dictionary of dictionaries.  

'data.move_dependencies['equal_prio'] is the dict of dependencies in which the dependent moves affect unit loads of equal priority class.  
'data.move_dependencies['equal_prio']['end-end'] is the list of move indexes that share the same drop-off access point and transport a unit load of the same priority class.  
'data.move_dependencies['equal_prio']['start-end'] is the list of move indexes where the first move's pick-up access point is the same as the drop-off access point of the second move.  
'data.move_dependencies['equal_prio']['end-start'] is the list of move indexes where the first move's drop-off access point is the same as the pick-up access point of the second move.  
'data.move_dependencies['equal_prio']['start-start'] is the list of move indexes where the first move's pick-up access point is the same as the pick-up access point of the second move.  

'data.move_dependencies['unequal_prio'] is the dict of dependencies in which the dependent moves affect unit loads of unequal priority class.  
It includes the same subdicts as 'equal_prio'.  

A dependency is encoded as a list of two indexes.  
The index refer to the position in the current_solution.  

'rng' is a NumPy random number generator (type: numpy.random.Generator).  

'scores' must be a list of lists.  
Each sublist should include a score for the pick-up access point and drop-off access point.  

Do not use 'while' loops.  
Do not give additional explanations.  
Don't create additional methods and please avoid nesting methods.  

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
Inputs: current_solution, data, rng  
Outputs: scores, k  

The input 'current_solution' is the current sequence of moves of each robot.  
The input 'data' is information about the moves in the sequence.  
The input 'rng' can be used to create randomness.  

The output 'scores' is the score for each move id of the current solution.  
The output 'k' is the number of moves to remove.  

'current_solution' is a two-level nested list with dictionaries in the second-level sublist.  
Each sublist represents the moves operated by an robot.  

Each dictionary represents one move:
current_solution[index]['id'] identifies the move.  
current_solution[index]['move'] is a list of an pick-up access point and an drop-off access point.  
current_solution[index]['start_time'] is the time the move begins.  
current_solution[index]['end_time'] is the time the move ends.  
current_solution[index]['robot'] identifies the robot performing the move.  

'data' is an object.
'data.ap_distance' is two-level nested list describing the distance matrix between access points.  
'data.virtual_lanes' represents the initial state as a list of objects.  
'data.virtual_lanes'[index].stacks represents the lane of the initial state warehouse.
'data.move_dependencies' is a dictionary of dictionaries.
'data.move_dependencies['equal_prio']['end-end'] is a list of dependency pairs.  
Each pair is a list of two move ids [id1, id2].
'data.move_dependencies['equal_prio']['start-end'] is a list of dependency pairs.  
Each pair is [id1, id2].
'data.move_dependencies['equal_prio']['end-start'] is a list of dependency pairs.  
Each pair is [id1, id2].
'data.move_dependencies['equal_prio']['start-start'] is a list of dependency pairs.  
Each pair is [id1, id2].
'data.move_dependencies['unequal_prio'] includes the same structure.
A dependency is encoded as a list of two ids.
'rng' is a NumPy random number generator (type: numpy.random.Generator).  

'scores' must be a list of tuples.  
Each tuple should include each move's id and the scoring.  

Do not use 'while' loops.  
Do not give additional explanations.  
Don't create additional methods and please avoid nesting methods.

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

Please help me create new heuristics that are totally different form from the given ones but can be motivated from them.  
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

