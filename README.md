# Knapsack-problem

        There are three methods to solve the knapsack problem.  
        The example is a sample to help me understand dynamic programming and reinforcement learning .
        
**Question**:

        Maximise the total value under the maximal weight constraint
        item = pd.DataFrame(data=[[1, 1],
                                  [6, 2],
                                  [18, 5],
                                  [22, 6],
                                  [28, 7]],
                                  columns=['Value', 'Weight'])
        limit_W = 11

**Answer**:

        The opitmal solution is [2, 3]

        The weight of [2, 3] is 11
        The value of [2, 3] is the maximal, 40.


**Method**:

        1. Greedy algorithm
        2. Dynamic programming
        3. Reinforcement learning
  
knapsack.ipynb includes 1. Greedy algorithm  
                        2. Dynamic programming  
                         2.1 Based on subproblem  
                         2.2 Based on q table  

knapsack.py    includes 1. Reinforcement learning  
                        2. Dynamic programming based on q table  
                        3. Dynamic programming based on v table  
