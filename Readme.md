# RandomGroupSpliter

## What is RandomGroupSpliter?

This class provide a simple way to split the multi dataset into train and test dataset.

## How to use it?

```python
from RandomGroupSpliter import RandomGroupSpliter
RGS = RandomGroupSplitter(noise = True, force_bound = False, tolerance = 0.1, split_rate = 0.75, label_matrix = label_matrix)
label_distribution, result_idx = RGS.get_one_combination() # get one combination
remain = RGS.get_remain_combination(result_idx) # get all remain combination based on the result_idx

top_k_distribution, top_k_group_combination = RGS.get_multi_combination(noOfCombination = 10, top_k = 1000) # get 10 combination and only consider top 1000 combination in each iteration 
tried = RGS.get_tried_combination() # get all tried combination
```

## caution

- **do not** set the `top_k` too large, it will be very time consuming.
- `top_k = 1000` will cause the program run about 30 seconds for 40 datasets.
- if the value of best result does not change for a long time, please stop the program since it may be some bug in the program.

## How to realize it?

by the idea of greedy algorithm, we can get the local optimal solution by the guide of the reward function.