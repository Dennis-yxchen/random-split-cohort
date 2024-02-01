import numpy as np
import pandas as pd
from heapq import heappush, heappop, heapify
from itertools import combinations as comb

def rewardFunction(current_situation, split_rate, total_sum_label ,noise = False):
    """
    this function is used to calculate the reward of the current situation
    if the reward is positive, it means the current situation is lower than the target
    give the number lower than the target a higher weight (5 times)
    """
    diff_num = total_sum_label * split_rate - current_situation
    if noise == True:
        diff_num = makeNoise(diff_num, scale = np.abs(np.mean(diff_num)/10))
    # if the reward is negative, increase its weight
    diff_num[diff_num > 0] *= 5
    return np.sum(diff_num)

def makeNoise(score, scale):
    """
    add noise to the score
    """
    gumbel_noise = np.random.gumbel(0, scale, size=score.shape) + score
    return gumbel_noise

def rewardFunction2(current_situation, split_rate, total_sum_label ,noise = False):
    """
    this function is used to calculate the reward of the current situation
    if the reward is positive, it means the current situation is lower than the target
    closer the number to the target, lower the reward / punishment
    exponential function
    """
    noise = False
    diff_num = total_sum_label * split_rate - current_situation
    if noise == True:
        diff_num = makeNoise(diff_num, scale = np.abs(np.mean(diff_num)/10))
    # if the diff is negative, increase its weight
    
    # normalize the diff
    
    diff_num = np.divide(diff_num, total_sum_label)
    
    diff_num[diff_num > 0] = np.exp(diff_num[diff_num > 0])
    diff_num[diff_num < 0] = -np.exp(diff_num[diff_num < 0])
    return np.sum(diff_num)

class RandomGroupSplitter:
    def __init__(self, noise, force_bound, tolerance, split_rate, label_matrix):
        """
        noise: whether to add noise to the reward function
        force_bound: whether to force the algorithm to split the group (to be-implement)
        tolerance: the tolerance of the algorithm (to be-implement)
        split_rate: the split rate of the labels
        label_matrix: the label matrix of the dataset, it should be a numpy array that each row is a label distribution in a dataset
        e.g. [[20,10,0,0], [0,0,20,10]] means there are two dataset, the first dataset has 20 labels in class 1 and 10 labels in class 2
        """
        self.noise = noise
        self.force_bound = force_bound
        self.tolerance = tolerance
        self.split_rate = split_rate
        self.label_matrix = label_matrix.astype(np.int64)
        self.total_sum_label = np.sum(self.label_matrix, axis = 0)
        self.tried_combination = []
        
    def get_one_combination(self):
        
        """
        return one combination of the label matrix
        return: 
            current_result: the label distribution of the target combination
            result_idx: the index of the target combination
        """
        
        end = False
        current_result = np.zeros(self.label_matrix.shape[1]).astype(np.int64)
        index = list(range(self.label_matrix.shape[0]))
        result_idx = []
        
        while not end:
            # new a queue
            heap = []
            for i in index:
                diff_result = rewardFunction(current_result + self.label_matrix[i], self.split_rate ,self.total_sum_label, noise= self.noise)
                heappush(heap, (diff_result, i))

            if heap[0][0] < 0:
                end = True
                break
            best_result = heappop(heap)
            current_result += self.label_matrix[best_result[1]]
            index.remove(best_result[1])
            # print(f"best_result: {best_result}")
            # print(f"label matrix: {self.label_matrix[best_result[1]]}")
            # print(f"current_result: {current_result}")
            result_idx.append(best_result[1])
            # print(result_idx)
        
        
        result_idx = sorted(result_idx)
        self.tried_combination.append(tuple(result_idx))
        
        return current_result, result_idx
    
    def get_multi_combination(self, noOfCombination, top_k = 1000):
        
        """
        return multiple combination of the label matrix
        noOfCombination: the number of combination you want to get
        top_k: the number of combination you want to try in each iteration
        
        return:
            top_k_distribution: the label distribution of the target combination
            top_k_group_combination: the index of the target combination
        """
        
        end = False
        combination_list = comb(range(self.label_matrix.shape[0]), 2)
        # print(list(combination_list))
        # print(label_matrix)
        counter = 0
        result = None
        while not end:
            counter += 1
            queue = []
            
            for i in list(combination_list):
                # print(i)
                # print(label_matrix[i, :])
                # print(np.sum(label_matrix[i, :], axis=0))
                diff_result = rewardFunction2(np.sum(self.label_matrix[i, :], axis=0), self.split_rate ,self.total_sum_label, noise= self.noise)
                heappush(queue, (diff_result, i))
                # print(queue)
            # print(len(queue))
            # print(queue)
            print(f"best result: {queue[0]}")
            if queue[0][0] < 0 or len(queue) < 50:
                end = True
                result = queue
                break
            k_top_index = [heappop(queue)[1] for _ in range(np.min((top_k, len(queue))))]
            # print(k_top_index)
            
            combination_list = list(comb(k_top_index, 2))
            # print(combination_list)
            # remove the duplicated index tuple, i.e. (1,2), (2,3) -> (1,2,3)
            combination_list = list(set([tuple(sorted(set(a + b))) for a, b in combination_list]))
            print(f"{counter}th iteration, len of combination_list: {len(combination_list)}")
            
            
            if counter == 20: 
                print(f"cannot converge")
                result = queue
                break
            
            # print(combination_list)
        top_k_group_combination = [heappop(result) for _ in range(noOfCombination)]
        
        top_k_distribution = np.zeros((noOfCombination, self.label_matrix.shape[1]))
        
        for idx,i in enumerate(top_k_group_combination):
            index = i[1]
            top_k_distribution[idx] = (np.sum(self.label_matrix[index,:], axis=0))
            
        # print(top_k_distribution)
        # print(top_k_group_combination)
        top_k_group_combination = [i[1] for i in top_k_group_combination]
        
        for i in top_k_group_combination:
            self.tried_combination.append(tuple(i))
        
        return top_k_distribution, top_k_group_combination

    def get_tried_combination(self):
        
        """
        return the unique tried combination
        """
        
        # print(self.tried_combination)
        return list(set(self.tried_combination))
    
    def get_remain_test_combination(self, train_combination):
        
        """
        return the test combination given the train combination
        """
        
        all_combination = list(range(self.label_matrix.shape[0]))
        tried_combination = train_combination
        remain_combination = sorted(list(set(all_combination) - set(tried_combination)))
        return remain_combination