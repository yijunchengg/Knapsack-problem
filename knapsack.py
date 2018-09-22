# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 21:13:21 2018

Knapsack problem
Maximise the total value under the maximal weight constraint


1. Reinforcement learning
2. Dynamic programming based on q value
3. Dynamic programming based on v value


@author: Yijun Cheng
@email: yijuncheng_csu@163.com

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
import itertools
item = pd.DataFrame(data=[[1, 1],
                          [6, 2],
                          [18, 5],
                          [22, 6],
                          [28, 7]],
                    columns=['Value', 'Weight'])

actions = list(range(len(item)))
limit_W = 11
gamma = 0.9
# %%
'''
**************************************************************
1. Q learning
'''


class RLforKnapsack():
    def __init__(self, limit_W, actions):
        self.limit_W = limit_W  # maximal weight
        self.epsilon = 0.9  # e-greedy algorithm
        self.gamma = 0.9  # reward decay
        self.alpha = 0.8  # learning_rate
        self.actions = actions
        self.q_table = pd.DataFrame(columns=actions)
        self.done = False

    def check_state(self, knapsack):
        if str(knapsack) not in self.q_table.index:
            # append new state to q table
            q_table_new = pd.Series([np.NAN]*len(self.actions),
                                    index=self.q_table.columns,
                                    name=str(knapsack))
            # 0-1 knapsack
            for i in list(set(self.actions).difference(set(knapsack))):
                q_table_new[i] = 0
            self.q_table = self.q_table.append(q_table_new)

    def choose_action(self, knapsack):
        self.check_state(knapsack)
        state_action = self.q_table.loc[str(knapsack), :]
        # random state_action in case there are two or more maximum
        state_action = state_action.reindex(
                np.random.permutation(state_action.index)
                )
        if np.random.uniform() < self.epsilon:
            # choose best action
            action = state_action.idxmax()  # the first maximun
        else:
            # choose random action
            action = np.random.choice(
                    list(set(self.actions).difference(set(knapsack)))
                    )
        return action

    def greedy_action(self, knapsack):
        # testing
        # choose best action
        state_action = self.q_table.loc[str(knapsack), :]
        state_action = state_action.reindex(
                np.random.permutation(state_action.index)
                )
        action = state_action.idxmax()
        return action

    def take_action(self, knapsack, action):
        # take the item
        knapsack_ = knapsack + [action]
        knapsack_.sort()
        self.check_state(knapsack_)
        return knapsack_

    def rewardWithPenalty(self, knapsack_, action):
        # constraint
        knapsack_W = np.sum([item['Weight'][i] for i in knapsack_])
        if knapsack_W > self.limit_W:
            r = -10
            self.done = True
        else:
            r = item['Value'][action]
        return r

    def update_qvalue(self, knapsack, knapsack_, action):
        self.done = False
        reward = self.rewardWithPenalty(knapsack_, action)
        q_predict = self.q_table.loc[str(knapsack), action]
        if len(knapsack) != len(self.actions):
            q_target = reward + self.gamma * self.q_table.loc[
                    str(knapsack_), :].max()
        else:
            q_target = reward  # no item can be added
        self.q_table.loc[str(knapsack), action] += self.alpha * (
                q_target - q_predict)
        return self.q_table, self.done


t1 = time()
plt.close('all')
RL = RLforKnapsack(limit_W=11, actions=actions)
for episode in range(1000):
    knapsack = []
    for step in range(5):
        action = RL.choose_action(knapsack)
        knapsack_ = RL.take_action(knapsack, action)
        q_table_RL, done = RL.update_qvalue(knapsack, knapsack_, action)
        knapsack = knapsack_
        if done:
            break
    plt.scatter(episode, q_table_RL.iloc[0, 3], c='r')
    plt.scatter(episode, q_table_RL.iloc[0, 4], c='b')
t2 = time()
plt.title([t2-t1, 'RL'])
plt.show()

# %% Policy based on q table
knapsack = []
# %%
action = RL.greedy_action(knapsack)
knapsack_ = RL.take_action(knapsack, action)
knapsack = knapsack_
np.sum([item['Weight'][i] for i in knapsack_])

# %%
'''
**************************************************************
2. Dynamic programming, q table
'''


def get_index(actions):
    index_t = []
    for i in range(len(actions)+1):
        index_t.extend(list(itertools.combinations(actions, i)))
    index_l = []
    for ind in index_t:
        index_l.append(list(ind))
    return index_l


class DPforKnapsack():
    def __init__(self, gamma, index_l, actions):
        self.gamma = 0.8
        self.index_l = index_l
        self.q_table = pd.DataFrame(np.zeros((len(self.index_l),
                                              len(actions))),
                                    index=[str(i) for i in self.index_l],
                                    columns=actions)

    def rewardWithPenalty(self, next_index, a):
        knapsack_W = np.sum([item['Weight'][i] for i in next_index])
        if knapsack_W > limit_W:
            r = -5
        else:
            r = item['Value'][a]
        return r

    def take_action(self, ind, a):
        next_index = self.index_l[ind]+[a]
        next_index.sort()
        return next_index

    def update_value(self, ind, action):
        if len(action) != 0:  # except [0, 1, 2, 3, 4, 5]
            for a in action:
                next_index = self.take_action(ind, a)
                reward = self.rewardWithPenalty(next_index, a)
                # np.max(self.q_table.loc[str(next_index)])
                q_pre = reward + gamma * self.q_table.loc[
                        str(next_index), :].max()
                self.q_table.loc[str(self.index_l[ind]), a] = q_pre
        return self.q_table


t1 = time()
plt.close('all')
index_l = get_index(actions)
DP = DPforKnapsack(gamma, index_l, actions)
for episode in range(5):
    for ind in range(len(index_l)):
        action = list(set(actions).difference(set(index_l[ind])))
        q_table_DP = DP.update_value(ind, action)
    plt.scatter(episode, q_table_DP.iloc[0, 3], c='r')
    plt.scatter(episode, q_table_DP.iloc[0, 4], c='b')
t2 = time()
plt.title([t2-t1, 'DP'])
plt.show()

# %%
'''
**************************************************************
3. Dynamic programming, v table
'''


def get_index(actions):
    index_t = []
    for i in range(len(actions)+1):
        index_t.extend(list(itertools.combinations(actions, i)))
    index_l = []
    for ind in index_t:
        index_l.append(list(ind))
    return index_l


def get_terminal(index_l):
    terminal = []
    for i in range(len(index_l)):
        knapsack_W = np.sum([item['Weight'][i] for i in index_l[i]])
        if knapsack_W > limit_W:
            terminal.append(index_l[i])
    return terminal


class DPforKnapsack():
    def __init__(self, gamma, index_l, actions):
        self.gamma = gamma
        self.index_l = index_l
        self.actions = actions
        self.v_table = pd.DataFrame(np.zeros(len(index_l)),
                                    index=[str(i) for i in index_l])
        self.q_table = pd.DataFrame(np.zeros((len(index_l),
                                              len(actions))),
                                    index=[str(i) for i in index_l],
                                    columns=self.actions)
        self.terminal = get_terminal(index_l)

    def rewardWithPenalty(self, next_index, a):
        knapsack_W = np.sum([item['Weight'][i] for i in next_index])
        if knapsack_W > limit_W:
            r = -5
        else:
            r = item['Value'][a]
        return r

    def take_action(self, ind, a):
        next_index = self.index_l[ind]+[a]
        next_index.sort()
        return next_index

    def update_value(self, ind, action):
        # get all value
        if len(action) != 0:
            v_pre = []
            for a in action:
                next_index = self.take_action(ind, a)
                if next_index in self.terminal:
                    v_pre.append(-5)
                else:
                    reward = self.rewardWithPenalty(next_index, a)
                    v_next = float(self.v_table.loc[str(next_index)])
                    v_pre.append(reward + self.gamma * v_next)
            self.v_table.loc[str(self.index_l[ind])] = max(v_pre)
        else:
            self.v_table.loc[str(self.index_l[ind])] = -5
        return self.v_table


t1 = time()
# value evaluation
index_l = get_index(actions)

DP = DPforKnapsack(gamma, index_l, actions)
for episode in range(5):
    for ind in range(len(index_l)):
        action = list(set(actions).difference(set(index_l[ind])))
        v_table = DP.update_value(ind, action)
    plt.scatter(episode, v_table.iloc[4], c='r')
    plt.scatter(episode, v_table.iloc[1], c='b')

# Policy
q_table_v = pd.DataFrame(np.zeros((len(index_l),
                                   len(actions))),
                         index=[str(i) for i in index_l],
                         columns=actions)
for i in range(len(index_l)):
    action = list(set(actions).difference(set(index_l[i])))
    if len(action) != 0:
        for j in action:
            next_index = DP.take_action(i, j)
            reward = DP.rewardWithPenalty(next_index, j)
            q_table_v.iloc[i, j] = reward + gamma * float(
                    v_table.loc[str(next_index)])
t2 = time()
print(t2 - t1, 'DP')
