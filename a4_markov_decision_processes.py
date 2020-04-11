#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
import numpy as np
import matplotlib.pyplot as plt
import pymdptoolbox.src.mdptoolbox as mdptoolbox
import pymdptoolbox.src.mdptoolbox.example as example
import sys
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import pandas as pd
import math


# In[2]:


np.random.seed(1)


# In[2]:


sys.path.append('./pymdptoolbox/src/mdptoolbox_mod')


# In[3]:


def plot_data(x, np_arr, title="Figure 1", x_label="Predicted Y", y_label="Test Y", color="blue", label=None, linestyle = 'solid', handlelength=3, marker='o'):
  plt.scatter(x ,np_arr, label=label, color = color, linestyle = linestyle, marker = marker )
  plt.title(title)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.legend(handlelength=handlelength)

def plot_line(x, np_arr, title="Figure 1", x_label="Predicted Y", y_label="Test Y", color="blue", label=None, linestyle = 'solid', handlelength=3):
  plt.plot(x ,np_arr, label=label, color = color, linestyle = linestyle )
  plt.title(title)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.legend(handlelength=handlelength)

def compute_daily_return_diff(df):
    """Compute and return the daily return values."""
    daily_returns = df.copy()
#     daily_returns = (df / df.shift(1)) - 1
    daily_returns = df.shift(1)-df
    daily_returns.iloc[0] = 0
    return daily_returns 

def calc_momentum(df,lookback):
    momentum = df.copy()
    momentum.values[lookback:,:] = (momentum.values[lookback:,:]/momentum.values[:-lookback,:])-1
    momentum.iloc[:lookback,:] = np.nan
    return momentum

def calc_sma(df, lookback):
    sma = df.cumsum()
    sma.values[lookback:,:] = (sma.values[lookback:,:]-sma.values[:-lookback,:])/lookback
    sma.iloc[:lookback,:] = np.nan
    return sma


# In[4]:


def vi(T, R, discount = 0.9, verbose=False):
    print('VALUE ITER')
    vi = mdptoolbox.mdp.ValueIteration(T, R, discount)
    vi.setVerbose()
    vi.run()
    if(verbose):
        print('Num of Iter')
        print(vi.iter)      
        print('Time')
        print(vi.time)
        print('V')
        print(sum(vi.V))

    return vi
def pi(T, R, discount = 0.9, verbose=False):
    print('POLICY ITER')
    pi = mdptoolbox.mdp.PolicyIteration(T, R, discount)
    pi.setVerbose()
    pi.run()
    # pi.policy
    if(verbose):
        print('Num of Iter')
        print(pi.iter)
        print('Time')
        print(pi.time)
        print('V')
        print(sum(pi.V))
    return pi
def ql(T, R, discount = 0.9, n_iter=10000, verbose=False, epsilon = 0.01, learning_rate=0.2):
    print('Q LEARNING')
    ql = mdptoolbox.mdp.QLearning(T, R, discount, n_iter, epsilon, learning_rate)
    ql.setVerbose()
    ql.run()
    if(verbose):
        print("Num of iter")
        print(len(ql.iter_time))         
        print('Time')
        print(ql.time)
        print('V')
        print(sum(ql.V))
 
    return ql


# In[5]:


def vp_graphs(p, v, problem_name, discount_rate):
    discount_rate = str(discount_rate)
    scaler = MinMaxScaler(feature_range=(0, 100))
    scaled_iter_variation_vi = scaler.fit_transform(np.array(v.iter_variation).reshape(-1, 1))
    plot_data(list(range(1, len(scaled_iter_variation_vi)+1)), scaled_iter_variation_vi, title="Policy/Value Variation (PI/VI, discount_rate="+discount_rate+", "+problem_name+")", x_label="Iteration", y_label="Variation", color="blue", label="VI", linestyle = 'solid', handlelength=3, marker='o')

    scaler = MinMaxScaler(feature_range=(0, 100))
    scaled_iter_variation_p = scaler.fit_transform(np.array(p.iter_policy_diff).reshape(-1, 1))
    plot_data(list(range(1, len(scaled_iter_variation_p)+1)), scaled_iter_variation_p, title="Policy/Value Variation (PI/VI, discount_rate="+discount_rate+", "+problem_name+")", x_label="Iteration", y_label="Variation", color="orange", label="PI", linestyle = 'solid', handlelength=3, marker='p')
    plt.show()
    plt.close()

    plot_data(list(range(1, len(v.iter_time)+1)), v.iter_time, title="Cumulative Run Time (PI/VI, discount_rate="+discount_rate+", "+problem_name+")", x_label="Iteration", y_label="Time", color="blue", label="VI", linestyle = 'solid', handlelength=3)
    plot_data(list(range(1, len(p.iter_time)+1)), p.iter_time, title="Cumulative Run Time (PI/VI, discount_rate="+discount_rate+", "+problem_name+")", x_label="Iteration", y_label="Time", color="orange", label="PI", linestyle = 'solid', handlelength=3)
    plt.show()
    plt.close()
    
    # Value Function 
    print(sum(v.V))
    print(sum(p.V))
    plot_data(list(range(1, len(v.V)+1)), v.V, title="Value Function (PI/VI, discount_rate="+discount_rate+", "+problem_name+")", x_label="State", y_label="Value", color="blue", label="VI", linestyle = 'solid', handlelength=3)
    plot_data(list(range(1, len(p.V)+1)), p.V, title="Value Function (PI/VI, discount_rate="+discount_rate+", "+problem_name+")", x_label="State", y_label="Value", color="orange", label="PI", linestyle = 'solid', handlelength=3)
    plt.show()
    plt.close()
    


# In[6]:


def q_graphs(q, problem_name):
# Q-Learning Policy Changes

    interval = 10000
    policy_changes = []
    for i in range(0,len(q.policies)-interval,interval):
        policy_changes.append((np.asarray(q.policies[i+interval])!=np.asarray(q.policies[i])).sum())
    plot_data(list(range(0,len(q.policies)-interval,interval)), policy_changes, title="Q-Learning Policy Changes ("+problem_name+")", x_label="Iteration", y_label="Difference", color="blue", label="", linestyle = 'solid', handlelength=3)
    plt.show()
    plt.close()

    # Q-Learning Q-Value Discounted Delta SMA

    df = pd.DataFrame(np.absolute(q.deltas))
    sma = calc_sma(df,50000)

    d = sma[0]
    # print(d)
    print(d.max())
    print(d.min())
    plot_line(list(range(1, len(d)+1)), d, title="Q-Value Delta SMA ("+problem_name+")", x_label="Iteration", y_label="Delta", color="blue", label=None, linestyle = 'solid', handlelength=3)
    plt.show()
    plt.close()


# In[7]:


def run_exp(T, R, problem_name, vipi_discount_rate=0.9, q_discount_rate=0.99, max_iter_rate = 500000, epsilon=0.03, learning_rate=0.2, verbose=True):
    print("STARTING ANALYSIS")

    v = vi(T,R,discount = vipi_discount_rate, verbose=True)
    p = pi(T,R, discount = vipi_discount_rate, verbose=True)

    q = ql(T, R, discount = q_discount_rate, n_iter=max_iter_rate, verbose=verbose, epsilon=epsilon, learning_rate=learning_rate)
    vipi_discount_rate = str(vipi_discount_rate)
    q_discount_rate = str(q_discount_rate)
    vp_graphs(p,v, problem_name, vipi_discount_rate)
    q_graphs(q, problem_name)

    print("Policy Length")
    print(len(p.policy))
    # print(len(v.policy))
    # print(len(q.policy))
    print("Policy Iteration - Value Iteration policy Diff: ")
    print((np.asarray(p.policy)!=np.asarray(v.policy)).sum())
    print("Policy Iteration - Q-Learning policy Diff: ")
    print((np.asarray(p.policy)!=np.asarray(q.policy)).sum())
    print("Value Iteration - Q-Learning policy Diff: ")
    print((np.asarray(v.policy)!=np.asarray(q.policy)).sum())

    # Value Function 
    print(sum(v.V))
    print(sum(p.V))
    print(sum(q.V))
    plot_data(list(range(1, len(v.V)+1)), v.V, title="Value Function, (discount_rate="+vipi_discount_rate+", "+problem_name+")", x_label="State", y_label="Value", color="blue", label="VI", linestyle = 'solid', handlelength=3)
    plot_data(list(range(1, len(p.V)+1)), p.V, title="Value Function, (discount_rate="+vipi_discount_rate+", "+problem_name+")", x_label="State", y_label="Value", color="orange", label="PI", linestyle = 'solid', handlelength=3)
    plot_data(list(range(1, len(q.V)+1)), q.V, title="Value Function, (discount_rate="+q_discount_rate+", "+problem_name+")", x_label="State", y_label="Value", color="green", label="Q-Learning", linestyle = 'solid', handlelength=3)
    plt.show()
    plt.close()
    return v,p,q
    


# In[8]:


def vipi_disc_expl(T, R, problem_name):

    print("VI PI DISCOUNT RATE EXPLORATION!:")
    p_vs= []
    v_vs= []
    pivi_policy_diff = []
    for discount in np.arange(0.1,1, 0.1):
        print("DISCOUNT RATE:")
        print(discount)
        v = vi(T,R, discount, verbose=True)
        p = pi(T,R, discount, verbose=True)
        p_vs.append(sum(v.V))
        v_vs.append(sum(p.V))
#         vp_graphs(p,v, problem_name, discount)
        policy_diff = (np.asarray(p.policy)!=np.asarray(v.policy)).sum()
        pivi_policy_diff.append(policy_diff)
        print("Policy Iteration - Value Iteration policy Diff: ")
        print(policy_diff)
    plot_data(list(np.arange(0.1,1, 0.1)), pivi_policy_diff, title="PI/VI Policy Difference ("+problem_name+")", x_label="Discount Rate", y_label="Difference", color="blue", label="", linestyle = 'solid', handlelength=3)
    plt.show()
    plt.close()
    plot_data(list(np.arange(0.1,1, 0.1)), p_vs, title="PI/VI Value Function ("+problem_name+")", x_label="Discount Rate", y_label="Value Sum", color="orange", label="PI", linestyle = 'solid', handlelength=3)
    plot_data(list(np.arange(0.1,1, 0.1)), v_vs, title="PI/VI Value Function ("+problem_name+")", x_label="Discount Rate", y_label="Value Sum", color="blue", label="VI", linestyle = 'solid', handlelength=3)
    plt.show()
    plt.close()


# In[9]:


def q_disc_expl(T, R, p, v,  problem_name, discount_rate=0.99, max_iter=400000, learning_rate=0.1):
    print("Q Learning DISCOUNT RATE EXPLORATION!:")
    q_vs= []
    q_time = []
    q_policy_diff = []
    for disc in np.arange(0.09 ,1, 0.1):
        print("DISCOUNT RATE:")
        print(disc)
        q = ql(T, R, discount = disc, n_iter=max_iter, verbose=True, epsilon=0.03, learning_rate=learning_rate)
        policy_diff = (np.asarray(p.policy)!=np.asarray(q.policy)).sum()
        q_vs.append(sum(q.V))
        q_time.append(q.time)
        q_policy_diff.append(policy_diff)
#         q_graphs(q, problem_name)
        print("Policy Iteration - Value Iteration policy Diff: ")
        print((np.asarray(p.policy)!=np.asarray(v.policy)).sum())
        print("Policy Iteration - Q-Learning policy Diff: ")
        print(policy_diff)
        print("Value Iteration - Q-Learning policy Diff: ")
        print((np.asarray(v.policy)!=np.asarray(q.policy)).sum())
    max_iter = str(max_iter)
    learning_rate = str(learning_rate)
    discount_rate = str(discount_rate)
    plot_data(list(np.arange(0.09,1, 0.1)), q_time, title="Q-Learning Cumulative Run Time, (learning_rate="+learning_rate+", max_iter="+max_iter+", "+problem_name+")", x_label="Discount Rate", y_label="Time", color="green", label="", linestyle = 'solid', handlelength=3)
    plt.show()
    plt.close()
    plot_data(list(np.arange(0.09,1, 0.1)), q_policy_diff, title="Q-Learning PI/VI Policy Difference, (learning_rate="+learning_rate+", max_iter="+max_iter+", "+problem_name+")", x_label="Discount Rate", y_label="Difference", color="blue", label="", linestyle = 'solid', handlelength=3)
    plt.show()
    plt.close()
    plot_data(list(np.arange(0.09,1, 0.1)), q_vs, title="Q-Learning Value Function, (learning_rate="+learning_rate+", max_iter="+max_iter+", "+problem_name+")", x_label="Discount Rate", y_label="Value Sum", color="orange", label="", linestyle = 'solid', handlelength=3)
    plt.show()
    plt.close()


# In[10]:


def q_lr_expl(T, R, p, v,  problem_name, discount_rate=0.99, max_iter=400000, learning_rate=0.1):
    print("Q Learning Learning RATE EXPLORATION!:")
    q_vs= []
    q_time = []
    q_policy_diff = []
    for rate in np.arange(0.1 ,0.99,0.1):
        print("Learning RATE:")
        print(rate)
        q = ql(T, R, discount = discount_rate, n_iter=max_iter, verbose=True, epsilon=0.03, learning_rate=rate)
        policy_diff = (np.asarray(p.policy)!=np.asarray(q.policy)).sum()
        q_vs.append(sum(q.V))
        q_time.append(q.time)
        q_policy_diff.append(policy_diff)
#         q_graphs(q, "FrozenLake")
        print("Policy Iteration - Value Iteration policy Diff: ")
        print((np.asarray(p.policy)!=np.asarray(v.policy)).sum())
        print("Policy Iteration - Q-Learning policy Diff: ")
        print(policy_diff)
        print("Value Iteration - Q-Learning policy Diff: ")
        print((np.asarray(v.policy)!=np.asarray(q.policy)).sum())
    max_iter = str(max_iter)
    learning_rate = str(learning_rate)
    discount_rate = str(discount_rate)
    plot_data(list(np.arange(0.1 ,0.99,0.1)), q_time, title="Q-Learning Cumulative Run Time, (discount_rate="+discount_rate+", max_iter="+max_iter+", "+problem_name+")", x_label="Learning Rate", y_label="Time", color="green", label="", linestyle = 'solid', handlelength=3)
    plt.show()
    plt.close()
    plot_data(list(np.arange(0.1 ,0.99,0.1)), q_policy_diff, title="Q-Learning PI/VI Policy Difference, (discount_rate="+discount_rate+", max_iter="+max_iter+", "+problem_name+")", x_label="Learning Rate", y_label="Difference", color="blue", label="", linestyle = 'solid', handlelength=3)
    plt.show()
    plt.close()
    plot_data(list(np.arange(0.1 ,0.99,0.1)), q_vs, title="Q-Learning Value Function, (discount_rate="+discount_rate+", max_iter="+max_iter+", "+problem_name+")", x_label="Learning Rate", y_label="Value Sum", color="orange", label="", linestyle = 'solid', handlelength=3)
    plt.show()
    plt.close()


# In[11]:


def q_max_iter_expl(T, R, p, v,  problem_name, discount_rate=0.99, max_iter=400000, learning_rate=0.1):
    print("Q Learning Maximum Iteration EXPLORATION!:")
    q_vs= []
    q_time = []
    q_policy_diff = []
    for rate in np.arange(50000 ,800001,50000):
        print("Max Iteration:")
        print(rate)
        q = ql(T, R, discount = discount_rate, n_iter=rate, verbose=True, epsilon=0.03, learning_rate=learning_rate)
        policy_diff = (np.asarray(p.policy)!=np.asarray(q.policy)).sum()
        q_vs.append(sum(q.V))
        q_time.append(q.time)
        q_policy_diff.append(policy_diff)
#         q_graphs(q, "FrozenLake")
        print("Policy Iteration - Value Iteration policy Diff: ")
        print((np.asarray(p.policy)!=np.asarray(v.policy)).sum())
        print("Policy Iteration - Q-Learning policy Diff: ")
        print(policy_diff)
        print("Value Iteration - Q-Learning policy Diff: ")
        print((np.asarray(v.policy)!=np.asarray(q.policy)).sum())
    max_iter = str(max_iter)
    learning_rate = str(learning_rate)
    discount_rate = str(discount_rate)
    plot_data(list(np.arange(50000, 800001,50000)), q_time, title="Q-Learning Cumulative Run Time, (discount_rate="+discount_rate+", learning_rate="+learning_rate+", "+problem_name+")", x_label="Max Iterations", y_label="Time", color="green", label="", linestyle = 'solid', handlelength=3)
    plt.show()
    plt.close()
    plot_data(list(np.arange(50000,800001,50000)), q_policy_diff, title="Q-Learning PI/VI Policy Difference, (discount_rate="+discount_rate+", learning_rate="+learning_rate+", "+problem_name+")", x_label="Max Iterations", y_label="Difference", color="blue", label="", linestyle = 'solid', handlelength=3)
    plt.show()
    plt.close()
    plot_data(list(np.arange(50000,800001,50000)), q_vs, title="Q-Learning Value Function, (discount_rate="+discount_rate+", learning_rate="+learning_rate+", "+problem_name+")", x_label="Max Iterations", y_label="Value Sum", color="orange", label="", linestyle = 'solid', handlelength=3)
    plt.show()
    plt.close()


# In[12]:


# FrozenLake experiment
env = gym.make('FrozenLake-v0')
action = env.action_space.sample()
problem_name = "FrozenLake"
R=np.zeros((env.nA,env.nS,env.nS))
T=np.zeros((env.nA,env.nS,env.nS))
for state in env.P:
#     print(state)
    for action in env.P[state]:
        for tup in env.P[state][action]:
#             print(tup)
            prob = tup[0]
            new_state = tup[1]
            reward = tup[2]
            R[action][state][new_state] = reward
            T[action][state][new_state] = T[action][state][new_state]+prob
# print(R)
# print(T)
v,p,q = run_exp(T, R, problem_name, vipi_discount_rate=0.9, q_discount_rate=0.9, max_iter_rate = 500000, epsilon=0.03, learning_rate=0.5, verbose=True)
vipi_disc_expl(T, R, problem_name)
q_disc_expl(T, R, p, v,  problem_name, discount_rate=0.9, max_iter=500000, learning_rate=0.5)
q_lr_expl(T, R, p, v,  problem_name, discount_rate=0.9, max_iter=500000, learning_rate=0.5)
q_max_iter_expl(T, R, p, v,  problem_name, discount_rate=0.9, max_iter=500000, learning_rate=0.5)


# In[12]:


# Forest Management experiment
T, R = example.forest(1000)
problem_name ="Forest Management"
v,p,q = run_exp(T, R, problem_name, vipi_discount_rate=0.9, q_discount_rate=0.99, max_iter_rate = 550000, epsilon=0.03, learning_rate=0.5, verbose=True)
vipi_disc_expl(T, R, problem_name)
q_disc_expl(T, R, p, v,  problem_name, discount_rate=0.99, max_iter=550000, learning_rate=0.5)
q_lr_expl(T, R, p, v,  problem_name, discount_rate=0.99, max_iter=550000, learning_rate=0.5)
q_max_iter_expl(T, R, p, v,  problem_name, discount_rate=0.99, max_iter=550000, learning_rate=0.5)

