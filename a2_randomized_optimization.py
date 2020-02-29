"""
ML (CS7641) - Randomized Optimization
Student Name: Gevorg Ter-Petrosyan
GT User ID: gterpetrosyan3
"""

#!/usr/bin/env python
# coding: utf-8

# In[24]:


import mlrose_hiive
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import random
import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler,OrdinalEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, multilabel_confusion_matrix
from sklearn.neural_network import MLPClassifier


# from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split, learning_curve, validation_curve


# In[3]:


def plot_data(x, np_arr, title="Figure 1", x_label="Predicted Y", y_label="Test Y", color="blue", label=None):
  plt.plot(x ,np_arr, label=label, color = color)
  plt.title(title)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.legend()


# In[4]:


def scatter_plot(x, y, title="Figure 1", x_label="Predicted Y", y_label="Test Y", color="black", marker="square", label=None):
  plt.scatter(x ,y, label=label, color = color, marker = marker)
  plt.title(title)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.legend()


# In[5]:


def sa(problem, init_state, max_attempts, max_iters):
    # Define decay schedule
    schedule = mlrose_hiive.ExpDecay()
    # Define initial state
#     init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    # Solve problem using simulated annealing
    best_state, best_fitness, fitness_curve  = mlrose_hiive.simulated_annealing(problem, schedule = schedule,
                                                          max_attempts = max_attempts, max_iters = max_iters,init_state = init_state, curve=True, random_state = 1)
    print('simulated annealing')                                                     
    print(best_state)
    print(best_fitness)
#     print(fitness_curve)
    return best_state, best_fitness, fitness_curve


# In[6]:


def rhc(problem, init_state, max_attempts, max_iters):
    best_state, best_fitness, fitness_curve  = mlrose_hiive.random_hill_climb(problem,
                                                          max_attempts = max_attempts, max_iters = max_iters, restarts = 100,
                                                          init_state = init_state, curve=True, random_state = 1)
    print('random hill climbing')
    print(best_state)
    print(best_fitness)
#     print(fitness_curve)
    return best_state, best_fitness, fitness_curve


# In[7]:


def gen(problem, init_state, max_attempts, max_iters):
    best_state, best_fitness, fitness_curve  = mlrose_hiive.genetic_alg(problem, pop_size=200, mutation_prob=0.1, 
                                                          max_attempts = max_attempts, max_iters = max_iters, curve=True, random_state = 1)
    print('genetic')
    print(best_state)
    print(best_fitness)
#     print(fitness_curve)
    return best_state, best_fitness, fitness_curve


# In[8]:


def mimic(problem, init_state, max_attempts, max_iters):
    best_state, best_fitness, fitness_curve  = mlrose_hiive.mimic(problem, pop_size=200, keep_pct=0.2, 
                                                          max_attempts = max_attempts, max_iters = max_iters, curve=True, random_state = 1)
    print('mimic')
    print(best_state)
    print(best_fitness)
#     print(fitness_curve)
    return best_state, best_fitness, fitness_curve


# In[283]:


# problem 1 Genetic - Knapsack
input_sizes = []
sa_times=[]
sa_scores=[]
rhc_times=[]
rhc_scores=[]
gen_times=[]
gen_scores=[]
mimic_times=[]
mimic_scores=[]
for i in range(10, 51, 10):
    if i==0:
        continue
    ex = {"weights": [random.randint(1, 20) for i in range(i)], "values": [random.randint(1, 10) for i in range(i)], "state": np.array([random.randint(0, 2) for i in range(i)])}
    input_sizes.append(i)
    weights = ex['weights']
    values = ex['values']
    state = ex['state']
    max_weight_pct = 0.6
    fitness = mlrose_hiive.Knapsack(weights, values, max_weight_pct)
    fitness.evaluate(state)
    problem = mlrose_hiive.DiscreteOpt(length = len(state), fitness_fn = fitness, maximize = True, max_val = int(max(state))+1)
    times = []
    best_scores = []

    start_time = time.time()
    best_state, best_fitness, fitness_curve = sa(problem,state, 30, 1000)
    elapsed_time = time.time() - start_time
    print(elapsed_time)
    times.append(elapsed_time*1000)
    best_scores.append(best_fitness)
    sa_times.append(elapsed_time*1000)
    sa_scores.append(best_fitness)
    plt.close()
    
    plot_data([i+1 for i in range(len(fitness_curve))], fitness_curve, title="Evaluations Required to Maximize Knapsack (Input Size = "+str(len(state))+")", x_label="Evaluations", y_label="Fitness Score", color="blue", label='Simulated annealing')

    start_time = time.time()
    best_state, best_fitness, fitness_curve = rhc(problem,state, 30, 1000)
    elapsed_time = time.time() - start_time
    print(elapsed_time)
    times.append(elapsed_time*1000)
    best_scores.append(best_fitness)
    rhc_times.append(elapsed_time*1000)
    rhc_scores.append(best_fitness)
    
    plot_data([i+1 for i in range(len(fitness_curve))], fitness_curve, title="Evaluations Required to Maximize Knapsack (Input Size = "+str(len(state))+")", x_label="Evaluations", y_label="Fitness Score", color="orange", label='Randomized hill climbing')
   
    
    start_time = time.time()
    best_state, best_fitness, fitness_curve = gen(problem,state, 30, 1000)
    elapsed_time = time.time() - start_time
    times.append(elapsed_time*1000)
    best_scores.append(best_fitness)
    gen_times.append(elapsed_time*1000)
    gen_scores.append(best_fitness)
    print(elapsed_time)
    
    plot_data([i+1 for i in range(len(fitness_curve))], fitness_curve, title="Evaluations Required to Maximize Knapsack (Input Size = "+str(len(state))+")", x_label="Evaluations", y_label="Fitness Score", color="red", label='Genetic algorithm')
   
    start_time = time.time()
    best_state, best_fitness, fitness_curve = mimic(problem,state, 30, 1000)
    elapsed_time = time.time() - start_time
    times.append(elapsed_time*1000)
    best_scores.append(best_fitness)
    mimic_times.append(elapsed_time*1000)
    mimic_scores.append(best_fitness)
    print(elapsed_time)
    
    plot_data([i+1 for i in range(len(fitness_curve))], fitness_curve, title="Evaluations Required to Maximize Knapsack (Input Size = "+str(len(state))+")", x_label="Evaluations", y_label="Fitness Score", color="green", label='Mimic')
   
    plt.savefig('a2_p1_evals_figure_'+str(len(input_sizes))+'.png')
    plt.show()
    plt.close()
    scatter_plot([times[0]], [best_scores[0]], title="Knapsack Convergence Times (Input Size = "+str(len(state))+")", x_label="Time (milliseconds)", y_label="Fitness Score", marker="s", color="blue", label="Simulated annealing")
    scatter_plot([times[1]], [best_scores[1]], title="Knapsack Convergence Times (Input Size = "+str(len(state))+")", x_label="Time (milliseconds)", y_label="Fitness Score", marker="s", color="orange", label="Randomized hill climbing")
    scatter_plot([times[2]], [best_scores[2]], title="Knapsack Convergence Times (Input Size = "+str(len(state))+")", x_label="Time (milliseconds)", y_label="Fitness Score", marker="s", color="red", label="Genetic algorithm")
    scatter_plot([times[3]], [best_scores[3]], title="Knapsack Convergence Times (Input Size = "+str(len(state))+")", x_label="Time (milliseconds)", y_label="Fitness Score", marker="s", color="green", label="Mimic")
    plt.savefig('a2_p1_conv_times_'+str(len(input_sizes))+"_input"+str(len(state))+'.png')
    plt.show()
    plt.close()

    
plt.close()
plot_data(input_sizes, sa_times, title="Knapsack Convergence Times", x_label="Input Size", y_label="Time", color="blue", label='Simulated annealing')
plot_data(input_sizes, rhc_times, title="Knapsack Convergence Times", x_label="Input Size", y_label="Time", color="orange", label='Randomized hill climbing')
plot_data(input_sizes, gen_times, title="Knapsack Convergence Times", x_label="Input Size", y_label="Time", color="red", label='Genetic algorithm')
plot_data(input_sizes, mimic_times, title="Knapsack Convergence Times", x_label="Input Size", y_label="Time", color="green", label='Mimic')
plt.savefig('a2_p1_conv_times_all_'+str(len(input_sizes))+'.png')
plt.show()
plt.close()


# In[341]:


# problem 3 mimic FlipFlop
input_sizes = []
sa_times=[]
sa_evals=[]
rhc_times=[]
rhc_evals=[]
gen_times=[]
gen_evals=[]
mimic_times=[]
mimic_evals=[]
for i in range(10, 51, 10):
    if i==0:
        continue
#     ex = {"weights": [random.randint(1, 20) for i in range(i)], "values": [random.randint(1, 10) for i in range(i)], "state": np.array([random.randint(0, 2) for i in range(i)])}
    input_sizes.append(i)
#     weights = ex['weights']
#     values = ex['values']
#     state = ex['state']
#     edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
#     state = np.array([random.randint(1, 20) for i in range(i)])
    state = np.array([random.randint(0, 2) for i in range(i)])
#     fitness = mlrose_hiive.FourPeaks(t_pct=0.15)
    fitness = mlrose_hiive.FlipFlop()
    problem = mlrose_hiive.DiscreteOpt(length = len(state), fitness_fn = fitness, maximize = True, max_val = int(max(state))+1)
#     problem = mlrose_hiive.DiscreteOpt(length = 5, fitness_fn = fitness, maximize = False, max_val = 2)
#     problem = mlrose_hiive.DiscreteOpt(length = len(state), fitness_fn = fitness, maximize = True, max_val = int(max(state))+1)
    times = []
    best_scores = []

    start_time = time.time()
    best_state, best_fitness, fitness_curve = sa(problem,state, 10, 1000)
    elapsed_time = time.time() - start_time
    print(elapsed_time)
    times.append(elapsed_time*1000)
    best_scores.append(best_fitness)
    sa_times.append(elapsed_time*1000)
    sa_evals.append(len(fitness_curve))
    plt.close()
    
    plot_data([i+1 for i in range(len(fitness_curve))], fitness_curve, title="Evaluations Required to Maximize FlipFlop (Input Size = "+str(len(state))+")", x_label="Evaluations", y_label="Fitness Score", color="blue", label='Simulated annealing')

    start_time = time.time()
    best_state, best_fitness, fitness_curve = rhc(problem,state, 10, 1000)
    elapsed_time = time.time() - start_time
    print(elapsed_time)
    times.append(elapsed_time*1000)
    best_scores.append(best_fitness)
    rhc_times.append(elapsed_time*1000)
    rhc_evals.append(len(fitness_curve))
    
    plot_data([i+1 for i in range(len(fitness_curve))], fitness_curve, title="Evaluations Required to Maximize FlipFlop (Input Size = "+str(len(state))+")", x_label="Evaluations", y_label="Fitness Score", color="orange", label='Randomized hill climbing')
   
    
    start_time = time.time()
    best_state, best_fitness, fitness_curve = gen(problem,state, 10, 1000)
    elapsed_time = time.time() - start_time
    times.append(elapsed_time*1000)
    best_scores.append(best_fitness)
    gen_times.append(elapsed_time*1000)
    gen_evals.append(len(fitness_curve))
    print(elapsed_time)
    
    plot_data([i+1 for i in range(len(fitness_curve))], fitness_curve, title="Evaluations Required to Maximize FlipFlop (Input Size = "+str(len(state))+")", x_label="Evaluations", y_label="Fitness Score", color="red", label='Genetic algorithm')
   
    start_time = time.time()
    best_state, best_fitness, fitness_curve = mimic(problem,state, 10, 1000)
    elapsed_time = time.time() - start_time
    times.append(elapsed_time*1000)
    best_scores.append(best_fitness)
    mimic_times.append(elapsed_time*1000)
    mimic_evals.append(len(fitness_curve))
    print(elapsed_time)
    
    plot_data([i+1 for i in range(len(fitness_curve))], fitness_curve, title="Evaluations Required to Maximize FlipFlop (Input Size = "+str(len(state))+")", x_label="Evaluations", y_label="Fitness Score", color="green", label='Mimic')
   
    plt.savefig('a2_p2_evals_figure_'+str(len(input_sizes))+'.png')
    plt.show()
    plt.close()
    scatter_plot([times[0]], [best_scores[0]], title="FlipFlop Convergence Times (Input Size = "+str(len(state))+")", x_label="Time (milliseconds)", y_label="Fitness Score", marker="s", color="blue", label="Simulated annealing")
    scatter_plot([times[1]], [best_scores[1]], title="FlipFlop Convergence Times (Input Size = "+str(len(state))+")", x_label="Time (milliseconds)", y_label="Fitness Score", marker="s", color="orange", label="Randomized hill climbing")
    scatter_plot([times[2]], [best_scores[2]], title="FlipFlop Convergence Times (Input Size = "+str(len(state))+")", x_label="Time (milliseconds)", y_label="Fitness Score", marker="s", color="red", label="Genetic algorithm")
    scatter_plot([times[3]], [best_scores[3]], title="FlipFlop Convergence Times (Input Size = "+str(len(state))+")", x_label="Time (milliseconds)", y_label="Fitness Score", marker="s", color="green", label="Mimic")
    plt.savefig('a2_p2_conv_times_'+str(len(input_sizes))+"_input"+str(len(state))+'.png')
    plt.show()
    plt.close()

    
plt.close()
plot_data(input_sizes, sa_times, title="FlipFlop Convergence Times", x_label="Input Size", y_label="Time", color="blue", label='Simulated annealing')
plot_data(input_sizes, rhc_times, title="FlipFlop Convergence Times", x_label="Input Size", y_label="Time", color="orange", label='Randomized hill climbing')
plot_data(input_sizes, gen_times, title="FlipFlop Convergence Times", x_label="Input Size", y_label="Time", color="red", label='Genetic algorithm')
plot_data(input_sizes, mimic_times, title="FlipFlop Convergence Times", x_label="Input Size", y_label="Time", color="green", label='Mimic')
plt.savefig('a2_p2_conv_times_all_'+str(len(input_sizes))+'.png')
plt.show()
plt.close()

plot_data(input_sizes, sa_evals, title="FlipFlop Evaluations", x_label="Input Size", y_label="Evaluations", color="blue", label='Simulated annealing')
plot_data(input_sizes, rhc_evals, title="FlipFlop Evaluations ", x_label="Input Size", y_label="Evaluations", color="orange", label='Randomized hill climbing')
plot_data(input_sizes, gen_evals, title="FlipFlop Evaluations", x_label="Input Size", y_label="Evaluations", color="red", label='Genetic algorithm')
plot_data(input_sizes, mimic_evals, title="FlipFlop Evaluations", x_label="Input Size", y_label="Evaluations", color="green", label='Mimic')
plt.savefig('a2_p2_evals_all_'+str(len(input_sizes))+'.png')
plt.show()
plt.close()


# In[284]:


# problem 2 - Simulated annealing  OneMax
input_sizes = []
sa_times=[]
sa_evals=[]
rhc_times=[]
rhc_evals=[]
gen_times=[]
gen_evals=[]
mimic_times=[]
mimic_evals=[]
for i in range(10, 51, 10):
    if i==0:
        continue
#     ex = {"weights": [random.randint(1, 20) for i in range(i)], "values": [random.randint(1, 10) for i in range(i)], "state": np.array([random.randint(0, 2) for i in range(i)])}
    input_sizes.append(i)
#     weights = ex['weights']
#     values = ex['values']
#     state = ex['state']
#     edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
#     state = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    state = np.array([random.randint(0, 2) for i in range(i)])
    fitness = mlrose_hiive.OneMax()
    problem = mlrose_hiive.DiscreteOpt(length = len(state), fitness_fn = fitness, maximize = True, max_val = int(max(state))+1)
#     problem = mlrose_hiive.DiscreteOpt(length = 5, fitness_fn = fitness, maximize = False, max_val = 2)
#     problem = mlrose_hiive.DiscreteOpt(length = len(state), fitness_fn = fitness, maximize = True, max_val = int(max(state))+1)
    times = []
    best_scores = []

    start_time = time.time()
    best_state, best_fitness, fitness_curve = sa(problem,state, 30, 1000)
    elapsed_time = time.time() - start_time
    print(elapsed_time)
    times.append(elapsed_time*1000)
    best_scores.append(best_fitness)
    sa_times.append(elapsed_time*1000)
    sa_evals.append(len(fitness_curve))
    plt.close()
    
    plot_data([i+1 for i in range(len(fitness_curve))], fitness_curve, title="Evaluations Required to Maximize OneMax (Input Size = "+str(len(state))+")", x_label="Evaluations", y_label="Fitness Score", color="blue", label='Simulated annealing')

    start_time = time.time()
    best_state, best_fitness, fitness_curve = rhc(problem,state, 30, 1000)
    elapsed_time = time.time() - start_time
    print(elapsed_time)
    times.append(elapsed_time*1000)
    best_scores.append(best_fitness)
    rhc_times.append(elapsed_time*1000)
    rhc_evals.append(len(fitness_curve))
    
    plot_data([i+1 for i in range(len(fitness_curve))], fitness_curve, title="Evaluations Required to Maximize OneMax (Input Size = "+str(len(state))+")", x_label="Evaluations", y_label="Fitness Score", color="orange", label='Randomized hill climbing')
   
    
    start_time = time.time()
    best_state, best_fitness, fitness_curve = gen(problem,state, 30, 1000)
    elapsed_time = time.time() - start_time
    times.append(elapsed_time*1000)
    best_scores.append(best_fitness)
    gen_times.append(elapsed_time*1000)
    gen_evals.append(len(fitness_curve))
    print(elapsed_time)
    
    plot_data([i+1 for i in range(len(fitness_curve))], fitness_curve, title="Evaluations Required to Maximize OneMax (Input Size = "+str(len(state))+")", x_label="Evaluations", y_label="Fitness Score", color="red", label='Genetic algorithm')
   
    start_time = time.time()
    best_state, best_fitness, fitness_curve = mimic(problem,state, 30, 1000)
    elapsed_time = time.time() - start_time
    times.append(elapsed_time*1000)
    best_scores.append(best_fitness)
    mimic_times.append(elapsed_time*1000)
    mimic_evals.append(len(fitness_curve))
    print(elapsed_time)
    
    plot_data([i+1 for i in range(len(fitness_curve))], fitness_curve, title="Evaluations Required to Maximize OneMax (Input Size = "+str(len(state))+")", x_label="Evaluations", y_label="Fitness Score", color="green", label='Mimic')
   
    plt.savefig('a2_p3_evals_figure_'+str(len(input_sizes))+'.png')
    plt.show()
    plt.close()
    scatter_plot([times[0]], [best_scores[0]], title="OneMax Convergence Times (Input Size = "+str(len(state))+")", x_label="Time (milliseconds)", y_label="Fitness Score", marker="s", color="blue", label="Simulated annealing")
    scatter_plot([times[1]], [best_scores[1]], title="OneMax Convergence Times (Input Size = "+str(len(state))+")", x_label="Time (milliseconds)", y_label="Fitness Score", marker="s", color="orange", label="Randomized hill climbing")
    scatter_plot([times[2]], [best_scores[2]], title="OneMax Convergence Times (Input Size = "+str(len(state))+")", x_label="Time (milliseconds)", y_label="Fitness Score", marker="s", color="red", label="Genetic algorithm")
    scatter_plot([times[3]], [best_scores[3]], title="OneMax Convergence Times (Input Size = "+str(len(state))+")", x_label="Time (milliseconds)", y_label="Fitness Score", marker="s", color="green", label="Mimic")
    plt.savefig('a2_p3_conv_times_'+str(len(input_sizes))+"_input"+str(len(state))+'.png')
    plt.show()
    plt.close()

    
plt.close()
plot_data(input_sizes, sa_times, title="OneMax Convergence Times", x_label="Input Size", y_label="Time", color="blue", label='Simulated annealing')
plot_data(input_sizes, rhc_times, title="OneMax Convergence Times", x_label="Input Size", y_label="Time", color="orange", label='Randomized hill climbing')
plot_data(input_sizes, gen_times, title="OneMax Convergence Times", x_label="Input Size", y_label="Time", color="red", label='Genetic algorithm')
plot_data(input_sizes, mimic_times, title="OneMax Convergence Times", x_label="Input Size", y_label="Time", color="green", label='Mimic')
plt.savefig('a2_p3_conv_times_all_'+str(len(input_sizes))+'.png')
plt.show()
plt.close()

plot_data(input_sizes, sa_evals, title="OneMax Evaluations", x_label="Input Size", y_label="Evaluations", color="blue", label='Simulated annealing')
plot_data(input_sizes, rhc_evals, title="OneMax Evaluations ", x_label="Input Size", y_label="Evaluations", color="orange", label='Randomized hill climbing')
plot_data(input_sizes, gen_evals, title="OneMax Evaluations", x_label="Input Size", y_label="Evaluations", color="red", label='Genetic algorithm')
plot_data(input_sizes, mimic_evals, title="OneMax Evaluations", x_label="Input Size", y_label="Evaluations", color="green", label='Mimic')
plt.savefig('a2_p3_evals_all_'+str(len(input_sizes))+'.png')
plt.show()
plt.close()


# In[11]:


diamondsDF = pd.read_csv('diamonds.csv')
# pre-process dataset
ord_encoder = OrdinalEncoder(categories=[['Fair','Good','Very Good','Premium','Ideal']])
diamondsDF["cut"] = ord_encoder.fit_transform(diamondsDF[['cut']])
# print(transformed_cut)
#  = transformed_cut
# print(diamondsDF)
hot_encoder = OneHotEncoder(handle_unknown='ignore')
diamonds_color_clarity = hot_encoder.fit_transform(diamondsDF[['color','clarity']]).toarray()
categories = []
for x in hot_encoder.categories_:
    categories.extend(x)
diamonds_color_clarity_df = pd.DataFrame(diamonds_color_clarity, columns=categories)
diamondsDF = diamondsDF.join(diamonds_color_clarity_df)
diamondsDF=diamondsDF.drop(['color','clarity'],axis=1)
# print(diamondsDF)
data = diamondsDF.loc[:, diamondsDF.columns != "cut"]
y_data = diamondsDF[['cut']]
x_train, x_test = train_test_split(data, test_size=0.9, shuffle=True, random_state=100)
y_train, y_test = train_test_split(y_data, test_size=0.9, shuffle=True, random_state=100)

hot_encoder = OneHotEncoder()
y_train_hot_encoded = hot_encoder.fit_transform(y_train.values.reshape(-1, 1)).todense()
y_test_hot_encoded = hot_encoder.fit_transform(y_test.values.reshape(-1, 1)).todense()

scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# In[372]:


mlp_learner = MLPClassifier(hidden_layer_sizes=(14, 14),activation='relu',solver='sgd', learning_rate = 'adaptive', learning_rate_init = 0.07)
train_sizes, train_scores, test_scores = learning_curve(mlp_learner, x_train_scaled, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.close()
plot_data(train_sizes, test_mean, title="Gradient Descent Neural Network Learning Curve(Diamonds)", x_label="Training Size", y_label="Accuracy Score", color="blue", label='Cross-validation score')
plot_data(train_sizes, train_mean, title="Gradient Descent Neural Network Learning Curve(Diamonds)", x_label="Training Size", y_label="Accuracy Score", color="orange", label='Training score')
# plt.show()
plt.savefig('a2_nn_sgd.png')
print(train_sizes)
print((train_sizes[np.argmax(test_mean)]))
print(test_mean[np.argmax(test_mean)])
print(train_mean[np.argmax(test_mean)])


# In[347]:


# print(x_train.shape)
# print(x_test.shape)
# print(y_train_hot_encoded.shape)
# print(y_test_hot_encoded.shape)
# print(y_train_hot_encoded)


# In[18]:


nn_model1 = mlrose_hiive.NeuralNetwork(hidden_nodes = [14, 14], activation = 'relu',                                  algorithm = 'genetic_alg', max_iters = 5000,                                  bias = True, is_classifier = True, learning_rate = 0.3,                                  early_stopping = False, clip_max = 5, max_attempts = 100,                                  random_state = 3)
print(x_train_scaled.shape)
print(y_train_hot_encoded.shape)

start_time = time.time()
nn_model1.fit(x_train_scaled, y_train_hot_encoded)
elapsed_time = time.time() - start_time
print(elapsed_time)
# start_time = time.time()
# # nn_model_sa.fit(x_train_scaled, y_train_hot_encoded)
# cv_scores = cross_validate(nn_model1, x_train_scaled, y_train_hot_encoded, cv=3, n_jobs=-1)
# print(cv_scores)
# print( np.mean(cv_scores['test_score']))
# elapsed_time = time.time() - start_time
# print(elapsed_time)


# In[13]:


# train_sizes, train_scores, test_scores = learning_curve(nn_model1, x_train_scaled, y_train_hot_encoded, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
# train_mean = np.mean(train_scores, axis=1)
# test_mean = np.mean(test_scores, axis=1)
# plt.close()
# plot_data(train_sizes, test_mean, title="Genetic Algorithm Neural Network Learning Curve (Diamonds)", x_label="Training Size", y_label="Accuracy Score", color="blue", label='Cross-validation score')
# plot_data(train_sizes, train_mean, title="Genetic Algorithm Neural Network Learning Curve (Diamonds)", x_label="Training Size", y_label="Accuracy Score", color="orange", label='Training score')
# plt.savefig('a2_nn_gen.png')
# print(train_sizes)
# print((train_sizes[np.argmax(test_mean)]))
# print(test_mean[np.argmax(test_mean)])
# print(train_mean[np.argmax(test_mean)])


# In[26]:


# nn_model1.fit(x_train_scaled, y_train_hot_encoded)
y_train_pred = nn_model1.predict(x_train_scaled)
y_train_accuracy = accuracy_score(y_train_hot_encoded, y_train_pred)

print(y_train_accuracy)

y_test_pred = nn_model1.predict(x_test_scaled)

y_test_accuracy = accuracy_score(y_test_hot_encoded, y_test_pred)

print(y_test_accuracy)

# confusion_matrix(y_train_hot_encoded.argmax(axis=1), y_train_pred.argmax(axis=1))
z = multilabel_confusion_matrix(y_train_hot_encoded, y_train_pred)

print(z)


# In[ ]:





# In[364]:



schedule = mlrose_hiive.ExpDecay()
nn_model_sa = mlrose_hiive.NeuralNetwork(hidden_nodes = [14, 14], activation = 'relu',                                  algorithm = 'simulated_annealing', max_iters = 10000,                                  bias = True, is_classifier = True, learning_rate = 0.3,                                  early_stopping = False, clip_max = 5, max_attempts = 100,                                  random_state = 3, schedule=schedule, curve=True)
print(x_train_scaled.shape)
print(y_train_hot_encoded.shape)

start_time = time.time()
cv_scores = cross_validate(nn_model_sa, x_train_scaled, y_train_hot_encoded, cv=5, n_jobs=-1)
print(cv_scores)
print( np.mean(cv_scores['test_score']))
elapsed_time = time.time() - start_time
print(elapsed_time)

# start_time = time.time()
# z=nn_model_sa.fit(x_train_scaled, y_train_hot_encoded)
# print(z)
# y_train_pred = nn_model_sa.predict(x_train_scaled)
# y_train_accuracy = accuracy_score(y_train_hot_encoded, y_train_pred)
# print(y_train_pred)
# print(y_train_hot_encoded)
# print(y_train_accuracy)
# y_test_pred = nn_model_sa.predict(x_test_scaled)
# y_test_accuracy = accuracy_score(y_test_hot_encoded, y_test_pred)
# print(y_test_accuracy)
# elapsed_time = time.time() - start_time
# print(elapsed_time)


# In[365]:


train_sizes, train_scores, test_scores = learning_curve(nn_model_sa, x_train_scaled, y_train_hot_encoded, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.close()
plot_data(train_sizes, test_mean, title="Simulated Annealing Neural Network Learning Curve (Diamonds)", x_label="Training Size", y_label="Accuracy Score", color="blue", label='Cross-validation score')
plot_data(train_sizes, train_mean, title="Simulated Annealing Neural Network Learning Curve (Diamonds)", x_label="Training Size", y_label="Accuracy Score", color="orange", label='Training score')
plt.show()
# plt.savefig('figure1cvc5.png')
print(train_sizes)
print((train_sizes[np.argmax(test_mean)]))
print(test_mean[np.argmax(test_mean)])
print(train_mean[np.argmax(test_mean)])


# In[367]:


plt.savefig('a2_nn_sa.png')


# In[354]:


# nn_model_gd = mlrose_hiive.NeuralNetwork(hidden_nodes = [100, 100], activation = 'relu', \
#                                  algorithm = 'gradient_descent', max_iters = 10000, \
#                                  bias = True, is_classifier = True, learning_rate = 0.07, \
#                                  early_stopping = False, clip_max = 5, max_attempts = 100, \
#                                  random_state = 3)
# print(x_train_scaled.shape)
# print(y_train_hot_encoded.shape)

# start_time = time.time()
# # nn_model_sa.fit(x_train_scaled, y_train_hot_encoded)
# cv_scores = cross_validate(nn_model_gd, x_train_scaled, y_train_hot_encoded, cv=5, n_jobs=-1)
# print(cv_scores)
# print( np.mean(cv_scores['test_score']))
# elapsed_time = time.time() - start_time
# print(elapsed_time)


# In[369]:


nn_model_rhc = mlrose_hiive.NeuralNetwork(hidden_nodes = [14, 14], activation = 'relu',                                  algorithm = 'random_hill_climb', max_iters = 10000,                                  bias = True, is_classifier = True, learning_rate = 0.3,                                  early_stopping = False, clip_max = 5, max_attempts = 100,                                  random_state = 3)
# print(x_train_scaled.shape)
# print(y_train_hot_encoded.shape)

# start_time = time.time()
# # nn_model_sa.fit(x_train_scaled, y_train_hot_encoded)
# cv_scores = cross_validate(nn_model_rhc, x_train_scaled, y_train_hot_encoded, cv=5, n_jobs=-1)
# print(cv_scores)
# print( np.mean(cv_scores['test_score']))
# elapsed_time = time.time() - start_time
# print(elapsed_time)


# In[370]:


train_sizes, train_scores, test_scores = learning_curve(nn_model_rhc, x_train_scaled, y_train_hot_encoded, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.close()
plot_data(train_sizes, test_mean, title="Random Hill Climb Neural Network Learning Curve (Diamonds)", x_label="Training Size", y_label="Accuracy Score", color="blue", label='Cross-validation score')
plot_data(train_sizes, train_mean, title="Random Hill Climb Neural Network Learning Curve (Diamonds)", x_label="Training Size", y_label="Accuracy Score", color="orange", label='Training score')
plt.savefig('a2_nn_rhc.png')
print(train_sizes)
print((train_sizes[np.argmax(test_mean)]))
print(test_mean[np.argmax(test_mean)])
print(train_mean[np.argmax(test_mean)])


# In[ ]:




