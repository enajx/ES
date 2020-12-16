import numpy as np
import multiprocessing as mp
import copy
import torch
import sys
import time
from os.path import join, exists
from os import mkdir

from fitness_functions import fitness_static


def compute_ranks(x):
  """
  Returns rank as a vector of len(x) with integers from 0 to len(x)
  """
  assert x.ndim == 1
  ranks = np.empty(len(x), dtype=int)
  ranks[x.argsort()] = np.arange(len(x))
  return ranks

def compute_centered_ranks(x):
  """
  Maps x to [-0.5, 0.5] and returns the rank
  """
  y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
  y /= (x.size - 1)
  y -= .5
  return y

def worker_process(arg):
    get_reward_func, weights, env = arg
    
    wp = np.array(weights)
    decay = - 0.01 * np.mean(wp**2) 
    r = get_reward_func(weights, env) + decay

    return r 


class EvolutionStrategyStatic(object):
    def __init__(self, weights, environment, population_size=500, sigma=0.1, learning_rate=0.2, decay=0.995, num_threads=-1):
        
        self.weights = weights
        self.environment = environment
        self.POPULATION_SIZE = population_size
        self.SIGMA = sigma
        self.learning_rate = learning_rate
        self.decay = decay
        self.num_threads = mp.cpu_count() if num_threads == -1 else num_threads
        self.update_factor = self.learning_rate / (self.POPULATION_SIZE * self.SIGMA)
        
        self.get_reward = fitness_static
        
    def _get_weights_try(self, w, p):
        
        weights_try = []
        for index, i in enumerate(p):
            jittered = np.float32(self.SIGMA * i)
            weights_try.append(w[index] + jittered)
        weights_try = np.array(weights_try)

        return weights_try   # weights_try[i] = w[i] + sigma * p[i]
 
    def get_weights(self):
        return self.weights

    def _get_population(self):
        population = []
        for i in range( int(self.POPULATION_SIZE/2) ):
            x = []
            x2 = []
            for w in self.weights:
                j = np.random.randn(*w.shape)
                x.append(j)
                x2.append(-j) 

            population.append(x)
            population.append(x2)
            
        population = np.array(population)

        return population    # [[w_i... w_92000], [w_j... w_92000], [...], ...]


    def _get_rewards(self, pool, population):
        
        # Multi-core
        if pool is not None:
            worker_args = []
            for p in population:

                weights_try1 = []

                for index, i in enumerate(p):
                    jittered = np.float32(self.SIGMA * i)
                    weights_try1.append(self.weights[index] + jittered)
                weights_try = np.array(weights_try1)
                worker_args.append( (self.get_reward, weights_try, self.environment) )
                
            rewards  = pool.map(worker_process, worker_args)
            
        # Single-core
        else:
            rewards = []
            for p in population:
                weights_try = np.array(self._get_weights_try(self.weights, p))   # weights_try[i] = self.weights[i] + sigma * p[i]
                rewards.append(self.get_reward(weights_try, self.environment))
        rewards = np.array(rewards)

        return rewards

    def _update_weights(self, rewards, population): 
        
        rewards = compute_centered_ranks(rewards)   # Project rewards to [-0.5, 0.5]

        std = rewards.std()
        if std == 0:
            raise ValueError('Variance should not be zero')


        rewards = (rewards - rewards.mean()) / std  # Normalize rewards
        for index, w in enumerate(self.weights):
            layer_population = np.array([p[index] for p in population])   # Array of all weights[i] for all the networks in the population
            
            self.update_factor = self.learning_rate / (self.POPULATION_SIZE * self.SIGMA)    
            self.weights[index] = w + self.update_factor * np.dot(layer_population.T, rewards).T 

        if self.learning_rate > 0.001:
            self.learning_rate *= self.decay

        #Decay sigma
        if self.SIGMA>0.01:
            self.SIGMA *= 0.999


    def run(self, iterations, print_step=10, path='weights'):
        
        id_ = str(int(time.time()))
        if not exists(path + '/' + id_):
            mkdir(path + '/' + id_)
            
        print('\n********************\n \nRUN: ' + id_ + '\n\n********************\n')
        
        pool = mp.Pool(self.num_threads) if self.num_threads > 1 else None
        
        generations_rewards = []
        
        for iteration in range(iterations):                     # Algorithm 2. Salimans, 2017: https://arxiv.org/abs/1703.03864
            
            population = self._get_population()                 # List of list of random nets [[w1, w2, .., w122888],[...],[...]] : Step 5
            rewards = self._get_rewards(pool, population)       # List of corresponding rewards for self.weights + jittered populations : Step 6
            self._update_weights(rewards, population)           # Updates self.weights : Steps 8->12 

            if (iteration + 1) % print_step == 0:
                rew_ = self.get_reward(self.weights, self.environment)
                print('iter %4i | reward: %3i |  update_factor: %f  lr: %f | sum_w: %i sum_abs_w: %i' % ( iteration + 1, rew_ , self.update_factor, self.learning_rate, int(np.sum(self.weights)) ,int(np.sum(abs(self.weights))) ), flush=True)
                torch.save(self.get_weights(), path + "/"+ id_ + "/" + self.environment + "__rew_" + str(int(rew_)) + "__pop_" + str(self.POPULATION_SIZE) + "__{}.dat".format(iteration))  

                generations_rewards.append(rew_)
                np.save(path + "/"+ id_ + '/Fitness_values_' + id_ + '_' + self.environment + '.npy', np.array(generations_rewards))

        if pool is not None:
            pool.close()
            pool.join()
            