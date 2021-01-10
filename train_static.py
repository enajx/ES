import time
import argparse
import sys
import torch
from os.path import join, exists
from os import mkdir
import gym
from gym.spaces import Discrete, Box
import pybullet_envs

from evolution_strategy_static import EvolutionStrategyStatic
from policies import MLP, CNN

torch.set_num_threads(1)
gym.logger.set_level(40)


def main(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--environment', type=str, default='LunarLander-v2', metavar='', help='Environment: any OpenAI Gym or pyBullet environment may be used')
    # parser.add_argument('--environment', type=str, default='CarRacing-v0', metavar='', help='Environment: any OpenAI Gym or pyBullet environment may be used')
    parser.add_argument('--popsize', type=int,  default = 32, metavar='', help='Population size.') 
    parser.add_argument('--print_every', type=int, default = 1, metavar='', help='Print and save every N steps.') 
    parser.add_argument('--lr', type=float,  default = 0.2, metavar='', help='ES learning rate.') 
    parser.add_argument('--decay', type=float,  default = 0.995, metavar='', help='ES decay.')  
    parser.add_argument('--sigma', type=float,  default = 0.1, metavar='', help='ES sigma: modulates the amount of noise used to populate each new generation') 
    parser.add_argument('--generations', type=int, default= 300, metavar='', help='Number of generations that the ES will run.')
    parser.add_argument('--folder', type=str, default='weights', metavar='', help='folder to store the evolved weights ')
    parser.add_argument('--threads', type=int, metavar='', default = -1, help='Number of threads used to run evolution in parallel.')    

    
    args = parser.parse_args()


    if not exists(args.folder):
        mkdir(args.folder)

    # Look up observation and action space dimension
    env = gym.make(args.environment)    
    if len(env.observation_space.shape) == 3:     # Pixel-based environment
        pixel_env = True
    elif len(env.observation_space.shape) == 1:   # State-based environment 
        pixel_env = False
        input_dim = env.observation_space.shape[0]
    elif isinstance(env.observation_space, Discrete):
        pixel_env = False
        input_dim = env.observation_space.n
    else:
        raise ValueError('Observation space not supported')

    if isinstance(env.action_space, Box):
        action_dim = env.action_space.shape[0]
    elif isinstance(env.action_space, Discrete):
        action_dim = env.action_space.n
    else:
        raise ValueError('Action space not supported')


    # Initialise policy network: with CNN layer for pixel envs and simple MLP for state-vector envs
    if pixel_env == True: 
        input_channels = 3
        p = CNN(input_channels, action_dim)      
    else:
        p = MLP(input_dim, action_dim)     


    # Initialise the EvolutionStrategy class
    print('\nInitilisating static-network ES for ' + str(args.environment))
    es = EvolutionStrategyStatic(p.get_weights(), args.environment, population_size=args.popsize, sigma=args.sigma, learning_rate=args.lr, decay=args.decay, num_threads=args.threads)

    # Start the evolution
    tic = time.time()
    print('\nStarting Evolution\n')
    es.run(args.generations, print_step=args.print_every, path=args.folder)
    toc = time.time()
    print('\nEvolution took: ', int(toc-tic), ' seconds\n')



if __name__ == '__main__':
    main(sys.argv)
