import gym
import torch
from torch import nn

import numpy as np
import pybullet_envs
from gym.spaces import Discrete, Box
from gym import wrappers as w
import pickle
import argparse
import sys
import time

from policies import MLP, CNN
from wrappers import ScaledFloatFrame, FireEpisodicLifeEnv


gym.logger.set_level(40)
        


def evaluate_static(environment : str, render : bool, evolved_parameters: [np.array]) -> None:
    """
    Evaluate an agent 'evolved_parameters' in an environment 'environment' during a lifetime.
    """
    
    with torch.no_grad():
                    
        # Load environment
        try:
            env = gym.make(environment, verbose = 0)
        except:
            env = gym.make(environment)
            
        bullet_env = 1 if environment[-12:-6] == 'Bullet' else 0
        if bullet_env and render:
            env.render()  
        
        # For environments with several intra-episode lives -eg. Breakout-
        try: 
            if 'FIRE' in env.unwrapped.get_action_meanings():
                env = FireEpisodicLifeEnv(env)
        except: 
            pass

        # Check if selected env is pixel or state-vector 
        if len(env.observation_space.shape) == 3:     # Pixel-based environment
            pixel_env = True
            env = w.ResizeObservation(env, 84)        # Resize and normilise input   
            env = ScaledFloatFrame(env)
            input_channels = 3
        elif len(env.observation_space.shape) == 1:   
            pixel_env = False
            input_dim = env.observation_space.shape[0]
        elif len(env.observation_space.shape) == 0:   
            pixel_env = False
            input_dim = env.observation_space.n
            
        # Determine action space dimension
        if isinstance(env.action_space, Box):
            action_dim = env.action_space.shape[0]
        elif isinstance(env.action_space, Discrete):
            action_dim = env.action_space.n
        else:
            raise ValueError('Only Box and Discrete action spaces supported')
        
        # Initialise policy network: with CNN layer for pixel envs and simple MLP for state-vector envs
        if pixel_env == True: 
            p = CNN(input_channels, action_dim)      
        else:
            p = MLP(input_dim, action_dim)     

        # Load weights into the policy network
        nn.utils.vector_to_parameters( torch.tensor (evolved_parameters, dtype=torch.float32 ),  p.parameters() )
            
        observation = env.reset() 
        if pixel_env: observation = np.swapaxes(observation,0,2) #(3, 84, 84)       

        # Burnout phase for the bullet quadruped so it starts off from the floor
        if environment == 'AntBulletEnv-v0':
            action = np.zeros(8)
            for _ in range(40):
                __ = env.step(action)        

        # Inner loop
        neg_count = 0
        rew_ep = 0
        t = 0
        while True:
            
            # For obaservation ∈ gym.spaces.Discrete, we one-hot encode the observation
            if isinstance(env.observation_space, Discrete): 
                observation = (observation == torch.arange(env.observation_space.n)).float()
            
            o3 = p([observation])

            # Bounding the action space
            if environment == 'CarRacing-v0':
                action = np.array([ torch.tanh(o3[0]), torch.sigmoid(o3[1]), torch.sigmoid(o3[2]) ]) 
                o3 = o3.numpy()
            elif environment[-12:-6] == 'Bullet':
                o3 = np.tanh(o3).numpy()
                action = o3
            else: 
                if isinstance(env.action_space, Box):
                    action = o3.numpy()                         
                    action = np.clip(action, env.action_space.low, env.action_space.high)  
                elif isinstance(env.action_space, Discrete):
                    action = np.argmax(o3).numpy()

            
            # Environment simulation step
            observation, reward, done, info = env.step(action)  
            if environment == 'AntBulletEnv-v0': reward = env.unwrapped.rewards[1] # Distance walked
            rew_ep += reward
            
            if (not bullet_env) and render:
                env.render()  
            
            if pixel_env: observation = np.swapaxes(observation,0,2) #(3, 84, 84)
                                       
            # Early stopping conditions
            if environment == 'CarRacing-v0':
                neg_count = neg_count+1 if reward < 0.0 else 0
                if (done or neg_count > 20):
                    break
            elif environment[-12:-6] == 'Bullet':
                if t > 200:
                    neg_count = neg_count+1 if reward < 0.0 else 0
                    if (done or neg_count > 30):
                        break
            else:
                if done:
                    break

            t += 1
            
        env.close()

    print('\n Episode cumulative rewards ', int(rew_ep))
    
    
def main(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--environment', type=str, default='CarRacing-v0', metavar='', help='Gym environment: any OpenAI Gym may be used')
    parser.add_argument('--path_weights', type=str, help='path to the evolved weights')
    parser.add_argument('--render', type=bool, default=1, help='path to the evolved weights')

    args = parser.parse_args()
    evolved_parameters = torch.load(args.path_weights) 
    
    # Run the environment
    evaluate_static(args.environment, args.render, evolved_parameters)
    
if __name__ == '__main__':
    main(sys.argv)

    
    
