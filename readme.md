 
---

<div align="center">    
 
# Evolution Strategies as Scalable Alternative to Reinforcement Learning 


</div>
 
This reposistory contains the code to train agents on any Gym, pyBullet, or MuJoCo environment using an Evolution Strategy (ES) algorithm. It's adapted from [this OpenAI implementation](https://github.com/openai/evolution-strategies-starter)
of the distributed Evolution-Strategy (ES) introduced in [Evolution Strategies as Scalable Alternative to Reinforcement Learning, Salimans et al. 2017](https://arxiv.org/abs/1703.03864).

This code was used to create the non-plastic baselines for our paper [Meta-Learning through Hebbian Plasticity in Random Networks](https://arxiv.org/abs/2007.02686).



## How to run   

First, install dependencies. Use `Python >= 3.8`:
```bash
# clone project   
git clone https://github.com/enajx/ES

# install dependencies   
cd ES 
pip install -r requirements.txt
 ```   
 Next, use `train_static.py` to train an agent. You can train any of OpenAI Gym's or pyBullet environments:
 ```bash

# train agent to solve the racing car
python train_static.py --environment CarRacing-v0


# train agent specifying evolution parameters, eg. 
python train_static.py --environment CarRacing-v0 --generations 300 --popsize 200 --print_every 1 --lr 0.2 --sigma 0.1 --decay 0.995 --threads -1

```

 Use `python train_static.py --help` to display all the training options:


 ```bash

train_static.py [--environment] [--popsize] [--print_every] [--lr] [--decay] [--sigma] [--generations] [--folder] [--threads]

arguments:
  --environment   Environment: any OpenAI Gym or pyBullet environment may be used
  --popsize       Population size.
  --print_every   Print and save every N steps.
  --lr            ES learning rate.
  --decay         ES decay.
  --sigma         ES sigma: modulates the amount of noise used to populate each new generation
  --generations   Number of generations that the ES will run.
  --folder        folder to store the evolved weights
  --threads       Number of threads used to run evolution in parallel.

```

Once trained, use `evaluate_static.py` to test the evolved agent:
 ```bash

python evaluate_static.py --environment CarRacing-v0 --path_weights weights.dat

```

When running on a headless server some environments will require a virtual display to run -eg. CarRacing-v0-, in this case run:
 ```bash

xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python train_static.py --environment CarRacing-v0

```

## Citation   

If you use the code for academic or commecial use, please cite the associated paper:

```

@inproceedings{Najarro2020,
	title = {{Meta-Learning through Hebbian Plasticity in Random Networks}},
	author = {Najarro, Elias and Risi, Sebastian},
	booktitle = {Advances in Neural Information Processing Systems},
	year = {2020},
	url = {https://arxiv.org/abs/2007.02686}
}

```   


## Some notes on training performance

In the paper we have tested the CarRacing-v0 and AntBulletEnv-v0 environments. For both of them we have written custom functions to bound the action activations;
the rest of the environments have a simple clipping mechanism to bound their actions. Environments with a continuous action space (ie. *Box*)
are likely to benefit from a continous scaling -rather than clipping- of their action spaces, either with a custom activation function or with 
Gym's RescaleAction wrapper.

Another element that greatly affects performance -if you have bounded computational resources- is the choice of a suitable early stop meachanism such that less CPU cycles are wasted, 
eg. for the CarRacing-v0 environment we use 20 consecutive steps with negative reward as an early stop signal.

Finally, some pixel-based environments would likely benefit from using grayscaling + stacked frames approach rather than feeding the network the three RGB channels as we do in our 
implementation, eg. by using Gym's [Frame stack wrapper](https://github.com/openai/gym/blob/master/gym/wrappers/frame_stack.py#L58) or the [Atari preprocessing wrapper](https://github.com/openai/gym/blob/master/gym/wrappers/atari_preprocessing.py#L12).
