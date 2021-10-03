import gym

import torch

import numpy as np
import random

from src.config import args
from src.utils import train, record_video

import argparse

# pass the environment name as a cmd argument
parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, required=True)
arguments = parser.parse_args()

# set the random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.random.manual_seed(args.seed)



if __name__ == "__main__":
    # extract the env name from the cmd args
    env_name = arguments.env
    # check if the environment exist
    all_envs = gym.envs.registry.all()
    env_ids = [env_spec.id for env_spec in all_envs]
    if env_name  not in env_ids:
        raise "Environment does not exist, check the envionment name"
    # train the model
    model = train(env_name)
    # record a video
    record_video(env_name, model)
    




