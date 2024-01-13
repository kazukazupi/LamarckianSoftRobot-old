import os
import numpy as np
import torch
import argparse
from visualize import visualize

from utils.config import Config

def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--robot_dir", type=str)
    parser.add_argument("--env_name", type=str)

    args = parser.parse_args()

    Config.env_name = args.env_name

    structure = np.load(os.path.join(args.robot_dir, "body.npy"))
    connections = np.load(os.path.join(args.robot_dir, "connections.npy"))

    print(structure)
    print(connections)

    actor_critic, obs_rms = torch.load(os.path.join(args.robot_dir, "actor_critic.pt"), map_location='cpu')

    visualize(
        structure=structure,
        connections=connections,
        actor_critic=actor_critic,
        obs_rm=obs_rms
    )

if __name__ == '__main__':
    main()