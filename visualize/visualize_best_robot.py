import argparse
import csv
import os
import numpy as np
import torch
from visualize import visualize

from utils.config import Config

import sys
sys.path.append('.')

def load(exp_dir, generation, id):

    while not os.path.exists(f'{exp_dir}/generation{str(generation).zfill(2)}/id{str(id).zfill(2)}'):
        generation -= 1

    robot_dir = f'{exp_dir}/generation{str(generation).zfill(2)}/id{str(id).zfill(2)}/robot_dir'
    print(f'loading from {robot_dir}')
    structure = np.load(os.path.join(robot_dir, 'structure.npy'), allow_pickle=True)
    connections = np.load(os.path.join(robot_dir, 'connections.npy'), allow_pickle=True)
    parameter = torch.load(os.path.join(robot_dir, 'parameter.pt'), map_location='cpu')

    
    return structure, connections, parameter

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-ex', '--exp_dir', type=str)
    parser.add_argument('-en', '--env_name', type=str)
    parser.add_argument('-m', '--movie_path', type=str, default=None)
    parser.add_argument('-n','--num_evals', type=int, default=1)

    args = parser.parse_args()

    exp_dir = args.exp_dir
    Config.env_name = args.env_name

    with open(os.path.join(exp_dir, 'fitness.csv')) as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                colname = row
            last_generation = row

    generation = int(last_generation[0])
    best_id = np.argmax([float(val) for val in last_generation[1:]])
    
    structure, connections, (actor_critic, obs_rms) = load(exp_dir, generation, best_id)

    visualize(
        structure=structure,
        connections=connections,
        actor_critic=actor_critic,
        obs_rm=obs_rms,
        movie_path=args.movie_path,
        num_evals=args.num_evals
    )

