import os

import numpy as np
import torch

def load(exp_dir:str, generation:int, id:int):

    while not os.path.exists(f'{exp_dir}/generation{str(generation).zfill(2)}/id{str(id).zfill(2)}'):
        generation -= 1

    robot_dir = f'{exp_dir}/generation{str(generation).zfill(2)}/id{str(id).zfill(2)}/robot_dir'

    return load_from_robot_dir(robot_dir)

def load_from_robot_dir(robot_dir:str):
    
    # TO DO: "structure"を"structure"→"body"に変更
    structure = np.load(os.path.join(robot_dir, 'structure.npy'))
    connections = np.load(os.path.join(robot_dir, 'connections.npy'))

    # TO DO: map_location を gpuにも対応できるようにする
    actor_critic, obs_rms = torch.load(os.path.join(robot_dir, 'parameter.pt'), map_location='cpu')

    return (structure, connections), (actor_critic, obs_rms)