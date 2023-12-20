import os
from typing import Any
import numpy as np
import torch
import argparse

import sys
sys.path.append('../myga')

import evogym.envs
from evogym import get_full_connectivity, is_connected
from ppo.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from visualize import visualize


def get_over_head(env_name:str):
    if env_name == "BridgeWalker-v0":
            return 3
    else:
        raise NotImplementedError()

# Return the coordinates of the mass point in order from top-left to bottom-right.
def mass_point_order(structure):

    structure = (structure != 0)
    (H, W) = structure.shape

    result = []
    for i in range(H):
        for j in range(W):
            if structure[i][j]:
                for pair in [[i, j], [i, j+1], [i+1, j], [i+1, j+1]]:
                    if (pair in result) == False:
                        result.append(pair)

    return result

class GetParamAction:
    
    def __init__(self, str1, str2, params1, params2) -> None:

        Y = str1.shape[0]
        X = str1.shape[1]

        self.params1 = params1
        self.params2 = params2

        self.coord_to_node1 = {}
        self.coord_to_node2 = {}
        
        counter1 = 0
        counter2 = 0
        for y in range(0, Y):
            for x in range(0, X):
                if str1[y][x] in [3, 4]:
                    self.coord_to_node1[(y, x)] = counter1
                    counter1 += 1
                if str2[y][x] in [3, 4]:
                    self.coord_to_node2[(y, x)] = counter2
                    counter2 += 1

    def __call__(self, x, y, mid, axis, key) -> Any:

        if axis == 0:
            if y < mid:
                node_num = self.coord_to_node1[(y, x)]
                return self.params1[key][node_num]
            else:
                node_num = self.coord_to_node2[(y, x)]
                return self.params2[key][node_num]
        else:
            if x < mid:
                node_num = self.coord_to_node1[(y, x)]
                return self.params1[key][node_num]
            else:
                node_num = self.coord_to_node2[(y, x)]
                return self.params2[key][node_num]

class GetParamState:

    def __init__(self, str1, str2, params1, params2, env_name) -> None:

        self.params1 = torch.transpose(params1['base.actor.0.weight'], 0, 1)
        self.params2 = torch.transpose(params2['base.actor.0.weight'], 0, 1)

        mpo1 = mass_point_order(str1)
        mpo2 = mass_point_order(str2)

        self.coord_to_node1 = {}
        self.coord_to_node2 = {}

        overhead = get_over_head(env_name)
        
        node_num = overhead
        for y, x in mpo1:
            self.coord_to_node1[(y, x)] = (node_num, node_num + len(mpo1))
            node_num += 1

        node_num = overhead
        for y, x in mpo2:
            self.coord_to_node2[(y, x)] = (node_num, node_num + len(mpo2))
            node_num += 1

        assert overhead + 2 * len(self.coord_to_node1) == self.params1.shape[0]
        assert overhead + 2 * len(self.coord_to_node2) == self.params2.shape[0]

    def non_coordinative_param(self, node_num):

        parent_to_inherit = np.random.choice([1, 2])
        if parent_to_inherit == 1:
            return self.params1[node_num]
        else:
            return self.params2[node_num]

    def __call__(self, x, y, axis, mid, index) -> Any:
        
        assert axis in [0, 1]
        assert index in [0, 1]
        
        if axis == 0:
            if y < mid:
                parent_to_inherit = 1
            elif y > mid:
                parent_to_inherit = 2
            else:
                if not (y, x) in self.coord_to_node1:
                    parent_to_inherit = 2
                elif not (y, x) in self.coord_to_node2:
                    parent_to_inherit = 1
                else:
                    parent_to_inherit = np.random.choice([1, 2])
        else:
            if x < mid:
                parent_to_inherit = 1
            elif x > mid:
                parent_to_inherit = 2
            else:
                if not (y, x) in self.coord_to_node1:
                    parent_to_inherit = 2
                elif not (y, x) in self.coord_to_node2:
                    parent_to_inherit = 1
                else:
                    parent_to_inherit = np.random.choice([1, 2])

        if parent_to_inherit == 1:
                node_num = self.coord_to_node1[(y, x)][index]
                return self.params1[node_num]
        
        elif parent_to_inherit == 2:
            node_num = self.coord_to_node2[(y, x)][index]
            return self.params2[node_num]
        
        else:
            raise ValueError(f"parent_to_inherit must be in [1, 2]. now {parent_to_inherit}")

def load(exp_dir, generation, id):

    while not os.path.exists(f'{exp_dir}/generation{str(generation).zfill(2)}/id{str(id).zfill(2)}'):
        generation -= 1

    robot_dir = f'{exp_dir}/generation{str(generation).zfill(2)}/id{str(id).zfill(2)}/robot_dir'
    structure = np.load(os.path.join(robot_dir, 'structure.npy'))
    parameter = torch.load(os.path.join(robot_dir, 'parameter.pt'), map_location='cpu')

    return structure, parameter

def crossover(exp_dir, generation, env_name:str, parent1_id, parent2_id):

    str1, (actor_critic1, _) = load(exp_dir, generation, parent1_id)
    str2, (actor_critic2, _) = load(exp_dir, generation, parent2_id)

    print(str1)
    print(str2)

    X = str1.shape[0]
    Y = str1.shape[1]

    count = 100

    while True:

        if count == 0:
            print('failed in connecting bodies')
            return None

        axis = np.random.choice([0, 1])
        axis = 1
        
        if axis == 0:
            mid = np.random.choice([y for y in range(1, Y)])
        else:
            mid = np.random.choice([x for x in range(1, X)])
        mid = 3

        if axis == 0:
            child_st = np.concatenate((str1[:mid], str2[mid:]), axis)
        else:
            child_st = np.concatenate((str1[:,:mid], str2[:,mid:]), axis)

        if is_connected(child_st):
            break

        count -= 1
    
    params1 = actor_critic1.state_dict()
    params2 = actor_critic2.state_dict()

    child_conn = get_full_connectivity(child_st)

    env = make_vec_envs(
        env_name=env_name,
        robot_structure=(child_st, child_conn),
        seed=1000,
        num_processes=1,
        gamma=None,
        log_dir=None,
        device='cpu',
        allow_early_resets=False
        )
    
    child_actor_critic = Policy(
        env.observation_space.shape,
        env.action_space,
        base_kwargs={'recurrent': False}
    )
    
    child_params = child_actor_critic.state_dict()

    # -----------------------------
    # copy weights of actor net
    # -----------------------------

    # copy weights of last layer
    get_param_a = GetParamAction(str1, str2, params1, params2)
    node_num = 0

    for y in range(0, Y):
        for x in range(0, X):
            if child_st[y][x] in [3, 4]:
                child_params['dist.fc_mean.weight'][node_num] = get_param_a(x, y, mid, axis, 'dist.fc_mean.weight')
                child_params['dist.fc_mean.bias'][node_num] = get_param_a(x, y, mid, axis, 'dist.fc_mean.bias')
                child_params['dist.logstd._bias'][node_num] = get_param_a(x, y, mid, axis, 'dist.logstd._bias')
                node_num += 1

    # copy weights of first layer
    overhead = get_over_head(env_name=env_name)
    mpo_c = mass_point_order(child_st)

    try:
        get_param_s = GetParamState(str1, str2, params1, params2, env_name)
    except:
        print('Something went wrong in GetParmState')
        return None

    child_params['base.actor.0.weight'] = torch.transpose(child_params['base.actor.0.weight'], 0, 1)
    node_num = 0
    while node_num < overhead:
        child_params['base.actor.0.weight'][node_num] = get_param_s.non_coordinative_param(node_num)
        node_num += 1
    for y, x in mpo_c:
        child_params['base.actor.0.weight'][node_num] = get_param_s(x, y, axis, mid, 0)
        node_num += 1
    for y, x in mpo_c:
        child_params['base.actor.0.weight'][node_num] = get_param_s(x, y, axis, mid, 1)
        node_num += 1

    if overhead + len(mpo_c) * 2 != child_params['base.actor.0.weight'].shape[0]:
        return None

    child_params['base.actor.0.weight'] = torch.transpose(child_params['base.actor.0.weight'], 0, 1)
    
    # copy weights of middle layer
    for i in range(64):
        for j in range(64):
            if np.random.random() < 0.5:
                child_params['base.actor.2.weight'][i][j] = params1['base.actor.2.weight'][i][j]
            else:
                child_params['base.actor.2.weight'][i][j] = params2['base.actor.2.weight'][i][j]
    
    for i in range(64):
        if np.random.random() < 0.5:
                child_params['base.actor.2.bias'][i] = params1['base.actor.2.bias'][i]
        else:
            child_params['base.actor.2.bias'][i] = params2['base.actor.2.bias'][i]

    for i in range(64):
        if np.random.random() < 0.5:
                child_params['base.actor.0.bias'][i] = params1['base.actor.0.bias'][i]
        else:
            child_params['base.actor.0.bias'][i] = params2['base.actor.0.bias'][i]
    
    child_actor_critic.load_state_dict(child_params)
    
    return child_st, child_actor_critic, axis, mid

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_dir', type=str)
    parser.add_argument('--generation', type=int)
    parser.add_argument('--id1', type=int)
    parser.add_argument('--id2', type=int)
    parser.add_argument('--env_name', type=str, default='BridgeWalker-v0')
    parser.add_argument('--movie_path', type=str, default=None)

    args = parser.parse_args()

    result = crossover(
        exp_dir=args.exp_dir,
        generation=args.generation,
        env_name=args.env_name,
        parent1_id=args.id1,
        parent2_id=args.id2
    )

    if result is None:
        print('cannot crossover.')
        return
    
    structure, actor_critic, axis, mid = result

    print(f'axis={axis}, mid={mid}')

    visualize(
        structure=structure,
        connections=get_full_connectivity(structure),
        env_name=args.env_name,
        actor_critic=actor_critic,
        obs_rm=None,
        movie_path=args.movie_path
    )


if __name__ == '__main__':
    main()
