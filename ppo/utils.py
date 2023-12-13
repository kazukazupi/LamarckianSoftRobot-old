import os
import copy

import numpy as np
import torch
import torch.nn as nn

from ppo.envs import VecNormalize

# Derived from
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail

class TerminationCondition():

    def __init__(self, max_iters):
        self.max_iters = max_iters

    def __call__(self, iters):
        return iters >= self.max_iters

    def change_target(self, max_iters):
        self.max_iters = max_iters

# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir):

    try:
        os.makedirs(log_dir)
    except:
        pass
        # files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        # for f in files:
        #     os.remove(f)


def get_mapping_table_action(structure_s:np.ndarray, structure_t:np.ndarray):

    actuator_coordinates_s = np.stack(np.where(structure_s >= 3), axis=-1)
    actuator_coordinates_t = np.stack(np.where(structure_t >= 3), axis=-1)

    mapping_table = []

    for coordinate_s in actuator_coordinates_s:
        result = np.transpose((actuator_coordinates_t - coordinate_s)) == 0 # compare with each x,y coordinate
        result = result[0] & result[1]
        result = np.where(result == True)
        if len(result[0]) > 0: mapping_table.append(result[0][0])
        else: mapping_table.append(-1)
    
    return mapping_table


# Return the coordinates of the mass point in order from top-left to bottom-right.
# TO DO: 現状、connection を考慮できていない。
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


def get_mapping_table_state(structure_s, structure_t, env_name = 'Walker-v0'):
    
    mass_point_order_s = mass_point_order(structure_s)
    mass_point_order_t = mass_point_order(structure_t)

    """
    mapping_table[i] = j indicates
    target structure's i-th mass point corresponds to
    source structure's j-th mass point.
    """
    mapping_table = []

    for mass_point in mass_point_order_t:
        try:
            mapping_table.append(mass_point_order_s.index(mass_point))
        except ValueError:
            mapping_table.append(-1)

    # here needs to be revised depending on the task
    if env_name == 'Walker-v0':
        return [0, 1] + list(map(lambda x: -1 if (x==-1) else x+2, mapping_table)) + list(map(lambda x: -1 if (x==-1) else x+2+len(mass_point_order_s), mapping_table))
    elif env_name == 'BridgeWalker-v0':
        return [0, 1, 2] + list(map(lambda x: -1 if (x==-1) else x+3, mapping_table)) + list(map(lambda x: -1 if (x==-1) else x+3+len(mass_point_order_s), mapping_table))
    elif env_name == 'BidirectionalWalker-v0':
        result = [0, 1] + list(map(lambda x: -1 if (x==-1) else x+2, mapping_table)) + list(map(lambda x: -1 if (x==-1) else x+2+len(mass_point_order_s), mapping_table))
        for i in range(3): result.append(i + 2 + 2 * len(mass_point_order_s))
        return result
    elif env_name in ['Carrier-v0', 'Carrier-v1', 'Pusher-v0']:
        result = [0, 1] + list(map(lambda x: -1 if (x==-1) else x+2, mapping_table)) + list(map(lambda x: -1 if (x==-1) else x+2+len(mass_point_order_s), mapping_table))
        for i in range(4): result.append(i + 2 + 2 * len(mass_point_order_s))
        return result
    elif env_name in ['ObstacleTraverser-v0', 'ObstacleTraverser-v1']:
        result = [0, 1, 2] + list(map(lambda x: -1 if (x==-1) else x+3, mapping_table)) + list(map(lambda x: -1 if (x==-1) else x+3+len(mass_point_order_s), mapping_table))
        for i in range(11): result.append(i + 3 + 2 * len(mass_point_order_s))
        return result
    else:
        raise NotImplementedError(f'func get_mapping_table_state is not implemeted for {env_name}')


def inherit_policy(child_actor_critic, child_structure:np.ndarray, parent_robot_dir:str, env_name='Walker-v0'):

    parent_structure = np.load(os.path.join(parent_robot_dir, "structure.npy"))
    mapping_table_action = get_mapping_table_action(parent_structure, child_structure)
    mapping_table_state = get_mapping_table_state(parent_structure, child_structure, env_name)
    
    # parent's parameter
    parent_actor_critic, parent_obs_rms = torch.load(os.path.join(parent_robot_dir, 'parameter.pt'), map_location=lambda storage, loc: storage)
    parent_dict = parent_actor_critic.state_dict()
    # child's parameter
    child_dict = copy.deepcopy(child_actor_critic.state_dict())

    # -----------------------------
    # copy weights of actor net
    # -----------------------------

    # copy weights of first layer
    child_dict['base.actor.0.weight'] = torch.transpose(child_dict['base.actor.0.weight'], 0, 1)
    parent_dict['base.actor.0.weight'] = torch.transpose(parent_dict['base.actor.0.weight'], 0, 1)
    for index_t, index_s in enumerate(mapping_table_state):
        if index_t == -1: continue
        child_dict['base.actor.0.weight'][index_t] = parent_dict['base.actor.0.weight'][index_s]
    child_dict['base.actor.0.weight'] = torch.transpose(child_dict['base.actor.0.weight'], 0, 1)

    # copy weights of last layer
    for index_s, index_t in enumerate(mapping_table_action):
        if index_t == -1: continue
        child_dict['dist.fc_mean.weight'][index_t] = parent_dict['dist.fc_mean.weight'][index_s]
        child_dict['dist.fc_mean.bias'][index_t] = parent_dict['dist.fc_mean.bias'][index_s]
        child_dict['dist.logstd._bias'][index_t] = parent_dict['dist.logstd._bias'][index_s]

    # copy weights of middle layer
    for key in ['base.actor.2.weight', 'base.actor.2.bias', 'base.actor.0.bias']:
        child_dict[key] = copy.deepcopy(parent_dict[key])
    
    # -----------------------------
    # copy weights of critic net
    # -----------------------------

    # copy weights of first layer
    child_dict['base.critic.0.weight'] = torch.transpose(child_dict['base.critic.0.weight'], 0, 1)
    parent_dict['base.critic.0.weight'] = torch.transpose(parent_dict['base.critic.0.weight'], 0, 1)
    for index_t, index_s in enumerate(mapping_table_state):
        if index_t == -1: continue
        child_dict['base.critic.0.weight'][index_t] = parent_dict['base.critic.0.weight'][index_s]
    child_dict['base.critic.0.weight'] = torch.transpose(child_dict['base.critic.0.weight'], 0, 1)

    # copy weights of middle, last layer
    for key in parent_dict.keys():
        if ('critic' in key) and (key != 'base.critic.0.weight') : child_dict[key] = copy.deepcopy(parent_dict[key])

    child_actor_critic.load_state_dict(child_dict)

    # # obs_rmsも変更する必要あり
    # print()
    # dim_obs_space = 53
    # child_obs_rms = copy.deepcopy(parent_obs_rms)
    # child_obs_rms.mean = np.full(shape=dim_obs_space, fill_value=np.mean(parent_obs_rms.mean))
    # child_obs_rms.var = np.full(shape=dim_obs_space, fill_value=np.mean(parent_obs_rms.var))

    # return child_obs_rms
