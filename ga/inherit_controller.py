import numpy as np
import os
import torch

from a2c_ppo_acktr.model import Policy
from ga.analyze_structure import get_overhead, get_overtail, get_mapping_table_action, get_mapping_table_state, get_mass_point_in_order_with_count, GetParamAction, GetParamState
from utils.config import Config

def get_controller(
        body: np.ndarray,
        observation_space_shape:tuple,
        action_space,
        parents,
        crossover_info=None
) -> Policy:
    
    # parents: list[Individual]
    
    # inherit is not allowed or has no parents
    if (not Config.inherit_en) or len(parents) == 0:

        actor_critic = Policy(
            obs_shape=observation_space_shape,
            action_space=action_space
        )
    
    # inherit is allowed and is emerged from mutation
    elif len(parents) == 1:

        parent = parents[0]
        parent_actor_critic = torch.load(
            os.path.join(parent.saving_dir, 'actor_critic.pt'),
            map_location='cpu'
        )[0]

        actor_critic = inherit_controller_mutation(
            parent_body=parent.body,
            parent_actor_critic=parent_actor_critic,
            child_body=body,
            child_observation_space_shape=observation_space_shape,
            child_action_space=action_space
        )

    # inherit is allowed and is emerged from crossover
    elif len(parents) == 2:

        assert crossover_info is not None
        axis = crossover_info['axis']
        mid = crossover_info['mid']

        parent1 = parents[0]
        parent2 = parents[1]

        parent1_actor_critic = torch.load(
            os.path.join(parent1.saving_dir, 'actor_critic.pt'),
            map_location='cpu'
        )[0]

        parent2_actor_critic = torch.load(
            os.path.join(parent2.saving_dir, 'actor_critic.pt'),
            map_location='cpu'
        )[0]

        actor_critic = inherit_controller_crossover(
            child_body=body,
            axis=axis,
            mid=mid,
            parent1_body=parent1.body,
            parent2_body=parent2.body,
            parent1_actor_critic=parent1_actor_critic,
            parent2_actor_critic=parent2_actor_critic,
            child_observation_space_shape=observation_space_shape,
            child_action_space=action_space
        )

    return actor_critic


def inherit_controller_mutation(
        parent_body: np.ndarray,
        parent_actor_critic: Policy,
        child_body: np.ndarray,
        child_observation_space_shape:tuple,
        child_action_space
) -> Policy:

    child_actor_critic = Policy(
        obs_shape=child_observation_space_shape,
        action_space=child_action_space
    )

    child_state_dict = child_actor_critic.state_dict()
    parent_state_dict = parent_actor_critic.state_dict()

    # i-th parent actor NN node corresponds to
    # j-th child actor NN node (where j = mapping_table[i])
    mapping_table_state = get_mapping_table_state(child_body, parent_body)
    mapping_table_action = get_mapping_table_action(parent_body, child_body)

    assert len(mapping_table_state) == parent_state_dict['base.critic.0.weight'].shape[1]
    assert len(mapping_table_action) == parent_state_dict['dist.fc_mean.weight'].shape[0]
    assert max(mapping_table_state) < child_state_dict['base.critic.0.weight'].shape[1]
    assert max(mapping_table_action) < child_state_dict['dist.fc_mean.weight'].shape[0]

    # ---------------------------
    #  copy weights of actor net
    # ---------------------------

    # copy weights of first layer
    child_state_dict['base.actor.0.weight'] = torch.transpose(child_state_dict['base.actor.0.weight'], 0, 1)
    parent_state_dict['base.actor.0.weight'] = torch.transpose(parent_state_dict['base.actor.0.weight'], 0, 1)
    for index_p, index_c in enumerate(mapping_table_state):
        if index_c == -1: continue
        child_state_dict['base.actor.0.weight'][index_c] = parent_state_dict['base.actor.0.weight'][index_p]
    child_state_dict['base.actor.0.weight'] = torch.transpose(child_state_dict['base.actor.0.weight'], 0, 1)
    
    # copy weights of the last layer
    for index_p, index_c in enumerate(mapping_table_action):
        if index_c == -1: continue
        child_state_dict['dist.fc_mean.weight'][index_c] = parent_state_dict['dist.fc_mean.weight'][index_p]
        child_state_dict['dist.fc_mean.bias'][index_c] = parent_state_dict['dist.fc_mean.bias'][index_p]
        child_state_dict['dist.logstd._bias'][index_c] = parent_state_dict['dist.logstd._bias'][index_p]

    # copy weights of the middle layer
    for key in ['base.actor.2.weight', 'base.actor.2.bias', 'base.actor.0.bias']:
        child_state_dict[key] = parent_state_dict[key]

    # ---------------------------
    #  copy weights of critic net
    # ---------------------------

    # copy weights of first layer
    child_state_dict['base.critic.0.weight'] = torch.transpose(child_state_dict['base.critic.0.weight'], 0, 1)
    parent_state_dict['base.critic.0.weight'] = torch.transpose(parent_state_dict['base.critic.0.weight'], 0, 1)
    for index_p, index_c in enumerate(mapping_table_state):
        if index_c == -1: continue
        child_state_dict['base.critic.0.weight'][index_c] = parent_state_dict['base.critic.0.weight'][index_p]
    child_state_dict['base.critic.0.weight'] = torch.transpose(child_state_dict['base.critic.0.weight'], 0, 1)

    # copy weights of middle, last layer
    for key in parent_state_dict.keys():
        if ('critic' in key) and (key != 'base.critic.0.weight') : child_state_dict[key] = parent_state_dict[key]

    child_actor_critic.load_state_dict(child_state_dict)

    return child_actor_critic

def inherit_controller_crossover(
        child_body:np.ndarray,
        axis:int,
        mid:int,
        parent1_body:np.ndarray,
        parent2_body:np.ndarray,
        parent1_actor_critic:Policy,
        parent2_actor_critic:Policy,
        child_observation_space_shape:tuple,
        child_action_space):

    X = parent1_body.shape[0]
    Y = parent1_body.shape[1]
    
    parent1_state_dict = parent1_actor_critic.state_dict()
    parent2_state_dict = parent2_actor_critic.state_dict()
    
    child_actor_critic = Policy(
        child_observation_space_shape,
        child_action_space,
        base_kwargs={'recurrent': False}
    )
    child_params = child_actor_critic.state_dict()

    # -----------------------------
    # copy weights of actor net
    # -----------------------------

    # copy weights of first layer
    overhead = get_overhead()
    overtail = get_overtail()
    child_mpio_with_count = get_mass_point_in_order_with_count(child_body)

    get_param_s = GetParamState(
        body1=parent1_body,
        body2=parent2_body,
        params1=parent1_actor_critic.state_dict(),
        params2=parent2_actor_critic.state_dict()
    )

    child_params['base.actor.0.weight'] = torch.transpose(child_params['base.actor.0.weight'], 0, 1)
    node_num = 0
    
    while node_num < overhead: # copy weight of connections from nodes receiving non-coordinative information
        child_params['base.actor.0.weight'][node_num] = get_param_s.get_head_param(node_num, 'base.actor.0.weight')
        node_num += 1
    for mass_point_with_count in child_mpio_with_count: # copy weight of connections from nodes receiving coordinative information
        child_params['base.actor.0.weight'][node_num] = get_param_s(mass_point_with_count, axis, mid, 0, 'base.actor.0.weight')
        node_num += 1
    for mass_point_with_count in child_mpio_with_count: # copy weight of connections from nodes receiving coordinative information
        child_params['base.actor.0.weight'][node_num] = get_param_s(mass_point_with_count, axis, mid, 1, 'base.actor.0.weight')
        node_num += 1
    for i in range(overtail): # copy weight of connections from nodes receiving non-coordinative information
        child_params['base.actor.0.weight'][i + overhead + len(child_mpio_with_count) * 2] = get_param_s.get_tail_param(i, 'base.actor.0.weight')
    
    assert overhead + overtail + len(child_mpio_with_count) * 2 == child_params['base.actor.0.weight'].shape[0]

    child_params['base.actor.0.weight'] = torch.transpose(child_params['base.actor.0.weight'], 0, 1)

    # copy weights of middle layer
    for i in range(64):
        for j in range(64):
            if np.random.random() < 0.5:
                child_params['base.actor.2.weight'][i][j] = parent1_state_dict['base.actor.2.weight'][i][j]
            else:
                child_params['base.actor.2.weight'][i][j] = parent2_state_dict['base.actor.2.weight'][i][j]
    
    for i in range(64):
        if np.random.random() < 0.5:
                child_params['base.actor.2.bias'][i] = parent1_state_dict['base.actor.2.bias'][i]
        else:
            child_params['base.actor.2.bias'][i] = parent2_state_dict['base.actor.2.bias'][i]

    for i in range(64):
        if np.random.random() < 0.5:
                child_params['base.actor.0.bias'][i] = parent1_state_dict['base.actor.0.bias'][i]
        else:
            child_params['base.actor.0.bias'][i] = parent2_state_dict['base.actor.0.bias'][i]

    # copy weights of last layer
    get_param_a = GetParamAction(parent1_body, parent2_body, parent1_state_dict, parent2_state_dict)
    node_num = 0

    for y in range(0, Y):
        for x in range(0, X):
            if child_body[y][x] in [3, 4]:
                child_params['dist.fc_mean.weight'][node_num] = get_param_a(x, y, mid, axis, 'dist.fc_mean.weight')
                child_params['dist.fc_mean.bias'][node_num] = get_param_a(x, y, mid, axis, 'dist.fc_mean.bias')
                child_params['dist.logstd._bias'][node_num] = get_param_a(x, y, mid, axis, 'dist.logstd._bias')
                node_num += 1

    # -----------------------------
    # copy weights of actor net
    # -----------------------------

    # copy weights of first layer
    child_params['base.critic.0.weight'] = torch.transpose(child_params['base.critic.0.weight'], 0, 1)
    node_num = 0
    while node_num < overhead:
        child_params['base.critic.0.weight'][node_num] = get_param_s.get_head_param(node_num, 'base.critic.0.weight')
        node_num += 1
    for mass_point_with_count in child_mpio_with_count:
        child_params['base.critic.0.weight'][node_num] = get_param_s(mass_point_with_count, axis, mid, 0, 'base.critic.0.weight')
        node_num += 1
    for mass_point_with_count in child_mpio_with_count:
        child_params['base.critic.0.weight'][node_num] = get_param_s(mass_point_with_count, axis, mid, 1, 'base.critic.0.weight')
        node_num += 1
    for i in range(overtail):
        child_params['base.critic.0.weight'][i + overhead + len(child_mpio_with_count) * 2] = get_param_s.get_tail_param(i, 'base.critic.0.weight')
    
    assert overhead + overtail + len(child_mpio_with_count) * 2 == child_params['base.critic.0.weight'].shape[0]
    child_params['base.critic.0.weight'] = torch.transpose(child_params['base.critic.0.weight'], 0, 1)

    # copy weights of middle layer
    for i in range(64):
        for j in range(64):
            if np.random.random() < 0.5:
                child_params['base.critic.2.weight'][i][j] = parent1_state_dict['base.critic.2.weight'][i][j]
            else:
                child_params['base.critic.2.weight'][i][j] = parent2_state_dict['base.critic.2.weight'][i][j]
    
    for i in range(64):
        if np.random.random() < 0.5:
                child_params['base.critic.2.bias'][i] = parent1_state_dict['base.critic.2.bias'][i]
        else:
            child_params['base.critic.2.bias'][i] = parent2_state_dict['base.critic.2.bias'][i]

    for i in range(64):
        if np.random.random() < 0.5:
                child_params['base.critic.0.bias'][i] = parent1_state_dict['base.critic.0.bias'][i]
        else:
            child_params['base.critic.0.bias'][i] = parent2_state_dict['base.critic.0.bias'][i]

    # copy weights of last layer
    for i in range(64):
        if np.random.random() < 0.5:
            child_params['base.critic_linear.weight'][0][i] = parent1_state_dict['base.critic_linear.weight'][0][i]
        else:
            child_params['base.critic_linear.weight'][0][i] = parent2_state_dict['base.critic_linear.weight'][0][i]

    if np.random.random() < 0.5:
        child_params['base.critic_linear.bias'] = parent1_state_dict['base.critic_linear.bias']
    else:
        child_params['base.critic_linear.bias'] = parent2_state_dict['base.critic_linear.bias']

    # return
    child_actor_critic.load_state_dict(child_params)
    return child_actor_critic
