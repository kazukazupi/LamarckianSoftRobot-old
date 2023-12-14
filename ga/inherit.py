import numpy as np
import torch

from a2c_ppo_acktr.model import Policy
from ga.structure_analize import get_mapping_table_action, get_mapping_table_state


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