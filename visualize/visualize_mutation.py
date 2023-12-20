import argparse

import evogym.envs

import sys
sys.path.append('.')

from utils.ga_utils import load
from ga.reproduction import mutate_structure
from ga.inherit import inherit_controller_mutation
from ppo.envs import make_vec_envs
from utils.config import Config
from visualize import visualize


def main() -> None:
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-ex', '--exp_dir', type=str)
    parser.add_argument('-g', "--generation", type=int)
    parser.add_argument('-i', '--id', type=int)
    parser.add_argument('-en', '--env_name', type=str)
    parser.add_argument('-m', '--movie_path', type=str, default=None)
    parser.add_argument('-n','--num_evals', type=int, default=1)
    
    args = parser.parse_args()

    exp_dir = args.exp_dir
    generation = args.generation
    id = args.id
    Config.env_name = args.env_name

    (body, _connections), (actor_critic, _obs_rms) = load(exp_dir, generation, id)

    mutated_structure = mutate_structure(body)

    envs = make_vec_envs(
        env_name=Config.env_name,
        robot_structure=mutated_structure,
        seed=1000,
        num_processes=1,
        gamma=None,
        log_dir=None,
        device='cpu',
        allow_early_resets=False
    )

    actor_critic = inherit_controller_mutation(
        parent_body=body,
        parent_actor_critic=actor_critic,
        child_body=mutated_structure[0],
        child_observation_space_shape=envs.observation_space.shape,
        child_action_space=envs.action_space
    )

    visualize(
        structure=mutated_structure[0],
        connections=mutated_structure[1],
        actor_critic=actor_critic,
        envs=envs,
        movie_path=args.movie_path,
        num_evals=args.num_evals,
    )
    

if __name__ == "__main__":
    main()