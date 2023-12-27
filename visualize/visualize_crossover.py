import argparse

import evogym.envs

import sys
sys.path.append('.')

from utils.utils import load
from ga.reproduction import crossover
from ga.inherit_controller import inherit_controller_crossover
from ppo.envs import make_vec_envs
from utils.config import Config
from visualize import visualize


def main() -> None:
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-ex', '--exp_dir', type=str)
    parser.add_argument('-en', '--env_name', type=str)
    parser.add_argument('-g', "--generation", type=int)
    parser.add_argument('--id1', type=int)
    parser.add_argument('--id2', type=int)
    parser.add_argument('-m', '--movie_path', type=str, default=None)
    parser.add_argument('-n','--num_evals', type=int, default=1)
    
    args = parser.parse_args()

    exp_dir = args.exp_dir
    generation = args.generation
    id1 = args.id1
    id2 = args.id2
    Config.env_name = args.env_name

    (body1, _), (actor_critic1, _) = load(exp_dir, generation, id1)
    (body2, _), (actor_critic2, _) = load(exp_dir, generation, id2)

    child_structure, (axis, mid) = crossover(body1, body2)

    envs = make_vec_envs(
        env_name=Config.env_name,
        robot_structure=child_structure,
        seed=1000,
        num_processes=1,
        gamma=None,
        log_dir=None,
        device='cpu',
        allow_early_resets=True
    )

    actor_critic = inherit_controller_crossover(
        child_body=child_structure[0],
        axis=axis,
        mid=mid,
        parent1_body=body1,
        parent2_body=body2,
        parent1_actor_critic=actor_critic1,
        parent2_actor_critic=actor_critic2,
        child_observation_space_shape=envs.observation_space.shape,
        child_action_space=envs.action_space
    )

    visualize(
        structure=child_structure[0],
        connections=child_structure[1],
        actor_critic=actor_critic,
        envs=envs,
        movie_path=args.movie_path,
        num_evals=args.num_evals,
    )
    

if __name__ == "__main__":
    main()