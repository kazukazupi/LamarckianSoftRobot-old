import argparse

from utils.ga_utils import load
from ga.reproduction import mutate_structure
from ga.inherit import inherit_controller_mutation
from ppo.envs import make_vec_envs
from utils.config import Config
import evogym.envs

def main() -> None:
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-ex', '--exp_dir', type=str)
    parser.add_argument('-g', "--generation", type=int)
    parser.add_argument('-i', '--id', type=int)
    parser.add_argument('-en', '--env_name', type=str)
    
    args = parser.parse_args()

    exp_dir = args.exp_dir
    generation = args.generation
    id = args.id

    (body, connections), (actor_critic, obs_rms) = load(exp_dir, generation, id)

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

    inherit_controller_mutation(
        parent_body=body,
        parent_actor_critic=actor_critic,
        child_body=mutated_structure[0],
        child_observation_space_shape=envs.observation_space.shape,
        child_action_space=envs.action_space
    )
    

if __name__ == "__main__":
    main()