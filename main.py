import argparse

from utils.ga_utils import load
from ga.reproduction import get_mass_point_in_order, get_over_head

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

    overhead = get_over_head(args.env_name)

    mpio = get_mass_point_in_order(body, connections)
    print(len(mpio))
    print(actor_critic.state_dict()['base.actor.0.weight'].shape[1])
    assert overhead + 2 * len(mpio) == actor_critic.state_dict()['base.actor.0.weight'].shape[1]
    

if __name__ == "__main__":
    main()