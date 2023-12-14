import glob
import os
import sys

sys.path.append('.')

from ga.inherit import get_mass_point_in_order, get_overhead, get_overtail
from utils.ga_utils import load_from_robot_dir
from utils.config import Config

def get_env_name(robot_dir:str):
    envs = ["BridgeWalker-v0", 'ObstacleTraverser-v0', 'ObstacleTraverser-v1', 'BidirectionalWalker-v0', 'Carrier-v0', 'Carrier-v1', 'Pusher-v0']
    for env in envs:
        if env in robot_dir:
            return env
        
    raise ValueError

def main():
    
    exp_dir = sys.argv[1]
    
    robot_dirs = sorted(glob.glob(os.path.join(exp_dir, '*/*/generation*/id*/robot_dir')))
    
    for robot_dir in robot_dirs:
        env_name = get_env_name(robot_dir)
        Config.env_name = env_name
        
        (body, connections), (actor_critic, obs_rms) = load_from_robot_dir(robot_dir)
        overhead = get_overhead()
        overtail = get_overtail()

        mpio = get_mass_point_in_order(body)
        assert overhead + overtail + 2 * len(mpio) == actor_critic.state_dict()['base.actor.0.weight'].shape[1]
        print(robot_dir, " ok.")
            
if __name__ == '__main__':
    main()