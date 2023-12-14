import numpy as np
import torch
import cv2
import evogym.envs

import sys
sys.path.append('.')

from utils.config import Config
from ppo.envs import make_vec_envs
from ppo.utils import get_vec_normalize

def visualize(
        structure:np.ndarray,
        connections:np.ndarray,
        actor_critic,
        obs_rm=None,
        movie_path=None,
        num_evals=1,
        envs=None
):

    if envs is None:
        envs = make_vec_envs(env_name=Config.env_name,
                            robot_structure=(structure, connections),
                            seed=100,
                            num_processes=1,
                            gamma=None,
                            log_dir=None,
                            device='cpu',
                            allow_early_resets=True)
    
    vec_norm = get_vec_normalize(envs)
    if obs_rm is not None:
        vec_norm.obs_rms = obs_rm

    eval_episode_rewards = []

    recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
    masks = torch.zeros(1, 1)

    obs = envs.reset()

    sum_reward = 0

    if movie_path is not None:
        best_frames = None
    else:
        envs.render('screen')

    frames= []

    while len(eval_episode_rewards) < num_evals:

        with torch.no_grad():
            _, action, _, recurrent_hidden_states = actor_critic.act(
                obs,
                recurrent_hidden_states,
                masks,
                deterministic=True)
            
        # Obser reward and next obs
        obs, reward, done, infos = envs.step(action)

        sum_reward += reward

        if movie_path is not None:
            frame = envs.render(mode="img")
            cv2.putText(
                frame,
                f"culminative reward: {round(sum_reward.numpy().flatten()[0], 2)}",
                (20, 50),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0)
            )
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)
        else:
            envs.render('screen')
        

        masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device='cpu')
        
        for info in infos:
            if 'episode' in info.keys():
                if movie_path is not None and (best_frames is None or sum_reward > max(eval_episode_rewards)):
                    best_frames = frames
                frames = []
                eval_episode_rewards.append(sum_reward)
                print(f'sum_reward: {sum_reward}')
                sum_reward = 0

    envs.close()

    if movie_path is not None:
        print('Writing Video...')
        frame = best_frames[0]
        shape = (frame.shape[1], frame.shape[0])
        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
        writer = cv2.VideoWriter(movie_path, fourcc, 50, shape)

        for frame in best_frames:
            writer.write(frame)

        writer.release()
