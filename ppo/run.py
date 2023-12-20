# import os, sys
import os
# sys.path.insert(1, os.path.join(sys.path[0], 'externals', 'pytorch_a2c_ppo_acktr_gail'))

import numpy as np
import time
import csv
from collections import deque
import torch

from ppo import utils
from ppo.evaluate import evaluate
from ppo.envs import make_vec_envs


from a2c_ppo_acktr import algo
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage

# Derived from
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail

def run_ppo(
    structure:np.ndarray,
    connections:np.ndarray,
    saving_dir:str,
    parent_robot_dir:str,
    args,
    en_print=True):

    termination_condition = utils.TerminationCondition(args.max_iters)

    if en_print:
        print(f'Starting training on \n{structure}\nat {saving_dir}...\n')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # ----------------------
    #  saving information
    # ----------------------

    # make saving_dir
    assert(not os.path.exists(saving_dir))
    os.makedirs(saving_dir)

    # save log at saving_dir/log.csv
    csv_f_path = os.path.join(saving_dir, 'log.csv')
    if en_print:
        print("write log at " + csv_f_path)
    with open(csv_f_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Updates', 'num episodes', 'num tipesteps', 'mean', 'median', 'min', 'max'])

    # save robot info at saving_dir/robot_dir
    robot_dir_path = os.path.join(saving_dir, 'robot_dir')
    assert(not os.path.exists(robot_dir_path))
    os.makedirs(robot_dir_path)
    np.save(os.path.join(robot_dir_path, 'structure.npy'), structure)
    np.save(os.path.join(robot_dir_path, 'connections.npy'), connections)
    parameter_f_path = os.path.join(robot_dir_path, 'parameter.pt')

    if en_print:
        print(f'save parameter at {parameter_f_path}')

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, (structure, connections), args.seed, args.num_processes,
                         args.gamma, robot_dir_path, device, False)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    if parent_robot_dir is not None:
        assert(args.inherit_flag)
        utils.inherit_policy(actor_critic, structure, parent_robot_dir, args.env_name)
        if en_print: print(f'inherit policy from {parent_robot_dir}')
    actor_critic.to(device)

    if args.algo == 'a2c':
        print('Warning: this code has only been tested with ppo.')
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        print('Warning: this code has only been tested with ppo.')
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))

        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4, subsample_frequency=20)
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    num_episdoes = 0

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    rewards_tracker = []
    avg_rewards_tracker = []
    sliding_window_size = 10
    max_determ_avg_reward = float('-inf')

    # end when (j == min{tc.max_iters, num_updates})
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            # track rewards
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    rewards_tracker.append(info['episode']['r'])
                    num_episdoes += 1
                    if len(rewards_tracker) < 10:
                        avg_rewards_tracker.append(np.average(np.array(rewards_tracker)))
                    else:
                        avg_rewards_tracker.append(np.average(np.array(rewards_tracker[-10:])))

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()
     
        # print status
        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            if en_print:
                print(
                    "Updates {}, num episodes {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                        .format(j, num_episdoes, total_num_steps,
                                int(total_num_steps / (end - start)),
                                len(episode_rewards), np.mean(episode_rewards),
                                np.median(episode_rewards), np.min(episode_rewards),
                                np.max(episode_rewards), dist_entropy, value_loss,
                                action_loss))
            
            with open(csv_f_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([j, num_episdoes, total_num_steps, round(np.mean(episode_rewards), 2), round(np.median(episode_rewards), 2), round(np.min(episode_rewards), 2), round(np.max(episode_rewards), 2)])
        
        # evaluate the controller and save it if it does the best so far
        if (len(episode_rewards) > 1 and (j % args.eval_interval == 0 or termination_condition(j))):
            
            obs_rms = utils.get_vec_normalize(envs).obs_rms
            determ_avg_reward = evaluate(args.num_evals, actor_critic, obs_rms, args.env_name, (structure, connections), args.seed,
                     args.num_processes, robot_dir_path, device)

            if en_print:
                print(f'Evaluated using {args.num_evals} episodes. Mean reward: {np.mean(determ_avg_reward)}\n')

            if determ_avg_reward > max_determ_avg_reward:
                max_determ_avg_reward = determ_avg_reward              
                
                if en_print: print(f'Saving {parameter_f_path} with avg reward {max_determ_avg_reward}\n')
                torch.save([
                    actor_critic,
                    getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
                ], parameter_f_path)

        # return upon reaching the termination condition
        if not termination_condition == None:
            if termination_condition(j):
                if en_print: print(f'met termination condition ({j})...terminating...\n')
                return max_determ_avg_reward

#python ppo_main_test.py --env-name "roboticgamedesign-v0" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 1 --num-steps 128 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01
#python ppo.py --env-name "roboticgamedesign-v0" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01 --log-dir "logs/"
