import argparse
import torch

class Config:

    env_name = "Walker-v0"
    shape = (5, 5)
    max_evaluations = 250
    population_size = 25
    mutation_rate = 0.1
    elite_rate_high = 0.6
    elite_rate_low = 0.1
    inherit_en = False
    print_en = True
    algo = 'ppo'
    gail = False
    gail_experts_dir = './gail_experts'
    gail_batch_size = 128
    gail_epoch = 5
    lr = 2.5e-4
    eps = 1e-5
    alpha = 0.99
    gamma = 0.99
    use_gae = True
    gae_lambda = 0.95
    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_grad_norm = 0.5
    seed = 1
    cuda_deterministic = False
    max_iters = 1000
    num_processes = 4
    num_steps = 128
    ppo_epoch = 4
    num_mini_batch = 4
    clip_param = 0.1
    log_interval = 10
    save_interval = 100
    num_evals = 1
    eval_interval = 50
    num_env_steps = 10e6
    log_dir = '/tmp/gym/'
    save_dir = './trained_models/'
    use_proper_time_limits = False
    recurrent_policy = False
    use_linear_lr_decay = True

    cuda = torch.cuda.is_available()

    assert algo in ['a2c', 'ppo', 'acktr']
    # assert (args.from_middle == False or args.from_gen1 == False)
    # if args.from_gen1:
    #     assert args.gen0_experiment_dir is not None
    if recurrent_policy:
        assert algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'
