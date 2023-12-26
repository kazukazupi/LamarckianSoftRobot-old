import argparse
import json
import torch
import os


class Config:
        
    @classmethod
    def initialize(cls):

        parser = argparse.ArgumentParser()

        parser.add_argument(
            '--exp-dir', type=str, default='result/experiment'
        )

        # GA params
        parser.add_argument(
            '--shape', default=(5, 5))
        parser.add_argument(
            '--max-evaluations', type=int, default=250)
        parser.add_argument(
            '--population-size', type=int, default=25)
        parser.add_argument(
            '--mutation-rate', default=0.1)
        parser.add_argument(
            '--elite-rate-high', default=0.6)
        parser.add_argument(
            '--elite-rate-low', default=0.1)
        parser.add_argument(
            '--inherit-en', default=False)
        
        # PPO params
        parser.add_argument(
            '--print-en', default=False
        )
        parser.add_argument(
            '--algo', default='ppo', help='algorithm to use: a2c | ppo | acktr')
        parser.add_argument(
            '--gail',
            action='store_true',
            default=False,
            help='do imitation learning with gail')
        parser.add_argument(
            '--gail-experts-dir',
            default='./gail_experts',
            help='directory that contains expert demonstrations for gail')
        parser.add_argument(
            '--gail-batch-size',
            type=int,
            default=128,
            help='gail batch size (default: 128)')
        parser.add_argument(
            '--gail-epoch', type=int, default=5, help='gail epochs (default: 5)')
        parser.add_argument(
            '--lr', type=float, default=2.5e-4, help='learning rate (default: 2.5e-4)')
        parser.add_argument(
            '--eps',
            type=float,
            default=1e-5,
            help='RMSprop optimizer epsilon (default: 1e-5)')
        parser.add_argument(
            '--alpha',
            type=float,
            default=0.99,
            help='RMSprop optimizer apha (default: 0.99)')
        parser.add_argument(
            '--gamma',
            type=float,
            default=0.99,
            help='discount factor for rewards (default: 0.99)')
        parser.add_argument(
            '--use-gae',
            action='store_true',
            default=True,
            help='use generalized advantage estimation')
        parser.add_argument(
            '--gae-lambda',
            type=float,
            default=0.95,
            help='gae lambda parameter (default: 0.95)')
        parser.add_argument(
            '--entropy-coef',
            type=float,
            default=0.01,
            help='entropy term coefficient (default: 0.01)')
        parser.add_argument(
            '--value-loss-coef',
            type=float,
            default=0.5,
            help='value loss coefficient (default: 0.5)')
        parser.add_argument(
            '--max-grad-norm',
            type=float,
            default=0.5,
            help='max norm of gradients (default: 0.5)')
        parser.add_argument(
            '--seed', type=int, default=1, help='random seed (default: 1)')
        parser.add_argument(
            '--cuda-deterministic',
            action='store_true',
            default=False,
            help="sets flags for determinism when using CUDA (potentially slow!)")
        parser.add_argument(
            '--max-iters',
            type=int,
            default=1000)
        parser.add_argument(
            '--num-processes',
            type=int,
            # default=1,
            default=4,
            help='how many training CPU processes to use (default: 1)')
        parser.add_argument(
            '--num-steps',
            type=int,
            default=128,
            help='number of forward steps in A2C / num steps to use in PPO (default: 128)')
        parser.add_argument(
            '--ppo-epoch',
            type=int,
            default=4,
            help='number of ppo epochs (default: 4)')
        parser.add_argument(
            '--num-mini-batch',
            type=int,
            default=4,
            help='number of batches for ppo (default: 4)')
        parser.add_argument(
            '--clip-param',
            type=float,
            default=0.1,
            help='ppo clip parameter (default: 0.1)')
        parser.add_argument(
            '--log-interval',
            type=int,
            default=10,
            help='log interval, one log per n updates (default: 10)')
        parser.add_argument(
            '--save-interval',
            type=int,
            default=100,
            help='save interval, one save per n updates (default: 100)')
        parser.add_argument(
            '--num-evals',
            type=int,
            default=1,
            help='number of times to evaluate each controller (for evaluation purposes not training). (default: 1) as most Evolution Gym environments are deterministic.')
        parser.add_argument(
            '--eval-interval',
            type=int,
            # default=None,
            default=50,
            help='eval interval, one eval per n updates (default: None)')
        parser.add_argument(
            '--num-env-steps',
            type=int,
            default=10e6,
            help='number of environment steps to train (default: 10e6)')
        parser.add_argument(
            '--env-name',
            default='Walker-v0',
            help='environment to train on (default: roboticgamedesign-v0)')
        parser.add_argument(
            '--log-dir',
            default='/tmp/gym/',
            help='directory to save agent logs (default: /tmp/gym)')
        parser.add_argument(
            '--save-dir',
            default='./trained_models/',
            help='directory to save agent logs (default: ./trained_models/)')
        parser.add_argument(
            '--use-proper-time-limits',
            action='store_true',
            default=False,
            help='compute returns taking into account time limits')
        parser.add_argument(
            '--recurrent-policy',
            action='store_true',
            default=False,
            help='use a recurrent policy')
        parser.add_argument(
            '--use-linear-lr-decay',
            action='store_true',
            default=True,
            help='use a linear schedule on the learning rate')
        args = parser.parse_args()

        args.cuda = torch.cuda.is_available()

        assert args.algo in ['a2c', 'ppo', 'acktr']
        if args.recurrent_policy:
            assert args.algo in ['a2c', 'ppo'], \
                'Recurrent policy is not implemented for ACKTR'

        args_dict = vars(args)
        
        for key, value in args_dict.items():
            setattr(cls, key, value)
    
    def __getattribute__(cls, arg):
        return cls.args_dict[arg]
    
    @classmethod
    def dump(cls, filename):
        args_dict = {key : value for key, value in cls.__dict__.items() if not key in ['initialize', 'dump'] and not key.startswith('__')}
        print(args_dict)
        with open(filename, 'w') as f:
            json.dump(args_dict, f, indent=4)
