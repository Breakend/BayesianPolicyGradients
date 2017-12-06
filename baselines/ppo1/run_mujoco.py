#!/usr/bin/env python
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import gym, logging
from baselines import logger
import sys
import tensorflow as tf

def train(env_id, num_timesteps, seed, dropout_on_V, dropout_tau_v, lengthscale_V, V_keep_prob, mc_samples, override_reg, optim_stepsize, vf_hid_size, activation_vf, sample_dropout):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)
    env = gym.make(env_id)

    #### this guy is deciding if we do dropout on V or not
    dropout_on_V=dropout_on_V

    pol_tau = 1.

    def policy_fn(name, ob_space, ac_space):

        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            ## MAIN CHANGES
            hid_size_V=vf_hid_size,
            hid_size_actor=64, num_hid_layers=2,
            V_keep_prob=V_keep_prob,mc_samples=mc_samples,\
            layer_norm=False,activation_critic=activation_vf,\
            activation_actor=tf.nn.relu , dropout_on_V=dropout_on_V, sample_dropout=sample_dropout)

    env = bench.Monitor(env, logger.get_dir() and
        osp.join(logger.get_dir(), "monitor.json"))
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_batch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=optim_stepsize, optim_batchsize=64,
            gamma=0.995, lam=0.97,
            ## MAIN CHANGES
            dropout_on_V=dropout_on_V,
            dropout_tau_V=dropout_tau_v,
            lengthscale_V=lengthscale_V,
            schedule='linear',
            override_reg=override_reg)
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ### Choose your environment
#    env_id='HalfCheetah-v1'
    env_id='Hopper-v1'
    parser.add_argument('--env', help='environment ID', default=env_id)
    parser.add_argument('--log_dir', help='logging directory', default="./logs/")
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--mc_samples', help='mc samples', type=int, default=50)
    parser.add_argument('--dropout_on_v', action="store_true")
    parser.add_argument('--dropout_tau_V', type=float, default=0.05)
    parser.add_argument('--keep_prob_V', type=float, default=0.95)
    parser.add_argument('--lengthscale_V', type=float, default=1e-3)
    parser.add_argument('--optim_stepsize', type=float, default=3e-4)
    parser.add_argument("--value_func_size", default=64, type=int)
    parser.add_argument("--activation_vf", type=str, default="tanh")
    parser.add_argument('--override_reg', type=float, default=None)
    parser.add_argument('--sample_dropout', action="store_true")
    activation_map = { "relu" : tf.nn.relu, "leaky_relu" : U.lrelu, "tanh" :tf.nn.tanh}

    args = parser.parse_args()
    activation_vf = activation_map[args.activation_vf]
    logger.configure(dir=args.log_dir)
    train(args.env, num_timesteps=2e6, seed=args.seed, dropout_on_V=args.dropout_on_v, dropout_tau_v=args.dropout_tau_V, lengthscale_V=args.lengthscale_V, V_keep_prob=args.keep_prob_V, mc_samples=args.mc_samples, override_reg=args.override_reg, optim_stepsize=args.optim_stepsize, vf_hid_size=args.value_func_size, activation_vf=activation_vf, sample_dropout=args.sample_dropout)


if __name__ == '__main__':
    main()
