


def main():
    import copy
    import glob
    import os
    import time

    import gym
    import numpy as np
    import torch
    torch.multiprocessing.set_start_method('spawn')

    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from gym.spaces import Discrete

    from arguments import get_args
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from envs import make_env
    from img_env import ImgEnv, IMG_ENVS
    from model import Policy
    from storage import RolloutStorage
    from utils import update_current_obs, eval_episode

    import algo

    print("#######")
    print("WARNING: All rewards are clipped or normalized so you need to use a monitor (see envs.py) or visdom plot to get true rewards")
    print("#######")

    args = get_args()
    if args.no_cuda:
        args.cuda = False
    print(args)
    assert args.algo in ['a2c', 'ppo', 'acktr']
    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    num_updates = int(args.num_frames) // args.num_steps // args.num_processes

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    toprint = ['seed', 'lr', 'nat', 'resnet']
    if args.env_name in IMG_ENVS:
        toprint += ['window', 'max_steps']
    toprint.sort()
    name = args.tag
    args_param = vars(args)
    os.makedirs(os.path.join(args.out_dir, args.env_name), exist_ok=True)
    for arg in toprint:
        if arg in args_param and (args_param[arg] or arg in ['gamma', 'seed']):
            if args_param[arg] is True:
                name += '{}_'.format(arg)
            else:
                name += '{}{}_'.format(arg, args_param[arg])
    model_dir = os.path.join(args.out_dir, args.env_name, args.algo)
    os.makedirs(model_dir, exist_ok=True)

    results_dict = {
        'episodes': [],
        'rewards': [],
        'args': args
    }
    torch.set_num_threads(1)
    eval_env = [make_env(args,
        args.env_name, args.seed, 0, None, args.add_timestep, natural=args.nat,
        clip_rewards=False, train=False)]
    envs = [make_env(args, args.env_name, args.seed, i, None,
            args.add_timestep, natural=args.nat, train=True)
                for i in range(args.num_processes)]
    if args.num_processes > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)
    eval_env = DummyVecEnv(eval_env)
    if len(envs.observation_space.shape) == 1:
        envs = VecNormalize(envs, gamma=args.gamma)

    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

    actor_critic = Policy(obs_shape, envs.action_space, args.recurrent_policy,
                          dataset=args.env_name, resnet=args.resnet,
                          pretrained=args.pretrained)
    if envs.action_space.__class__.__name__ == "Discrete":
        action_shape = 1
    else:
        action_shape = envs.action_space.shape[0]

    if args.cuda:
        actor_critic.cuda()

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                         args.value_loss_coef, args.entropy_coef, lr=args.lr,
                               eps=args.eps,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, acktr=True)

    action_space = envs.action_space
    if args.env_name in IMG_ENVS:
        action_space = np.zeros(2)
    # obs_shape = envs.observation_space.shape
    rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, action_space, actor_critic.state_size)
    current_obs = torch.zeros(args.num_processes, *obs_shape)

    obs = envs.reset()
    update_current_obs(obs, current_obs, obs_shape, args.num_stack)
    rollouts.observations[0].copy_(current_obs)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([args.num_processes, 1])
    final_rewards = torch.zeros([args.num_processes, 1])

    if args.cuda:
        current_obs = current_obs.cuda()
        rollouts.cuda()

    start = time.time()
    for j in range(num_updates):
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, states = actor_critic.act(
                        rollouts.observations[step],
                        rollouts.states[step],
                        rollouts.masks[step])
            cpu_actions = action.squeeze(1).cpu().numpy()

            # Obser reward and next obs
            obs, reward, done, info = envs.step(cpu_actions)
            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            episode_rewards += reward

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            if args.cuda:
                masks = masks.cuda()

            if current_obs.dim() == 4:
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_obs *= masks

            update_current_obs(obs, current_obs, obs_shape, args.num_stack)
            rollouts.insert(current_obs, states, action, action_log_prob, value, reward, masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.observations[-1],
                                                rollouts.states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        if j % args.save_interval == 0:
            torch.save((actor_critic.state_dict(), results_dict), os.path.join(
                model_dir, name + 'model.pt'))

        if j % args.log_interval == 0:
            end = time.time()
            total_reward = eval_episode(eval_env, actor_critic, args)
            results_dict['rewards'].append(total_reward)
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print("Updates {}, num timesteps {}, FPS {}, reward {:.1f} entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       np.mean(results_dict['rewards'][-10:]), dist_entropy,
                       value_loss, action_loss))


if __name__ == "__main__":
    main()
