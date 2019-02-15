import os
import gym
import glob
import numpy as np
import torch
import torchvision.transforms as T
from gym.spaces.box import Box
from torchvision import datasets

from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind, \
    WarpFrame, FrameStack, ClipRewardEnv, MaxAndSkipEnv
from natural_env import PixelMujoCoEnv, ReplaceBackgroundEnv, \
    ReplaceMuJoCoBackgroundEnv, KINETICS_PATH, KINETICS_PATH_TEST
from img_env import ImgEnv, DetectionEnv, CITYSCAPE, IMG_ENVS
from matting import *
from imgsource import *
from cityscapes import Cityscapes
try:
    import dm_control2gym
except ImportError:
    pass

try:
    import roboschool
except ImportError:
    pass

try:
    import pybullet_envs
except ImportError:
    pass


def make_env(args, env_id, seed, rank, log_dir, add_timestep, train=True,
             natural=False, clip_rewards=True, loader=None):
    def _thunk():
        if train:
            vid_path = KINETICS_PATH
        else:
            vid_path = KINETICS_PATH_TEST
        if env_id in IMG_ENVS:
            if env_id == 'mnist':
                channels = 2
            else:
                channels = 4
            env = ImgEnv(env_id, max_steps=args.max_steps, channels=channels,
                         window=args.window, train=train)
        elif env_id in ['cityscapes']:
            env = DetectionEnv(env_id, max_steps=200, train=train)
        elif env_id.startswith("dm"):
            _, domain, task = env_id.split('.')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        else:
            env = gym.make(env_id)
        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        is_mujoco = hasattr(gym.envs, 'mujoco') and isinstance(
            env.unwrapped, gym.envs.mujoco.MujocoEnv)
        if is_atari:
            env = make_atari(env_id)
        if natural and is_atari:
            env = ReplaceBackgroundEnv(
                env,
                BackgroundMattingWithColor((0, 0, 0)),
                #RandomColorSource(shape2d)
                #RandomImageSource(shape2d, glob.glob(COCO_PATH))
                RandomVideoSource(env.observation_space.shape[:2],
                                  glob.glob(vid_path)))
        elif natural and is_mujoco:
            env.observation_space = Box(
                low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)
            env = ReplaceMuJoCoBackgroundEnv(
                env,
                BackgroundMattingWithColor((0, 0, 0)),
                #RandomColorSource(shape2d)
                #RandomImageSource(shape2d, glob.glob(COCO_PATH))
                RandomVideoSource(env.observation_space.shape[:2],
                                  glob.glob(vid_path)))
        elif is_mujoco:
            env.observation_space = Box(
                low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)
            env = PixelMujoCoEnv(env)
        env.seed(seed + rank)

        obs_shape = env.observation_space.shape
        if add_timestep and len(
                obs_shape) == 1 and str(env).find('TimeLimit') > -1:
            env = AddTimestep(env)

        if log_dir is not None:
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)))

        if is_atari:
            env = wrap_deepmind(env, clip_rewards=clip_rewards)
        if is_mujoco:
            env = ClipRewardEnv(WarpFrame(MaxAndSkipEnv(env, skip=4)))
        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = WrapPyTorch(env)

        return env

    return _thunk


class AddTimestep(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(AddTimestep, self).__init__(env)
        self.observation_space = Box(
            self.observation_space.low[0],
            self.observation_space.high[0],
            [self.observation_space.shape[0] + 1],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return np.concatenate((observation, [self.env._elapsed_steps]))


class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)
