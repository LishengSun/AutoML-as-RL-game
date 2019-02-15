#!/usr/bin/env python


import time
import numpy as np
import glob
import gym
from gym import spaces


from matting import *
from imgsource import *

KINETICS_PATH = '/datasets01/kinetics/070618/400/train/driving_car/*.mp4'
KINETICS_PATH_TEST = '/datasets01/kinetics/070618/400/val/driving_car/*.mp4'


class ReplaceBackgroundEnv(gym.ObservationWrapper):

    viewer = None

    def __init__(self, env, bg_matting, natural_source):
        """
        The source must produce a image with a shape that's compatible to env's observation
        """
        super(ReplaceBackgroundEnv, self).__init__(env)
        self._bg_matting = bg_matting
        self._natural_source = natural_source

    def observation(self, obs):
        mask = self._bg_matting.get_mask(obs)
        img = self._natural_source.get_image()
        obs[mask] = img[mask]
        self._last_ob = obs
        return obs

    def reset(self):
        self._natural_source.reset()
        return super(ReplaceBackgroundEnv, self).reset()

    # copied from gym/envs/atari/atari_env.py
    def render(self, mode='human'):
        img = self._last_ob
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return env.viewer.isopen


class ReplaceMuJoCoBackgroundEnv(gym.ObservationWrapper):
    # observation_space = spaces.Box(
    #     low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)
    viewer = None

    def __init__(self, env, bg_matting, natural_source):
        """
        The source must produce a image with a shape that's compatible to env's
        observation
        """
        super(ReplaceMuJoCoBackgroundEnv, self).__init__(env)
        self._bg_matting = bg_matting
        self._natural_source = natural_source

    def observation(self, obs):
        obs = self.unwrapped.sim.render(
            camera_name='track', width=128, height=128, depth=False)[::-1,:,:]
        # obs = obs.transpose(1, 2, 0)
        mask = self._bg_matting.get_mask(obs)
        img = self._natural_source.get_image()
        obs[mask] = img[mask]
        self._last_ob = obs
        return obs

    def reset(self):
        self._natural_source.reset()
        return super(ReplaceMuJoCoBackgroundEnv, self).reset()


class PixelMujoCoEnv(gym.ObservationWrapper):
    # observation_space = spaces.Box(
    #     low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)
    viewer = None
    def __init__(self, env):
        """
        The source must produce a image with a shape that's compatible to
        env's observation
        """
        super(PixelMujoCoEnv, self).__init__(env)

    def observation(self, obs):
        obs = self.unwrapped.sim.render(
            camera_name='track', width=128, height=128, depth=False)[::-1,:,:]
        # obs = obs.transpose(1, 2, 0)
        self._last_ob = obs
        return obs

    def reset(self):
        super(PixelMujoCoEnv, self).reset()
        return self.observation(None)


if __name__ == '__main__':
    COCO_PATH = '/datasets01/COCO/060817/val2014/*.jpg'
    """
    Background color per games:
    Breakout: 0,0,0
    """

    env = gym.make('BreakoutDeterministic-v4')
    shape2d = env.observation_space.shape[:2]

    env = ReplaceBackgroundEnv(
        env,
        BackgroundMattingWithColor((0, 0, 0)),
        #RandomColorSource(shape2d)
        #RandomImageSource(shape2d, glob.glob(COCO_PATH))
        RandomVideoSource(shape2d, glob.glob(KINETICS_PATH))
    )
    env.reset()
    while True:
        act = env.action_space.sample()
        ob, r, isOver, info = env.step(act)
        env.render()
        time.sleep(0.03)
        if isOver:
            env.reset()
