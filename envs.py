import cv2
import gym
import numpy as np
from gym.spaces.box import Box

def create_atari_env(env_id, render_mode=None):
    env = gym.make(env_id, render_mode=render_mode)
    if len(env.observation_space.shape) > 1:
        env = AtariPreprocessing(env)
    return env

def _preprocessing(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # RGB to Gray
    frame = cv2.pyrDown(frame, (120, 84))             # down-sampling
    frame = cv2.resize(frame, (84,84))                # resize
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)                            # Normalize
    frame = frame.reshape(1, 84, 84)    
    return frame

class AtariPreprocessing(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(AtariPreprocessing, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [1, 84, 84])

    def observation(self, observation_n):
        return _preprocessing(observation_n)