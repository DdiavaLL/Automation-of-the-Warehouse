import random

import numpy as np
import math
import gym
from gym import error, spaces, utils
from gym.utils import seeding

def categorical_sample(prob_n, np_random):
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    if csprob_n is None:
        return (csprob_n > np_random.rand()).argmax()
    return random.randint(0, 6)

class StoreHouse(gym.Env):
    metadata = {'render.modes': ['human']}
    """
        Storehouse environment.
        Description:
            The robot controls the work of the warehouse, loading and unloading boxes.
            The goal is to fill the warehouse as much as possible without losing the ability to unload boxes.

        Actions:
            Type: Discrete(2)
            Num	Action
            0	Lift the box.
            1	Put the box.
            2   Step left.
            3   Step right.
            4   Step up.
            5   Step down.
        """
    def __init__(self):
        length = 5
        width = 5
        self.length = length
        self.width = width
        if width / 2 == 0:
            self.current_step = (length - 1, int(width / 2))
            self.assist = int(width / 2)
        else:
            self.current_step = (length - 1, int(width / 2 + 1))
            self.assist = int(width / 2 + 1)
        size = self.length * self.width
        self.ROBOT_CARRY = True

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Discrete(self.length * self.width)
        self.reward_range = (-1, size)

        # Store what the agent tried
        self.curr_episode = -1
        self.action_episode_memory = []

        # Init gym_storehouse
        self.state = (np.zeros((1, length, width), dtype=int), self.ROBOT_CARRY)


    def step(self, action):
        self._take_action(action)
        reward = self.compute_reward()
        if reward == -10:
            done = True
        else:
            done = False

        # !!!!
        seed = None
        np_random, seed = seeding.np_random(seed)
        obs = self.next_observation()
        help = categorical_sample(obs, np_random)
        return help, reward, done, {}

    def _take_action(self, action):
        action_type = action
        print(f'Action type: {action_type}')

        if action_type == 0:    #Lift the box
            if self.current_step == (self.length - 1, self.assist):
                self.state = (self.state[0], True)
        elif action_type == 1:  #Put the box
            if self.state[1] == True:
                self.state = (self.state[0], False)
                self.state[0][0, self.current_step[0], self.current_step[1]] = 1
        elif action_type == 2:  #Step left
            if self.current_step[1] > 0 and self.current_step[1] <= 4:
                if self.state[0][0, self.current_step[0], self.current_step[1] - 1] == 0:
                    self.current_step = (self.current_step[0], self.current_step[1] - 1)
        elif action_type == 3:  #Step right
            if self.current_step[1] >= 0 and self.current_step[1] < 4:
                if self.state[0][0, self.current_step[0], self.current_step[1] + 1] == 0:
                    self.current_step = (self.current_step[0], self.current_step[1] + 1)
        elif action_type == 4:  #Step up
            if self.current_step[0] > 0 and self.current_step[0] <= 4:
                if self.state[0][0, self.current_step[0] - 1, self.current_step[1]] == 0:
                    self.current_step = (self.current_step[0] - 1, self.current_step[1])
        elif action_type == 5:  #Step down
            if self.current_step[0] >= 0 and self.current_step[0] < 4:
                if self.state[0][0, self.current_step[0] + 1, self.current_step[1]] == 0:
                    self.current_step = (self.current_step[0] + 1, self.current_step[1])

    def reset(self):
        self.ROBOT_CARRY = True
        seed = None
        np_random, seed = seeding.np_random(seed)
        if self.width / 2 == 0:
            self.current_step = (self.length - 1, int(self.width / 2))
            self.assist = int(self.width / 2)
        else:
            self.current_step = (self.length - 1, int(self.width / 2 + 1))
            self.assist = int(self.width / 2 + 1)

        self.state = (np.zeros((1, self.length, self.width), dtype=int), self.ROBOT_CARRY)
        help = categorical_sample(self.next_observation(), np_random)
        return int(help)

    def next_observation(self):
        return self.state

    def render(self, mode = 'human'):
        print(f'Step: {self.current_step}')
        print(f'State: {self.state}')
        print('--------------------')

    def compute_reward(self):
        first, second, third, fourth = 0, 0, 0, 0
        if self.current_step == (0, 0):
            second = self.state[0][0, self.current_step[0] + 1, self.current_step[1]]
            fourth = self.state[0][0, self.current_step[0], self.current_step[1] + 1]
            if second == 1 and fourth == 1:
                return - 1000
        elif self.current_step == (self.length - 1, 0):
            first = self.state[0][0, self.current_step[0] - 1, self.current_step[1]]
            fourth = self.state[0][0, self.current_step[0], self.current_step[1] + 1]
            if first == 1 and fourth == 1:
                return - 1000
        elif self.current_step == (0, self.width - 1):
            third = self.state[0][0, self.current_step[0], self.current_step[1] - 1]
            second = self.state[0][0, self.current_step[0] + 1, self.current_step[1]]
            if third == 1 and second == 1:
                return -10
        elif self.current_step == (self.length - 1, self.width - 1):
            first = self.state[0][0, self.current_step[0] - 1, self.current_step[1]]
            third = self.state[0][0, self.current_step[0], self.current_step[1] - 1]
            if first == 1 and third == 1:
                return -10
        elif self.current_step[0] == 0:
            second = self.state[0][0, self.current_step[0] + 1, self.current_step[1]]
            third = self.state[0][0, self.current_step[0], self.current_step[1] - 1]
            fourth = self.state[0][0, self.current_step[0], self.current_step[1] + 1]
            if second == 1 and third == 1 and fourth == 1:
                return -10
        elif self.current_step[1] == self.width - 1:
            first = self.state[0][0, self.current_step[0] - 1, self.current_step[1]]
            second = self.state[0][0, self.current_step[0] + 1, self.current_step[1]]
            third = self.state[0][0, self.current_step[0], self.current_step[1] - 1]
            if first == 1 and second == 1 and third == 1:
                return -10
        elif self.current_step[0] == self.length - 1:
            first = self.state[0][0, self.current_step[0] - 1, self.current_step[1]]
            third = self.state[0][0, self.current_step[0], self.current_step[1] - 1]
            fourth = self.state[0][0, self.current_step[0], self.current_step[1] + 1]
            if first == 1 and third == 1 and fourth == 1:
                return -10
        elif self.current_step[1] == 0:
            first = self.state[0][0, self.current_step[0] - 1, self.current_step[1]]
            second = self.state[0][0, self.current_step[0] + 1, self.current_step[1]]
            fourth = self.state[0][0, self.current_step[0], self.current_step[1] + 1]
            if first == 1 and second == 1 and fourth == 1:
                return -10
        else:
            first = self.state[0][0, self.current_step[0] - 1, self.current_step[1]]
            second = self.state[0][0, self.current_step[0] + 1, self.current_step[1]]
            third = self.state[0][0, self.current_step[0], self.current_step[1] - 1]
            fourth = self.state[0][0, self.current_step[0], self.current_step[1] + 1]
            if first == 1 and second == 1 and third == 1 and fourth == 1:
                return -10

        if second == 1:
            if self.current_step[0] == self.length - 1:
                if self.current_step[1] == self.assist:
                    return 1
        return np.count_nonzero(self.state[0] == 1)

    def rand_coord(self):
        randx = random.randint(0, self.length)
        randy = random.randint(0, self.width)
        return randx, randy

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None