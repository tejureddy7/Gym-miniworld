import numpy as np
import math
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box, Ball #Add
from gym import spaces
import random

class HallwayEnv(MiniWorldEnv):
    """
    Environment in which the goal is to go to a red box
    at the end of a hallway
    """

    def __init__(self, length=12, **kwargs):
        assert length >= 2
        self.length = length

        super().__init__(
            max_episode_steps=250,
            **kwargs
        )

        # Allow only the movement actions

        self.action_space = spaces.Discrete(self.actions.move_forward+1)

    def _gen_world(self):

        #self.boxIdx = random.randint(0, 1)
        # print("test",self.boxIdx)
        # Create a long rectangular room
        self.room = self.add_rect_room(
            min_x=-1, max_x=-1 + self.length,
            min_z=-2, max_z=2
        )
        print("Room position",self.room.min_x, self.room.max_x, self.room.min_z, self.room.max_z)

        # Place the box at the end of the hallway

        self.ball = self.place_entity(
            Box(color='blue'),
            min_x=self.room.max_x - 2
        )


        #Add Place a Red sphere at another endd of the hallway
        # self.ball = self.place_entity(
        #     Ball(color='red',size =0.9),
        #     min_x=self.room.min_x - 2
        # )

        #place a Cone at the final end

        # Place the agent a random distance away from the goal
        self.agent = self.place_agent(
            dir=self.rand.float(-math.pi/4, math.pi/4),
            max_x=self.room.max_x - 2
        )


    def step(self, action):
        obs, reward, done, info = super().step(action)

        # if self.boxIdx==0:
        #     if self.near(self.box):
        #         reward += self._reward()
        #         done = True
        # else:
        if self.near(self.ball):
                reward += self._reward()
                done = True
                print("success")

        x, _, z = self.agent.pos
        if x <= self.room.min_x or x >= self.room.max_x:
            done = True
            reward = 0
            print("collision")
        if z <= self.room.min_z or z >= self.room.max_z:
            done = True
            reward = 0
            print("collision")




        return obs, reward, done, info
