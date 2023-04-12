import numpy as np
import math
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box, Ball

class OneRoomEnv(MiniWorldEnv):
    """
    Environment in which the goal is to go to a red box
    placed randomly in one big room.
    """

    def __init__(self, size=10, **kwargs):
        assert size >= 2
        self.size = size

        super().__init__(
            max_episode_steps=180,
            **kwargs
        )

    def _gen_world(self):
        self.add_rect_room(
            min_x=0,
            max_x=self.size,
            min_z=0,
            max_z=self.size,
            wall_tex="cardboard",
            floor_tex="asphalt",
            no_ceiling=False,
        )


        obj_types = [Ball, Box ] #, Key]

        # for obj in range(self.num_objs):
        #     # obj_type1 = self.rand.choice(obj_types)
        #     obj_type1 = Ball
        #     obj_type2 = Box
        #
        #     color = self.rand.color()

            # if obj_type1 == Ball:

        self.box = self.place_entity(Box(color="blue", size=0.9)) #edit color
            # if obj_type2 == Box:
        # print("thispoint?")
        self.ball = self.place_entity(Ball(color="red", size=0.9)) #edit
            # if obj_type == Key:
            #     self.place_entity(Key(color=color)) # edit

        self.place_agent()

        self.num_picked_up = 0
    '''

    def _gen_world(self):
        room = self.add_rect_room(
            min_x=0,
            max_x=self.size,
            min_z=0,
            max_z=self.size
        )

        self.box = self.place_entity(Box(color='red'))
        self.place_agent()
    '''

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.near(self.box):
            reward += self._reward()
            done = True

        return obs, reward, done, info
