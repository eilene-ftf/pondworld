from __future__ import annotations

import numpy as np

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import WorldObj, Box, Wall, Door, Key
from minigrid.core.constants import COLOR_TO_IDX, DIR_TO_VEC, OBJECT_TO_IDX, TILE_PIXELS
from minigrid.core.actions import Actions
#from minigrid.utils.window import Window
from minigrid.minigrid_env import MiniGridEnv

from typing import Any, Callable

import importlib.resources

from . import Fly, FroggyGrid

OBJECT_TO_IDX = OBJECT_TO_IDX.copy()
OBJECT_TO_IDX['fly'] = 7 # alias box to fly
OBJECT_TO_IDX["lit"] = 11
del OBJECT_TO_IDX["box"]
IDX_TO_OBJECT = {v: k for k, v in OBJECT_TO_IDX.items()}


def rand_col():
    return np.random.choice(list(COLOR_TO_IDX.keys()))

class House(MiniGridEnv):
    def __init__(
        self,
        size = 16,
        max_steps: int | None = None,
        **kwargs
    ):
        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=False,
            max_steps=max_steps,
            **kwargs,
        )
              
    def render(self):        
        super().render()
    
    @staticmethod
    def _gen_mission():
        return "eat all the flies"
    
    def gen_obs_grid(self, agent_view_size=None):
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        if agent_view_size is None, self.agent_view_size is used
        """

        topX, topY, botX, botY = self.get_view_exts(agent_view_size)

        agent_view_size = agent_view_size or self.agent_view_size

        grid = self.grid.slice(topX, topY, agent_view_size, agent_view_size)

        for i in range(self.agent_dir + 1):
            grid = grid.rotate_left()

        # Process occluders and visibility
        # Note that this incurs some performance cost
        if not self.see_through_walls:
            vis_mask = grid.process_vis(
                agent_pos=(agent_view_size // 2, agent_view_size - 1)
            )
        else:
            vis_mask = np.ones(shape=(grid.width, grid.height), dtype=bool)

        # Make it so the agent sees what it's carrying
        # We do this by placing the carried object at the agent's position
        # in the agent's partially observable view
        agent_pos = grid.width // 2, grid.height - 1
        if self.carrying:
            grid.set(*agent_pos, self.carrying)
        else:
            grid.set(*agent_pos, None)

        return grid, vis_mask
    
    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = FroggyGrid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        #self.put_obj(Goal(), width - 2, height - 2)
        
        # Put a wall in the middle
        door_spot = np.random.randint(1, height-1)
        door_colour = rand_col()
        for i in range(height):
            if i != door_spot:
                self.put_obj(Wall(), width//2, i)
            else:
                self.put_obj(Door(door_colour, is_locked=True), width//2, door_spot)
        
        north_options = np.argwhere(self.grid.encode()[:width//2, :, 0] == 1)
        south_options = np.argwhere(self.grid.encode()[width//2+1:, :, 0] == 1) + (width//2+1, 0)
        
        north_places = north_options[np.random.choice(
            range(len(north_options)), 
            size=8, 
            replace=False
        )]
        
        south_places = south_options[np.random.choice(
            range(len(south_options)), 
            size=2, 
            replace=False
        )]

        self.agent_pos = south_places[0]
        self.agent_dir = np.random.randint(4)
        
        self.put_obj(Key(door_colour), *south_places[1])
        
        for place in north_places:
            self.put_obj(Fly(rand_col()), *place)

        self.saw_fly = self.gen_obs()['image'][self.agent_view_size//2, -2, 0] == OBJECT_TO_IDX['fly']
        
        self.mission = self._gen_mission()
        
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        if action == Actions.toggle and self.saw_fly:
            reward = self._reward()
        
        if not (self.grid.encode()[:, :, 0] == OBJECT_TO_IDX['fly']).any():
            terminated = True
        
        self.saw_fly = self.gen_obs()['image'][self.agent_view_size//2, -2, 0] == OBJECT_TO_IDX['fly']
        
        return obs, reward, terminated, truncated, info
