from __future__ import annotations

import numpy as np

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import WorldObj, Box
from minigrid.core.constants import COLOR_TO_IDX, DIR_TO_VEC, OBJECT_TO_IDX, TILE_PIXELS
from minigrid.core.actions import Actions
from minigrid.utils.window import Window
from minigrid.minigrid_env import MiniGridEnv


import cairosvg
import cv2

from minigrid.utils.rendering import (
    downsample,
    fill_coords,
    highlight_img,
    point_in_rect,
    point_in_triangle,
    rotate_fn,
)

from typing import Any, Callable


def render_svg(img, fname, rot=0):
    with open(fname, 'r') as f:
        b = f.read()
    im = cairosvg.svg2png(b, output_width=img.shape[0], output_height=img.shape[1])
    decoded = cv2.imdecode(np.frombuffer(im, np.uint8), -1)
    
    decoded = np.rot90(decoded, k=rot, axes=(0, 1))
    
    for y in range(img.shape[1]):
        for x in range(img.shape[0]):
            if decoded[x, y, 3] > 0:
                img[x, y] = decoded[x, y, :3][::-1]
                
    return img
    
class Fly(Box):
    def render(self, img):
        render_svg(img, f'/home/renee/Documents/TA-grad_stuff/cgsc_3601/pondworld/assets/fly-{self.color}.svg')

    
class Pond(MiniGridEnv):
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
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )
                
    @staticmethod
    def _gen_mission():
        return "eat all the flies"
    
    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = FroggyGrid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        #self.put_obj(Goal(), width - 2, height - 2)
        
        options = np.argwhere(self.grid.encode()[:, :, 0] == 1)
        
        places = options[np.random.choice(
            range(len(options)), 
            size=9, 
            replace=False
        )]

        self.agent_pos = places[0]
        self.agent_dir = np.random.randint(4)
        
        for place in places[1:]:
            self.put_obj(Fly(np.random.choice(list(COLOR_TO_IDX.keys()))), *place)

        self.saw_fly = self.gen_obs()['image'][self.agent_view_size//2, -2, 0] == OBJECT_TO_IDX['box']
        
        self.mission = self._gen_mission()
        
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        if action == Actions.toggle and self.saw_fly:
            reward = self._reward()
        
        if not (self.grid.encode()[:, :, 0] == OBJECT_TO_IDX['box']).any():
            terminated = True
        
        self.saw_fly = self.gen_obs()['image'][self.agent_view_size//2, -2, 0] == OBJECT_TO_IDX['box']
        
        return obs, reward, terminated, truncated, info
    

class FroggyGrid(Grid):
    def render(
        self,
        tile_size: int,
        agent_pos: tuple[int, int],
        agent_dir: int | None = None,
        highlight_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """

        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)

                agent_here = np.array_equal(agent_pos, (i, j))
                assert highlight_mask is not None
                tile_img = FroggyGrid.render_tile(
                    cell,
                    agent_dir=agent_dir if agent_here else None,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size,
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img
    
    @classmethod
    def render_tile(
        cls,
        obj: WorldObj | None,
        agent_dir: int | None = None,
        highlight: bool = False,
        tile_size: int = TILE_PIXELS,
        subdivs: int = 3,
    ) -> np.ndarray:
        """
        Render a tile and cache the result
        """

        # Hash map lookup key for the cache
        key: tuple[Any, ...] = (agent_dir, highlight, tile_size)
        key = obj.encode() + key if obj else key

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        img = np.zeros(
            shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8
        )

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if obj is not None:
            obj.render(img)

        # Overlay the agent on top
        if agent_dir is not None:
            render_svg(img, f'/home/renee/Documents/TA-grad_stuff/cgsc_3601/pondworld/assets/frog.svg', rot=(-agent_dir + 1) % 4)

        # Highlight the cell if needed
        if highlight:
            highlight_img(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img