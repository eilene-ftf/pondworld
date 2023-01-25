from __future__ import annotations

import gymnasium as gym

from minigrid.minigrid_env import MiniGridEnv
from minigrid.utils.window import Window
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

from minigrid.manual_control import ManualControl

from enum import Enum
from dataclasses import dataclass

import numpy as np

@dataclass
class EnvState:
    reward: float
    terminated: bool
    truncated: bool

class FrogControl(ManualControl):
    def __init__(
        self,
        env: MiniGridEnv,
        agent_view: bool = False,
        window: Window = None,
        seed = None,
        textmode: bool = False,
        emojis: bool = False
    ) -> None:
        super().__init__(env, agent_view, window, seed)
        
        self.actions = {
            'forward': MiniGridEnv.Actions.forward,
            'left': MiniGridEnv.Actions.left,
            'right': MiniGridEnv.Actions.right,
            'interact': MiniGridEnv.Actions.toggle
        }

        self.textmode = textmode
        self.emojis = emojis
        
    def start(self):
        self.reset(self.seed)
        self.window.show(block=False)
    
    def reset(self, seed=None):
        self.env.reset(seed=seed)
        self.obs = self.env.gen_obs()
        
        self.my_coord = (self.obs['image'].shape[0]//2, self.obs['image'].shape[1] - 1)

        if hasattr(self.env, "mission"):
            self.mission = self.env.mission
            self.window.set_caption(self.env.mission)

        self.redraw()    
        
    def key_handler(self, _):
        pass
    
    def move(self, action) -> EnvState:
        #if not isinstance(action, Action):
        #    raise TypeError("Action must be one of Action.forward, Action.left, Action.right, or Action.interact, defined in this module")
        
        return self.step(self.actions[action])
        
    def end(self):
        self.window.close()
        
    def step(self, action: MiniGridEnv.Actions) -> EnvState:
        obs, reward, terminated, truncated, _ = self.env.step(action)
        self.obs = obs # what the froggie can observe

        if terminated:
            self.reset(self.seed)
        elif truncated:
            self.reset(self.seed)
        else:
            self.redraw()
            
        return EnvState(reward, terminated, truncated)
    
    def _taxicab(self, trg_coord, axis: int | None = None):
        if isinstance(axis, int):
            return np.array(self.my_coord)[axis] - np.array(trg_coord)[axis]
        return (np.array(self.my_coord) - np.array(trg_coord)).sum() 
    
    def look(self):
        if self.textmode:
            #print("\033c", end='')
            world = self.env.grid.encode()[:, :, 0]
            world[self.env.agent_pos] = 8
            s = {7: '🪰', 1: ' ', 2: '🧱', 8: '🐸'} if self.emojis {7: '°', 1: ' ', 2: '#', 8: '♦'}
            for row in world:
                print(' '.join([s[t] for t in row]))

        if self.obs['image'][self.my_coord[0], -2, 0] == 7:
            return 1  # there is a fly directly in front
        
        flies = np.argwhere(self.obs['image'][:, :, 0] == 7)
        closest = np.argmin(map(self._taxicab, flies)) if flies.any() else None
        
        if closest is None: 
            return 0 # there are no flies
        
        dist_x, dist_y = [self._taxicab(flies[closest], axis=i) for i in range(2)]
        
        if dist_y >= np.abs(dist_x):
            return 2 # fly is farther ahead
        elif dist_x > 0:
            return 3 # fly is farther to the left
        else: # dist_x < 0
            return 4 # fly is farther to the right
        
