from __future__ import annotations

import gymnasium as gym

from minigrid.minigrid_env import MiniGridEnv
from minigrid.utils.window import Window
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from minigrid.core.constants import OBJECT_TO_IDX

from minigrid.manual_control import ManualControl

from enum import Enum
from dataclasses import dataclass

import numpy as np

from minigrid.core.world_object import Key, Door
from minigrid.core.world_object import Box as Fly

OBJECT_TO_IDX = OBJECT_TO_IDX.copy()
OBJECT_TO_IDX["fly"] = 7
OBJECT_TO_IDX["lit"] = 11
del OBJECT_TO_IDX["box"]
IDX_TO_OBJECT = {v: k for k, v in OBJECT_TO_IDX.items()}

WORLD_OBJECTS = {'key':  Key, 
                 'door': Door, 
                 'fly':  Fly
                }

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
        emojis: bool = False,
        see_through_walls: bool = False
    ) -> None:
        if not textmode:
            super().__init__(env, agent_view, window, seed, see_through_walls)
        else:
            self.env = env
            self.agent_view = agent_view
            self.seed = seed
            self.env.see_through_walls = see_through_walls
        
        self.actions = {
            'forward': MiniGridEnv.Actions.forward,
            'left': MiniGridEnv.Actions.left,
            'right': MiniGridEnv.Actions.right,
            'interact': MiniGridEnv.Actions.toggle,
            'pickup': MiniGridEnv.Actions.pickup
        }

        self.textmode = textmode
        self.emojis = emojis
        
    def start(self):
        self.reset(self.seed)
        if not self.textmode:
            self.window.show(block=False)
    
    def reset(self, seed=None):
        self.env.reset(seed=seed)
        self.obs = self.env.gen_obs()
        
        self.my_coord = (self.obs['image'].shape[0]//2, self.obs['image'].shape[1] - 1)
        
        if hasattr(self.env, "mission"):
            self.mission = self.env.mission
            if not self.textmode:
                self.window.set_caption(self.env.mission)

        if not self.textmode:
            self.redraw()
        
    def key_handler(self, _):
        pass
    
    def move(self, action) -> EnvState:
        #if not isinstance(action, Action):
        #    raise TypeError("Action must be one of Action.forward, Action.left, Action.right, or Action.interact, defined in this module")
        return self.step(self.actions[action])
        
    def end(self):
        if not self.textmode:
            self.window.close()
        
    def step(self, action: MiniGridEnv.Actions) -> EnvState:
        obs, reward, terminated, truncated, _ = self.env.step(action)
        self.obs = obs # what the froggie can observe

        if terminated:
            self.reset(self.seed)
        elif truncated:
            self.reset(self.seed)
        elif not self.textmode:
            self.redraw()
            
        return EnvState(reward, terminated, truncated)
    
    def _taxicab(self, trg_coord, axis: int | None = None):
        if isinstance(axis, int):
            return np.array(self.my_coord)[axis] - np.array(trg_coord)[axis]
        return (np.array(self.my_coord) - np.array(trg_coord)).sum() 
    
    def look(self):
        if self.textmode:
            #print("\033c", end='')
            
            # This needs to be used to process occlusion eventually
            _, view = self.env.gen_obs_grid()
            
            viewrot = np.rot90(view, k=self.env.agent_dir+1)
            
            world = self.env.grid.encode()[:, :, 0]
            world[tuple(self.env.agent_pos)] = 10
            
            vision_pos = (self.env.agent_view_size//2, self.env.agent_view_size-1)
            
            exts = [
                ( # down
                    self.env.agent_pos[0],
                    self.env.agent_pos[0] + self.env.agent_view_size - 1,
                    self.env.agent_pos[1] - self.env.agent_view_size//2,
                    self.env.agent_pos[1] + self.env.agent_view_size//2
                ),
                ( # right
                    self.env.agent_pos[0] - self.env.agent_view_size//2,
                    self.env.agent_pos[0] + self.env.agent_view_size//2,
                    self.env.agent_pos[1],
                    self.env.agent_pos[1] + self.env.agent_view_size - 1
                ),
                ( # up
                    self.env.agent_pos[0] - self.env.agent_view_size + 1,
                    self.env.agent_pos[0],
                    self.env.agent_pos[1] - self.env.agent_view_size//2,
                    self.env.agent_pos[1] + self.env.agent_view_size//2
                ),
                ( # left
                    self.env.agent_pos[0] - self.env.agent_view_size//2,
                    self.env.agent_pos[0] + self.env.agent_view_size//2,
                    self.env.agent_pos[1] - self.env.agent_view_size + 1,
                    self.env.agent_pos[1]
                )                    
            ]
            
            viewY = (
                max(0, exts[self.env.agent_dir][0]),
                min(world.shape[0], exts[self.env.agent_dir][1] + 1)
            )
            
            viewX = (
                max(0, exts[self.env.agent_dir][2]),
                min(world.shape[1], exts[self.env.agent_dir][3] + 1)
            )
            
            for y, i in enumerate(range(*viewY)):
                for x, j in enumerate(range(*viewX)):
                    
                    ylit = y - min(0, exts[self.env.agent_dir][0])
                    xlit = x - min(0, exts[self.env.agent_dir][2])
                    
                    #print(f'[{ylit} {xlit}]')
                
                    if world[i, j] == 1 and viewrot[ylit, xlit]:
                        world[i, j] = 11
            
            jc = '' if self.emojis else ' '
            idxs = [OBJECT_TO_IDX[k] for k in ('fly', 'empty', 'wall', 'agent', 'lit', 'door', 'key')]
            objects = ('🪰', '⬛', '🧱', '🐸', '🟨', '🚪', '🗝️') if self.emojis else ('°',' ','#','♦', '_', '▥', '%')
            s = {k: v for k, v in zip(idxs, objects)}
            for row in world:
                print(jc.join([s[t] for t in row]))
                
            print(self.env.mission)
            
            dirs = ['⬇️', '➡️', '⬆️', '⬅️',] if self.emojis else ['🖣', '☛', '🖢', '☚']
            
            print(f'frog compass: {dirs[self.env.agent_dir]}')

            
        state = {}
        for obj in ['key', 'door', 'fly']:
            if isinstance(self.env.carrying, WORLD_OBJECTS[obj]):
                state[obj] = 5 # the object is held
                continue
                
            if self.obs['image'][self.my_coord[0], -2, 0] == OBJECT_TO_IDX[obj]:
                state[obj] = 1  # the object is directly in front
                continue

            objs = np.argwhere(view*self.obs['image'][:, :, 0] == OBJECT_TO_IDX[obj])
            closest = np.argmin(map(self._taxicab, objs)) if objs.any() else None

            if closest is None: 
                state[obj] = 0 # there are no objects
                continue

            dist_x, dist_y = [self._taxicab(objs[closest], axis=i) for i in range(2)]

            if dist_y >= np.abs(dist_x) and self.obs['image'][self.my_coord[0], -2, 0] != OBJECT_TO_IDX['wall']:
                state[obj] = 2 # object is farther ahead
            elif dist_x > 0:
                state[obj] = 3 # fly is farther to the left
            else: # dist_x < 0
                state[obj] = 4 # fly is farther to the right
                
           
        #print(state)
        return state
