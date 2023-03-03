from __future__ import annotations

import gymnasium as gym

from minigrid.minigrid_env import MiniGridEnv
from minigrid.utils.window import Window
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX

from minigrid.manual_control import ManualControl

from enum import Enum
from dataclasses import dataclass

import numpy as np

from minigrid.core.world_object import Key, Door
from minigrid.core.world_object import Box as Fly

code_obj = 0
code_col = 20
code_state = 30
code_sep = 40
codes = (code_obj, code_col, code_state)

OBJECT_TO_IDX = OBJECT_TO_IDX.copy()
OBJECT_TO_IDX["fly"] = 7
OBJECT_TO_IDX["lit"] = 11
OBJECT_TO_IDX["viewfinder"] = 0
del OBJECT_TO_IDX["box"]

for k, col in COLOR_TO_IDX.items():
    OBJECT_TO_IDX[k] = col + code_col
    
for k, state in STATE_TO_IDX.items():
    OBJECT_TO_IDX[k] = state + code_state
    
OBJECT_TO_IDX["separator"] = code_sep

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
        
        self.jc = '' if self.emojis else ' '
        self.col_idxs = list(COLOR_TO_IDX.values()).sort() # r, g, b, p, y, w
        self.cols = ('🟥', '🟩', '🟦', '🟪', '🟨', '⬜') if self.emojis else ('R', 'G', 'B', 'P', 'Y', 'W')
        self.states = ('⬛', '🔓', '🔒') if self.emojis else (' ', 'U', 'L')
        self.obj_idxs = [OBJECT_TO_IDX[k] for k in ('fly', 'empty', 'wall', 'agent', 'lit', 'door', 'key', 'viewfinder', 'red', 'green', 'blue', 'yellow', 'purple', 'grey', 'open', 'closed', 'locked', 'separator')]
        self.objects = ('🪰', '⬛', '🧱', '🐸', '🟨', '🚪', '🗝️', '⬛') if self.emojis else ('°',' ','#','♦', '_', '▥', '🗝', ' ')
        self.all_strs = self.objects + self.cols + self.states + ('💠',) if self.emojis else ('=',) 
        self.obj_str = {k: v for k, v in zip(self.obj_idxs, self.all_strs)}
        self.dirs = ['➡️', '⬇️', '⬅️', '⬆️'] if self.emojis else ['☛', '🖣',  '☚', '🖢']
        
        
    def start(self):
        self.reset(self.seed)
        if not self.textmode:
            self.window.show(block=False)
    
    def reset(self, seed=None):
        self.env.reset(seed=seed)
        self.obs = self.env.gen_obs()
        
        self.my_coord = (self.obs['image'].shape[0]//2, self.obs['image'].shape[1] - 1)
        self.view_less_me = np.ones(shape=self.obs['image'].shape[:2], dtype=bool)
        self.view_less_me[self.my_coord] = False
        
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
                    
                
                    if world[i, j] == 1 and viewrot[ylit, xlit]:
                        world[i, j] = 11
            
            cam_image = self.obs['image'][:, :, 0] * view
            
            # Wondering if I should print the colour of grey walls too
            for i, j in np.array(np.where(cam_image == OBJECT_TO_IDX['wall'])).T:
                if self.obs['image'][i, j, 1] != OBJECT_TO_IDX['grey'] - code_col:
                    cam_image[i, j] = self.obs['image'][i, j, 1] + code_col
            
            frog_cam = np.zeros((world.shape[0], view.shape[1]), dtype=int)
            frog_cam[world.shape[0]-view.shape[1]:, :] = np.rot90(cam_image, k=-1)
            
            for i, j in np.array(np.where(frog_cam == OBJECT_TO_IDX['empty'])).T:
                frog_cam[i, j] = OBJECT_TO_IDX['lit']
                
            
            cam_break = world.shape[0]-view.shape[1]
            
            for i, j in np.array(np.where(frog_cam[world.shape[0]-view.shape[1]:, :] == 0)).T:
                frog_cam[cam_break+i, j] = OBJECT_TO_IDX['empty']
                
            frog_cam[-1, frog_cam.shape[1]//2] = OBJECT_TO_IDX['agent']
            
            
            viewfinder = np.zeros((cam_break-1, view.shape[0]))
            vo = 0
            for vo, (i, j) in enumerate(np.array(np.where(sum(OBJECT_TO_IDX[k] == cam_image for k in ('fly', 'key', 'door')))).T):
                viewfinder[-(vo//2) - 1, vo%2*4:vo%2*4+3] = self.obs['image'][i, j, :] + codes
                
            frog_cam[:cam_break-1, :] = viewfinder
            
            frog_cam[cam_break-1, :] = code_sep
           
            # Take the transpose of the world so it lines up with our moves
            for i, (world_row, cam_row) in enumerate(zip(world.T, frog_cam)):
                cam_str = self.jc.join(self.obj_str[t] for t in cam_row)
                
                if i < cam_break - 1:
                    if (frog_cam[i+1] != OBJECT_TO_IDX['separator']).any() and (frog_cam[i+1] != OBJECT_TO_IDX['viewfinder']).any():
                        cam_str = 'viewfinder:'
                    elif (cam_row == OBJECT_TO_IDX['viewfinder']).all():
                        cam_str = ''
                    
                if i == cam_break - 1:
                    cam_str = 'live frog reaction:'
                print(self.jc.join(self.obj_str[t] for t in world_row) + '    ' + cam_str)
                
            print(self.env.mission)
            
            
            print(f'frog compass: {self.dirs[self.env.agent_dir]}')
    
        ca, cl, cr = (self.obs['image'][self.my_coord[0], -2, 0] == OBJECT_TO_IDX['wall'], 
                      self.obs['image'][self.my_coord[0]+1, -1, 0] == OBJECT_TO_IDX['wall'],
                      self.obs['image'][self.my_coord[0]-1, -1, 0] == OBJECT_TO_IDX['wall']
                     )
    
        state = {}
        state['wall'] = ca * 1 + cl * 2 + cr * 3
            
        state['wall'] += 1 * (state['wall'] > 1) + 2 * (state['wall'] > 3 or (state['wall'] == 3 and ca)) 
        
            
        for obj in ['key', 'door', 'fly']:
            if (isinstance(self.env.carrying, WORLD_OBJECTS[obj]) and not 
                OBJECT_TO_IDX[obj] in (self.obs['image'][:, :, 0] * self.view_less_me)):
                state[obj] = 5 # the object is held
                continue
                
            if self.obs['image'][self.my_coord[0], -2, 0] == OBJECT_TO_IDX[obj]:
                state[obj] = 1  # the object is directly in front
                state[f'{obj}_colour'] = IDX_TO_OBJECT[self.obs['image'][self.my_coord[0], -2, 1] + code_col]
                continue

            objs = np.argwhere(view*self.obs['image'][:, :, 0] == OBJECT_TO_IDX[obj])
            closest = np.argmin(map(self._taxicab, objs)) if objs.any() else None
            
            #state[f'{obj}_colour']
                                                       
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
                
            state[f'{obj}_colour'] = IDX_TO_OBJECT[self.obs['image'][(*objs[closest], 1)] + code_col]
                
        return state
