from pondworld.envs.pond import Pond, Fly, FroggyGrid
from pondworld.envs.house import House
from pondworld.envs.cellar import Cellar
from gymnasium.envs.registration import register

def registration():

    register(
        id='pond-v0',
        entry_point='pondworld.envs:Pond',
    )

    register(
        id='house-v0',
        entry_point='pondworld.envs:House',
    )

    register(
        id='cellar-v0',
        entry_point='pondworld.envs:Cellar',
    )
