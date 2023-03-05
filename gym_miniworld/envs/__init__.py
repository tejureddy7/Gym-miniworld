import gym

from .remotebot import *
from .hallway import *
from .oneroom import *
from .tmaze import *
from .fourrooms import *
from .threerooms import *
from .wallgap import *
from .putnext import *
from .maze import *
from .collecthealth import *
from .simtoreal1 import *
from .simtoreal2 import *
# from .pickupobs import *
print("done")
# Registered environment ids
env_ids = []

def register_envs():
    module_name = __name__
    global_vars = globals()

    # Iterate through global names
    for global_name in sorted(list(global_vars.keys())):
        if not global_name.endswith('Env'):
            continue
        if global_name == 'MiniWorldEnv':
            continue

        env_name = global_name.split('Env')[0]
        env_class = global_vars[global_name]

        # Register the environment with OpenAI Gym
        gym_id = 'MiniWorld-%s-v0' % (env_name)
        entry_point = '%s:%s' % (module_name, global_name)

        gym.envs.registration.register(
            id=gym_id,
            entry_point=entry_point,
        )

        env_ids.append(gym_id)

        print('Registered env:', gym_id)

register_envs()
