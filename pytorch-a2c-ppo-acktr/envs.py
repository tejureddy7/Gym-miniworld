import os

import gym
import numpy as np
import torch
from gym.spaces.box import Box

from vec_env import VecEnvWrapper
from vec_env.dummy_vec_env import DummyVecEnv
from vec_env.subproc_vec_env import SubprocVecEnv
import cv2
try:
    import dm_control2gym
except ImportError:
    pass

try:
    import roboschool
except ImportError:
    pass

try:
    import pybullet_envs
except ImportError:
    pass

try:
    import gym_miniworld
except ImportError:
    pass


def make_env(env_id, seed, rank, log_dir, add_timestep, allow_early_resets):
    def _thunk():
        env = gym.make(env_id)
        env.seed(seed + rank)

        env = Yolowrapper(env)

        obs_shape = env.observation_space.shape

        if add_timestep and len(
                obs_shape) == 1 and str(env).find('TimeLimit') > -1:
            env = AddTimestep(env)

        #if log_dir is not None:
        #    env = bench.Monitor(env, os.path.join(log_dir, str(rank)),
        #                        allow_early_resets=allow_early_resets)

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env)
        print("Agent position",env.agent.pos)

        return env

    return _thunk

def make_vec_envs(env_name, seed, num_processes, gamma, log_dir, add_timestep, device, allow_early_resets):
    envs = [make_env(env_name, seed, i, log_dir, add_timestep, allow_early_resets) for i in range(num_processes)]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    envs = VecPyTorch(envs, device)

    #if len(envs.observation_space.shape) == 3:
    #    envs = VecPyTorchFrameStack(envs, 4, device)

    return envs


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
    def observation(self, observation):
        if self.env._elapsed_steps > 0:
            observation[-2:0] = 0
        return observation


class AddTimestep(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(AddTimestep, self).__init__(env)
        self.observation_space = Box(
            self.observation_space.low[0],
            self.observation_space.high[0],
            [self.observation_space.shape[0] + 1],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return np.concatenate((observation, [self.env._elapsed_steps]))


class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 1, 0)

class Yolowrapper(gym.ObservationWrapper):
    """
    Transpose the observation image tensors for PyTorch
    """

    def __init__(self, env=None):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        #observation space for single object
        self.observation_space = gym.spaces.Box(low=-100000, high=100000, shape=(1*10,))
        # load model
        self.model_yolo = torch.hub.load('yolov5','custom', path='yolov5/Lastweightscolored.pt', source='local') #Add path to yolov5

    def observation(self, observation):

        # cv2.imshow("OutputWindow",observation)
        # cv2.waitKey(1) #blue
        #obs = cv2.cvtColor(observation, cv2.COLOR_BGRA2RGB)
        obs = observation
        #cv2.imshow("ResultWindow",obs)
        #cv2.waitKey(1) #red
        results = self.model_yolo(obs)
        BBox_Coordinates = results.pandas().xyxy[0].sort_values('xmin')
        #print(BBox_Coordinates)

       #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        obswrite = cv2.cvtColor(observation, cv2.COLOR_BGRA2RGB)
        cv2.imwrite('1.png',obswrite)
       #results = self.model_yolo('C:/Users/EEHPC/Airlearning_project2/airlearning-rl2/yoloimagestest/1.png')

        #print("coordinates",BBox_Coordinates)
        x1,y1,x2,y2 = 0,0,0,0
        a1,b1,a2,b2 = 0,0,0,0
        p1,q1,p2,q2 = 0,0,0,0
        if(len(BBox_Coordinates)!=0):
            #For Blue box
            if(0 in BBox_Coordinates['class'].values): #blue box
                n1 = BBox_Coordinates.index[BBox_Coordinates['class'] == 0].values
                n1 = n1[0]
                x1 = int(BBox_Coordinates['xmin'][n1])
                y1 = int(BBox_Coordinates['ymin'][n1])
                x2 = int(BBox_Coordinates['xmax'][n1])
                y2 = int(BBox_Coordinates['ymax'][n1])
                print("Cube detected")
                cv2.rectangle(obs, (x1, y1), (x2, y2), (255, 255, 255), 2)
            # else:
            #     print("Cube was not detected but ..")

            #For Yellow Cone
            # if(1 in BBox_Coordinates['class'].values): #cone
            #    print("Cone detected") #Sphere incase of nano yolo
            #    n2 = BBox_Coordinates.index[BBox_Coordinates['class'] == 1].values
            #    n2 = n2[0]
            #    a1 = int(BBox_Coordinates['xmin'][n2])
            #    b1 = int(BBox_Coordinates['ymin'][n2])
            #    a2 = int(BBox_Coordinates['xmax'][n2])
            #    b2 = int(BBox_Coordinates['ymax'][n2])
            #    cv2.rectangle(obs, (a1, b1), (a2, b2), (0, 255, 0), 2)

            #For Red Sphere
            if(2 in BBox_Coordinates['class'].values): #red sphere
                print("Sphere detected")
                n3 = BBox_Coordinates.index[BBox_Coordinates['class'] == 2].values
                n3 = n3[0]
                p1 = int(BBox_Coordinates['xmin'][n3])
                q1 = int(BBox_Coordinates['ymin'][n3])
                p2 = int(BBox_Coordinates['xmax'][n3])
                q2 = int(BBox_Coordinates['ymax'][n3])
                #cv2.rectangle(obs, (p1, q1), (p2, q2), (255, 255, 255), 2)

            #single object-box detection
            #x1,y1,x2,y2] #,a1,b1,a2,b2] #cube


            #see bounding box on images
            #cv2.imshow("ResultOutputWindow",obs)
            #cv2.waitKey(1)
            # Create another policy for highlevel

            BBox = [p1,q1,p2,q2,x1,y1,x2,y2] #Sphere and cube
        else:

            BBox = [0,0,0,0,0,0,0,0] #,0,0,0,0]
            print("No Detection happened")
            #print("target color", self.target_color)
            #to decide goal vector
        goal_vec = [0 ,0 ]
        print("Goal", self.boxIdx)
        goal_vec[self.boxIdx] = 1
        obs = BBox + goal_vec

        #obs = BBox
        print(obs)
        #obs = BBox

        return  obs





class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        actions = actions.squeeze(1).cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
        return obs, reward, done, info


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device):
        self.venv = venv
        self.nstack = nstack
        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.low.shape[0]
        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)
        self.stackedobs = np.zeros((venv.num_envs,) + low.shape)
        self.stackedobs = torch.from_numpy(self.stackedobs).float()
        self.stackedobs = self.stackedobs.to(device)
        observation_space = gym.spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stackedobs[:, :-self.shape_dim0] = self.stackedobs[:, self.shape_dim0:]
        for (i, new) in enumerate(news):
            if new:
                self.stackedobs[i] = 0
        self.stackedobs[:, -self.shape_dim0:] = obs
        return self.stackedobs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        self.stackedobs.fill_(0)
        self.stackedobs[:, -self.shape_dim0:] = obs
        return self.stackedobs

    def close(self):
        self.venv.close()
