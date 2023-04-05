"""
Author: Steve Paul 
Date: 8/23/22 """

import random
import warnings
from MRTA_Collective_Transport_Env import MRTA_Collective_Transport_Env
import torch
import numpy as np
import gc
import time

gc.collect()
warnings.filterwarnings('ignore')
torch.cuda.empty_cache()


def as_tensor(observation):
    for key, obs in observation.items():
        observation[key] = torch.tensor(obs)
    return observation


def compute_objective_fn_value(env, X_dec):
    index_track = np.zeros((env.n_agents), dtype=int)
    obs = env.get_encoded_state()
    obs = as_tensor(obs)
    for i in range(1000000):
        start = time.time()
        robot_taking_decision = env.agent_taking_decision
        action = X_dec[index_track[robot_taking_decision], robot_taking_decision]
        index_track[robot_taking_decision] += 1
        dec_time = time.time() - start
        time_list.append(dec_time)
        obs, reward, done, _ = env.step(action)
        obs = as_tensor(obs)
        if done:
            return reward.item()


trained_model_n_loc = 51
trained_model_n_robots = 6
# loc_test_multipliers = [.5,1,2,5,10]
loc_test_multipliers = [1]
# robot_test_multipliers = [.5,1,2]
robot_test_multipliers = [1]
for loc_mult in loc_test_multipliers:
    for rob_mult in robot_test_multipliers:
        n_robots_test = int(rob_mult * loc_mult * trained_model_n_robots) + 1
        n_loc_test = int(trained_model_n_loc * loc_mult)

        nAllTasks = n_loc_test
        nRobot = n_robots_test
        total_rewards_list = []
        distance_list = []
        total_tasks_done_list = []
        time_list = []
        env = MRTA_Collective_Transport_Env(
            n_locations=nAllTasks,
            n_agents=nRobot,
            max_task_size=10,
            enable_dynamic_tasks=False,
            display=False,
            enable_topological_features=True,
        )
        max_dec_per_robot = 20

        # this is a sample decision variable
        X_dec = np.zeros((max_dec_per_robot, nRobot), dtype=int)
        for i in range(max_dec_per_robot):
            for j in range(nRobot):
                X_dec[i, j] = random.randint(0, nAllTasks-1)

        obj_fn_value = compute_objective_fn_value(env, X_dec)






