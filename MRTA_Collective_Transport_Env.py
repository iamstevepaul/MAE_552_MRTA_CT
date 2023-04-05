"""
Author: Steve Paul 
Date: 7/19/22 """

import numpy as np
import gym
import time
from gym import Env
from collections import defaultdict
from gym.spaces import Discrete, MultiBinary, Box, Dict
import random
import matplotlib.pyplot as plt
import torch
from topology import *
import scipy.sparse as sp
from persim import wasserstein, bottleneck
import math


class MRTA_Collective_Transport_Env(Env):

    def __init__(self,
                 n_locations=100,
                 visited=[],
                 n_agents=2,
                 agents=[],
                 agents_location=[],
                 total_distance_traveled=0.0,
                 max_capacity=6,
                 max_range=4,
                 max_task_size = 10,
                 enable_dynamic_tasks=False,
                 n_initial_tasks=30,
                 display=False,
                 enable_topological_features=False
                 ):
        super(MRTA_Collective_Transport_Env, self).__init__()
        self.n_locations = n_locations
        self.action_space = Discrete(1)
        self.locations = np.random.random((n_locations, 2))
        self.depot = self.locations[0, :]
        self.visited = visited
        self.n_agents = n_agents
        self.agents = agents
        self.agents_location = agents_location
        self.agents_prev_location = np.zeros((n_agents, 1), dtype=int)
        self.agents_next_location = np.zeros((n_agents, 1), dtype=int)
        self.agents_distance_traveled = np.zeros((n_agents, 1))
        self.total_distance_traveled = total_distance_traveled
        self.agent_taking_decision = 0
        self.current_location_id = 0
        self.nodes_visited = np.zeros((n_locations, 1))
        self.enable_dynamic_tasks = enable_dynamic_tasks
        self.agents_distance_to_destination = np.zeros((n_agents, 1))

        self.distance_matrix = np.linalg.norm(self.locations[:, None, :] - self.locations[None, :, :], axis=-1)
        self.time = 0.0
        self.agent_speed = 0.01
        self.agents_next_decision_time = np.zeros((n_agents, 1))
        self.agents_prev_decision_time = np.zeros((n_agents, 1))
        self.agents_destination_coordinates = np.ones((n_agents, 1)) * self.depot
        self.total_reward = 0.0
        self.total_length = 0  # number of steps taken?

        self.max_capacity = max_capacity
        self.max_range = max_range
        self.agents_current_range = torch.ones((1, n_agents), dtype=torch.float32) * max_range
        self.agents_current_payload = torch.ones((1, n_agents), dtype=torch.float32) * max_capacity
        self.time_deadlines = (torch.tensor(np.random.random((1, n_locations))) * .7 + .3) * 550
        self.time_deadlines[0, 0] = 1000000

        self.max_task_size = max_task_size
        self.task_size = torch.tensor(np.random.random_integers(low=1, high=self.max_task_size + 1, size=(1, n_locations)), dtype=torch.int) #creating a zero dimensional tensor
        self.original_task_size = np.copy(self.task_size)
        # fraction of the task that has been completed
        self.task_completion = torch.zeros((1, n_locations), dtype=torch.float32)

        self.task_done = torch.zeros((1, n_locations), dtype=torch.float32)  # change to int?
        self.deadline_passed = torch.zeros((1, n_locations), dtype=torch.float32)
        self.depot_id = 0
        self.active_tasks = ((self.nodes_visited == 0).nonzero())[0]
        self.available_tasks = torch.zeros((n_locations, 1), dtype=torch.float32)

        if not self.enable_dynamic_tasks:
            n_initial_tasks = n_locations
        self.n_initial_tasks = n_initial_tasks
        self.available_tasks[0: n_initial_tasks, 0] = 1
        self.time_start = self.time_deadlines * (torch.rand((n_locations, 1)) * .5).T
        self.time_start[0, 0:self.n_initial_tasks] = 0
        self.display = display
        self.enable_topological_features = enable_topological_features

        self.task_graph_node_dim = self.generate_task_graph()[0].shape[1]
        self.agent_node_dim = self.generate_agents_graph()[0].shape[1]

        if self.enable_topological_features:
            self.observation_space = Dict(
                dict(
                    depot=Box(low=0, high=1, shape=(1, 2)),
                    mask=Box(low=0, high=1, shape=self.nodes_visited.shape),
                    topo_laplacian=Box(low=0, high=1, shape=(n_locations - 1, n_locations - 1)),
                    task_graph_nodes=Box(low=0, high=1, shape=(n_locations - 1, self.task_graph_node_dim)),
                    agents_graph_nodes=Box(low=0, high=1, shape=(n_agents, self.agent_node_dim)),
                    agent_taking_decision=Box(low=0, high=n_agents, shape=(1,), dtype=int),
                ))
            self.topo_laplacian = None
            state = self.get_encoded_state()
            topo_laplacian = self.get_topo_laplacian(state)
            state["topo_laplacian"] = topo_laplacian
            self.topo_laplacian = topo_laplacian
        else:
            self.observation_space = Dict(
                dict(
                    depot=Box(low=0, high=1, shape=(1, 2)),
                    mask=Box(low=0, high=1, shape=self.nodes_visited.shape),
                    task_graph_nodes=Box(low=0, high=1, shape=(n_locations - 1, self.task_graph_node_dim)),
                    task_graph_adjacency=Box(low=0, high=1, shape=(n_locations - 1, n_locations - 1)),
                    agents_graph_nodes=Box(low=0, high=1, shape=(n_agents, self.agent_node_dim)),
                    agent_taking_decision=Box(low=0, high=n_agents, shape=(1,), dtype=int)
                ))

        self.distance = 0.0
        self.done = False

    # Unchanged
    def get_state(self):
        # include locations visited into the state
        return np.concatenate((np.concatenate((self.locations, self.agents_destination_coordinates,
                                               self.agents_destination_coordinates[self.agent_taking_decision,
                                               :].reshape(1, 2)), axis=0).reshape(-1, 1),
                               self.nodes_visited.reshape(-1, 1)))

    # Unchanged
    def get_encoded_state(self):
        mask = self.get_mask()
        task_graph_nodes, task_graph_adjacency = self.generate_task_graph()
        agents_graph_nodes, agents_graph_adjacency = self.generate_agents_graph()
        if self.enable_topological_features:
            state = {
                'depot': self.depot.reshape(1, 2),
                'mask': mask,
                'task_graph_nodes': task_graph_nodes,
                'topo_laplacian': self.topo_laplacian,
                'agents_graph_nodes': agents_graph_nodes,
                'agent_taking_decision': self.agent_taking_decision,
            }
            # topo_laplacian = self.get_topo_laplacian(state)
        else:
            state = {
                'depot': self.depot.reshape(1, 2),
                'mask': mask,
                'task_graph_nodes': task_graph_nodes,
                'task_graph_adjacency': task_graph_adjacency,
                'agents_graph_nodes': agents_graph_nodes,
                'agent_taking_decision': self.agent_taking_decision,
            }
        return state

    # Unchanged
    def var_preprocess(self, adj, r):
        adj_ = adj + sp.eye(adj.shape[0])
        adj_ = adj_ ** r
        adj_[adj_ > 1] = 1
        rowsum = adj_.sum(1).A1
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5))
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).T.dot(degree_mat_inv_sqrt).tocsr()
        return adj_normalized


    # Unchanged
    def get_topo_laplacian(self, data):
        # active_tasks = ((data['nodes_visited'] == 0).nonzero())[0]
        X_loc = (data['task_graph_nodes'][:,:3].numpy())[None, :]
        # X_loc = X_loc[:, active_tasks[1:] - 1, :]
        # distance_matrix = ((((X_loc[:, :, None] - X_loc[:, None]) ** 2).sum(-1)) ** .5)[0]
        distance_matrix = torch.cdist(torch.tensor(X_loc), torch.tensor(X_loc), p=2)[0]
        threshold = 2/((X_loc.shape[1]*8*3.14)**.33)
        adj_ = np.float32(distance_matrix < threshold)

        adj_ = adj_ * (self.available_tasks[1:, :].T).numpy()
        adj_ = adj_ * (self.available_tasks[1:, :]).numpy()

        dt = defaultdict(list)
        for i in range(adj_.shape[0]):
            n_i = adj_[i, :].nonzero()[0].tolist()

            dt[i] = n_i

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(dt))
        adj_array = adj.toarray().astype(np.float32)
        var_laplacian = self.var_preprocess(adj=adj, r=2).toarray()

        secondorder_subgraph = k_th_order_weighted_subgraph(adj_mat=adj_array, w_adj_mat=distance_matrix, k=2)

        reg_dgms = list()
        for i in range(len(secondorder_subgraph)):
            # print(i)
            tmp_reg_dgms = simplicial_complex_dgm(secondorder_subgraph[i])
            if tmp_reg_dgms.size == 0:
                reg_dgms.append(np.array([]))
            else:
                reg_dgms.append(np.unique(tmp_reg_dgms, axis=0))

        reg_dgms = np.array(reg_dgms)

        row_labels = np.where(var_laplacian > 0.)[0]
        col_labels = np.where(var_laplacian > 0.)[1]

        topo_laplacian_k_2 = np.zeros(var_laplacian.shape, dtype=np.float32)

        for i in range(row_labels.shape[0]):
            tmp_row_label = row_labels[i]
            tmp_col_label = col_labels[i]
            tmp_wasserstin_dis = wasserstein(reg_dgms[tmp_row_label], reg_dgms[tmp_col_label])
            topo_laplacian_k_2[tmp_row_label, tmp_col_label] = 1. / (tmp_wasserstin_dis + 1)
            topo_laplacian_k_2[tmp_col_label, tmp_row_label] = 1. / (tmp_wasserstin_dis + 1)

        return topo_laplacian_k_2

    def step(self, action):
        agent_taking_decision = self.agent_taking_decision
        current_location_id = self.current_location_id

        self.total_length = self.total_length + 1

        reward = 0.0
        info = {}
        travel_distance = self.distance_matrix[current_location_id, action]
        self.agents_current_range[0, agent_taking_decision] -= travel_distance
        self.agents_prev_decision_time[agent_taking_decision, 0] = self.time
        self.visited.append((action, self.agent_taking_decision))
        if action == self.depot_id:  # agent returns to depot and refills
            self.agents_current_payload[0, agent_taking_decision] = self.max_capacity
            self.agents_current_range[0, agent_taking_decision] = self.max_range
            self.nodes_visited[action] = 0
        if self.nodes_visited[action] != 1 and action != self.depot_id:
            # update agent status
            distance_covered = self.total_distance_traveled + travel_distance
            self.total_distance_traveled = distance_covered
            self.agents_distance_traveled[agent_taking_decision] += travel_distance

            current_agent_payload = self.agents_current_payload[0, agent_taking_decision].item()
            self.agents_current_payload[0, agent_taking_decision] -= self.task_size[0, action].item()
            if self.agents_current_payload[0, agent_taking_decision] < 0:
                self.agents_current_payload[0, agent_taking_decision] = 0

            # update node_visited status
            if self.time_deadlines[0, action] < torch.tensor(self.time):
                self.deadline_passed[0, action] = 1
            else:
                old_demand = float(self.original_task_size[0, action].item())
                self.task_size[0, action] -= int(current_agent_payload)
                if self.task_size[0, action].item() < 0:
                    self.task_size[0, action] = 0
                new_demand = float(self.task_size[0, action].item())
                self.task_completion[0, action] = 1 - new_demand/old_demand
            if math.isclose(self.task_completion[0, action], 1):
                self.task_done[0, action] = 1
                self.nodes_visited[action] = 1
            self.total_reward += reward #is this just zero?

        self.agents_next_location[agent_taking_decision] = action
        self.agents_prev_location[agent_taking_decision] = current_location_id
        self.agents_destination_coordinates[agent_taking_decision] = self.locations[action].copy()
        self.agents_distance_to_destination[agent_taking_decision] = travel_distance
        self.agents_next_decision_time[agent_taking_decision] = self.time + travel_distance / self.agent_speed
        if self.display:
            self.render(action)
        self.agent_taking_decision = np.argmin(self.agents_next_decision_time)
        self.current_location_id = self.agents_next_location[self.agent_taking_decision][0].copy()
        self.time = self.agents_next_decision_time[self.agent_taking_decision][0].copy()
        deadlines_passed_ids = (self.time_deadlines < torch.tensor(self.time)).nonzero()
        if deadlines_passed_ids.shape[0] != 0:
            self.deadline_passed[0, deadlines_passed_ids[:, 1]] = 1
            self.nodes_visited[deadlines_passed_ids[:, 1], 0] = 1
        self.active_tasks = ((self.nodes_visited == 0).nonzero())[0]
        self.available_tasks = (self.time_start <= self.time).to(torch.float32).T

        if sum(self.nodes_visited) == self.n_locations - 1:
            final_distance_to_depot = torch.cdist(torch.tensor(self.agents_destination_coordinates), torch.tensor(self.depot[None, :])).sum().item()
            if self.task_done.sum() == self.n_locations - 1:
                reward = -(self.total_distance_traveled + final_distance_to_depot) / (1.41 *self.n_locations)
            else:
                reward = -((self.n_locations - 1) - self.task_done.sum()) / ((self.n_locations - 1))
            self.total_reward = reward
            self.done = True
            info = {"is_success": self.done,
                     "episode": {
                         "r": self.total_reward,
                         "l": self.total_length
                     }
                     }
        return self.get_encoded_state(), reward, self.done, info

    # Unchanged
    def get_mask(self):
        agent_taking_decision = self.agent_taking_decision
        mask = self.nodes_visited.copy()
        current_location_id = self.current_location_id
        if self.agents_current_payload[0, agent_taking_decision] == 0:
            mask[1:, 0] = 1
            mask[0, 0] = 0
        elif current_location_id == self.depot_id:
            mask[0, 0] = 1
        else:
            unreachbles = (self.distance_matrix[0, :] + self.distance_matrix[current_location_id, :] >
                           self.agents_current_range[0, agent_taking_decision].item()).nonzero()
            if unreachbles[0].shape[0] != 0:
                mask[unreachbles[0], 0] = 1
            mask = np.logical_or(mask, (self.deadline_passed.T).numpy()).astype(mask.dtype)
            if mask[1:, 0].prod() == 1:  # if no other feasible locations, then go to depot
                mask[0, 0] = 0
        if mask.prod() != 0.0:
            mask[0,0] = 0
        mask = mask*(self.available_tasks).numpy() # making unavailable tasks
        return mask

    def generate_task_graph(self):

        locations = torch.tensor(self.locations)
        time_deadlines = (self.time_deadlines.T)
        task_size = (self.task_size.T)
        task_completion = self.task_completion.T #this is new
        deadlines_passed = self.deadline_passed.T
        node_properties = torch.cat((locations, time_deadlines, task_size, task_completion, deadlines_passed), dim=1) # ERROR: zero-dimensional tensor (at position 2) cannot be concatenated
        node_properties = node_properties[1:, :]  # excluding the depot
        node_properties[:, 0:4] = node_properties[:, 0:4] / node_properties[:, 0:4].max(
            dim=0).values  # normalizing all except deadline_passed
        distance_matrix = torch.cdist(node_properties, node_properties)
        adjacency_matrix = 1 / (1 + torch.cdist(node_properties, node_properties))
        adjacency_matrix = adjacency_matrix * (distance_matrix > 0).to(torch.float32)  # setting diagonal elements as 0
        node_properties = node_properties[:, :] * self.available_tasks[1:, :]  # masking the unavailable tasks
        adjacency_matrix = adjacency_matrix * (self.available_tasks[1:, :].T)
        adjacency_matrix = adjacency_matrix * self.available_tasks[1:, :]
        return node_properties, adjacency_matrix

    # Unchanged
    def generate_agents_graph(self):
        node_properties = torch.cat((torch.tensor(self.agents_destination_coordinates), self.agents_current_range.T,
                                     self.agents_current_payload.T, torch.tensor(self.agents_next_decision_time)),
                                    dim=1)
        distance_matrix = torch.cdist(node_properties, node_properties)
        adjacency_matrix = 1 / (1 + torch.cdist(node_properties, node_properties))
        adjacency_matrix = adjacency_matrix * (distance_matrix > 0).to(torch.float32)  # setting diagonal elements as 0
        return node_properties, adjacency_matrix

    # Unchanged
    def render(self, action):
        # Show the locations

        plt.plot(self.locations[0, 0], self.locations[0, 1], 'bo')
        for i in range(1, self.n_locations):
            if self.available_tasks[i, 0] == 1:
                if self.task_done[0, i] == 1:
                    plt.plot(self.locations[i, 0], self.locations[i, 1], 'go')
                elif self.nodes_visited[i, 0] == 0 and self.deadline_passed[0, i] == 0:
                    plt.plot(self.locations[i, 0], self.locations[i, 1], 'ro')
                elif self.deadline_passed[0, i] == 1:
                    plt.plot(self.locations[i, 0], self.locations[i, 1], 'ko')
        plt.plot(self.locations[action, 0], self.locations[action, 1], 'mo')
        prev_loc = self.locations[self.agents_prev_location][:, 0, :]
        next_loc = self.locations[self.agents_next_location][:, 0, :]
        diff = next_loc - prev_loc
        velocity = np.zeros((self.n_agents, 2))
        for i in range(self.n_agents):
            if diff[i, 0] == 0 and diff[i, 1] == 0:
                velocity[i, 0] = 0
                velocity[i, 1] = 0
            else:
                direction = diff[i, :] / (np.linalg.norm(diff[i, :]))
                velocity[i, :] = direction * self.agent_speed

        prev_time = self.time
        current_agent_locations = prev_loc + (prev_time - self.agents_prev_decision_time) * velocity

        agent_taking_decision = np.argmin(self.agents_next_decision_time)
        # current_location_id = self.agents_next_location[agent_taking_decision][0].copy()
        next_time = self.agents_next_decision_time[agent_taking_decision][0].copy()
        delta_t = (next_time - prev_time) / 10
        curr_time = prev_time
        for i in range(10):
            current_agent_locations = current_agent_locations + velocity * delta_t
            plt.plot(current_agent_locations[:, 0], current_agent_locations[:, 0], 'mv')
            curr_time = curr_time + delta_t
            deadlines_passed_ids = (self.time_deadlines < torch.tensor(curr_time)).nonzero()
            time.sleep(0.01)

        # print(prev_loc)
        # print(next_loc)
        # print("***********")
        for i in range(self.n_agents):
            plt.arrow(prev_loc[i, 0], prev_loc[i, 1], diff[i, 0], diff[i, 1])
        plt.draw()
        time.sleep(1)
        plt.clf()
        #   Grey as unavailable
        #   Red as active
        #   Green as done
        #   Black as deadline passed and not completed
        # Current location of the robots
        # Show arrow for destination
        # Encircle robot taking decision
        # encircle decision taken
        # Show movement inbetween decision-making

    def reset(self):
        self.action_space = Discrete(1)
        self.locations = np.random.random((self.n_locations, 2))
        self.depot = self.locations[0, :]
        self.visited = []
        self.agents = []
        self.agent_taking_decision = 1
        self.agents_location = []
        self.agents_prev_location = np.zeros((self.n_agents, 1), dtype=int)
        self.agents_next_location = np.zeros((self.n_agents, 1), dtype=int)
        self.agents_distance_travelled = np.zeros((self.n_agents, 1))
        self.total_distance_travelled = 0.0
        self.agent_taking_decision = 0
        self.current_location_id = 0
        self.nodes_visited = np.zeros((self.n_locations, 1))
        self.agents_distance_to_destination = np.zeros((self.n_agents, 1))

        self.distance_matrix = np.linalg.norm(self.locations[:, None, :] - self.locations[None, :, :], axis=-1)
        self.time = 0.0
        self.agent_speed = 0.01
        self.agents_next_decision_time = np.zeros((self.n_agents, 1))
        self.agents_prev_decision_time = np.zeros((self.n_agents, 1))
        self.agents_destination_coordinates = np.ones((self.n_agents, 1)) * self.depot
        self.total_reward = 0.0
        self.total_length = 0

        self.agents_current_range = torch.ones((1, self.n_agents), dtype=torch.float32) * self.max_range
        self.agents_current_payload = torch.ones((1, self.n_agents), dtype=torch.float32) * self.max_capacity
        self.time_deadlines = (torch.tensor(np.random.random((1, self.n_locations))) * .7 + .3) * 550
        self.time_deadlines[0, 0] = 1000000  # large number for depot,

        self.task_size = torch.tensor(np.random.random_integers(low=1, high=self.max_task_size + 1, size=(1, self.n_locations)), dtype=torch.int)
        self.original_task_size = np.copy(self.task_size)
        self.task_completion = torch.zeros((1, self.n_locations), dtype=torch.float32)

        self.task_done = torch.zeros((1, self.n_locations), dtype=torch.float32)
        self.deadline_passed = torch.zeros((1, self.n_locations), dtype=torch.float32)
        self.active_tasks = ((self.nodes_visited == 0).nonzero())[0]
        self.done = False

        if not self.enable_dynamic_tasks:  # this conditional might be unnecessary
            n_initial_tasks = self.n_locations
        else:
            n_initial_tasks = self.n_initial_tasks
        self.n_initial_tasks = n_initial_tasks
        self.available_tasks[0: n_initial_tasks, 0] = 1  # set the initial tasks available
        self.time_start = self.time_deadlines*(torch.rand((self.n_locations,1))*.5).T
        self.time_start[0,0:self.n_initial_tasks] = 0
        state = self.get_encoded_state()
        if self.enable_topological_features:
            self.topo_laplacian = None

            topo_laplacian = self.get_topo_laplacian(state)
            state["topo_laplacian"] = topo_laplacian
            self.topo_laplacian = topo_laplacian
        return state

