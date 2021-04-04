from abc import ABC, abstractmethod
from collections import defaultdict
import random
import sys
from typing import List, Dict, DefaultDict

import numpy as np
from gym.spaces import Space, Box
from gym.spaces.utils import flatdim

from rl2021.exercise5.matrix_game import actions_to_onehot

def obs_to_tuple(obs):
    return tuple([tuple(o) for o in obs])


class MultiAgent(ABC):
    """Base class for multi-agent reinforcement learning

    **DO NOT CHANGE THIS BASE CLASS**

    """

    def __init__(
        self,
        num_agents: int,
        action_spaces: List[Space],
        observation_spaces: List[Space],
        gamma: float,
        **kwargs
    ):
        """Constructor of base agent for Q-Learning

        Initializes basic variables of MARL agents
        namely epsilon, learning rate and discount rate.

        :param num_agents (int): number of agents
        :param action_spaces (List[Space]): action spaces of the environment for each agent
        :param observation_spaces (List[Space]): observation spaces of the environment for each agent
        :param gamma (float): discount factor (gamma)

        :attr n_acts (List[int]): number of actions for each agent
        """

        self.num_agents = num_agents
        self.action_spaces = action_spaces
        self.observation_spaces = observation_spaces
        self.n_acts = [flatdim(action_space) for action_space in action_spaces]

        self.gamma: float = gamma

    @abstractmethod
    def act(self, obs: List[np.ndarray]) -> List[int]:
        """Chooses an action for all agents given observations

        :param obs (List[np.ndarray] of float with dim (observation size)):
            received observations representing the current environmental state for each agent
        :return (List[int]): index of selected action for each agent
        """
        ...

    @abstractmethod
    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ...

    @abstractmethod
    def learn(self):
        ...


class IndependentQLearningAgents(MultiAgent):
    """Agent using the Independent Q-Learning algorithm

    ** YOU NEED TO IMPLEMENT THE FUNCTIONS IN THIS CLASS **
    """

    def __init__(self, learning_rate: float =0.5, epsilon: float =1.0, **kwargs):
        """Constructor of IndependentQLearningAgents

        :param learning_rate (float): learning rate for Q-learning updates
        :param epsilon (float): epsilon value for all agents


        :attr q_tables (List[DefaultDict]): tables for Q-values mapping (OBS, ACT) pairs of observations
            and actions to respective Q-values for all agents

        Initializes some variables of the Independent Q-Learning agents, namely the epsilon, discount rate
        and learning rate
        """  

        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        # initialise Q-tables for all agents
        self.q_tables: List[DefaultDict] = [defaultdict(lambda: 0) for i in range(self.num_agents)]


    def act(self, obss: List[np.ndarray]) -> List[int]:
        """Implement the epsilon-greedy action selection here

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**
        :param obss (List[np.ndarray] of float with dim (observation size)):
            received observations representing the current environmental state for each agent
        :return (List[int]): index of selected action for each agent
        """
        actions = []
        print('self.n_acts=== ', max(self.n_acts))
        
        num_actions = self.action_spaces[0].n # or self.n_acts[0] or max(self.n_acts) ??? Does it matter?
        ### PUT YOUR CODE HERE ###
        for current_agent in range(self.num_agents):

            if np.random.random() < self.epsilon:
                actions.append(np.random.randint(0,num_actions,1,dtype=int)[0])
            else:
                q_tab = self.q_tables[current_agent]
                print('q_tab === ', q_tab)
                q_values = [q_tab[(obss[current_agent], act)] for act in range(num_actions)]
                print('q_values === ', q_values)
                max_acts_arg = np.argmax(q_values)
                print('max_acts_arg === ', max_acts_arg)            
                # max_val = max(q_values)
                # print('max_val === ', max_val) 
                # max_acts = [idx for idx, act_val in enumerate(q_values) if act_val == max_val]
                # print('max_acts === ', max_acts)
                actions.append(max_acts_arg)
        
        # Checking the test for IQL
        action = list(self.action_spaces.sample())
        obs = list(self.observation_spaces.sample())
        for i, (o, a) in enumerate(zip(obs, action)):
            var = self.q_tables[i][(o, a)]
            print(f"loop {i} === {var} and type {type(var)}")

        # raise NotImplementedError("Needed for Q5")
        return actions 

    def learn(
        self, obss: List[np.ndarray], actions: List[int], rewards: List[float], n_obss: List[np.ndarray], dones: List[bool]
    ) -> List[float]:
        """Updates the Q-tables based on agents' experience

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        :param obss (List[np.ndarray] of float with dim (observation size)):
            received observations representing the current environmental state for each agent
        :param action (List[int]): index of applied action of each agent
        :param rewards (List[float]): received reward for each agent
        :param n_obss (List[np.ndarray] of float with dim (observation size)):
            received observations representing the next environmental state for each agent
        :param dones (List[bool]): flag indicating whether a terminal state has been reached for each agent
        :return (List[float]): updated Q-values for current observation-action pair of each agent
        """
        updated_values = [] 

        for current_agent in range(self.num_agents): 
            q_tab = self.q_tables[current_agent] 
            key = (obss[current_agent],actions[current_agent]) 

            q_values = [q_tab[(obss[current_agent], act)] for act in range(3)] 
            # print('q_values === ', q_values)
            q_values_max = np.max(q_values)
            # print('q_values_max === ', q_values_max) 

            targ = rewards[current_agent] + self.gamma * q_tab[(n_obss[current_agent],q_values_max)] 
            update = q_tab[key] + self.learning_rate * (targ - q_tab[key])
            
            q_tab[key] = update 
            updated_values.append(update)

        # raise NotImplementedError("Needed for Q5")
        return updated_values


    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ### PUT YOUR CODE HERE ###
        # raise NotImplementedError("Needed for Q5")


class JointActionLearning(MultiAgent):
    """Agents using the Joint Action Learning algorithm with Opponent Modelling

    ** YOU NEED TO IMPLEMENT THE FUNCTIONS IN THIS CLASS **
    """

    def __init__(self, learning_rate: float =0.5, epsilon: float =1.0, **kwargs):
        """Constructor of JointActionLearning

            :param learning_rate (float): learning rate for Q-learning updates
            :param epsilon (float): epsilon value for all agents

            :attr q_tables (List[DefaultDict]): tables for Q-values mapping (OBS, ACT) pairs of
                observations and joint actions to respective Q-values for all agents
            :attr models (List[DefaultDict[DefaultDict]]): each agent holding model of other agent
                mapping observation to other agent actions to count of other agent action

            Initializes some variables of the Joint Action Learning agents, namely the epsilon, discount
            rate and learning rate
        """

        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.n_acts = [flatdim(action_space) for action_space in self.action_spaces]

        # initialise Q-tables for all agents
        self.q_tables: List[DefaultDict] = [defaultdict(lambda: 0) for _ in range(self.num_agents)]

        # initialise models for each agent mapping state to other agent actions to count of other agent action
        self.models = [defaultdict(lambda: defaultdict(lambda: 0)) for _ in range(self.num_agents)] 

        # count observations - count for each agent
        self.c_obss = [defaultdict(lambda: 0) for _ in range(self.num_agents)]


    def act(self, obss: List[np.ndarray]) -> List[int]:
        """Implement the epsilon-greedy action selection here
            YOU MUST IMPLEMENT THIS FUNCTION FOR Q5

            :param obss (List[np.ndarray] of float with dim (observation size)):
                received observations representing the current environmental state for each agent
            :return (List[int]): index of selected action for each agent
        """
        joint_action = []
        ### PUT YOUR CODE HERE ###
        for current_agent in range(self.num_agents):
            c_obss_agent = self.c_obss[current_agent]
            q_tab = self.q_tables[current_agent]
            model = self.models[current_agent]

            if np.random.random() < self.epsilon or self.c_obss[current_agent][0] == 0:
                joint_action.append(np.random.randint(0,3,1,dtype=int)[0])
                #print('little bastard === ', joint_action)
            else:
                aggregates = []
                for own_action in range(3):
                    agg = 0
                    for others_action in range(3):
                        action_key = (own_action, others_action)
                        action_key_alt = (others_action, own_action)
                        
                        if c_obss_agent[obss[current_agent]] != 0:
                            if current_agent == 0: #own turn
                                agg += model[obss[current_agent]][others_action] / c_obss_agent[obss[current_agent]]*q_tab[(obss[current_agent],action_key)]
                            else:
                                agg += model[obss[current_agent]][others_action] / c_obss_agent[obss[current_agent]]*q_tab[(obss[current_agent],action_key_alt)]
                        else:
                            agg = 0

                        # print(f'others_action === {others_action}')
                        # print(f'c_obss_agent[0] === {c_obss_agent[0]}')
                        #print(f"epsilon: {self.epsilon}")
                        #print(f"c_obss: {self.c_obss[current_agent]}")
                        #print('q_tab[(0,action_key)] === ', q_tab[(0,action_key)])
                        # agg = agg + model[0][others_action]/c_obss_agent[0]*q_tab[(0,action_key)]
                    #print('agg ===  ', agg)
                    aggregates.append(agg)
                action = np.argmax(aggregates)
                #print(f"action:{action}, aggregates: {aggregates}")
                ########################## 
                joint_action.append(action)
            
        # print(f'actions - act === {joint_action}')
        # raise NotImplementedError("Needed for Q5")

        print(f' joint action returned === {joint_action}')
        return joint_action

    def learn(
        self, obss: List[np.ndarray], actions: List[int], rewards: List[float], n_obss: List[np.ndarray], dones: List[bool]
    ) -> List[float]:
        """Updates the Q-tables and models based on agents' experience

            YOU MUST IMPLEMENT THIS FUNCTION FOR Q5

            :param obss (List[np.ndarray] of float with dim (observation size)):
                received observations representing the current environmental state for each agent
            :param action (List[int]): index of applied action of each agent
            :param rewards (List[float]): received reward for each agent
            :param n_obss (List[np.ndarray] of float with dim (observation size)):
                received observations representing the next environmental state for each agent
            :param dones (List[bool]): flag indicating whether a terminal state has been reached for each agent
            :return (List[float]): updated Q-values for current observation-action pair of each agent

        """
        #print("im in learn")
        updated_values = []
        for current_agent in range(self.num_agents):
            current_agent_action = actions[current_agent]
            other_agent_action = actions[1-current_agent]

            q_tab = self.q_tables[current_agent]
            model = self.models[current_agent]
            c_obss_agent = self.c_obss[current_agent]

            action_key = tuple(actions)
            model[obss[current_agent]][other_agent_action] += 1
            c_obss_agent[obss[current_agent]] +=1

            ##################
            joint_action = []
            for own_action in range(3):
                agg = 0 
                for others_action in range(3):
                    current_action_key = (own_action, others_action)
                    #print(f"action_key: {action_key}, q_tab value: {q_tab[(0,action_key)]}, N(s): {c_obss_agent[0]}, model: {model[0][others_action]}")

                    action_key_own = (own_action, others_action)
                    action_key_alt = (others_action, own_action)
                    
                    if c_obss_agent[n_obss[current_agent]] != 0:
                        if current_agent == 0: #own turn
                            agg += model[n_obss[current_agent]][others_action] / c_obss_agent[n_obss[current_agent]]*q_tab[(n_obss[current_agent],action_key_own)]
                        else:
                            agg += model[n_obss[current_agent]][others_action] / c_obss_agent[n_obss[current_agent]]*q_tab[(n_obss[current_agent],action_key_alt)]
                    else:
                        agg = 0
                    
                    # agg = agg + model[0][others_action]/c_obss_agent[0]*q_tab[(,current_action_key)]
                    # agghhh it matters which way around the choices are!! agh!

                joint_action.append(agg)
            ##################

            action_ = np.max(joint_action)
            #print('aggregates ==== ', joint_action)
            
            print(f"action_key: {action_key}")
            #print(f"q_tab before: {q_tab[(0,action_key)]}")
            update = q_tab[(obss[current_agent],action_key)] + self.learning_rate * (rewards[current_agent] + self.gamma * action_ - q_tab[(obss[current_agent],action_key)])
            # print(f"update: {update}")
            update = float(update)
            # print(f"update_after: {update}")
            #print(f"q_tab after: {q_tab[(0,action_key)]}")
            q_tab[(0,action_key)] = update
            updated_values.append(update)

        # print('actions - learn ===  ',actions)
        return updated_values
        # raise NotImplementedError("Needed for Q5")


    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        YOU MUST IMPLEMENT THIS FUNCTION FOR Q5

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ### PUT YOUR CODE HERE ###
        # raise NotImplementedError("Needed for Q5")
        #         ### PUT YOUR CODE HERE ###
        #raise NotImplementedError("Needed for Q5")
        t = 1 if timestep == 0 else timestep
        self.epsilon = max(0.01, self.epsilon/(2**int(t/20)))
        self.learning_rate = max(0.001, self.learning_rate/(2**int(t/20)))