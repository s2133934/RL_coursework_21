from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np
import random
from typing import List, Dict, DefaultDict
from gym.spaces import Space
from gym.spaces.utils import flatdim


class Agent(ABC):
    """Base class for Q-Learning agent

    **ONLY CHANGE THE BODY OF THE act() FUNCTION**

    """

    def __init__(
        self,
        action_space: Space,
        obs_space: Space,
        gamma: float,
        epsilon: float,
        **kwargs
    ):
        """Constructor of base agent for Q-Learning

        Initializes basic variables of the Q-Learning agent
        namely the epsilon, learning rate and discount rate.

        :param action_space (int): action space of the environment
        :param obs_space (int): observation space of the environment
        :param gamma (float): discount factor (gamma)
        :param epsilon (float): epsilon for epsilon-greedy action selection

        :attr n_acts (int): number of actions
        :attr q_table (DefaultDict): table for Q-values mapping (OBS, ACT) pairs of observations
            and actions to respective Q-values
        """

        self.action_space = action_space
        self.obs_space = obs_space
        self.n_acts = flatdim(action_space)

        self.epsilon: float = epsilon
        self.gamma: float = gamma

        self.q_table: DefaultDict = defaultdict(lambda: 0)

        self.epsilon_initial = self.epsilon ##### CHECK THIS

    def act(self, obs: np.ndarray) -> int:
        """Implement the epsilon-greedy action selection here

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**

        :param obs (np.ndarray of float with dim (observation size)):
            received observation representing the current environmental state
        :return (int): index of selected action
        """
        # print("Im in Agent class - act function")
        
        act_vals = [self.q_table[(obs, act)] for act in range(self.n_acts)]
        max_val = max(act_vals)
        max_acts = [idx for idx, act_val in enumerate(act_vals) if act_val == max_val]

        if random.random() < self.epsilon:
            return random.randint(0, self.n_acts - 1) # choose random integer between 0 and n_acts-1 / EXPLORE
        else:
            return random.choice(max_acts) # Choose randomly between actions of max value - EXPLOIT


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


class QLearningAgent(Agent):
    """Agent using the Q-Learning algorithm

    """

    def __init__(self, alpha: float, **kwargs):
        """Constructor of QLearningAgent

        Initializes some variables of the Q-Learning agent, namely the epsilon, discount rate
        and learning rate alpha.

        :param alpha (float): learning rate alpha for Q-learning updates
        """

        super().__init__(**kwargs)
        self.alpha: float = alpha


    def learn(self, obs: np.ndarray, action: int, reward: float, n_obs: np.ndarray, done: bool) -> float:
        """Updates the Q-table based on agent experience

            **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**

            :param obs (np.ndarray of float with dim (observation size)):
                received observation representing the current environmental state
            :param action (int): index of applied action
            :param reward (float): received reward
            :param n_obs (np.ndarray of float with dim (observation size)):
                received observation representing the next environmental state

            :param done (bool): flag indicating whether a terminal state has been reached
            :return (float): updated Q-value for current observation-action pair
        """
        # print('Im in Q learning Agent - learn function')

        act_vals = [self.q_table[(obs, act)] for act in range(self.n_acts)]
        max_val = max(act_vals)
        max_acts = [idx for idx, act_val in enumerate(act_vals) if act_val == max_val]
        n_action = random.choice(max_acts)
        # n_action (single integer):
        # action chosen by taking the maximum action values in q_table for a the next state
    
        target_value = reward + self.gamma * (1 - done) * self.q_table[(n_obs, n_action)]
        self.q_table[(obs, action)] += self.alpha * (target_value - self.q_table[(obs, action)])

        return self.q_table[(obs, action)]

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

            **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**

            This function is called before every episode and allows you to schedule your
            hyperparameters.

            :param timestep (int): current timestep at the beginning of the episode
            :param max_timestep (int): maximum timesteps that the training loop will run for

            returns: none (only updates parameters)
        """
        ### PUT YOUR CODE HERE ###
        
        decay_epsilon = 0.07
        # self.epsilon = 1.0-(min(1.0, timestep/(decay_epsilon*max_timestep)))*0.95
        self.epsilon = self.epsilon_initial * (1.0 - (min(1.0, timestep / (decay_epsilon * max_timestep))) * 0.95)
        # raise NotImplementedError("Needed for Q2")


class MonteCarloAgent(Agent):
    """
    Agent using the Monte-Carlo algorithm for training
    """

    def __init__(self, **kwargs):
        """Constructor of MonteCarloAgent

        Initializes some variables of the Monte-Carlo agent, namely epsilon,
        discount rate and an empty observation-action pair dictionary.

        :attr sa_counts (Dict[(Obs, Act), int]): dictionary to count occurrences observation-action pairs
        """
        super().__init__(**kwargs)
        self.sa_counts = {}

    def learn(self, obses: List[np.ndarray], actions: List[int], rewards: List[float]) -> Dict:

        """
        Updates the Q-table based on agent experience

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**

        :param obses (List(np.ndarray) with numpy arrays of float with dim (observation size)):
            list of received observations representing environmental states of trajectory (in
            the order they were encountered) 
        :param actions (List[int]): list of indices of applied actions in trajectory (in the
            order they were applied)
        :param rewards (List[float]): list of received rewards during trajectory (in the order
            they were received)

        :return (Dict): (for me "updated_values") A dictionary containing the updated Q-value of all the 
        updated state-action pairs indexed by the state action pair.
        """
        updated_values = {}
        ### PUT YOUR CODE HERE ###
        G = 0 #start with gains = 0 

        #create an episode, and denote episode[i] as a step = [S,A,R]
        # episode = [ [obses[i],actions[i],rewards[i]] for i in range(len(rewards)) ]
        episode_sa_pairs = [ (obses[i],actions[i]) for i in range(len(rewards)) ] 
        
        # state_action = [] 
        # for i in range(len(obses)): 
        #     state_action.append((obses[i],actions[i]))
        #     sa_pairs_list.append((obses[i],actions[i])) 
        # print('state_action=== ', type(state_action))
        # print('sa_pairs_list == ', sa_pairs_list)
        # print('episode_sa_pairs == ', type(episode_sa_pairs))
        # print('obses == \n', obses, '\n len(obses) == ', len(obses))
        # print('actions == \n', actions, '\n len(actions) == ', len(actions))
        # print('rewards == \n', rewards, '\n len(rewards) == ', len(rewards))
        # print('length q_table == ', len(self.q_table)) 
        # print('sa_counts == ', self.sa_counts)

        #for all steps of the episode (uses obses as same length as actions and rewards)
        # k=0
        for current_step in range(len(obses)-1,-1,-1):
            G = self.gamma * G + rewards[current_step]
            # print('G == ', G)            
            # if state of current step doesnt appear in the rest of the list (which is reversed) 
            # then it is the first time that state is encountered (tested in terminal)
            # if obses[current_step] not in [x for x in obses[::-1]][current_step+1:] == True:
            # edit: must be the sa pair, not just the state in obses, so created a list of the sa pairs 
            # in episode - tested in terminal
            current_state = episode_sa_pairs[current_step][0] 
            # print('current_State == ', current_state)
            current_action = episode_sa_pairs[current_step][1]
            # print('current_action == ', current_action)
            key = (current_state,current_action)
            # if episode_sa_pairs[current_step] not in [x for x in episode_sa_pairs[::-1]][current_step+1:]:
            if key not in episode_sa_pairs[:current_step]:
                # k+=1 
                # print('found a first instance .... ', current_step)

                # print('type(key) == ', key)
                # returns(state,action).append(G)
                # returns[current_state].append(G)

                if key not in self.sa_counts.keys():
                    self.sa_counts[key] = 1
                    # print('created key in sa_counts')
                else:
                    self.sa_counts[key] += 1
                    # print('found key in sa_counts, adding 1')
                # print('self.sa_counts', self.sa_counts)
                # Q(state,action) = average(returns(state,action)) GLOBAL
                # print('values == ', self.q_table[key], self.sa_counts[key])

                updated_values[key] = self.q_table[key] + ( (G - self.q_table[key]) / self.sa_counts[key] )
                
                self.q_table[key] = updated_values[key]

                # A_star = np.argmax(Q(state,action)) 
                # This is the epsilon soft bit
                # for all a in actions:
        # for i, step in enumerate(episode[::-1]):
        #     G = self.gamma*G + step[2]

        #     if step[0] not in [x[0] for x in episode[::-1][len(episode)-i:]]:
        #         idx = (step[0][0], step[0][1])
        #         returns[idx].append(G)
        #         self.q_table[(obs, action)] = np.average(returns[idx])
        #         # newValue = np.average(returns[idx])

        #         updated_values[idx[0], idx[1]] = newValue

        # raise NotImplementedError("Needed for Q2")
        return updated_values

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        decay_epsilon = 0.001
        self.epsilon = 1.0-(min(1.0, timestep/(decay_epsilon*max_timestep)))*0.45
        # self.epsilon = self.epsilon_initial * (1.0 - (min(1.0, timestep / (decay_epsilon * max_timestep))) * 0.95)
        
        # SOMEVALUE = 80000
        # if timestep > SOMEVALUE:
        #     self.gamma = 0.999999
        # self.gamma = 0.9 + 0.1 * timestep/max_timestep    

        # raise NotImplementedError("Needed for Q2")

# if __name__ == "__main__":
    # print('Script found')
    # Agent().act(np.array([0,0]))