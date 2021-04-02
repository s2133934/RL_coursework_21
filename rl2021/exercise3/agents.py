from abc import ABC, abstractmethod
from copy import deepcopy
import gym
import numpy as np
import os.path
from torch import Tensor
from torch.distributions.categorical import Categorical
import torch.nn
from torch.optim import Adam
from typing import Dict, Iterable, List

from rl2021.exercise3.networks import FCNetwork
from rl2021.exercise3.replay import Transition, ReplayBuffer


class Agent(ABC):
    """Base class for Deep RL Exercise 3 Agents

    **DO NOT CHANGE THIS CLASS**

    :attr action_space (gym.Space): action space of used environment
    :attr observation_space (gym.Space): observation space of used environment
    :attr saveables (Dict[str, torch.nn.Module]):
        mapping from network names to PyTorch network modules

    Note:
        see http://gym.openai.com/docs/#spaces for more information on Gym spaces
    """

    def __init__(self, action_space: gym.Space, observation_space: gym.Space):
        """The constructor of the Agent Class

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        """
        self.action_space = action_space
        self.observation_space = observation_space

        self.saveables = {} 

    def save(self, path: str, suffix: str = "") -> str:
        """Saves saveable PyTorch models under given path

        The models will be saved in directory found under given path in file "models_{suffix}.pt"
        where suffix is given by the optional parameter (by default empty string "")

        :param path (str): path to directory where to save models
        :param suffix (str, optional): suffix given to models file
        :return (str): path to file of saved models file
        """
        torch.save(self.saveables, path)
        return path

    def restore(self, save_path: str):
        """Restores PyTorch models from models file given by path

        :param save_path (str): path to file containing saved models
        """
        dirname, _ = os.path.split(os.path.abspath(__file__))
        save_path = os.path.join(dirname, save_path)
        checkpoint = torch.load(save_path)
        for k, v in self.saveables.items():
            v.load_state_dict(checkpoint[k].state_dict())

    @abstractmethod
    def act(self, obs: np.ndarray):
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
    def update(self):
        ...


class DQN(Agent):
    """The DQN agent for exercise 3

    **YOU NEED TO IMPLEMENT FUNCTIONS IN THIS CLASS**

    :attr critics_net (FCNetwork): fully connected DQN to compute Q-value estimates
    :attr critics_target (FCNetwork): fully connected DQN target network
    :attr critics_optim (torch.optim): PyTorch optimiser for DQN critics_net
    :attr learning_rate (float): learning rate for DQN optimisation
    :attr update_counter (int): counter of updates for target network updates
    :attr target_update_freq (int): update frequency (number of iterations after which the target
        networks should be updated)
    :attr batch_size (int): size of sampled batches of experience
    :attr gamma (float): discount rate gamma
    """

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.Space,
        learning_rate: float,
        hidden_size: Iterable[int],
        target_update_freq: int,
        batch_size: int,
        gamma: float,
        **kwargs,
    ):
        """The constructor of the DQN agent class

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param learning_rate (float): learning rate for DQN optimisation
        :param hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected DQNs
        :param target_update_freq (int): update frequency (number of iterations after which the target
            networks should be updated)
        :param batch_size (int): size of sampled batches of experience
        :param gamma (float): discount rate gamma
        """
        super().__init__(action_space, observation_space)

        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.n

        # ######################################### #
        #  BUILD YOUR NETWORKS AND OPTIMIZERS HERE  #
        # ######################################### #
        
        self.critics_net = FCNetwork((STATE_SIZE, *hidden_size, ACTION_SIZE), output_activation=None)

        self.critics_target = deepcopy(self.critics_net)

        self.critics_optim = Adam(self.critics_net.parameters(), lr=learning_rate, eps=1e-3) 

        # ############################################# #
        # WRITE ANY HYPERPARAMETERS YOU MIGHT NEED HERE #
        # ############################################# #
        self.learning_rate = learning_rate
        self.update_counter = 0
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.gamma = gamma

        self.epsilon = 1
        # ########################################## ADDED:::

        self.epsilon_initial = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995 #0.995

        self.loss_func = torch.nn.MSELoss()
        # Chose MSE loss as we are characterising the error as the distance between the target and the actual value

        # ###########################################
        self.saveables.update(
            {
                "critics_net": self.critics_net,
                "critics_target": self.critics_target,
                "critic_optim": self.critics_optim,
            }
        )

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        #Set a hard minimum on epsilon to ensure some exploration all of the time and a non negative epsilon!
        # self.episilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)
        # https://github.com/adamprice97/cartpole/blob/master/cartpole.py
        # self.exploration_rate *= EXPLORATION_DECAY

        epsilon_decayed = self.epsilon * self.epsilon_decay
        # self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)
        self.epsilon = max(self.epsilon_min, epsilon_decayed) 
        # raise NotImplementedError("Needed for Q3")

    def act(self, obs: np.ndarray, explore: bool):
        """Returns an action (should be called at every timestep)

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        When explore is False you should select the best action possible (greedy). However, during
        exploration, you should be implementing an exploration strategy (like e-greedy). Use
        schedule_hyperparameters() for any hyperparameters that you want to change over time.

        :param obs (np.ndarray): observation vector from the environment
        :param explore (bool): flag indicating whether we should explore
        :return (sample from self.action_space): action the agent should perform
        """
        ### PUT YOUR CODE HERE ###

        obs = torch.from_numpy(obs)
        obs = obs.type(torch.FloatTensor)


        if explore == True:
            # Agent follow epsilon greedy policy:
            if np.random.random() < self.epsilon:
                # j = np.random.choice(3) 
                # return np.random.choice(self.action_space)
                return self.action_space.sample()
            else:
                actions = self.critics_net.forward(obs)
                return torch.argmax(actions).item()

        elif explore == False:
            # Agent is greedy
            actions = self.critics_net.forward(obs)
            return torch.argmax(actions).item()

        # raise NotImplementedError("Needed for Q3")

    def update(self, batch: Transition) -> Dict[str, float]:
        """Update function for DQN

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        This function is called after storing a transition in the replay buffer. This happens
        every timestep. It should update your network and return the Q-loss in the form of a
        dictionary.

        :param batch (Transition): batch vector from replay buffer
        :return (Dict[str, float]): dictionary mapping from loss names to loss values
        """
        ### PUT YOUR CODE HERE ###
        q_loss = 0.0

        action_batch = batch.actions
        action_batch = action_batch.type(torch.LongTensor)
        # Evaluate current Q-values and predicted Q-values
        q_current = self.critics_net.forward(batch.states).gather(1,action_batch) # shape[10,1]

        q_next = self.critics_target.forward(batch.next_states).detach() # shape[10,2] 

        q_max = torch.max(q_next,dim=1)[0].view(self.batch_size,1) # shape [10] -> [10,1]

        q_target = batch.rewards + self.gamma * ((1-batch.done) * q_max) # shape[10,1]

        # Loss Function
        q_loss = self.loss_func(q_current,q_target) #MSELoss(Current Q values, Target Q Values)
        q_loss.clamp(min=-1,max=1)

        self.critics_optim.zero_grad()
        q_loss.backward()
        self.critics_optim.step()

        self.update_counter += 1

        if self.update_counter % self.target_update_freq ==0:
            self.critics_target.hard_update(self.critics_net)
       
        return {"q_loss": q_loss.detach()}


class Reinforce(Agent):
    """ The Reinforce Agent for Ex 3

    ** YOU NEED TO IMPLEMENT THE FUNCTIONS IN THIS CLASS **
    :attr learning_rate (float): learning rate for DQN optimisation
    :attr gamma (float): discount rate gamma
    :attr policy (FCNetwork): fully connected network for policy
    :attr policy_optim (torch.optim): PyTorch optimiser for policy network
    """

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.Space,
        learning_rate: float,
        hidden_size: Iterable[int],
        gamma: float,
        **kwargs,
    ):
        """
        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param learning_rate (float): learning rate for DQN optimisation
        :param hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected DQNs
        :param gamma (float): discount rate gamma
        """
        super().__init__(action_space, observation_space)
        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.n

        # ######################################### #
        #  BUILD YOUR NETWORKS AND OPTIMIZERS HERE  #
        # ######################################### #

        ### DO NOT CHANGE THE OUTPUT ACTIVATION OF THIS POLICY ###
        self.policy = FCNetwork(
            (STATE_SIZE, *hidden_size, ACTION_SIZE), output_activation=torch.nn.modules.activation.Softmax
        )

        self.policy_optim = Adam(self.policy.parameters(), lr=learning_rate, eps=1e-3)

        # ############################################# #
        # WRITE ANY EXTRA HYPERPARAMETERS YOU NEED HERE #
        # ############################################# #
        self.learning_rate = learning_rate
        self.gamma = gamma

        # ############################### #
        # WRITE ANY AGENT PARAMETERS HERE #
        # ############################### #
        self.learning_rate_min = 1e-4
        self.learning_rate_decay = 0.995
        self.learning_rate = 0.001
        self.learning_rate_intial = 0.001
        # ###############################################
        self.saveables.update(
            {
                "policy": self.policy,
            }
        )

    def schedule_hyperparameters(self, timestep: int, max_timesteps: int):
        """Updates the hyperparameters 

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ### PUT YOUR CODE HERE ###
        # raise NotImplementedError("Needed for Q3")
        self.learning_rate = self.learning_rate_intial * (1.0- (min(1.0,timestep / (0.07 * max_timesteps))) * 0.995)

    def act(self, obs: np.ndarray, explore: bool):
        """Returns an action (should be called at every timestep)

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        Select an action from the model's stochastic policy by sampling a discrete action
        from the distribution specified by the model output

        :param obs (np.ndarray): observation vector from the environment
        :param explore (bool): flag indicating whether we should explore
        :return (sample from self.action_space): action the agent should perform
        """
        ### PUT YOUR CODE HERE ###

        obs = torch.tensor(obs, dtype = torch.float32)
        
        p = self.policy.forward(obs)
        # Take a random choice from our probability distribution:
        sample = np.random.choice(range(self.action_space.n), p= p.detach().numpy(), size = 1)[0]

        return sample

        # raise NotImplementedError("Needed for Q3")

    def update(
        self, rewards: List[float], observations: List[np.ndarray], actions: List[int],
        ) -> Dict[str, float]:
        """Update function for policy gradients

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        :param rewards (List[float]): rewards of episode (from first to last)
        :param observations (List[np.ndarray]): observations of episode (from first to last)
        :param actions (List[int]): applied actions of episode (from first to last)
        :return (Dict[str, float]): dictionary mapping from loss names to loss values
            losses
        """
        ### PUT YOUR CODE HERE ###
        # raise NotImplementedError("Needed for Q3")
        p_loss = 0.0

        obs = torch.tensor(np.array(observations), dtype = torch.float32)
        G=0
        loss = []
        for t in range(len(observations)-1,-1,-1):
            G = self.gamma * G + rewards[t]
            loss.append(- self.gamma * G * torch.log(self.policy.forward(obs[t])[actions[t]]))
        
        p_loss = sum(loss)/len(rewards)

        # Reset Gradient
        self.policy_optim.zero_grad()
        p_loss.backward()
        self.policy_optim.step()

        return{"p_loss": loss}
