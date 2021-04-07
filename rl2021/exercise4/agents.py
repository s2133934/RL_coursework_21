import os
import gym
import numpy as np
from torch.optim import Adam
from typing import Dict, Iterable
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from rl2021.exercise3.agents import Agent
from rl2021.exercise3.networks import FCNetwork, Tanh2
from rl2021.exercise3.replay import Transition


class DDPG(Agent):
    """ DDPG

        ** YOU NEED TO IMPLEMENT THE FUNCTIONS IN THIS CLASS **

        :attr critic (FCNetwork): fully connected critic network
        :attr critic_optim (torch.optim): PyTorch optimiser for critic network
        :attr policy (FCNetwork): fully connected actor network for policy
        :attr policy_optim (torch.optim): PyTorch optimiser for actor network
        :attr gamma (float): discount rate gamma
        """

    def __init__(
            self,
            action_space: gym.Space,
            observation_space: gym.Space,
            gamma: float,
            critic_learning_rate: float,
            policy_learning_rate: float,
            critic_hidden_size: Iterable[int],
            policy_hidden_size: Iterable[int],
            tau: float,
            **kwargs,
    ):
        """
        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q4**

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param gamma (float): discount rate gamma
        :param critic_learning_rate (float): learning rate for critic optimisation
        :param policy_learning_rate (float): learning rate for policy optimisation
        :param critic_hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected critic
        :param policy_hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected policy
        :param tau (float): step for the update of the target networks
        """
        super().__init__(action_space, observation_space)
        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.shape[0]

        # ######################################### #
        #  BUILD YOUR NETWORKS AND OPTIMIZERS HERE  #
        # ######################################### #
        self.actor = FCNetwork(
            (STATE_SIZE, *policy_hidden_size, ACTION_SIZE), output_activation=Tanh2
        )
        self.actor_target = FCNetwork(
            (STATE_SIZE, *policy_hidden_size, ACTION_SIZE), output_activation=Tanh2
        )

        self.actor_target.hard_update(self.actor)

        self.critic = FCNetwork(
            (STATE_SIZE + ACTION_SIZE, *critic_hidden_size, 1), output_activation=None
        )
        self.critic_target = FCNetwork(
            (STATE_SIZE + ACTION_SIZE, *critic_hidden_size, 1), output_activation=None
        )
        self.critic_target.hard_update(self.critic)

        self.policy_optim = Adam(self.actor.parameters(), lr=policy_learning_rate, eps=1e-3)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_learning_rate, eps=1e-3)


        # ############################################# #
        # WRITE ANY EXTRA HYPERPARAMETERS YOU NEED HERE #
        # ############################################# #
        self.gamma = gamma
        self.critic_learning_rate = critic_learning_rate
        self.policy_learning_rate = policy_learning_rate
        self.tau = tau

        # ################################################### #
        # DEFINE A GAUSSIAN THAT WILL BE USED FOR EXPLORATION #
        # ################################################### #

        ### PUT YOUR CODE HERE ###
        # Just need to take a sample and add some noise right??
        action_space_sample = self.action_space.sample()
        self.noise = Normal(0, 0.1 * torch.ones(len(action_space_sample)))
        # raise NotImplementedError("Needed for Q4")
        

        # ############################### #
        # WRITE ANY AGENT PARAMETERS HERE #
        # ############################### #

        self.saveables.update(
            {
                "actor": self.actor,
                "actor_target": self.actor_target,
                "critic": self.critic,
                "critic_target": self.critic_target,
                "policy_optim": self.policy_optim,
                "critic_optim": self.critic_optim,
            }
        )

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


    def schedule_hyperparameters(self, timestep: int, max_timesteps: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q4**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ### PUT YOUR CODE HERE ###
        # Tried a few things here but they just made it way worse!?
        # raise NotImplementedError("Needed for Q4")

    def act(self, obs: np.ndarray, explore: bool):
        """Returns an action (should be called at every timestep)

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q4**

        When explore is False you should select the best action possible (greedy). However, during exploration,
        you should be implementing exporation using the self.noise variable that you should have declared in the __init__.
        Use schedule_hyperparameters() for any hyperparameters that you want to change over time.

        :param obs (np.ndarray): observation vector from the environment
        :param explore (bool): flag indicating whether we should explore
        :return (sample from self.action_space): action the agent should perform
        """
        obs = torch.tensor(obs, dtype = torch.float32)

        if explore == False:
            # Be greedy!
            sample = self.actor.forward(obs).detach()
            # sample = torch.argmax(actions).item()

        if explore == True:
            # Use self.noise
            sample = (self.actor.forward(obs) + self.noise.sample()).detach()

        sample.clamp(min=-2,max=2)

        return sample
        
    def update(self, batch: Transition) -> Dict[str, float]:
        """Update function for DQN
        
        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q4**

        This function is called after storing a transition in the replay buffer. This happens
        every timestep. It should update your networks and return the q_loss and the policy_loss in the form of a
        dictionary.

        :param batch (Transition): batch vector from replay buffer
        :return (Dict[str, float]): dictionary mapping from loss names to loss values
        """        
        q_loss = 0.0 #tensor.type(torch.float32)?????
        p_loss = 0.0        

        # print("previous state shape", batch.next_states.shape)
        # print("previous state type", batch.next_states.type)

        actor_targ_output = self.actor_target.forward(batch.next_states)
        critic_target_output = self.critic_target(torch.cat([batch.next_states,actor_targ_output], dim = 1))

        y_i = batch.rewards + self.gamma * ((1-batch.done) * critic_target_output)

        critic_output = self.critic.forward(torch.cat([batch.states,batch.actions], dim = 1))

        q_loss = 1/len(batch.actions) * torch.sum((y_i - critic_output)**2)

        self.critic_optim.zero_grad()
        q_loss.backward(retain_graph = True) 
        self.critic_optim.step()

        current_action = self.actor.forward(batch.states)
        critic_output_p_loss = self.critic.forward(torch.cat([batch.states, current_action], dim = 1))
        
        p_loss = -1/len(batch.actions) * torch.sum(critic_output_p_loss)

        self.policy_optim.zero_grad() 
        p_loss.backward()
        # -torch.sum(critic_output).backward(retain_graph = True) ??
        self.policy_optim.step()

        self.critic_target.soft_update(self.critic,self.tau)
        self.actor_target.soft_update(self.actor,self.tau)

        return {"q_loss": q_loss,
                "p_loss": p_loss}
