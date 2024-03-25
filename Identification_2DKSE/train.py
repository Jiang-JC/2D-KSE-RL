import torch
import torch.nn.functional as F
import numpy as np
import utils
import model

BATCH_SIZE = 200
LEARNING_RATE = 0.001
LEARNING_RATE_EXPLORATION =0.001
GAMMA = 0.99
TAU = 0.001

class Trainer:

    def __init__(self, state_dim, feature_dim, action_dim, action_lim, ram, device, Test):
        """
        :param state_dim: Dimensions of state (int)
        :param action_dim: Dimension of action (int)
        :param action_lim: Used to limit action in [-action_lim,action_lim]
        :param ram: replay memory buffer object
        :return:
        """
        self.state_dim   = state_dim
        self.feature_dim  = feature_dim
        self.action_dim  = action_dim
        self.action_lim  = action_lim
        self.count       = 0
        self.update      = 0
        self.ram         = ram
        self.noise_size = 2

        self.device = device

        self.actor = model.Actor(self.state_dim, self.action_dim, self.action_lim).to(device)
        self.target_actor = model.Actor(self.state_dim, self.action_dim, self.action_lim).to(device)
        self.actor_pert   = model.Actor(self.state_dim, self.action_dim, self.action_lim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),LEARNING_RATE)

        self.critic = model.Critic(self.state_dim, self.action_dim).to(device)
        self.target_critic = model.Critic(self.state_dim, self.action_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),LEARNING_RATE)

        utils.hard_update(self.target_actor, self.actor)
        utils.hard_update(self.target_critic, self.critic)

    def get_action(self, state, Test=False, noise=True, param=True):
        if Test:
            noise = None
            param = None
        """
        gets the action from actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        state = torch.from_numpy(state).to(self.device)
        self.actor.eval()
        
        new_action = self.actor.forward(state).detach().data.cpu().numpy()

        self.actor.train()
            
        # Gaussion Noise
        if state.ndim == 1:
            new_action += np.random.normal(loc=0.0, scale = self.noise_size * self.action_lim, size=self.action_dim)
            new_action = np.clip(new_action, -1.2 * self.action_lim, 1.2 * self.action_lim)
        else:
            for parallel_pos in range(len(state)):
                new_action[parallel_pos] += np.random.normal(loc=0.0, scale = self.noise_size * self.action_lim, size=self.action_dim)
                new_action[parallel_pos] = np.clip(new_action[parallel_pos], -1.2 * self.action_lim, 1.2 * self.action_lim)

        return new_action
    
    def get_icm_reward(self, s1, a1, s2, parallel):
        
        s1 = torch.from_numpy(s1).to(self.device)
        a1 = torch.from_numpy(a1).to(self.device)
        s2 = torch.from_numpy(s2).to(self.device)

        s1_encoder = self.Feature(s1)
        s2_encoder = self.Feature(s2)

        predict_s2_encoder = self.Forward(s1_encoder, a1)

        Forward_loss = []
        for parallel_pos in range(parallel):
            Forward_loss.append(F.mse_loss(predict_s2_encoder[parallel_pos], s2_encoder[parallel_pos]).detach().data.cpu().numpy())

        return np.float32(Forward_loss)
    
    def optimize(self,Test):
        if Test:
            return
        """
        Samples a random batch from replay memory and performs optimization
        :return:
        """
        self.count = self.count+1
        s1,a1,r1,s2 = self.ram.sample(BATCH_SIZE)

        s1 = torch.from_numpy(s1).to(self.device)
        a1 = torch.from_numpy(a1).to(self.device)
        r1 = torch.from_numpy(r1).to(self.device)
        s2 = torch.from_numpy(s2).to(self.device)

        # ---------------------- optimize critic ----------------------
        # Use target actor exploitation policy here for loss evaluation
        a2 = self.target_actor.forward(s2).detach()
        next_val = torch.squeeze(self.target_critic.forward(s2, a2).detach())
        
        y_expected = r1 + GAMMA*next_val
        y_expected = torch.squeeze(y_expected)
        y_predicted = torch.squeeze(self.critic.forward(s1, a1))
        # compute critic loss, and update the critic
        loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # ---------------------- optimize actor ----------------------
        pred_a1 = self.actor.forward(s1)

        loss_actor = -1*torch.mean(self.critic.forward(s1, pred_a1))
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        utils.soft_update(self.target_actor, self.actor, TAU)
        utils.soft_update(self.target_critic, self.critic, TAU)

    def perturb_actor_parameters(self,Test):
        if Test:
            return
        """Apply parameter noise to actor model, for exploration"""
        utils.hard_update(self.actor_pert, self.actor)
        params = self.actor_pert.state_dict()
        for name in params:
            if 'ln' in name: 
                pass
            param = params[name]
            param += (torch.randn(param.shape) * self.param_noise.current_stddev).to(self.device)

    def save_models(self):
        """
        saves the target actor and critic models
        :param episode_count: the count of episodes iterated
        :return:
        """
        torch.save(self.actor.state_dict(), './Models/cur_actor.pt')
        torch.save(self.critic.state_dict(), './Models/cur_critic.pt')

        print('Models saved successfully')

    def save_best_models(self):
        """
        saves the target actor and critic models
        :param episode_count: the count of episodes iterated
        :return:
        """
        torch.save(self.actor.state_dict(), './Models/best_actor.pt')
        torch.save(self.critic.state_dict(), './Models/best_critic.pt')

        print('Models saved successfully')


    def load_models(self):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        :param episode: the count of episodes iterated (used to find the file name)
        :return:
        """
        self.actor.load_state_dict(torch.load('./Models/best_actor.pt'))
        print('Models loaded succesfully')
        
