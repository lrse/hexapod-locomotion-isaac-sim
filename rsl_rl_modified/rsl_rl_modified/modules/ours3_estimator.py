import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as torchd
from torch.distributions import Normal, Categorical


class Ours3Estimator(nn.Module):
    def __init__(self,
                 temporal_steps,
                 num_one_step_obs,
                 num_critic_obs,
                 latent_dim=16,
                 critic_latent_dim=32,
                 enc_hidden_dims=[128, 64],
                 tar_hidden_dims=[128, 64],
                 critic_enc_hidden_dims=[128],#, 64],
                 prop_pred_hidden_dims=[64],
                 pred_hidden_dims=[64],
                 activation='elu',
                 learning_rate=1e-3,
                 max_grad_norm=10.0,
                 init_noise_std=1.0,
                 **kwargs):
        if kwargs:
            print("Estimator_CL.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(Ours3Estimator, self).__init__()
        activation = get_activation(activation)

        self.temporal_steps = temporal_steps
        self.num_one_step_obs = num_one_step_obs
        self.num_critic_obs = num_critic_obs
        self.max_grad_norm = max_grad_norm
        self.latent_dim = latent_dim
        self.critic_latent_dim = critic_latent_dim
        
        # Encoder
        enc_input_dim = self.temporal_steps * self.num_one_step_obs
        enc_output_dim = 3 + latent_dim + critic_latent_dim # vel + observation latent vector + privileged information latent vector
        enc_layers = []
        for l in range(len(enc_hidden_dims)):
            enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[l]), activation]
            enc_input_dim = enc_hidden_dims[l]
        enc_layers += [nn.Linear(enc_input_dim, enc_output_dim)]
        self.encoder = nn.Sequential(*enc_layers)

        # Target
        tar_input_dim = self.num_one_step_obs
        tar_layers = []
        for l in range(len(tar_hidden_dims)):
            tar_layers += [nn.Linear(tar_input_dim, tar_hidden_dims[l]), activation]
            tar_input_dim = tar_hidden_dims[l]
        tar_layers += [nn.Linear(tar_input_dim, latent_dim)]
        self.target = nn.Sequential(*tar_layers)

        # Critic Encoder (Privileged Information Encoder)
        critic_enc_input_dim = self.num_critic_obs
        critic_enc_layers = []
        for l in range(len(critic_enc_hidden_dims)):
            critic_enc_layers += [nn.Linear(critic_enc_input_dim, critic_enc_hidden_dims[l]), activation]
            critic_enc_input_dim = critic_enc_hidden_dims[l]
        critic_enc_layers += [nn.Linear(critic_enc_input_dim, critic_latent_dim)]
        self.critic_encoder = nn.Sequential(*critic_enc_layers)

        # Propioceptive Observation Predictor
        prop_pred_input_dim = latent_dim
        prop_pred_layers = []
        for l in range(len(prop_pred_hidden_dims)):
            prop_pred_layers += [nn.Linear(prop_pred_input_dim, prop_pred_hidden_dims[l]), activation]
            prop_pred_input_dim = prop_pred_hidden_dims[l]
        prop_pred_layers += [nn.Linear(prop_pred_input_dim, latent_dim)]
        self.propioceptive_predictor = nn.Sequential(*prop_pred_layers)

        # Privileged Info Predictor
        pred_input_dim = critic_latent_dim
        pred_layers = []
        for l in range(len(pred_hidden_dims)):
            pred_layers += [nn.Linear(pred_input_dim, pred_hidden_dims[l]), activation]
            pred_input_dim = pred_hidden_dims[l]
        pred_layers += [nn.Linear(pred_input_dim, critic_latent_dim)]
        self.predictor = nn.Sequential(*pred_layers)

        # Encoder noise
        self.std = nn.Parameter(init_noise_std * torch.ones(enc_output_dim))
        self.distribution = None

        # Optimizer
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
    
    @property
    def encoder_mean(self):
        return self.distribution.mean

    @property
    def encoder_std(self):
        return self.distribution.stddev
    
    def update_distribution(self, obs_history):
        mean = self.encoder(obs_history)
        self.distribution = Normal(mean, mean*0. + self.std)

    def enc(self, obs_history=None, **kwargs):
        self.update_distribution(obs_history)
        return self.distribution.sample()

    def forward(self, obs_history):
        parts = self.encoder(obs_history.detach())
        vel, latent_vectors = parts[..., :3], parts[..., 3:]
        return vel.detach(), latent_vectors.detach()

    def update(self, obs_history, critic_obs, next_critic_obs, lr=None):
        if lr is not None:
            self.learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
                
        vel = next_critic_obs[:, 0:3].detach()
        next_obs = next_critic_obs.detach()[:, 3:self.num_one_step_obs+3]

        z = self.enc(obs_history)
        pred_vel, z_s, z_c = z[..., :3], z[..., 3:self.latent_dim+3], z[..., self.latent_dim+3:]

        # Propioceptive Contrast Loss
        z_next_obs = self.target(next_obs)
        p_next_obs = self.propioceptive_predictor(z_next_obs)
        p_s = self.propioceptive_predictor(z_s)
        # Calculate negative cosine similarities:
        propioceptive_contrast_loss = -(cosine_similarity(p_s, z_next_obs.detach()).mean() + 
                                        cosine_similarity(p_next_obs, z_s.detach()).mean()) * 0.5

        # Contrast Loss
        z_critic = self.critic_encoder(critic_obs)
        p_c = self.predictor(z_c)
        p_critic = self.predictor(z_critic)
        # Calculate negative cosine similarities:
        contrast_loss = -(cosine_similarity(p_c, z_critic.detach()).mean() + cosine_similarity(p_critic, z_c.detach()).mean()) * 0.5
        
        # Estimation Loss
        estimation_loss = F.mse_loss(pred_vel, vel)
        
        losses = estimation_loss + propioceptive_contrast_loss + contrast_loss

        self.optimizer.zero_grad()
        losses.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return estimation_loss.item(), propioceptive_contrast_loss.item(), contrast_loss.item()

def cosine_similarity(pi, zj):
    return F.normalize(pi, dim=-1, p=2) * F.normalize(zj, dim=-1, p=2)

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "silu":
        return nn.SiLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None