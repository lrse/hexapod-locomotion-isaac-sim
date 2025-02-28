import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as torchd
from torch.distributions import Normal, Categorical


class Ours2Estimator(nn.Module):
    def __init__(self,
                 temporal_steps,
                 num_one_step_obs,
                 num_critic_obs,
                 latent_dim=16,
                 critic_latent_dim=32,
                 enc_hidden_dims=[128, 64],
                 tar_hidden_dims=[128, 64],
                 critic_enc_hidden_dims=[128, 64],
                 pred_hidden_dims=[64],
                 activation='elu',
                 learning_rate=1e-3,
                 max_grad_norm=10.0,
                 num_prototype=32,
                 temperature=3.0,
                 init_noise_std=1.0,
                 **kwargs):
        if kwargs:
            print("Estimator_CL.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(Ours2Estimator, self).__init__()
        activation = get_activation(activation)

        self.temporal_steps = temporal_steps
        self.num_one_step_obs = num_one_step_obs
        self.num_critic_obs = num_critic_obs
        self.max_grad_norm = max_grad_norm
        self.latent_dim = latent_dim
        self.critic_latent_dim = critic_latent_dim
        self.temperature = temperature

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

        # Prototype
        self.proto = nn.Embedding(num_prototype, latent_dim)

        # Critic Encoder (Privileged Information Encoder)
        critic_enc_input_dim = self.num_critic_obs
        critic_enc_layers = []
        for l in range(len(critic_enc_hidden_dims) - 1):
            critic_enc_layers += [nn.Linear(critic_enc_input_dim, critic_enc_hidden_dims[l]), activation]
            critic_enc_input_dim = critic_enc_hidden_dims[l]
        critic_enc_layers += [nn.Linear(critic_enc_input_dim, critic_latent_dim)]
        self.critic_encoder = nn.Sequential(*critic_enc_layers)

        # Predictor
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
        mu = self.encoder_mean
        sigma = self.encoder_std
        z_t = self.target(next_obs)
        pred_vel, z_s, z_c = z[..., :3], z[..., 3:self.latent_dim+3], z[..., self.latent_dim+3:]

        # Swap Loss
        z_s = F.normalize(z_s, dim=-1, p=2)
        z_t = F.normalize(z_t, dim=-1, p=2)

        with torch.no_grad():
            w = self.proto.weight.data.clone()
            w = F.normalize(w, dim=-1, p=2)
            self.proto.weight.copy_(w)

        score_s = z_s @ self.proto.weight.T
        score_t = z_t @ self.proto.weight.T

        with torch.no_grad():
            q_s = sinkhorn(score_s)
            q_t = sinkhorn(score_t)

        log_p_s = F.log_softmax(score_s / self.temperature, dim=-1)
        log_p_t = F.log_softmax(score_t / self.temperature, dim=-1)

        swap_loss = -0.5 * (q_s * log_p_t + q_t * log_p_s).mean()

        # Contrast Loss
        z_critic = self.critic_encoder(critic_obs)
        p_c = self.predictor(z_c)
        p_critic = self.predictor(z_critic)
        # Calculate negative cosine similarities:
        def cosine_similarity(pi, zj):
            return F.normalize(pi, dim=-1, p=2) * F.normalize(zj, dim=-1, p=2)
        contrast_loss = -(cosine_similarity(p_c, z_critic.detach()).mean() + cosine_similarity(p_critic, z_c.detach()).mean()) * 0.5
        
        # Estimation Loss
        estimation_loss = F.mse_loss(pred_vel, vel)
        
        losses = estimation_loss + swap_loss + contrast_loss

        self.optimizer.zero_grad()
        losses.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return estimation_loss.item(), swap_loss.item(), contrast_loss.item()

@torch.no_grad()
def sinkhorn(out, eps=0.05, iters=3):
    Q = torch.exp(out / eps).T
    K, B = Q.shape[0], Q.shape[1]
    Q /= Q.sum()

    for it in range(iters):
        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B
    return (Q * B).T

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