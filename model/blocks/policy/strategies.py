from abc import ABC, abstractmethod
import torch
import numpy as np
import cma
from torch import nn
from model.layers import FeedForward
from solver import build_optimizer
from torch.nn.parameter import Parameter
from optimizer import Optimizer
from utils.index import Index


class BaseStrategy(ABC, nn.Module):

    def __init__(self, cfg, agent, reinforce_loss_weight=1.0,
                 min_reinforce_loss_weight=0.0, min_sd=0, soft_relu_beta=0.2,
                 adam_betas=(0.9, 0.999)):

        super(BaseStrategy, self).__init__()

        self.cfg = cfg
        self.method = cfg.MODEL.POLICY.METHOD

        # Get action dimension, horizon length, and batch size
        self.action_dim = agent.action_space.sample().shape[0]
        self.state_dim = agent.observation_space.sample().shape[0]
        self.horizon = cfg.MODEL.POLICY.MAX_HORIZON_STEPS
        self.dim = (self.action_dim, self.horizon)
        self.batch_size = cfg.MODEL.BATCH_SIZE

        # Set initial values
        self.mean = []
        self.clamped_action = []
        self.sd = []
        #self.clamped_sd = []
        self.min_sd = min_sd
        self.soft_relu_beta = soft_relu_beta
        self.best_actions = []

        # Set loss parameters
        self.gamma = cfg.MODEL.POLICY.GAMMA
        self.min_reinforce_loss_weight = min_reinforce_loss_weight
        self.reinforce_loss_weight = reinforce_loss_weight
        self.eps = np.finfo(np.float32).eps.item()
        self.loss_functions = {"IR": self.IR_loss, "PR": self.PR_loss, "H": self.H_loss, "SIR": self.SIR_loss}
        self.log_prob = []

        # Set optimizer parameters
        self.optimizer = None
        self.learning_rate = cfg.SOLVER.BASE_LR
        self.adam_betas = adam_betas
        self.optimize_functions = {"default": self.standard_optimize, "H": self.H_optimize}

        # Get references in case we want to track episode and step
        self.step_idx = agent.get_step_idx()
        self.episode_idx = agent.get_episode_idx()

    @abstractmethod
    def optimize(self, batch_loss):
        pass

    #def optimize(self, batch_loss):
    #    loss = self._optimize(batch_loss)
    #    return loss

    @abstractmethod
    def forward(self, state):
        pass

    #def forward(self, state):
    #    action = self._forward(state)
    #    return action

    @staticmethod
    def clip(x, mean, limit):
        xmin = mean - limit
        xmax = mean + limit
        return torch.max(torch.min(x, xmax), xmin)

    @staticmethod
    def soft_relu(x, beta=0.1):
        return (torch.sqrt(x**2 + beta**2) + x) / 2.0

    # From Tassa 2012 synthesis and stabilization of complex behaviour
    def clamp_sd(self, sd):
        #return sd
        return torch.exp(sd)
        #return self.min_sd + self.soft_relu(sd - self.min_sd, self.soft_relu_beta)

    def get_clamped_sd(self):
        return self.clamp_sd(self.sd).detach().numpy()

    def get_clamped_action(self):
        return self.clamped_action

    def initialise_mean(self, loc=0.0, sd=0.1, seed=0):
        if self.dim is not None:
            #return np.zeros(self.dim, dtype=np.float64)
            if seed > 0:
                np.random.seed(seed)
            return np.asarray(np.random.normal(loc, sd, self.dim), dtype=np.float64)

    def initialise_sd(self, factor=1.0):
        if self.dim is not None:
            return factor*np.ones(self.dim, dtype=np.float64)

    # Initialise mean / sd or get dim from initial values

    def calculate_reinforce_loss(self, batch_loss, stepwise_loss=False):
        # Initialise a tensor for returns
        returns = torch.empty(batch_loss.shape, dtype=torch.float64)

        # Calculate returns
        for episode_idx, episode_loss in enumerate(batch_loss):
            R = 0
            for step_idx in range(1, len(episode_loss)+1):
                R = -episode_loss[-step_idx] + self.gamma*R
                returns[episode_idx, -step_idx] = R

        # Remove baseline
        advantages = ((returns - returns.mean(dim=0)) / (returns.std(dim=0) + self.eps)).detach()

        # Return REINFORCE loss
        #reinforce_loss = torch.mean((-advantages * self.log_prob).sum(dim=1))
        #reinforce_loss = torch.sum((-advantages * self.log_prob))
        avg_stepwise_loss = (-advantages * self.log_prob).mean(dim=0)
        if stepwise_loss:
            return avg_stepwise_loss
        else:
            #return torch.sum(avg_stepwise_loss)
            return torch.mean(torch.sum((-advantages * self.log_prob), dim=1))

    def calculate_objective_loss(self, batch_loss, stepwise_loss=False):
        #batch_loss = (batch_loss.mean(dim=0) - batch_loss) / (batch_loss.std(dim=0) + self.eps)
        #batch_loss = batch_loss.mean() - batch_loss

        # Initialise a tensor for returns
        #ret = torch.empty(batch_loss.shape, dtype=torch.float64)
        # Calculate returns
        #for episode_idx, episode_loss in enumerate(batch_loss):
        #    R = torch.zeros(1, dtype=torch.float64)
        #    for step_idx in range(1, len(episode_loss)+1):
        #        R = -episode_loss[-step_idx] + self.gamma*R.detach()
        #        ret[episode_idx, -step_idx] = R
        #avg_stepwise_loss = torch.mean(-ret, dim=0)

        #means = torch.mean(batch_loss, dim=0)
        #idxs = means > torch.mean(means).detach()
        #idxs = sums < np.mean(sums)

        #advantages = torch.zeros((1, batch_loss.shape[1]))
        #advantages = []
        #idxs = batch_loss < torch.mean(batch_loss)
        #for i in range(batch_loss.shape[1]):
        #    avg = torch.mean(batch_loss[:,i]).detach().numpy()
        #    idxs = batch_loss[:,i].detach().numpy() < avg
        #    advantages[0,i] = torch.mean(batch_loss[idxs,i], dim=0)
        #    if torch.any(idxs[:,i]):
        #        advantages.append(torch.mean(batch_loss[idxs[:,i],i], dim=0))

        #batch_loss = (batch_loss - batch_loss.mean(dim=0).detach()) / (batch_loss.std(dim=0).detach() + self.eps)
        #advantages = batch_loss / (batch_loss.std(dim=0).detach() + self.eps)

        if stepwise_loss:
            return torch.mean(batch_loss, dim=0)
        else:
            #return torch.sum(advantages[advantages<0])
            return torch.mean(torch.sum(batch_loss, dim=1))
            #return torch.sum(torch.mean(batch_loss, dim=0))
            #return torch.sum(batch_loss)
            #return torch.sum(torch.stack(advantages))

        # Remove baseline
        #advantages = ((ret - ret.mean(dim=0)) / (ret.std(dim=0) + self.eps))
        #advantages = ret

        # Return REINFORCE loss
        #avg_stepwise_loss = -advantages.mean(dim=0)
        #if stepwise_loss:
        #    return avg_stepwise_loss
        #else:
        #    return torch.sum(avg_stepwise_loss)


    def PR_loss(self, batch_loss):

        # Get REINFORCE loss
        reinforce_loss = self.calculate_reinforce_loss(batch_loss)

        # Get objective loss
        objective_loss = self.calculate_objective_loss(batch_loss)

        # Return a sum of the objective loss and REINFORCE loss
        #loss = objective_loss + self.reinforce_loss_weight*reinforce_loss
        loss = objective_loss
        return loss, {"objective_loss": [objective_loss.detach().numpy()],
                      "reinforce_loss": [reinforce_loss.detach().numpy()],
                      "total_loss": [loss.detach().numpy()]}

    def IR_loss(self, batch_loss):

        # Get REINFORCE loss
        reinforce_loss = self.calculate_reinforce_loss(batch_loss)

        # Get objective loss
        objective_loss = self.calculate_objective_loss(batch_loss)

        # Return an interpolated mix of the objective loss and REINFORCE loss
        clamped_sd = self.clamp_sd(self.sd)
        mix_factor = 0.5*((clamped_sd - self.min_sd) / (self.initial_clamped_sd - self.min_sd)).mean().detach().numpy()
        mix_factor = self.min_reinforce_loss_weight + (1.0 - self.min_reinforce_loss_weight)*mix_factor
        mix_factor = np.maximum(np.minimum(mix_factor, 1), 0)

        loss = (1.0 - mix_factor) * objective_loss + mix_factor * reinforce_loss
        return loss, {"objective_loss": [objective_loss.detach().numpy()],
                      "reinforce_loss": [reinforce_loss.detach().numpy()],
                      "total_loss": [loss.detach().numpy()]}

    def SIR_loss(self, batch_loss):

        # Get REINFORCE loss
        reinforce_loss = self.calculate_reinforce_loss(batch_loss, stepwise_loss=True)

        # Get objective loss
        objective_loss = self.calculate_objective_loss(batch_loss, stepwise_loss=True)

        # Return an interpolated mix of the objective loss and REINFORCE loss
        clamped_sd = self.clamp_sd(self.sd)
        mix_factor = 0.5*((clamped_sd - self.min_sd) / (self.initial_clamped_sd - self.min_sd)).detach()
        mix_factor = self.min_reinforce_loss_weight + (1.0 - self.min_reinforce_loss_weight)*mix_factor
        mix_factor = torch.max(torch.min(mix_factor, torch.ones_like(mix_factor)), torch.zeros_like(mix_factor))

        loss = ((1.0 - mix_factor) * objective_loss + mix_factor * reinforce_loss).sum()
        return loss, {"objective_loss": [objective_loss.detach().numpy()],
                      "reinforce_loss": [reinforce_loss.detach().numpy()],
                      "total_loss": [loss.detach().numpy()]}

    def H_loss(self, batch_loss):

        # Get REINFORCE loss
        reinforce_loss = self.calculate_reinforce_loss(batch_loss)

        # Get objective loss
        objective_loss = torch.sum(batch_loss, dim=1)

        # Get idx of best sampled action values
        best_idx = objective_loss.argmin()

        # If some other sample than the mean was best, update the mean to it
        if best_idx != 0:
            self.best_actions = self.clamped_action[:, :, best_idx].copy()

        return reinforce_loss, objective_loss[best_idx], \
               {"objective_loss": [objective_loss.detach().numpy()],
                "reinforce_loss": [reinforce_loss.detach().numpy()],
                "total_loss": [reinforce_loss.detach().numpy()]}

    def get_named_parameters(self, name):
        params = {}
        for key, val in self.named_parameters():
            if key.split(".")[0] == name:
                params[key] = val

        # Return a generator that behaves like self.named_parameters()
        return ((x, params[x]) for x in params)

    def standard_optimize(self, batch_loss):

        # Get appropriate loss
        loss, stats = self.loss_functions[self.method](batch_loss)

        self.optimizer.zero_grad()
        loss.backward()
        #nn.utils.clip_grad_norm_(self.parameters(), 0.1)
        self.optimizer.step()

        # Empty log probs
        self.log_prob = torch.empty(self.batch_size, self.horizon, dtype=torch.float64)

        return stats

    def H_optimize(self, batch_loss):

        # Get appropriate loss
        loss, best_objective_loss, stats = self.loss_functions[self.method](batch_loss)

        # Adapt the mean
        self.optimizer["mean"].zero_grad()
        best_objective_loss.backward(retain_graph=True)
        self.optimizer["mean"].step()

        # Adapt the sd
        self.optimizer["sd"].zero_grad()
        loss.backward()
        self.optimizer["sd"].step()

        # Empty log probs
        self.log_prob = torch.empty(self.batch_size, self.horizon, dtype=torch.float64)

        return stats


class VariationalOptimization(BaseStrategy):

    def __init__(self, *args, **kwargs):
        super(VariationalOptimization, self).__init__(*args, **kwargs)

        # Initialise mean and sd
        self.mean_network = True
        if self.mean_network:
            # Set a feedforward network for means
            self.mean = FeedForward(
                self.state_dim,
                self.cfg.MODEL.POLICY.LAYERS,
                self.action_dim
            )

        else:
            # Set tensors for means
            self.mean = Parameter(torch.from_numpy(
                self.initialise_mean(self.cfg.MODEL.POLICY.INITIAL_ACTION_MEAN, self.cfg.MODEL.POLICY.INITIAL_ACTION_SD, seed=1111)
            ))
            self.register_parameter("mean", self.mean)

        # Set tensors for standard deviations
        self.sd = Parameter(torch.from_numpy(self.initialise_sd(self.cfg.MODEL.POLICY.INITIAL_LOG_SD)))
        self.initial_clamped_sd = self.clamp_sd(self.sd.detach())
        self.register_parameter("sd", self.sd)
        #self.clamped_sd = np.zeros((self.action_dim, self.horizon), dtype=np.float64)
        self.clamped_action = np.zeros((self.action_dim, self.horizon, self.batch_size), dtype=np.float64)

        # Initialise optimizer
        if self.method == "H":
            # Separate mean and sd optimizers (not sure if actually necessary)
            self.optimizer = {"mean": build_optimizer(self.cfg, self.get_named_parameters("mean")),
                              "sd": build_optimizer(self.cfg, self.get_named_parameters("sd"))}
            self.best_actions = np.empty(self.sd.shape)
            self.best_actions.fill(np.nan)
        else:
            self.optimizer = build_optimizer(self.cfg, self.named_parameters())

        # We need log probabilities for calculating REINFORCE loss
        self.log_prob = torch.empty(self.batch_size, self.horizon, dtype=torch.float64)

    def forward(self, state):

        # Get clamped sd
        clamped_sd = self.clamp_sd(self.sd[:, self.step_idx])

        #if self.training:
        #    self.clamped_sd[:, self.step_idx, self.episode_idx] = clamped_sd.detach().numpy()

        # Get mean of action value
        if self.mean_network:
            mean = self.mean(state).double()
        else:
            mean = self.mean[:, self.step_idx]

        if not self.training:
            return mean

        # Get normal distribution
        dist = torch.distributions.Normal(mean, 0.05)

        # Sample action
        if self.method == "H" and self.episode_idx == 0:
            if np.all(np.isnan(self.best_actions[:, self.step_idx])):
                action = mean
            else:
                action = torch.from_numpy(self.best_actions[:, self.step_idx])
        else:
            #action = dist.rsample()
            action = mean

            # Clip action
            #action = self.clip(action, mean, 2.0*clamped_sd)

        self.clamped_action[:, self.step_idx, self.episode_idx-1] = action.detach().numpy()
        # Get log prob for REINFORCE loss calculations
        self.log_prob[self.episode_idx-1, self.step_idx] = dist.log_prob(action.detach()).sum()

        return action

    def optimize(self, batch_loss):
        return self.optimize_functions.get(self.method, self.standard_optimize)(batch_loss)


class CMAES(BaseStrategy):

    def __init__(self, *args, **kwargs):
        super(CMAES, self).__init__(*args, **kwargs)

        # Make sure batch size is larger than one
        assert self.batch_size > 1, "Batch size must be >1 for CMA-ES"

        # Set up CMA-ES options
        cmaes_options = {"popsize": self.batch_size, "CMA_diagonal": True}

        # Initialise mean and flatten it
        self.mean = self.initialise_mean()
        self.mean = np.reshape(self.mean, (self.mean.size,))

        # Initialise CMA-ES
        self.optimizer = cma.CMAEvolutionStrategy(self.mean, 0.05, inopts=cmaes_options)
        self.actions = []

    def forward(self, state):

        # If we've hit the end of minibatch we need to sample more actions
        if self.step_idx == 0 and self.episode_idx - 1 == 0 and self.training:
            self.actions = self.optimizer.ask()

        # Get action
        action = self.actions[self.episode_idx-1][self.step_idx]

        return torch.DoubleTensor(np.asarray(action))

    def optimize(self, batch_loss):
        loss = batch_loss.sum(axis=1)
        self.optimizer.tell(self.actions, loss.detach().numpy())
        return {"objective_loss": [loss.detach().numpy().mean()], "total_loss": [loss.detach().numpy().mean()]}


class Perttu(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super(Perttu, self).__init__(*args, **kwargs)

        # Initialise optimizer object
        self.optimizer = \
            Optimizer(mode=self.method,
                      initialMean=np.random.normal(self.cfg.MODEL.POLICY.INITIAL_ACTION_MEAN,
                                                   self.cfg.MODEL.POLICY.INITIAL_ACTION_SD,
                                                   (self.action_dim, self.horizon)),
                      initialSd=0.25*np.ones((self.action_dim, self.horizon)),
                      #initialSd=0.2*np.ones((1,1)),
                      learningRate=self.cfg.SOLVER.BASE_LR,
                      adamBetas=(0.9, 0.99),
                      minReinforceLossWeight=0.0,
                      nBatch=self.cfg.MODEL.BATCH_SIZE)

    def forward(self, state):

        # If we've hit the end of minibatch we need to sample more actions
        if self.step_idx == 0 and self.episode_idx-1 == 0 and self.training:
            if self.method == "CMA-ES":
                samples = self.optimizer.ask()
                self.actions = torch.empty(self.action_dim, self.horizon, self.batch_size)
                for ep_idx, ep_actions in enumerate(samples):
                    self.actions[:, :, ep_idx] = torch.reshape(ep_actions, (self.action_dim, self.horizon))
            else:
                self.actions = self.optimizer.ask().permute(1, 2, 0)

        # Get action
        action = self.actions[:, self.step_idx, self.episode_idx-1]

        return action.double()

    def optimize(self, batch_loss):
        loss, meanFval = self.optimizer.tell(batch_loss)
        return {"objective_loss": float(meanFval), "total_loss": float(loss)}

    def get_clamped_sd(self):
        return self.optimizer.getClampedSd().detach().numpy()

    def get_clamped_action(self):
        # Might need to reshape
        return self.actions.detach().numpy()