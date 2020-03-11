from abc import ABC, abstractmethod
import torch
import numpy as np
import cma
from torch import nn
from model.layers import FeedForward
from solver import build_optimizer
from torch.nn.parameter import Parameter
from optimizer import Optimizer


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
        self.sd_threshold = 0.0001

        # Set loss parameters
        self.gamma = cfg.MODEL.POLICY.GAMMA
        self.min_reinforce_loss_weight = min_reinforce_loss_weight
        self.reinforce_loss_weight = reinforce_loss_weight
        self.eps = np.finfo(np.float32).eps.item()
        self.loss_functions = {"IR": self.IR_loss, "PR": self.PR_loss, "H": self.H_loss,
                               "SIR": self.SIR_loss, "R": self.R_loss}
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

    @abstractmethod
    def forward(self, state):
        pass

    @staticmethod
    def clip(x, mean, limit):
        xmin = mean - limit
        xmax = mean + limit
        return torch.max(torch.min(x, xmax), xmin)

    @staticmethod
    def clip_negative(x):
        #return torch.max(x, torch.zeros(x.shape, dtype=torch.double))
        return torch.abs(x)

    @staticmethod
    def soft_relu(x, beta=0.1):
        return (torch.sqrt(x**2 + beta**2) + x) / 2.0

    # From Tassa 2012 synthesis and stabilization of complex behaviour
    def clamp_sd(self, sd):
        #return torch.max(sd, 0.001*torch.ones(sd.shape, dtype=torch.double))
        #return torch.exp(sd)
        #return self.min_sd + self.soft_relu(sd - self.min_sd, self.soft_relu_beta)
        return sd

    def clamp_action(self, action):
        #return self.soft_relu(action, self.soft_relu_beta)
        # return torch.clamp(action, min=0.0, max=1.0)  # NB! Kills gradients at borders
        #return torch.exp(action)

        if self.cfg.MODEL.POLICY.NETWORK:
            #for idx in range(len(action)):
            #    if idx < 10:
            #        action[idx] = action[idx]/10
            #    else:
            #        action[idx] = torch.exp(action[idx])/20
            pass
        #else:
        #    for idx in range(len(action)):
        #        if idx < 10 or action[idx] >= 0:
        #            continue
        #        elif action[idx] < 0:
        #            action[idx].data -= action[idx].data

        return action

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
        ## Calculate returns
        #for episode_idx, episode_loss in enumerate(batch_loss):
        #    R = torch.zeros(1, dtype=torch.float64)
        #    for step_idx in range(1, len(episode_loss)+1):
        #        R = -episode_loss[-step_idx] + self.gamma*R.detach()
        #        ret[episode_idx, -step_idx] = R
        #avg_stepwise_loss = torch.mean(-ret, dim=0)
        #advantages = ret - ret.mean(dim=0)

        #means = torch.mean(batch_loss, dim=0)
        #idxs = (batch_loss > torch.mean(batch_loss)).detach()
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

        #batch_loss = (batch_loss - batch_loss.mean(dim=1).detach()) / (batch_loss.std(dim=1).detach() + self.eps)
        #batch_loss = (batch_loss - batch_loss.mean(dim=0).detach()) / (batch_loss.std(dim=0).detach() + self.eps)
        #batch_loss = (batch_loss - batch_loss.mean(dim=0).detach())

        #fvals = torch.empty(batch_loss.shape[0])
        #for episode_idx, episode_loss in enumerate(batch_loss):
        #    nans = torch.isnan(episode_loss)
        #    fvals[episode_idx] = torch.sum(episode_loss[nans == False])

        nans = torch.isnan(batch_loss)
        batch_loss[nans] = 0

        if stepwise_loss:
            return torch.mean(batch_loss, dim=0)
        else:
            #return torch.sum(batch_loss[0][100])
            #return torch.sum(advantages[advantages<0])
            #idxs = torch.isnan(batch_loss) == False
            #return torch.sum(batch_loss[idxs])
            #return torch.mean(fvals)
            #return torch.mean(torch.sum(batch_loss, dim=1))

            return torch.sum(torch.mean(batch_loss, dim=0))
            #return batch_loss[0][-1]
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

    def R_loss(self, batch_loss):

        # Get objective loss
        objective_loss = self.calculate_objective_loss(batch_loss)

        return objective_loss, {"objective_loss": float(torch.mean(torch.sum(batch_loss, dim=1)).detach().numpy()),
                                "total_loss": float(objective_loss.detach().numpy())}
                                #"total_loss": batch_loss.detach().numpy()}


    def PR_loss(self, batch_loss):

        # Get REINFORCE loss
        reinforce_loss = self.calculate_reinforce_loss(batch_loss)

        # Get objective loss
        objective_loss = self.calculate_objective_loss(batch_loss)

        # Return a sum of the objective loss and REINFORCE loss
        loss = objective_loss + self.reinforce_loss_weight*reinforce_loss

        return loss, {"objective_loss": float(torch.mean(torch.sum(batch_loss, dim=1)).detach().numpy()),
                      "reinforce_loss": float(reinforce_loss.detach().numpy()),
                      "total_loss": float(loss.detach().numpy())}

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
        if self.cfg.SOLVER.OPTIMIZER == "sgd":
            nn.utils.clip_grad_value_(self.parameters(), 1)

        #total_norm_sqr = 0
        #total_norm_sqr += self.mean._layers["linear_layer_0"].weight.grad.norm() ** 2
        #total_norm_sqr += self.mean._layers["linear_layer_0"].bias.grad.norm() ** 2
        #total_norm_sqr += self.mean._layers["linear_layer_1"].weight.grad.norm() ** 2
        #total_norm_sqr += self.mean._layers["linear_layer_1"].bias.grad.norm() ** 2
        #total_norm_sqr += self.mean._layers["linear_layer_2"].weight.grad.norm() ** 2
        #total_norm_sqr += self.mean._layers["linear_layer_2"].bias.grad.norm() ** 2

        #gradient_clip = 0.01
        #scale = min(1.0, gradient_clip / (total_norm_sqr ** 0.5 + 1e-4))
        #print("lr: ", scale * self.cfg.SOLVER.BASE_LR)

        #if total_norm_sqr ** 0.5 > gradient_clip:

        #    for param in self.parameters():
        #    param.data.sub_(scale * self.cfg.SOLVER.BASE_LR)
        #        if param.grad is not None:
        #            param.grad = (param.grad * gradient_clip) / (total_norm_sqr ** 0.5)
                    #pass
        self.optimizer.step()

        #print("ll0 weight: {} {}".format(self.mean._layers["linear_layer_0"].weight_std.min(),
        #                        self.mean._layers["linear_layer_0"].weight_std.max()))
        #print("ll1 weight: {} {}".format(self.mean._layers["linear_layer_1"].weight_std.min(),
        #                        self.mean._layers["linear_layer_1"].weight_std.max()))
        #print("ll2 weight: {} {}".format(self.mean._layers["linear_layer_2"].weight_std.min(),
        #                        self.mean._layers["linear_layer_2"].weight_std.max()))
        #print("ll0 bias: {} {}".format(self.mean._layers["linear_layer_0"].bias_std.min(),
        #                        self.mean._layers["linear_layer_0"].bias_std.max()))
        #print("ll1 bias: {} {}".format(self.mean._layers["linear_layer_1"].bias_std.min(),
        #                        self.mean._layers["linear_layer_1"].bias_std.max()))
        #print("ll2 bias: {} {}".format(self.mean._layers["linear_layer_2"].bias_std.min(),
        #                        self.mean._layers["linear_layer_2"].bias_std.max()))
        #print("")

        # Make sure sd is not negative
        idxs = self.sd < self.sd_threshold
        self.sd.data[idxs] = self.sd_threshold

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
        if self.cfg.MODEL.POLICY.NETWORK:
            # Set a feedforward network for means
            self.mean = FeedForward(
                self.state_dim,
                self.cfg.MODEL.POLICY.LAYERS,
                self.action_dim
            )
        else:
            # Set tensors for means
            self.mean = Parameter(torch.from_numpy(
                self.initialise_mean(self.cfg.MODEL.POLICY.INITIAL_ACTION_MEAN, self.cfg.MODEL.POLICY.INITIAL_ACTION_SD)
            ))
            self.register_parameter("mean", self.mean)

        # Set tensors for standard deviations
        self.sd = Parameter(torch.from_numpy(self.initialise_sd(self.cfg.MODEL.POLICY.INITIAL_SD)))
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

        # Get mean of action value
        if self.cfg.MODEL.POLICY.NETWORK:
            mean = self.mean(state).double()
        else:
            mean = self.mean[:, self.step_idx]

        if not self.training:
            return self.clamp_action(mean)

        # Get normal distribution
        dist = torch.distributions.Normal(mean, clamped_sd)

        # Sample action
        if self.method == "H" and self.episode_idx == 0:
            if np.all(np.isnan(self.best_actions[:, self.step_idx])):
                action = mean
            else:
                action = torch.from_numpy(self.best_actions[:, self.step_idx])
        elif self.batch_size > 1:
            action = dist.rsample()
        else:
            action = mean

            # Clip action
            action = self.clamp_action(action)
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

        # We want to store original (list of batches) actions for tell
        self.orig_actions = []

        # Initialise CMA-ES
        self.optimizer = cma.CMAEvolutionStrategy(self.mean, self.cfg.MODEL.POLICY.INITIAL_SD, inopts=cmaes_options)
        self.actions = []

    def forward(self, state):

        # If we've hit the end of minibatch we need to sample more actions
        if self.step_idx == 0 and self.episode_idx - 1 == 0 and self.training:
            self.orig_actions = self.optimizer.ask()
            self.actions = torch.empty(self.action_dim, self.horizon, self.batch_size, dtype=torch.float64)
            for ep_idx, ep_actions in enumerate(self.orig_actions):
                self.actions[:, :, ep_idx] = torch.from_numpy(np.reshape(ep_actions, (self.action_dim, self.horizon)))

        # Get action
        action = self.actions[:, self.step_idx, self.episode_idx-1]

        return action

    def optimize(self, batch_loss):
        loss = batch_loss.sum(axis=1)
        self.optimizer.tell(self.orig_actions, loss.detach().numpy())
        return {"objective_loss": float(loss.detach().numpy().mean()), "total_loss": float(loss.detach().numpy().mean())}

    def get_clamped_sd(self):
        return np.asarray(self.sd)

    def get_clamped_action(self):
        return self.actions.detach().numpy()


class Perttu(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super(Perttu, self).__init__(*args, **kwargs)

        # Initialise optimizer object
        self.optimizer = \
            Optimizer(mode=self.method,
                      initialMean=np.random.normal(self.cfg.MODEL.POLICY.INITIAL_ACTION_MEAN,
                                                   self.cfg.MODEL.POLICY.INITIAL_ACTION_SD,
                                                   (self.action_dim, self.horizon)),
                      initialSd=self.cfg.MODEL.POLICY.INITIAL_SD*np.ones((self.action_dim, self.horizon)),
                      #initialSd=self.cfg.MODEL.POLICY.INITIAL_SD*np.ones((1, 1)),
                      learningRate=self.cfg.SOLVER.BASE_LR,
                      adamBetas=(0.9, 0.99),
                      minReinforceLossWeight=0.0,
                      nBatch=self.cfg.MODEL.BATCH_SIZE,
                      solver=self.cfg.SOLVER.OPTIMIZER)

    def forward(self, state):

        # If we've hit the end of minibatch we need to sample more actions
        if self.training:
            if self.step_idx == 0 and self.episode_idx-1 == 0:
                samples = self.optimizer.ask()
                self.actions = torch.empty(self.action_dim, self.horizon, self.batch_size)
                for ep_idx, ep_actions in enumerate(samples):
                    self.actions[:, :, ep_idx] = torch.reshape(ep_actions, (self.action_dim, self.horizon))

            # Get action
            action = self.actions[:, self.step_idx, self.episode_idx-1]

        else:
            if self.method != "CMA-ES":
                if self.step_idx == 0:
                    samples = self.optimizer.ask(testing=~self.training)
                    for ep_idx, ep_actions in enumerate(samples):
                        self.actions[:, :, ep_idx] = torch.reshape(ep_actions, (self.action_dim, self.horizon))

            # Get action
            action = self.actions[:, self.step_idx, 0]

        return action.double()

    def optimize(self, batch_loss):
        loss, meanFval = self.optimizer.tell(batch_loss)
        return {"objective_loss": float(meanFval), "total_loss": float(loss)}

    def get_clamped_sd(self):
        return self.optimizer.getClampedSd().reshape([self.optimizer.original_dim, self.optimizer.steps]).detach().numpy()

    def get_clamped_action(self):
        return self.actions.detach().numpy()
