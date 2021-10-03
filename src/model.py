import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import MultivariateNormal, Categorical


# policy model 
class MLP(nn.Module):
  '''
  Simple MLP model for a policy
  '''
  def __init__(self, n_input, n_hiddens, n_actions, action_type="discrete"):
    '''
    n_input: size of the observation input
    n_hiddens: list contains the size of each hidden layer
    n_actions: size of the action vector
    action_type: type of the action (discrete or continuous)
    '''
    super().__init__()
    self.action_type = action_type
    layers = []
    layers.append(nn.Linear(n_input, n_hiddens[0]))
    layers.append(nn.ReLU())
    for i in range(len(n_hiddens)-1):
      layers.append( nn.Linear(n_hiddens[i], n_hiddens[i+1]) )
      if n_hiddens[i] != n_hiddens[-2]:
        layers.append( nn.ReLU() )
    self.model = nn.Sequential(*layers)
    if action_type == "discrete":
      self.dis_actions = nn.Linear(n_hiddens[-1], n_actions)
    else:
      self.mean = nn.Linear(n_hiddens[-1], n_actions)
      self.log_var = nn.Linear(n_hiddens[-1], n_actions)

  # forward pass
  def forward(self, x):
    if self.action_type == "discrete":
      return self.dis_actions(self.model(x))
    else:
      x = self.model(x)
      mean = self.mean(x)
      log_var = self.log_var(x)
      return mean, log_var
  
  # sample action from the policy
  def sample_action(self, x):
    if self.action_type == "discrete":
      x = self.forward(x)
      self.actions_probs = F.softmax(x, dim=-1)
      action = torch.multinomial(self.actions_probs, 1)
      log_prob = F.log_softmax(x)[action] 
      return action, log_prob
    else:
      mean, log_var = self.forward(x)
      covar = torch.exp(log_var)
      self.actions_dist = MultivariateNormal(mean, torch.diag(covar))
      action = self.actions_dist.sample()
      log = self.action_dis.log_prob(action)
      return action, log
  # return the entropy of the policy
  def entropy(self):
    if self.action_type == "discrete":
      return Categorical(self.actions_probs).entropy()
    else:
      return self.actions_dist.entropy()

