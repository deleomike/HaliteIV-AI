import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PYTORCH USING: ", device)


def ActorModel(num_actions, in_):
    common = tf.keras.layers.Dense(128, activation='tanh')(in_)
    common = tf.keras.layers.Dense(32, activation='tanh')(common)
    common = tf.keras.layers.Dense(num_actions, activation='softmax')(common)

    return common


def CriticModel(in_):
    common = tf.keras.layers.Dense(128)(in_)
    common = tf.keras.layers.ReLU()(common)
    common = tf.keras.layers.Dense(32)(common)
    common = tf.keras.layers.ReLU()(common)
    common = tf.keras.layers.Dense(1)(common)

    return common


class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128).cuda(device=device)
        self.linear2 = nn.Linear(128, 256).cuda(device=device)
        self.linear3 = nn.Linear(256, self.action_size).cuda(device=device)

    def forward(self, state):
        output = F.relu(self.linear1(state)).cuda(device=device)
        output = F.relu(self.linear2(output)).cuda(device=device)
        output = self.linear3(output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution



class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns