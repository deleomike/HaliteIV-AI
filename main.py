from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cProfile

from tqdm import tqdm

import torch
import torch.optim as optim
import tensorflow as tf

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import sys
import PIL.Image

import logging

import random
import matplotlib.pyplot as plt
import seaborn as sns

from kaggle_environments import evaluate, make
from kaggle_environments.envs.halite.helpers import *


#########################################################3
from Models import ActorModel, CriticModel, Actor, Critic, compute_returns
from Agents import simple_agent, advanced_agent
############################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PYTORCH USING: ", device)

pr = cProfile.Profile()
pr.enable()

seed=123
torch.manual_seed(seed)

tf.compat.v1.set_random_seed(seed)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=10, inter_op_parallelism_threads=10)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)
logging.disable(sys.maxsize)
global ship_

env = make("halite", debug=True)
env.run(["random"])
env.render(mode="ipython",width=800, height=600)


def getDirTo(fromPos, toPos, size):
    fromX, fromY = divmod(fromPos[0], size), divmod(fromPos[1], size)
    toX, toY = divmod(toPos[0], size), divmod(toPos[1], size)
    if fromY < toY: return ShipAction.NORTH
    if fromY > toY: return ShipAction.SOUTH
    if fromX < toX: return ShipAction.EAST
    if fromX > toX: return ShipAction.WEST




trainer = env.train([None, "random"])
observation = trainer.reset()
flatBoardLen = len(observation['halite'])
while not env.done:
    my_action = simple_agent(observation, env.configuration)
    print("My Action", my_action)
    observation = trainer.step(my_action)[0]
    print("Reward gained",observation.players[0][0])


env.render(mode="ipython",width=800, height=600)





input_ = tf.keras.layers.Input(shape=[441,])
model = tf.keras.Model(inputs=input_, outputs=[ActorModel(5,input_),CriticModel(input_)])
model.summary()

lr = 0.0001

optimizer = tf.keras.optimizers.Adam(lr=7e-4)

huber_loss = tf.keras.losses.Huber()
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
num_actions = 5
eps = np.finfo(np.float32).eps.item()
gamma = 0.99  # Discount factor for past rewards
env = make("halite", debug=True)
trainer = env.train([None,"random"])

def trainIters(actor, critic, n_iters):
    optimizerA = optim.Adam(actor.parameters())
    optimizerC = optim.Adam(critic.parameters())
    for iter in tqdm(range(n_iters)):
        kag_state = env.reset()
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0
        env.reset()

        for i in range(1000):
            env.render()

            state = kag_state[0]['observation']['halite']

            state = torch.FloatTensor(state).to(device)
            dist, value = actor(state), critic(state)

            action = dist.sample()
            next_state, reward, done, _ = env.step(action.cpu().numpy())

            log_prob = dist.log_prob(action).unsqueeze(0)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            masks.append(torch.tensor([1-done], dtype=torch.float, device=device))

            state = next_state

            if done:
                print('Iteration: {}, Score: {}'.format(iter, i))
                break


        next_state = torch.FloatTensor(next_state).to(device)
        next_value = critic(next_state)
        returns = compute_returns(next_value, rewards, masks)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        optimizerA.zero_grad()
        optimizerC.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        optimizerA.step()
        optimizerC.step()
    torch.save(actor, 'model/actor.pkl')
    torch.save(critic, 'model/critic.pkl')
    env.close()

def train_torch(actor, critic, reward_thresh=550):
    optimizerA = optim.Adam(actor.parameters(), lr=lr)
    optimizerC = optim.Adam(critic.parameters(), lr=lr)

    while not env.done:
        state = trainer.reset()
        episode_reward = 0
        log_probs = []

        for timestep in range(1, env.configuration.episodeSteps + 200):
            ##########################################################
            ####      Part 1: Calculate Action from Model         ####
            ##########################################################

            # convert board into a tensor
            state_ = torch.tensor(state.halite).cuda(device=device)

            # predictions
            actions_prob = actor.forward(state_)
            critic_value = critic.forward(state_)

            # store critic value
            critic_value_history.append(critic_value)

            # Get sample action from action pdf
            action = np.random.choice(num_actions, p=np.squeeze(actions_prob))

            # log probability
            log_prob = actions_prob.log_prob(action).unsqueeze(0)
            log_probs.append(log_probs)

            # store the action probabilities normalized with log
            action_probs_history.append(torch.math.log(actions_prob[0, action]))

            ##########################################################
            ### Part 2: Apply the sampled action in our environment ##
            ##########################################################

            # Get opponent's action
            action = advanced_agent(state, env.configuration, action)

            # step the environment forward
            next_state = trainer.step(action)[0]

            # get the rewards for the action
            gain = next_state.players[0][0] / 5000

            # save the reward
            rewards_history.append(torch.tensor([gain]), type=torch.float32, device=device)

            episode_reward += gain

            state = next_state

            if env.done:
                state = trainer.reset()
                # Update running reward to check condition for solving

        #Increment the total reward
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)
        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()
        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )

        optimizerA.zero_grad()
        optimizerC.zero_grad()
        # actor_loss.backward()
        # critic_loss.backward()
        optimizerA.step()
        optimizerC.step()

    torch.save(actor, 'model_data/actor.pkl')
    torch.save(critic, 'model/critic.pkl')


def train_tf(model, rewardThresh):
    episode_count = 0
    huber_loss = tf.keras.losses.Huber()
    action_probs_history = []
    critic_value_history = []
    rewards_history = []
    running_reward = 0
    num_actions = 5
    eps = np.finfo(np.float32).eps.item()
    gamma = 0.99  # Discount factor for past rewards
    env = make("halite", debug=True)
    trainer = env.train([None, "random"])
    while not env.done:
        state = trainer.reset()
        episode_reward = 0
        with tf.GradientTape() as tape:
            for timestep in range(1, env.configuration.episodeSteps + 200):
                # of the agent in a pop up window.
                state_ = tf.convert_to_tensor(state.halite)
                state_ = tf.expand_dims(state_, 0)
                # Predict action probabilities and estimated future rewards
                # from environment state
                action_probs, critic_value = model(state_)
                critic_value_history.append(critic_value[0, 0])

                # Sample action from action probability distribution
                action = np.random.choice(num_actions, p=np.squeeze(action_probs))
                action_probs_history.append(tf.math.log(action_probs[0, action]))

                # Apply the sampled action in our environment
                action = advanced_agent(state, env.configuration, action)
                state = trainer.step(action)[0]
                gain = state.players[0][0] / 5000
                rewards_history.append(gain)
                episode_reward += gain

                if env.done:
                    state = trainer.reset()
                    # Update running reward to check condition for solving
            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

            # Calculate expected value from rewards
            # - At each timestep what was the total reward received after that timestep
            # - Rewards in the past are discounted by multiplying them with gamma
            # - These are the labels for our critic
            returns = []
            discounted_sum = 0
            for r in rewards_history[::-1]:
                discounted_sum = r + gamma * discounted_sum
                returns.insert(0, discounted_sum)
            # Normalize
            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
            returns = returns.tolist()
            # Calculating loss values to update our network
            history = zip(action_probs_history, critic_value_history, returns)
            actor_losses = []
            critic_losses = []
            for log_prob, value, ret in history:
                # At this point in history, the critic estimated that we would get a
                # total reward = `value` in the future. We took an action with log probability
                # of `log_prob` and ended up recieving a total reward = `ret`.
                # The actor must be updated so that it predicts an action that leads to
                # high rewards (compared to critic's estimate) with high probability.
                diff = ret - value
                actor_losses.append(-log_prob * diff)  # actor loss

                # The critic must be updated so that it predicts a better estimate of
                # the future rewards.
                critic_losses.append(
                    huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
                )
            # Backpropagation
            loss_value = sum(actor_losses) + sum(critic_losses)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Clear the loss and reward history
            action_probs_history.clear()
            critic_value_history.clear()
            rewards_history.clear()

        # Log details
        episode_count += 1
        if episode_count % 10 == 0:
            template = "running reward: {:.2f} at episode {}"
            print(template.format(running_reward, episode_count))

        if running_reward > rewardThresh:  # Condition to consider the task solved
            print("Solved at episode {}!".format(episode_count))
            break

#train_tf(model=model, rewardThresh=100)

numActions = 4
stateSize = flatBoardLen
actor = Actor(state_size=stateSize, action_size=numActions).cuda(device=device)
critic = Critic(state_size=stateSize, action_size=numActions).cuda(device=device)

trainIters(actor, critic, 100)

pr.disable()

# after your program ends
pr.print_stats(sort="calls")

exit(0)
# while not env.done:
#     state_ = tf.convert_to_tensor(state.halite)
#     state_ = tf.expand_dims(state_, 0)
#     action_probs, critic_value = model(state_)
#     critic_value_history.append(critic_value[0, 0])
#     action = np.random.choice(num_actions, p=np.squeeze(action_probs))
#     action_probs_history.append(tf.math.log(action_probs[0, action]))
#     action = advanced_agent(state, env.configuration, action)
#     state = trainer.step(action)[0]