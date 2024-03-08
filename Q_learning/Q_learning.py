import gym
import random
import numpy as np
import time
from collections import deque
import pickle
from collections import defaultdict


EPISODES =  20000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999


def default_Q_value():
    return 0


def epsilon_greedy_policy(Q_table, state, epsilon):
    # With probability epsilon, select a random action
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        # Otherwise, select the action with highest Q-value for the current state
        Q_values = [Q_table[(state, action)] for action in range(env.action_space.n)]
        return np.argmax(Q_values)


def update_Q_value(Q_table, state, action, reward, next_state, alpha, gamma):
    # Compute the Q-learning target value
    max_next_Q = max([Q_table[(next_state, a)] for a in range(env.action_space.n)])
    Q_table[(state, action)] = (1 - alpha) * Q_table[(state, action)] + alpha * (reward + gamma * max_next_Q)



if __name__ == "__main__":

    random.seed(1)
    np.random.seed(1)
    env = gym.envs.make("FrozenLake-v1")
    env.seed(1)
    env.action_space.np_random.seed(1)

    # You will need to update the Q_table in your iteration
    Q_table = defaultdict(default_Q_value) # starts with a pessimistic estimate of zero reward for each state.
    episode_reward_record = deque(maxlen=100)

    for i in range(EPISODES):
        episode_reward = 0
        done = False
        obs = env.reset()

        while not done:
            # Select action using epsilon-greedy policy
            action = epsilon_greedy_policy(Q_table, obs, EPSILON)

            # Take the action and observe the new state and reward
            next_obs, reward, done, info = env.step(action)

            # Update the Q-value
            update_Q_value(Q_table, obs, action, reward, next_obs, LEARNING_RATE, DISCOUNT_FACTOR)

            # Update the state
            obs = next_obs

            # Update the total reward
            episode_reward += reward

        # This is the correct place for the epsilon decay, after the while loop, meaning after the episode is finished
        EPSILON = max(EPSILON * EPSILON_DECAY, 0.01)

        # Record the reward
        episode_reward_record.append(episode_reward)

        # Printing the average reward
        if i % 100 == 0 and i > 0:
            average_reward = sum(episode_reward_record) / len(episode_reward_record)
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(average_reward))
            print("EPSILON: " + str(EPSILON))


    #### DO NOT MODIFY ######
    model_file = open('Q_TABLE.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    model_file.close()
    #########################