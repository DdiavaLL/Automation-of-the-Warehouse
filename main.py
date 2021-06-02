import gym
import gym_storehouse
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt

def QLearning():
    env = gym.make('gym_storehouse-v0')
    n_episodes = 1000

    max_iter_episode = 100
    exploration_proba = 1
    exploration_decreasing_decay = 0.001
    min_exploration_proba = 0.01
    gamma = 0.99
    lr = 0.1
    total_rewards_episode = list()
    rewards_per_episode = list()
    n_observations = env.observation_space.n
    n_actions = env.action_space.n
    Q_table = np.zeros((n_observations,n_actions))
    for e in range(n_episodes):
        current_state = env.reset()
        done = False
        total_episode_reward = 0

        for i in range(max_iter_episode):
            env.render()
            print("EPISODE! ", i)

            if np.random.uniform(0,1) < exploration_proba:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q_table[current_state,:])

            next_state, reward, done, _ = env.step(action)

            Q_table[current_state, action] = (1-lr) * Q_table[current_state, action] +lr*(reward + gamma*max(Q_table[next_state,:]))
            total_episode_reward = total_episode_reward + reward
            if done:
                break
            current_state = next_state
        exploration_proba = max(min_exploration_proba, np.exp(-exploration_decreasing_decay*e))
        rewards_per_episode.append(total_episode_reward)

    print("Training finished!")
    with open("storehouse_qlearning.pkl", 'wb') as f:
        pickle.dump(Q_table, f)
    print(Q_table)

def SARSA():
    env = gym.make('gym_storehouse-v0')

    epsilon = 0.9
    total_episodes = 1000
    max_steps = 100
    lr_rate = 0.81
    gamma = 0.96

    Q = np.zeros((env.observation_space.n, env.action_space.n))

    def choose_action(state):
        action = 0
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])
        return action

    def learn(state, state2, reward, action, action2):
        predict = Q[state, action]
        target = reward + gamma * Q[state2, action2]
        Q[state, action] = Q[state, action] + lr_rate * (target - predict)

    rewards = 0

    for episode in range(total_episodes):
        t = 0
        state = env.reset()
        action = choose_action(state)

        while t < max_steps:
            env.render()

            state2, reward, done, info = env.step(action)

            action2 = choose_action(state2)

            learn(state, state2, reward, action, action2)

            state = state2
            action = action2

            t += 1
            rewards+=1

            if done:
                break

        print("EPISODE!", episode)
    with open("storehouse_sarsa.pkl", 'wb') as f:
        pickle.dump(Q, f)
    print(Q)

def QAgent():
    env = gym.make('gym_storehouse-v0')
    number = []
    n_reward = []
    total_reward = 0

    with open("storehouse_qlearning.pkl", 'rb') as f:
        Q = pickle.load(f)

    def choose_action(state):
        action = np.argmax(Q[state, :])
        return action

    for episode in range(200):
        number.append(episode)
        state = env.reset()
        print("*** Episode: ", episode)
        t = 0
        while t < 3000:
            env.render()

            action = choose_action(state)

            state2, reward, done, info = env.step(action)
            t += 1
            state = state2
            total_reward = reward
            if t == 2999:
                n_reward.append(total_reward)
            if done:
                n_reward.append(total_reward)
                break

    print(number)
    print(n_reward)
    plt.plot(number, n_reward)
    plt.xlabel('n_epoch')
    plt.ylabel('aver. score')
    plt.show()

def SARSAgent():
    env = gym.make('gym_storehouse-v0')
    number = []
    n_reward = []
    total_reward = 0

    with open("storehouse_sarsa.pkl", 'rb') as f:
        Q = pickle.load(f)

    def choose_action(state):
        action = np.argmax(Q[state, :])
        return action

    for episode in range(200):
        number.append(episode)
        state = env.reset()
        print("*** Episode: ", episode)
        t = 0
        while t < 4000:
            env.render()

            action = choose_action(state)

            state2, reward, done, info = env.step(action)
            t += 1
            state = state2
            total_reward = reward
            print("step", t)
            print("episode", episode)
            if t == 3999:
                n_reward.append(total_reward)
            if done:
                n_reward.append(total_reward)
                break

    print(number)
    print(n_reward)
    plt.plot(number, n_reward)
    plt.xlabel('n_epoch')
    plt.ylabel('aver. score')
    plt.show()

def main():
    QLearning()
    QAgent()
    SARSA()
    SARSAgent()


if __name__ == "__main__":
    main()