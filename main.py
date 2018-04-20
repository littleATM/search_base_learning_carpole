import gym
import random

ENV = 'CartPole-v0'
action_list = [1 for i in range(1000)]
state_list = [0 for i in range(1000)]
flag_list = [0 for i in range(1000)]
import gym
import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model

ENV = 'CartPole-v0'


def build_model():
    # Neural Net for Deep-Q learning Model
    model = Sequential()
    model.add(Dense(24, input_dim=4, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(2, activation='linear'))
    model.compile(loss="mse",
                  optimizer=Adam(lr=0.001, clipvalue=1))
    return model


def main():
    env = gym.make(ENV)
    action = env.action_space.n
    env.observation_space
    model = build_model()
    episode = 0
    max_r = 0
    s = env.reset()
    main_step = 0
    while True:
        main_step = main_step+1
        #episode = episode + 1


        #left child episode
        s_, r, done, info = env.step(0)
        left_r = 0
        while True:
            if done:
                #print("left_r", left_r)
                #print("max_r", max_r, "epside", episode)
                if max_r < left_r:
                    max_r = left_r
                    #print("max_r", max_r, "epside", episode)
                break
            else:
                act_values = model.predict(np.reshape(s_, [1, 4]))
                # print(s_)
                a = np.argmax(act_values[0])
                s_, r, done, info = env.step(a)
                left_r = left_r + 1
            if left_r > max_r:
                pass
                #env.render()
            if left_r > max_r and left_r % 100 == 0:
                #print("left_new_max", left_r)  # episode too long print step
                pass
            if left_r>100:#200/2
                break



        #right child episode
        env.reset()  # never mind
        env.set_state(s[0], s[1], s[2], s[3])  # set previous s
        s_, r, done, info = env.step(1)
        right_r = 0
        while True:
            if done:
                #print("right_r", right_r)
                #print("max_r", max_r, "epside", episode)
                if max_r < right_r:
                    max_r = right_r
                    #print("max_r:", max_r, "epside", episode)
                break
            else:
                act_values = model.predict(np.reshape(s_, [1, 4]))
                a = np.argmax(act_values[0])
                s_, r, done, info = env.step(a)
                right_r = right_r + 1
            if right_r > max_r:
                pass
                #env.render()
            if right_r > max_r and right_r % 100 == 0:
                pass
                #print("right_new_max", right_r)
            if right_r >100:#200/2
                break
        #main episode
        env.reset()
        env.set_state(s[0], s[1], s[2], s[3])
        if left_r >= right_r:
            target_f = model.predict(np.reshape(s, [1, 4]))
            target_f[0][0] = 1  # left-right  gap larger , converge faster
            target_f[0][1] = -1
            model.fit(np.reshape(s, [1, 4]), target_f, epochs=1, verbose=0)
            s, r, done, info = env.step(0)
        elif right_r > left_r:
            target_f = model.predict(np.reshape(s, [1, 4]))
            target_f[0][0] = -1
            target_f[0][1] = 1
            model.fit(np.reshape(s, [1, 4]), target_f, epochs=1, verbose=0)
            s, r, done, info = env.step(1)
        env.render()
        print("main_step",main_step)
        if done or main_step>200:
            s = env.reset()
            main_step = 0

    pass


if __name__ == "__main__":
    main()
