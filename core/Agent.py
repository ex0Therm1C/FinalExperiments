import tensorflow.keras as keras
import numpy as np
import os

class DDQN_1:

    def __init__(self, env, gamma=0.9, fromCheckpoints=None):
        self.gamma = gamma
        self.env = env
        self.model1 = self.createModel(fromCheckpoints)
        self.model2 = self.createModel(None)
        self.model2.set_weights(self.model1.get_weights())

    def createModel(self, fromCheckpoint):
        model = keras.models.Sequential([
            keras.layers.Input(self.env.stateSpace),
            keras.layers.Dense(24, activation=keras.layers.LeakyReLU(),
                               kernel_regularizer=keras.regularizers.l2(0.001),
                               kernel_initializer='he_uniform'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(12, activation=keras.layers.LeakyReLU(),
                               kernel_regularizer=keras.regularizers.l2(0.001),
                               kernel_initializer='he_uniform'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(self.env.actionSpace)
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                      loss=keras.losses.mse)

        if fromCheckpoint is not None and os.path.exists(fromCheckpoint + '.index'):
            print('load model from ', fromCheckpoint)
            model.load_weights(fromCheckpoint)

        return model

    def predict(self, inputs, greedParameter=1):
        q = self.model1.predict(inputs)
        if greedParameter <= 0:
            return q, np.argmax(q, axis=1)
        exp = np.exp(q / greedParameter)
        softmax = exp / np.sum(exp, axis=1).reshape(-1, 1)
        a = np.zeros((q.shape[0], 1))
        for i in range(q.shape[0]):
            try:
                a[i] = np.random.choice(self.env.actionSpace, 1, p=softmax[i])
            except ValueError:
                print('softmax error with q', q[i], 'softmax', softmax[i])
                a[i] = np.argmax(q[i])
        return q, a

    def fit(self, memoryBatch, callbacks=list(), lr=None):
        state = memoryBatch[0]
        actions = memoryBatch[1]
        rewards = memoryBatch[2]
        nextState = memoryBatch[3]
        dones = memoryBatch[4]

        Q1 = self.model1.predict(state)
        qPrime1 = self.model1.predict(nextState)
        qPrime2 = self.model2.predict(nextState)

        nextAction = np.argmax(qPrime1, axis=1)

        for i in range(Q1.shape[0]):
            Q1[i, actions[i]] = rewards[i]
            if dones[i] == 0:
                # van hasselt 2016
                Q1[i, actions[i]] += self.gamma * qPrime2[i, nextAction[i]]
        cb = []
        if len(callbacks) > 0: cb.append(callbacks[0])

        if lr is not None: self.model1.optimizer.lr = lr

        hist = self.model1.fit(x=state, y=Q1, epochs=1, verbose=0, callbacks=cb)
        if len(cb) > 0:
            self.model2.set_weights(self.model1.get_weights())

        return hist.history['loss'][0]


class DDQN_2:

    def __init__(self, env, gamma=0.9, fromCheckpoints=None):
        self.gamma = gamma
        self.env = env
        self.model1 = self.createModel(fromCheckpoints)
        self.model2 = self.createModel(None)
        self.model2.set_weights(self.model1.get_weights())

    def createModel(self, fromCheckpoint):
        model = keras.models.Sequential([
            keras.layers.Input(self.env.stateSpace),
            keras.layers.Dense(48, activation=keras.layers.LeakyReLU(),
                               kernel_regularizer=keras.regularizers.l2(0.001),
                               kernel_initializer='he_uniform'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(24, activation=keras.layers.LeakyReLU(),
                               kernel_regularizer=keras.regularizers.l2(0.001),
                               kernel_initializer='he_uniform'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(self.env.actionSpace)
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                      loss=keras.losses.mse)

        if fromCheckpoint is not None and os.path.exists(fromCheckpoint + '.index'):
            print('load model from ', fromCheckpoint)
            model.load_weights(fromCheckpoint)

        return model

    def predict(self, inputs, greedParameter=1):
        q = self.model1.predict(inputs)
        if greedParameter <= 0:
            return q, np.argmax(q, axis=1)
        exp = np.exp(q / greedParameter)
        softmax = exp / np.sum(exp, axis=1).reshape(-1, 1)
        a = np.zeros((q.shape[0], 1))
        for i in range(q.shape[0]):
            try:
                a[i] = np.random.choice(self.env.actionSpace, 1, p=softmax[i])
            except ValueError:
                print('softmax error with q', q[i], 'softmax', softmax[i])
                a[i] = np.argmax(q[i])
        return q, a

    def fit(self, memoryBatch, callbacks=list(), lr=None):
        state = memoryBatch[0]
        actions = memoryBatch[1]
        rewards = memoryBatch[2]
        nextState = memoryBatch[3]
        dones = memoryBatch[4]

        Q1 = self.model1.predict(state)
        qPrime1 = self.model1.predict(nextState)
        qPrime2 = self.model2.predict(nextState)

        nextAction = np.argmax(qPrime1, axis=1)

        for i in range(Q1.shape[0]):
            Q1[i, actions[i]] = rewards[i]
            if dones[i] == 0:
                # van hasselt 2016
                Q1[i, actions[i]] += self.gamma * qPrime2[i, nextAction[i]]
        cb = []
        if len(callbacks) > 0:
            cb.append(callbacks[0])
        if lr is not None:
            self.model1.optimizer.lr = lr
        hist = self.model1.fit(x=state, y=Q1, epochs=1, verbose=0, callbacks=cb)
        if len(cb) > 0:
            self.model2.set_weights(self.model1.get_weights())

        return hist.history['loss'][0]
