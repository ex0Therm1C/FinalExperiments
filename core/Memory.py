import os, shutil
import numpy as np

class Memory:

    def __init__(self, env, maxLength):
        self.env = env
        self.maxLength = maxLength
        self.state = np.zeros([0, env.stateSpace])
        self.actions = np.zeros((0, 1), dtype=int)
        self.rewards = np.zeros((0, 1))
        self.newState = np.zeros([0, env.stateSpace])
        self.dones = np.zeros((0, 1), dtype=int)

    def addMemory(self, state, action, reward, newState, done):
        self.state = np.append(self.state, state[0].reshape([1, self.env.stateSpace]), axis=0)
        self.actions = np.append(self.actions, np.array(action, dtype=int).reshape([1, 1]), axis=0)
        self.rewards = np.append(self.rewards, np.array(reward).reshape([1, 1]), axis=0)
        self.newState = np.append(self.newState, newState[0].reshape([1, self.env.stateSpace]), axis=0)
        self.dones = np.append(self.dones, np.array(done, dtype=int).reshape([1, 1]), axis=0)

        if len(self.actions) > self.maxLength:
            offset = len(self.actions) - self.maxLength
            self.state = self.state[offset:]
            self.actions = self.actions[offset:]
            self.rewards = self.rewards[offset:]
            self.newState = self.newState[offset:]
            self.dones = self.dones[offset:]

    def sampleMemory(self, size):
        idx = np.random.choice(len(self.actions), min(len(self.actions), size))
        return (self.state[idx],
                self.actions[idx], self.rewards[idx],
                self.newState[idx],
                self.dones[idx])

    def writeToDisk(self, saveFolder='memory'):
        if os.path.exists(saveFolder):
            shutil.rmtree(saveFolder)
        os.mkdir(saveFolder)

        np.save(os.path.join(saveFolder, 'state.npy'), self.state)
        np.save(os.path.join(saveFolder, 'actions.npy'), self.actions)
        np.save(os.path.join(saveFolder, 'rewards.npy'), self.rewards)
        np.save(os.path.join(saveFolder, 'newState.npy'), self.newState)
        np.save(os.path.join(saveFolder, 'dones.npy'), self.dones)

    def loadFromDisk(self, saveFolder='memory'):
        if os.path.exists(saveFolder):
            self.state = np.load(os.path.join(saveFolder, 'state.npy'))
            self.actions = np.load(os.path.join(saveFolder, 'actions.npy'))
            self.rewards = np.load(os.path.join(saveFolder, 'rewards.npy'))
            self.newState = np.load(os.path.join(saveFolder, 'newState.npy'))
            self.dones = np.load(os.path.join(saveFolder, 'dones.npy'))
            return True
        return False

    def __len__(self):
        return len(self.actions)