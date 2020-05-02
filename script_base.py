import numpy as np
import os, shutil, gc
from time import time
import tensorflow
import tensorflow.keras as keras
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


def labelToOneHot(x, numLabels=10):
    result = np.zeros((len(x), numLabels), dtype=np.float32)
    # result[np.arange(len(x)), x[:,0]] = 1
    result[np.arange(len(x)), x] = 1
    return result

def loadMNIST(numTest=1000):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(len(x_train), 28, 28, 1)
    x_train = np.array(x_train, dtype=float) / 255
    y_train = labelToOneHot(y_train, numLabels=10)

    x_test = x_test.reshape(len(x_test), 28, 28, 1)
    x_test = np.array(x_test, dtype=float)[:numTest] / 255
    y_test = labelToOneHot(y_test[:numTest], numLabels=10)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test


def ImageClassifier(inputShape=[28,28,1], numClasses=10):
    model = keras.models.Sequential([
        keras.layers.Input(inputShape),
        keras.layers.Conv2D(64, 3, strides=[3, 3], activation='relu'),
        keras.layers.Conv2D(32, 3, activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(24, activation=keras.activations.relu),
        keras.layers.Dense(numClasses, activation='softmax')
    ])
    model.compile(optimizer=tensorflow.keras.optimizers.Adam(),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=[keras.metrics.categorical_accuracy])
    return model


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


class ICGameBase:

    def __init__(self, dataset, budget, maxInteractions, modelFunction, labelCost, verbose):
        self.x_train = dataset[0]
        self.y_train = dataset[1]
        self.x_test = dataset[2]
        self.y_test = dataset[3]
        self.nClasses = self.y_train.shape[1]

        self.budget = budget
        self.maxInteractions = maxInteractions
        self.labelCost = labelCost
        self.verbose = verbose

        self.es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=2)
        self.bestMdlFile = '/tmp/best_mdl.h5'
        self.bestModelCB = keras.callbacks.ModelCheckpoint(self.bestMdlFile, save_best_only=True,
                                                           monitor='val_loss', mode='min')

        self.classifier = modelFunction(inputShape=self.x_train.shape[1:],
                                        numClasses=self.y_train.shape[1])
        self.initialWeights = self.classifier.get_weights()
        self.modelFunction = modelFunction

    def _initLabeledDataset(self, pointsPerClass=5):
        del self.xLabeled
        del self.yLabeled
        del self.xUnlabeled
        del self.yUnlabeled
        gc.collect()
        self.xLabeled, self.yLabeled = [], []

        ids = np.arange(self.x_train.shape[0], dtype=int)
        np.random.shuffle(ids)
        addedDataPerClass = np.zeros(self.y_train.shape[1])
        usedIds = []
        for i in ids:
            label = np.argmax(self.y_train[i])
            if addedDataPerClass[label] < pointsPerClass:
                self.xLabeled.append(self.x_train[i])
                self.yLabeled.append(self.y_train[i])
                usedIds.append(i)
                addedDataPerClass[label] += 1
            if sum(addedDataPerClass) >= pointsPerClass * len(addedDataPerClass):
                break
        unusedIds = [i for i in np.arange(self.x_train.shape[0]) if i not in usedIds]
        self.xLabeled = np.array(self.xLabeled)
        self.yLabeled = np.array(self.yLabeled)
        self.xUnlabeled = np.array(self.x_train[unusedIds])
        self.yUnlabeled = np.array(self.y_train[unusedIds])

    def _fitClassifier(self, epochs=50):
        self.classifier.set_weights(self.initialWeights)
        self.bestModelCB.best = np.inf
        train_history = self.classifier.fit(self.xLabeled, self.yLabeled, epochs=epochs, verbose=0,
                                            callbacks=[self.es, self.bestModelCB],
                                            validation_data=(self.x_test, self.y_test))
        if os.path.exists(self.bestMdlFile):
            self.classifier.load_weights(self.bestMdlFile)
            os.remove(self.bestMdlFile)
        else:
            print('no best file')
        yHat = self.classifier.predict(self.x_test)
        yHat = np.argmax(yHat, axis=1)

        self.report = classification_report(np.argmax(self.y_test, axis=1), yHat, output_dict=True)
        self.perClassPrec = [self.report[str(c)]['precision'] for c in range(self.nClasses)]
        self.perClassRec = [self.report[str(c)]['recall'] for c in range(self.nClasses)]
        self.perClassF1 = [self.report[str(c)]['f1-score'] for c in range(self.nClasses)]

        unique, counts = np.unique(np.argmax(self.yLabeled, axis=1), return_counts=True)
        d = dict(zip(unique, counts))
        self.perClassIntances = [0 for _ in range(self.nClasses)]
        for cls, count in d.items():
            self.perClassIntances[int(cls)] = count

        return np.mean(self.perClassF1), np.min(train_history.history['val_loss'])


def plot(lossCurve, rewardCurve, expCurve, imgCurve, outDir, showPlots=False):
    # loss
    avrgCurve = []
    for i in range(1, len(lossCurve)):
        avrgCurve.append(np.mean(lossCurve[max(0, i-20):i]))
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.axhline(y=0, color='k')
    ax1.set_ylim(0, 1)
    ax1.plot(np.arange(len(avrgCurve)), avrgCurve, label='loss')
    ax1.legend(fontsize='small')
    # rewards
    ax2 = ax1.twinx()
    avrgCurve = []
    for i in range(1, len(rewardCurve)):
        avrgCurve.append(np.mean(rewardCurve[max(0, i-30):i]))
    ax2.plot(np.arange(len(avrgCurve)), avrgCurve, c='red', label='reward')
    #ax2.set_ylim(-0.03, 0.03)
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax2.legend(fontsize='small')
    # new game lines
    for i in range(1, len(imgCurve)):
        if imgCurve[i] < imgCurve[i-1]:
            plt.axvline(x=i, color='k', linestyle='--', linewidth=1)
    fig.tight_layout()
    plt.savefig(os.path.join(outDir, 'prog.png'), dpi=200)
    if showPlots: plt.show()
    plt.close('all')

    # expectation
    avrgCurve = []
    for i in range(1, len(expCurve)):
        avrgCurve.append(np.mean(expCurve[max(0, i-30):i]))
    plt.clf()
    plt.axhline(y=0, color='k')
    #plt.ylim(-0.6, 0.6)
    plt.plot(np.arange(len(avrgCurve)), avrgCurve, label='expected reward')
    plt.legend(fontsize='small')
    # new game lines
    for i in range(1, len(imgCurve)):
        if imgCurve[i] < imgCurve[i-1]:
            plt.axvline(x=i, color='k', linestyle='--', linewidth=1)
    plt.savefig(os.path.join(outDir, 'exp.png'), dpi=200)
    if showPlots: plt.show()
    plt.close('all')


def scoreAgent(agent, env, numImgs, imgAddedCond, printInterval=50):
    state = env.reset()
    lossProg = []
    f1Prog = []
    i = 0
    done = False
    while not done:
        Q, a = agent.predict(state, greedParameter=0)
        state, reward, done, _ = env.step(a)

        if imgAddedCond(Q, a):
            for _ in range(env.imgsToAvrg):
                f1Prog.append(env.currentTestF1)
                lossProg.append(env.currentTestLoss)

        if i % printInterval == 0 and len(f1Prog) > 0:
            print('%d | %d : %1.3f'%(i, env.addedImages, f1Prog[-1]))
        i += 1

    print('stopping with', env.addedImages)
    if env.addedImages >= numImgs:
        return f1Prog, lossProg
    print('not converged')
    raise AssertionError('not converged')


def saveFile(name, file):
    if os.path.exists(name + '.npy'):
        os.remove(name + '.npy')
    np.save(name + '.npy', file)


def parameterPlan(val1, val2, warmup, conversion):
    plan1 = np.full(warmup, val1)
    plan2 = np.linspace(val1, val2, conversion)
    return np.concatenate([plan1, plan2])





##################################################################
##################################################################
# Training

x_train, y_train, x_test, y_test = loadMNIST()

OUTPUT_FOLDER = 'output'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
memDir = os.path.join(OUTPUT_FOLDER, 'memory')
cacheDir = os.path.join(OUTPUT_FOLDER, 'cache')
bestCacheDir = os.path.join(OUTPUT_FOLDER, 'cacheBest')

lossCurve = []
if os.path.exists(os.path.join(OUTPUT_FOLDER, 'lossCurve.npy')): lossCurve = np.load(os.path.join(OUTPUT_FOLDER, 'lossCurve.npy'))
expCurve = []
if os.path.exists(os.path.join(OUTPUT_FOLDER, 'expCurve.npy')): expCurve = np.load(os.path.join(OUTPUT_FOLDER, 'expCurve.npy'))
rewardCurve = []
if os.path.exists(os.path.join(OUTPUT_FOLDER, 'rewardCurve.npy')): rewardCurve = np.load(os.path.join(OUTPUT_FOLDER, 'rewardCurve.npy'))
imgCurve = []
if os.path.exists(os.path.join(OUTPUT_FOLDER, 'imgCurve.npy')): imgCurve = np.load(os.path.join(OUTPUT_FOLDER, 'imgCurve.npy'))
