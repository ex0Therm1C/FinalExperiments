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


def loadMNISTLocal(numTest=1000):
    x_train = np.load('../mnist/x_train.npy').reshape(-1, 28, 28, 1)
    x_train = np.array(x_train, dtype=float) / 255
    y_train = labelToOneHot(np.load('../mnist/y_train.npy'), numLabels=10)
    x_test = np.load('../mnist/x_test.npy').reshape(-1, 28, 28, 1)[:numTest]
    x_test = np.array(x_test, dtype=float) / 255
    y_test = labelToOneHot(np.load('../mnist/y_test.npy')[:numTest], numLabels=10)
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
        self.bestMdlFile = 'output/best_mdl.h5'
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


class ImageClassificationGame_2(ICGameBase):

    def __init__(self, dataset, modelFunction, rewardShaping, sampleSize=5, budget=800, verbose=True,
                 labelCost=0.0, imgsToAvrg=5, gameLength=50, maxInteractions=300, rewardScaling=10):
        del gameLength
        del rewardScaling
        del imgsToAvrg
        self.imgsToAvrg = 1
        self.sampleSize = sampleSize
        self.rewardShaping = rewardShaping

        super(ImageClassificationGame_2, self).__init__(dataset, budget, maxInteractions,
                                                        modelFunction, labelCost, verbose)

        self.stateSpace = self._calcSateSpace()
        self.actionSpace = sampleSize + 1

        self.xLabeled, self.yLabeled, self.xUnlabeled, self.yUnlabeled = [], [], [], []
        self._initLabeledDataset(5)
        self.currentTestF1, self.currentTestLoss = self._fitClassifier()
        self.initialF1 = self.currentTestF1


    def _calcSateSpace(self):
        space = 0

        modelMetrics = 3
        space += len(self.classifier.get_weights()) * modelMetrics

        predMetrics = 2
        space += self.sampleSize * predMetrics

        otherMetrics = 1
        space += otherMetrics

        return space


    def _createState(self):
        # prediction metrics
        bVsSB = []
        entropy = []
        for i in range(self.currentStateIds.shape[1]):
            x = self.xUnlabeled[self.currentStateIds[:, i]]
            pred = self.classifier.predict(x)
            struct = np.sort(pred, axis=1)

            entropy.append(-np.sum(struct * np.log(struct), axis=1).reshape([1, -1]))
            bVsSB.append(1 - (struct[:, -1] - struct[:, -2]).reshape([1, -1]))

        meanBVsSB = np.mean(np.stack(bVsSB), axis=0)
        meanEntropy = np.mean(np.stack(entropy), axis=0)

        # model metrics
        weights = self.classifier.get_weights()
        modelMetrics = list()
        for layer in weights:
            modelMetrics += [np.mean(layer), np.std(layer), np.linalg.norm(layer)]  # , np.linalg.norm(layer, ord=2)]
        modelMetrics = np.array(modelMetrics)

        state = np.concatenate([np.array(np.mean(self.perClassF1)).reshape([1, -1]),
                                modelMetrics.reshape([1, -1]),
                                meanBVsSB,
                                meanEntropy], axis=1)
        return state


    def reset(self, pointsPerClass=5):
        self.numInteractions = 0
        self.addedImages = 0
        self.currentStateIds = np.random.choice(self.xUnlabeled.shape[0], (self.sampleSize, self.imgsToAvrg))

        self.classifier = self.modelFunction(inputShape=self.x_train.shape[1:],
                                             numClasses=self.y_train.shape[1])
        self.initialWeights = self.classifier.get_weights()

        self._initLabeledDataset(pointsPerClass)
        self.currentTestF1, self.currentTestLoss = self._fitClassifier()
        self.initialF1 = self.currentTestF1

        return self._createState()


    def step(self, action):
        self.numInteractions += 1

        if int(action) >= self.sampleSize:
            # replace random image
            self.currentStateIds[np.random.randint(0, self.sampleSize)] = np.random.choice(len(self.xUnlabeled), self.imgsToAvrg)
            newTestF1 = self.currentTestF1 # reward of 0
        else:
            self.addedImages += self.imgsToAvrg
            # add images to dataset
            indices = self.currentStateIds[int(action)]
            for a in range(len(indices)):
                idx = indices[a]
                self.xLabeled = np.append(self.xLabeled, self.xUnlabeled[idx:idx + 1], axis=0)
                self.yLabeled = np.append(self.yLabeled, self.yUnlabeled[idx:idx + 1], axis=0)
                self.xUnlabeled = np.delete(self.xUnlabeled, idx, axis=0)
                self.yUnlabeled = np.delete(self.yUnlabeled, idx, axis=0)
                # adjust indices of current state
                for i in range(self.currentStateIds.shape[0]):
                    if i != int(action):
                        for j in range(self.currentStateIds.shape[1]):
                            if idx < self.currentStateIds[i, j]:
                                self.currentStateIds[i, j] -= 1
                            elif idx == self.currentStateIds[i, j]:
                                self.currentStateIds[i, j] = np.random.randint(len(self.xUnlabeled))
                    else:
                        for j in range(a+1, self.currentStateIds.shape[1]):
                            if idx < self.currentStateIds[i, j]:
                                self.currentStateIds[i, j] -= 1
            # replace missing images
            self.currentStateIds[int(action)] = np.random.choice(len(self.xUnlabeled), self.imgsToAvrg)
            # retrain classifier
            newTestF1, self.currentTestLoss = self._fitClassifier()

        done = self.addedImages >= self.budget or \
               self.numInteractions >= self.maxInteractions

        if self.rewardShaping:
            reward = newTestF1 - self.currentTestF1
            self.currentTestF1 = 0.7 * self.currentTestF1 + 0.3 * newTestF1
        elif done:
            reward = self.currentTestF1 - self.initialF1
        else:
            reward = 0

        if done and self.verbose:
            print('inital F1 %1.4f \t current F1 %1.4f \t labeled Images %d \t reward %1.4f' % (
                   self.initialF1, self.currentTestF1, self.xLabeled.shape[0], reward))
        return self._createState(), reward, done, {}


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
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.001),
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


def imgWasAdded_2(Q, a):
    return a < Q.shape[1] - 1



##################################################################
##################################################################
# Training

x_train, y_train, x_test, y_test = loadMNIST()

lossCurve = []
expCurve = []
rewardCurve = []
imgCurve = []

OUTPUT_FOLDER = 'output'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
memDir = os.path.join(OUTPUT_FOLDER, 'memory')
cacheDir = os.path.join(OUTPUT_FOLDER, 'cache')
bestCacheDir = os.path.join(OUTPUT_FOLDER, 'cacheBest')

REWARD_SHAPING = True
BATCH_SIZE = 64
BUDGET = 800
SAMPLE_SIZE = 5
C = 10
minLoss = 0.4
EVAL_ITERATIONS = 10
name = 'DDQN_exp_2'
MIN_INTERACTIONS = 0#6000
MAX_INTERACTIONS_PER_GAME = 2000
exploration, conversion = 2000, 1000

greed = parameterPlan(1, 0.2, warmup=exploration, conversion=conversion)
print('planned interactions', MIN_INTERACTIONS)
print(exploration, 'exploration', conversion, 'conversion')
learningRate = parameterPlan(0.001, 0.0001, exploration, conversion)

env = ImageClassificationGame_2(dataset=(x_train, y_train, x_test, y_test),
                                modelFunction=ImageClassifier, budget=BUDGET,
                                rewardShaping=REWARD_SHAPING, maxInteractions=MAX_INTERACTIONS_PER_GAME,
                                sampleSize=SAMPLE_SIZE, labelCost=0.0)
memory = Memory(env, maxLength=1000)
memory.loadFromDisk(memDir)

ckptPath1 = os.path.join(cacheDir, 'ckpt')
cp_callback = keras.callbacks.ModelCheckpoint(ckptPath1, verbose=0, save_freq=1, save_weights_only=True)
if os.path.exists(ckptPath1 + '.index'):
    model = DDQN_2(env, fromCheckpoints=ckptPath1)
else:
    model = DDQN_2(env)

totalSteps = 0
startTime = time()
while totalSteps < MIN_INTERACTIONS:
    state = env.reset()
    epochLoss, epochExpMean, epochRewards = 0, 0, 0
    steps, done = 0, False
    print(totalSteps, '_______________________________')
    while not done:
        g = greed[np.clip(totalSteps, 0, len(greed)-1)]
        Q, a = model.predict(state, greedParameter=g)
        epochExpMean += np.mean(Q[:,:-1])

        a = a[0]
        statePrime, reward, done, _ = env.step(a)
        epochRewards += reward

        memory.addMemory(state, a, reward, statePrime, done)
        memSample = memory.sampleMemory(BATCH_SIZE)

        if totalSteps % C == 0 or done:
            callbacks = [cp_callback]
        else:
            callbacks = []
        lr = learningRate[np.clip(totalSteps, 0, len(learningRate)-1)]
        epochLoss += model.fit(memSample, callbacks=callbacks, lr= lr)

        totalSteps += 1
        steps += 1
        state = statePrime

    memory.writeToDisk(memDir)

    # logging ##################
    lossPerStep = epochLoss / steps
    if lossPerStep < minLoss:
        print('saving best model with loss', lossPerStep)
        minLoss = lossPerStep
        shutil.rmtree(bestCacheDir, ignore_errors=True)
        shutil.copytree(cacheDir, bestCacheDir)
    timePerStep = (time()-startTime) / float(totalSteps)
    etaSec = timePerStep * (MIN_INTERACTIONS - totalSteps - 1)
    print('ETA %d min | \t steps %d \t loss per step %1.6f \t total steps %d \t greed %1.4f \t lr %1.6f \n'%(
          int(etaSec / 60), steps, lossPerStep, totalSteps, g, lr))
    lossCurve.append(lossPerStep)
    expCurve.append(epochExpMean / steps)
    rewardCurve.append(epochRewards)
    imgCurve.append(env.xLabeled.shape[0])
    plot(lossCurve, rewardCurve, expCurve, imgCurve, OUTPUT_FOLDER)

saveFile(os.path.join(OUTPUT_FOLDER, 'lossCurve.npy'), np.array(lossCurve))
saveFile(os.path.join(OUTPUT_FOLDER, 'expCurve.npy'), np.array(expCurve))
saveFile(os.path.join(OUTPUT_FOLDER, 'rewardCurve.npy'), np.array(rewardCurve))
saveFile(os.path.join(OUTPUT_FOLDER, 'imgCurve.npy'), np.array(imgCurve))

############################################################
############################################################
# Evaluation

env = ImageClassificationGame_2(dataset=(x_train, y_train, x_test, y_test),
                                modelFunction=ImageClassifier, budget=BUDGET,
                                rewardShaping=REWARD_SHAPING, maxInteractions=3000,
                                sampleSize=SAMPLE_SIZE, labelCost=0.0)
env.verbose = False

ckptPath = os.path.join(bestCacheDir, 'ckpt')
agent = DDQN_2(env, fromCheckpoints=ckptPath)
lossCurves = []
f1Curves = []

for i in range(EVAL_ITERATIONS):
    print('%d ########################' % (i))
    try:
        f1, loss = scoreAgent(agent, env, BUDGET, imgWasAdded_2)
        lossCurves.append(loss)
        f1Curves.append(f1)
    except AssertionError:
        pass
    except Exception as e:
        print(e)

[print(len(l)) for l in lossCurves]
lossCurves = np.array(lossCurves)
lossCurves = np.mean(lossCurves, axis=0)
f1Curves = np.array(f1Curves)
f1Curves = np.mean(f1Curves, axis=0)

file = os.path.join(OUTPUT_FOLDER, name)
saveFile(file + '_f1', f1Curves)
saveFile(file + '_loss', lossCurves)