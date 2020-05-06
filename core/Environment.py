import tensorflow.keras as keras
import gc, os
import numpy as np
from sklearn.metrics import classification_report


class ICGameBase:

    def __init__(self, dataset, budget, maxInteractions, modelFunction,  labelCost, verbose):
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
                                            callbacks=[self.es, self.bestModelCB], validation_data=(self.x_test, self.y_test))
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
        


class ImageClassificationGame_1(ICGameBase):

    def __init__(self, dataset, modelFunction, rewardShaping, sampleSize=5, budget=800, verbose=True,
                 labelCost=0.0, imgsToAvrg=5, gameLength=50, maxInteractions=300, rewardScaling=10):
        del gameLength
        del rewardScaling
        del imgsToAvrg
        self.sampleSize = 1
        self.imgsToAvrg = 1
        self.rewardShaping = rewardShaping

        super(ImageClassificationGame_1, self).__init__(dataset, budget, maxInteractions,
                                                        modelFunction, labelCost, verbose)

        self.stateSpace = self._calcSateSpace()
        self.actionSpace = 2

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
            if not self.rewardShaping:
                # With reward shaping we need to do this after the reward is calculated
                self.currentTestF1 = 0.7 * self.currentTestF1 + 0.3 * newTestF1

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
            if not self.rewardShaping:
                # With reward shaping we need to do this after the reward is calculated
                self.currentTestF1 = 0.7 * self.currentTestF1 + 0.3 * newTestF1

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



class ImageClassificationGame_3(ICGameBase):

    def __init__(self, dataset, modelFunction, sampleSize=10, budget=300, verbose=True,
                 labelCost=0.0, imgsToAvrg=5, gameLength=50, maxInteractions=30, rewardScaling=10):
        self.sampleSize = sampleSize
        self.imgsToAvrg = imgsToAvrg
        self.gameLength = gameLength
        self.rewardScaling = rewardScaling

        super(ImageClassificationGame_3, self).__init__(dataset, budget, maxInteractions,
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
        self.initialF1 = self.currentTestF1
        self.currentStateIds = np.random.choice(self.xUnlabeled.shape[0], (self.sampleSize, self.imgsToAvrg))

        if self.xLabeled.shape[0] - (pointsPerClass * self.nClasses) >= self.budget:
            # full reset
            self.currentTestF1 = 0

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
            self.currentStateIds[np.random.randint(0, self.sampleSize)] = np.random.choice(len(self.xUnlabeled),
                                                                                           self.imgsToAvrg)
        else:
            indices = self.currentStateIds[int(action)]
            for a in range(len(indices)):
                idx = indices[a]
                self.xLabeled = np.append(self.xLabeled, self.xUnlabeled[idx:idx + 1], axis=0)
                self.yLabeled = np.append(self.yLabeled, self.yUnlabeled[idx:idx + 1], axis=0)
                self.xUnlabeled = np.delete(self.xUnlabeled, idx, axis=0)
                self.yUnlabeled = np.delete(self.yUnlabeled, idx, axis=0)
                self.addedImages += 1
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
            # replace missing image
            self.currentStateIds[int(action)] = np.random.choice(len(self.xUnlabeled), self.imgsToAvrg)
            # retrain classifier
            newTestF1, self.currentTestLoss = self._fitClassifier()
            self.currentTestF1 = 0.7 * self.currentTestF1 + 0.3 * newTestF1

        done = self.numInteractions >= self.maxInteractions
        reward = 0
        if self.addedImages >= self.gameLength:
            done = True
            reward = (self.currentTestF1 - self.initialF1) * self.rewardScaling

        if done and self.verbose:
            print('inital F1 %1.4f \t current F1 %1.4f \t labeled Images %d \t reward %1.4f' % (
                   self.initialF1, self.currentTestF1, self.xLabeled.shape[0], reward))
        return self._createState(), reward, done, {}
