import numpy as np
import os, shutil
from time import time
import tensorflow.keras as keras

from core.Data import loadMNIST
from core.Environment import ImageClassificationGame_2
from core.ImageClassifier import ImageClassifier
from core.Agent import DDQN_2
from core.Memory import Memory
from core.Plotting import plot
from core.Evaluation import scoreAgent, imgWasAdded_2

def saveFile(name, file):
    if os.path.exists(name + '.npy'):
        os.remove(name + '.npy')
    np.save(name + '.npy', file)

def parameterPlan(val1, val2, warmup, conversion):
    plan1 = np.full(warmup, val1)
    plan2 = np.linspace(val1, val2, conversion)
    return np.concatenate([plan1, plan2])


#################################################################
#################################################################
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
BUDGET = 50#800
SAMPLE_SIZE = 5
C = 10
minLoss = 0.4
EVAL_ITERATIONS = 2#10
name = 'DDQN_exp_2'
MIN_INTERACTIONS = 50#4000
MAX_INTERACTIONS_PER_GAME = 300
exploration, conversion = 1500, 1000

greed = parameterPlan(1, 0.2, warmup=exploration, conversion=conversion)
print('planned interactions', MIN_INTERACTIONS)
print(exploration, 'exploration', conversion, 'conversion')
learningRate = parameterPlan(0.00005, 0.000001, exploration, conversion)

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
                                rewardShaping=REWARD_SHAPING, maxInteractions=BUDGET,
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
