import matplotlib.pyplot as plt
import numpy as np
import os

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
    plt.savefig(os.path.join(outDir, 'exp.png'))
    if showPlots: plt.show()
    plt.close('all')