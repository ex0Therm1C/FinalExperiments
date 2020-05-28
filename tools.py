import numpy as np
import os

OUTPUT_FOLDER = 'output'
EXPERIMENT = 'Exp1'

lossCurve = list(np.load(os.path.join(EXPERIMENT, OUTPUT_FOLDER, 'DDQN_exp_1_f1.npy')))
print(len(lossCurve))