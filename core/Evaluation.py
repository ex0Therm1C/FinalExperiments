def scoreAgent(agent, env, numImgs, printInterval=50):
    state = env.reset()
    lossProg = []
    f1Prog = []
    i = 0
    done = False
    while not done:
        Q, a = agent.predict(state, greedParameter=0)
        state, reward, done, _ = env.step(a)

        if a < Q.shape[1] - 1:
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