import numpy as np

class Controller:
    def __init__(self, learner, delta):
        self.learner = learner
        self.delta = delta

    def fit(self, iterator):
        C1, C2 = np.log(4/self.delta)/2, 8*np.log(32/self.delta)
        n_per_class = {k: 0 for k in [(0,0), (0,1), (1,0), (1,1)]}
        n = 0
        data_x, data_y, data_A = [], [], []
        for x, y, A in iterator:
            n_per_class[(y, A)] += 1
            n += 1
            data_x.append(x)
            data_y.append(y)
            data_A.append(A)
            if np.min(list(n_per_class.values())) - np.sqrt(n * C1) >= C2:
                break
        
        self.alpha = 4*np.sqrt(np.log(256/self.delta)/np.min(list(n_per_class.values())))
        print(f'The learner needs to access {n} samples.')
        self.learner.fit(np.array(data_x), np.array(data_y), np.array(data_A), self.alpha)

