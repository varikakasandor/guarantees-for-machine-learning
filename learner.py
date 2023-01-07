import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class NDLearner:
    def __init__(self):
        pass

    def fit(self, x, y, A, alpha):
        '''
        Finds the empirical minimalizer that is alpha-discriminatory.
        '''
        pass

    def __call__(self, x, A=None):
        '''
        Returns the prediction on the data x.
        '''
        pass

class NDFunction:
    def __init__(self):
        pass

    def __call__(self, x):
        pass

    def loss(self, x, y, A):
        prediction = self.__call__(x)
        loss = np.average(prediction==y)
        ya_pairs = [(0,0), (0,1), (1,0), (1,1)]
        n_per_class = {}
        gamma = {}
        for k in ya_pairs:
            _y, _a = k
            n_per_class[k] = np.sum((y==_y)&(A==_a))
            gamma[k] = np.sum((y==_y)&(A==_a)&(prediction==1)).astype(np.float32)/n_per_class[k] if n_per_class[k]!= 0 else 0.5
        gamma_loss = max(abs(gamma[(0,0)] - gamma[(0,1)]), abs(gamma[(1,0)] - gamma[(1,1)]))
        #print(n_per_class)
        #print(gamma)
        return loss, gamma_loss, (n_per_class, gamma)

class WrappedFun(NDFunction):
    def __init__(self, fun):
        self.fun = fun
    
    def __call__(self, x):
        return self.fun(x)

class FiniteLearner(NDLearner):
    'For set of NDFunctions finds the empirical best.'
    def __init__(self, funs):
        self.funs = [] #[WrappedFun(lambda x: 1)] # The constant funtion is added to makes sure that self.funs is non-empty and that there is a non-discriminatory predictor
        self.funs.extend(funs)
        self.fun = funs[0]

    def fit(self, x, y, A, alpha):
        current_loss = 1e9
        losses = {i: [] for i in range(10)}
        for f in tqdm(self.funs):
            loss, gamma_loss, _ = f.loss(x, y, A)
            if gamma_loss <0.1:
                losses[int(gamma_loss*100)].append(loss)
            if gamma_loss <= alpha and loss < current_loss:
                current_loss = loss
                self.fun = f

        print(f'A {alpha}-discriminatory empirical minimizer has been found with test loss: {current_loss}.')
        plt.boxplot(list(losses.values()))
        plt.savefig('gamma.png')

class Majority(NDFunction):
    def __init__(self, weight, funs):
        self.weight = weight
        self.funs = funs
    
    def __call__(self, x):
        ans = 0
        for f, w in zip(self.funs, self.weight):
            ans += f(x)*w
        return ans>0.

def compose(funs, num):
    new_funs = []
    for _ in range(num):
        weight = np.random.rand(len(funs)) - 0.5
        new_funs.append(Majority(weight, funs))
    return new_funs

