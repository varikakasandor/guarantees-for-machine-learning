import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class NDFunction:
    def __init__(self):
        pass

    def __call__(self, x):
        pass

    def loss(self, x, y, A):
        prediction = self.__call__(x)
        loss = np.average(prediction == y)
        ya_pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
        n_per_class = {}
        gamma = {}
        beta_loss = 0
        for k in ya_pairs:
            _y, _a = k
            n_per_class[k] = np.sum((y == _y) & (A == _a))
            n_per_target = np.sum(y == _y)
            gamma[k] = np.sum((y == _y) & (A == _a) & (prediction == 1)).astype(np.float32) / n_per_class[k] if \
                n_per_class[k] != 0 else 0.5
            p_y_hat_given_y = np.sum((y == _y) & (prediction == 1)).astype(np.float32) / n_per_target if \
                n_per_target != 0 else 0.5
            beta_loss = max(beta_loss, abs(gamma[k] - p_y_hat_given_y) * (n_per_class[k] / len(x)))
        alpha_loss = max(abs(gamma[(0, 0)] - gamma[(0, 1)]), abs(gamma[(1, 0)] - gamma[(1, 1)]))
        return loss, alpha_loss, beta_loss, (n_per_class, gamma)


class WrappedFun(NDFunction):
    def __init__(self, fun):
        super().__init__()
        self.fun = fun

    def __call__(self, x):
        return self.fun(x)


class FiniteAlphaLearner(WrappedFun):
    'For set of NDFunctions finds the empirical best.'

    def __init__(self, funs):
        self.funs = []  # [WrappedFun(lambda x: 1)] # The constant funtion is added to makes sure that self.funs is non-empty and that there is a non-discriminatory predictor
        self.funs.extend(funs)
        WrappedFun.__init__(self, funs[0])

    def fit(self, x, y, A, alpha, min_ya_p_ya=0.25):
        current_loss, current_alpha_loss = 1e9, 0.
        losses = {i: [] for i in range(10)}
        for f in tqdm(self.funs):
            loss, alpha_loss, _, _ = f.loss(x, y, A)
            if alpha_loss < 0.1:
                losses[int(alpha_loss * 100)].append(loss)
            if alpha_loss <= alpha and loss < current_loss:
                current_loss = loss
                current_alpha_loss = alpha_loss
                self.fun = f

        # print(
        #     f'A {alpha}-discriminatory empirical minimizer has been found with {current_loss} test loss and {current_alpha_loss} gamma loss.')
        # plt.boxplot(list(losses.values()))
        # plt.savefig('gamma_vs_mistakes.png')
        #
        # plt.clf()
        # plt.plot(list(map(lambda x: np.min(x), losses.values())))
        # plt.savefig('empirical_bests.png')
        return current_loss, current_alpha_loss


class FiniteBetaLearner(WrappedFun):
    'For set of NDFunctions finds the empirical best.'

    def __init__(self, funs):
        self.funs = []  # [WrappedFun(lambda x: 1)] # The constant funtion is added to makes sure that self.funs is non-empty and that there is a non-discriminatory predictor
        self.funs.extend(funs)
        WrappedFun.__init__(self, funs[0])

    def fit(self, x, y, A, alpha, min_ya_p_ya=0.25):
        beta = alpha * min_ya_p_ya
        current_loss, current_alpha_loss, current_beta_loss = 1e9, 0., 0.
        for f in tqdm(self.funs):
            loss, alpha_loss, beta_loss, _ = f.loss(x, y, A)
            if beta_loss <= beta and loss < current_loss:
                current_loss = loss
                current_alpha_loss = alpha_loss
                current_beta_loss = beta_loss
                self.fun = f
        return current_loss, current_alpha_loss


class Majority(NDFunction):
    def __init__(self, weight, funs):
        super().__init__()
        self.weight = weight
        self.funs = funs

    def __call__(self, x):
        ans = 0
        for f, w in zip(self.funs, self.weight):
            ans += f(x) * w
        return ans > 0.


def compose(funs, num):
    new_funs = []
    for _ in range(num):
        weight = np.random.rand(len(funs)) - 0.5
        new_funs.append(Majority(weight, funs))
    return new_funs
