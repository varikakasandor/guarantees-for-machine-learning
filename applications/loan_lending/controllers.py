import numpy as np
from sklearn.model_selection import train_test_split
from utils import find_min_ya_p_ya

class ExtendedAlgorithm:
    def __init__(self, learner, delta):
        self.learner = learner
        self.delta = delta

    def fit(self, iterator):
        C1, C2 = np.log(4 / self.delta) / 2, 8 * np.log(32 / self.delta)
        n_per_class = {k: 0 for k in [(0, 0), (0, 1), (1, 0), (1, 1)]}
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

        self.alpha = 4 * np.sqrt(np.log(256 / self.delta) / np.min(list(n_per_class.values())))
        print(f'The learner needs to access {n} samples.')
        self.learner.fit(np.array(data_x), np.array(data_y), np.array(data_A), self.alpha)


class OneStepAlgorithm:
    def __init__(self, learner, delta):
        self.learner = learner
        self.delta = delta

    def fit(self, dataset, test_size=0.3):
        X, Y, A = dataset
        n = len(X)
        X_train, X_test, Y_train, Y_test, A_train, A_test = train_test_split(X, Y, A, test_size=test_size, random_state=23)
        min_ya_p_ya = find_min_ya_p_ya(Y, A)
        alpha = 2 * np.sqrt(np.log(64 / self.delta) / (n * min_ya_p_ya))
        objective_loss_train, alpha_loss_train, gamma_train = self.learner.fit(X_train, Y_train, A_train, alpha, min_ya_p_ya)
        objective_loss_test, alpha_loss_test, _, tmp = self.learner.loss(X_test, Y_test, A_test)
        _, gamma_test = tmp
        print(f"The learner has objective loss {objective_loss_train} and alpha loss {alpha_loss_train} on the "
              f"training set.")
        print(f"The learner has objective loss {objective_loss_test} and alpha loss {alpha_loss_test} on the test set.")
        print(f"The learnt majority function has weights {self.learner.fun.weight}")
        print(f"Gamma on train is = {gamma_train}")
        print(f"Gamma on test is = {gamma_test}")
