import numpy as np
import pandas as pd
from learners import WrappedFun, FiniteAlphaLearner, FiniteBetaLearner, compose
from controllers import ExtendedAlgorithm, OneStepAlgorithm



# def test1():
#     N = 10000
#     fun1 = WrappedFun(lambda x: np.sum(x, axis=-1) > 2)
#     fun2 = WrappedFun(lambda x: np.sum(x, axis=-1) < 3)
#     x = np.random.randint(2, size=(N, 4))
#     y = np.random.randint(2, size=(N)) & (fun1(x) | fun2(x))
#     A = np.random.randint(2, size=(N))
#     # print(x, y.astype(np.int32), A)
#     print(*fun1.loss(x, y, A), sep='\n')
#
#     learner = FiniteAlphaLearner([fun1, fun2])
#     learner.fit(x, y, A, 0.1)


if __name__ == '__main__':
    np.random.seed(23)
    df = pd.read_csv('./data/application_train.csv',
                     usecols=['TARGET', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CODE_GENDER', 'CNT_CHILDREN',
                              'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'OWN_CAR_AGE'])
    df = df.dropna()
    df = pd.concat((df[df.TARGET == 1], df[df.TARGET == 0].sample(frac=0.1)))
    df = df.sample(frac=1)  # Shuffle dataset
    # print(df.describe())
    y = df['TARGET'].to_numpy()
    x = np.concatenate(
        ((df[['FLAG_OWN_CAR', 'FLAG_OWN_REALTY']] == 'Y').to_numpy().astype(np.int32),
         df[['CNT_CHILDREN']].to_numpy(),
         df[['AMT_CREDIT', 'AMT_INCOME_TOTAL', 'OWN_CAR_AGE']].to_numpy().astype(np.int32),
         (df[['CODE_GENDER']] == 'M').to_numpy().astype(np.int32),
         ),
        axis=-1)

    A = (df['CODE_GENDER'] == 'M').to_numpy()
    # print(x[:10], y[:10])

    function_class = compose([
        lambda x: x[:, 0],
        lambda x: x[:, 1],
        lambda x: x[:, 2] > 0,
        lambda x: x[:, 3] > 1000000,
        lambda x: x[:, 3] > 500000,
        lambda x: x[:, 3] > 100000,
        lambda x: x[:, 4] > 200000,
        lambda x: x[:, 5] > 10,
        lambda x: x[:, 5] > 20,
        lambda x: 3 * x[:, 6],
        # lambda x: x[:, 3]>50000,
    ], 10000)
    beta_learner = FiniteBetaLearner(function_class)
    alpha_learner = FiniteAlphaLearner(function_class)
    ctrl_alpha = OneStepAlgorithm(alpha_learner, 1e-30) # ExtendedAlgorithm(learner, 1e-30)
    ctrl_alpha.fit((x, y, A))
    ctrl_beta = OneStepAlgorithm(beta_learner, 1e-30)
    ctrl_beta.fit((x, y, A))