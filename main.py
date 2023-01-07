import numpy as np
import pandas as pd
from learner import WrappedFun, FiniteLearner, compose

def test1():

    N=10000
    fun1 = WrappedFun(lambda x: np.sum(x, axis=-1)>2)
    fun2 = WrappedFun(lambda x: np.sum(x, axis=-1)<3)
    x = np.random.randint(2, size=(N, 4))
    y = np.random.randint(2, size=(N))&(fun1(x)|fun2(x))
    A = np.random.randint(2, size=(N))
    #print(x, y.astype(np.int32), A)
    print(*fun1.loss(x, y, A), sep='\n')

    learner = FiniteLearner([fun1, fun2])
    learner.fit(x, y, A, 0.1)

if __name__ == '__main__':
    df = pd.read_csv('data/application_train.csv', usecols=['TARGET', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CODE_GENDER', 'CNT_CHILDREN', 'AMT_CREDIT'])
    df = df.dropna()
    y = df['TARGET'].to_numpy()
    x = np.concatenate(
        ((df[['FLAG_OWN_CAR', 'FLAG_OWN_REALTY']]=='Y').to_numpy().astype(np.int32),
        df[['CNT_CHILDREN']].to_numpy(),
        df[['AMT_CREDIT']].to_numpy().astype(np.int32),
        ),
        axis=-1)


    A = (df['CODE_GENDER']=='M').to_numpy()
    print(x[:10], y[:10])

    learner = FiniteLearner(compose([
        #lambda x: x[:, 0], 
        #lambda x: x[:, 1],
        lambda x: x[:, 2]>0,
        lambda x: x[:, 3]>400000,
        lambda x: x[:, 3]>300000,
        lambda x: x[:, 3]>200000,
        lambda x: x[:, 3]>100000,
        lambda x: x[:, 3]>50000,
        ], 4000))
    learner.fit(x, y, A, 0.1)
