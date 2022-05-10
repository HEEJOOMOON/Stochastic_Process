import numpy as np
import pandas as pd
from typing import Union
import statsmodels.api as sm
from scipy.stats import t as T
from typing import Optional, Union

class Ornstein_Uhlenbeck:

    def __init__(
            self,
            series: Union[pd.Series, np.array],
            period: Union[str, int, float],
            method: str='least_square'):

        if type(series) == pd.Series:
            self.time_series = np.array(series)
        elif type(series) == np.array:
            self.time_series = series
        else: raise TypeError('pd.Series, np.array or None(for simulation)')

        self.mu = None
        self.sigma = None
        self.theta = None
        self.t, self.i = None, None
        self.method = method
        self.confidence_interval_ = None
        self.half_life_ = None

        if type(period) == str:
            if period == 'daily':
                self.dt = 1/252
            elif period == 'monthly':
                self.dt = 1/12
            elif period == 'year':
                self.dt = 1
        elif type(period) == int or float:
            self.dt = period
        else:
            raise TypeError('daily, monthly, year, int or float')


    def fit(self):

        if self.method == 'least_square':
            X = self.time_series[1:]
            X = sm.add_constant(X)
            Y = self.time_series[:-1].reshape(-1, 1)
            model = sm.OLS(Y, X)
            res = model.fit()
            alpha, phi = res.params[0], res.params[1]
            epsilon = res.mse_resid
            self.theta = -np.log(phi)/self.dt
            self.mu = alpha/(1-phi)
            self.sigma = np.sqrt(epsilon/self.dt)
            self.half_life_ = np.log(2)/self.theta

        elif self.method == 'max_likelihood':
            None


    def predict(
            self,
            t: int,
            i: int=10000,
            cl: float=.95
    ):

        self.t, self.i = t, i
        S = self.time_series[-1]*np.exp(-self.theta*t*self.dt) + \
            self.mu*(1 - np.exp(-self.theta*t*self.dt)) + \
            self.sigma*np.sqrt((1 - np.exp(-2*self.theta*t*self.dt))/(2*self.theta))*np.random.normal(loc=.0, scale=1.0, size=i)
        mean = np.mean(S)
        std = np.std(S)
        dof = len(S)-1
        self.confidence_interval_ = T.interval(cl, dof, loc=mean, scale=std)
        return S


    @staticmethod
    def simulation(mu: Optional[float],
                   sigma: Union[float, int],
                   theta: Union[float, int],
                   initial: Optional[float],
                   n: int = 1000,
                   T: int = 1000,
                   dt: float = 1/252,
                   ):
        X = np.zeros((T, n))
        if initial:
            X[0] = initial
        for i in range(1, T):
            X[i] = X[i-1] + theta*(mu - X[i-1])*dt + sigma*np.random.normal(loc=0.0, scale=1.0, size=n)
        return X


if __name__ == '__main__':
    import FinanceDataReader as fdr
    import matplotlib.pyplot as plt


    df = fdr.DataReader('KO', '2017-01-01', '2022-01-01').Close
    model = Ornstein_Uhlenbeck(df, period='daily')
    model.fit()
    paths = []
    for i in range(20):
        paths.append(model.predict(t=i))
    paths = np.array(paths)
    plt.plot(paths[:, :20])
    plt.show()
    pred = model.predict(20)
    plt.hist(pred)
    plt.show()
    print(model.confidence_interval_)
    print(model.half_life_)