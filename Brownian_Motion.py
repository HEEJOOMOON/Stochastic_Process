import numpy as np
import pandas as pd
from typing import Union
from scipy.stats import t as T

class Brownian:

    def __init__(self, series: Union[pd.Series, np.array],
                 process: str,
                 period: Union[str, int, float],
                 ):

        if type(series) == pd.Series:
            self.time_series = np.array(series)
        else: self.time_series = series
        self.process = process
        self.mu, self.std = None, None
        self.t, self.i = None, None
        self.confidence_level_ = None

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
            raise KeyError('daily, monthly, year, int or float')


    def fit(self):

        if self.process == 'standard':
            tmp = np.diff(self.time_series)

        elif self.process == 'arithmetic':
            tmp = np.diff(self.time_series)
            self.mu = np.mean(tmp)/self.dt
            self.std = np.std(tmp)/np.sqrt(self.dt)

        elif self.process == 'geometric':
            tmp = np.diff(np.log(self.time_series))
            self.mu = lambda dt: np.mean(tmp)/self.dt * dt
            self.std = lambda dt: np.std(tmp)/np.sqrt(self.dt) * dt

        else:
            raise KeyError('standard, arithmetic or geometric')


    def _standard(self, S, t, cl):

        path = S[0] + np.sqrt(t)*self.dt*np.random.normal(loc=0.0, scale=1.0, size=len(S))
        mean = np.mean(path)
        std = np.std(path)
        dof = len(S)-1
        self.confidence_level_ = T.interval(cl, dof, loc=mean, scale=std)
        return path


    def _abm(self, S, t, cl):

        path = S[0] + self.mu*t*self.dt + np.sqrt(self.std**2) * np.sqrt(t*self.dt)*np.random.normal(loc=0.0, scale=1.0, size=len(S))
        mean = np.mean(path)
        std = np.std(path)
        dof = len(S)-1
        self.confidence_level_ = T.interval(cl, dof, loc=mean, scale=std)
        return path


    def _gbm(self, S, t, cl):

        path = S[0] * np.exp(self.mu(t * self.dt) + self.std(t * self.dt) * np.random.normal(loc=0.0, scale=1.0, size=len(S)))
        mean = np.mean(np.log(path))
        std = np.std(np.log(path))
        dof = len(S)-1
        self.confidence_level_ = np.exp(T.interval(cl, dof, loc=mean, scale=std))
        return path


    def predict(self, t: int, i: int=10000,
                confidence_level: float=.95,
                ):

        self.t = t
        self.i = i

        S = np.zeros(i)
        S[0] = self.time_series[-1]

        if self.process == 'standard':
            return self._standard(S, t, confidence_level)

        elif self.process == 'arithmetic':
            return self._abm(S, t, confidence_level)

        elif self.process == 'geometric':
            return self._gbm(S, t, confidence_level)


if __name__ == '__main__':
    import yfinance as yf
    from Brownian_Motion import Brownian
    import matplotlib.pyplot as plt

    df = yf.download('KO', '2017-01-01', '2022-01-01').Close
    model = Brownian(df, process='standard', period='daily')
    model.fit()
    paths = []
    for i in range(100):
        paths.append(model.predict(t=i))
    plt.plot(paths)
    plt.show()
    prediction = model.predict(t=100)
    print(model.confidence_level_)
    plt.hist(prediction)
    plt.show()