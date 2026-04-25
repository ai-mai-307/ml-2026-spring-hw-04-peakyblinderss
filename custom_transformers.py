# библиотеки
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# обработка аномалий
class OutlierClipper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # считаем квартили и IQR по каждому признаку
        self.q1 = np.nanpercentile(X, 25, axis=0)
        self.q3 = np.nanpercentile(X, 75, axis=0)
        self.iqr = self.q3 - self.q1
        return self
    def transform(self, X):
        X_new = X.copy()
        # если на вход пришёл pandas DataFrame
        if hasattr(X_new, "columns"):
            for i, col in enumerate(X_new.columns):
                if self.iqr[i] == 0:
                    continue
                lower = self.q1[i] - 1.5 * self.iqr[i]
                upper = self.q3[i] + 1.5 * self.iqr[i]
                X_new[col] = np.clip(X_new[col], lower, upper)
        # если на вход пришёл numpy array
        else:
            X_new = np.asarray(X_new).copy()
            for i in range(X_new.shape[1]):
                if self.iqr[i] == 0:
                    continue
                lower = self.q1[i] - 1.5 * self.iqr[i]
                upper = self.q3[i] + 1.5 * self.iqr[i]
                X_new[:, i] = np.clip(X_new[:, i], lower, upper)
        return X_new
numeric_outlier = OutlierClipper()

# циклическое преобразование
class CyclicalTimeEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.seconds_in_day = 86400
        self.seconds_in_hour = 3600
        self.hours_in_day = 24
        self.day_in_week = 7
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = np.array(X)  # ожидаем числовой timestamp в секундах
        hour = (X[:,0] //self.seconds_in_hour) %  self.hours_in_day
        day = X[:,0] // self.seconds_in_day
        weekday = day % self.day_in_week
        hour_sin = np.sin(2 * np.pi * hour /  self.hours_in_day)
        hour_cos = np.cos(2 * np.pi * hour / self.hours_in_day)
        weekday_sin = np.sin(2 * np.pi * weekday / self.day_in_week)
        weekday_cos = np.cos(2 * np.pi * weekday / self.day_in_week)
        return np.column_stack([hour_sin, hour_cos, weekday_sin, weekday_cos])
timestamp_encoder = CyclicalTimeEncoder()