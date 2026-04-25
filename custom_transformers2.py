# библиотеки
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DateTimeExtractor(BaseEstimator, TransformerMixin):
    
    # Извлекает временные признаки из datetime столбца 'tpep_pickup_datetime'
    # Создает циклические признаки для времени суток (sin/cos) и календарные признаки
    
    def __init__(self):
        self.seconds_in_day = 24 * 60 * 60      # Секунд в сутках
        self.seconds_in_hour = 3600             # Секунд в часе
        self.seconds_in_minute = 60             # Секунд в минуте
        self.feature_names_in_ = None
    
    def fit(self, X, y=None):
    # Сохраняет имена входных признаков для get_feature_names_out
        if hasattr(X, 'columns'):
            self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X):
    # Преобразует datetime в признаки:
    #   - pickup_dayofweek: день недели (0=понедельник, 6=воскресенье)
    #   - pickup_hour: час pickup
    #   - pickup_month: месяц
    #   - pickup_day: день месяца
    #   - time_sin, time_cos: циклические признаки времени суток
        
    # Циклические признаки кодируют время непрерывно (учитывая переход через полночь):
    # time_of_day = (hour*3600 + minute*60 + second) / 86400
    # time_rad = 2π * time_of_day

        X_new = X.copy()
        X_new['tpep_pickup_datetime'] = pd.to_datetime(X_new['tpep_pickup_datetime'])
        
        # Календарные признаки
        X_new['pickup_dayofweek'] = X_new['tpep_pickup_datetime'].dt.dayofweek
        X_new['pickup_hour'] = X_new['tpep_pickup_datetime'].dt.hour
        X_new['pickup_month'] = X_new['tpep_pickup_datetime'].dt.month
        X_new['pickup_day'] = X_new['tpep_pickup_datetime'].dt.day
        
        # Циклическое кодирование времени суток
        time_of_day = (X_new['tpep_pickup_datetime'].dt.hour * self.seconds_in_hour +
                      X_new['tpep_pickup_datetime'].dt.minute * self.seconds_in_minute +
                      X_new['tpep_pickup_datetime'].dt.second) / self.seconds_in_day
        time_of_day_rad = 2 * np.pi * time_of_day
        X_new['time_sin'] = np.sin(time_of_day_rad)
        X_new['time_cos'] = np.cos(time_of_day_rad)
        
        return X_new.drop(columns=['tpep_pickup_datetime'])
    
    def get_feature_names_out(self, input_features=None):
    # Возвращает имена выходных колонок
        if input_features is None:
            original_features = [col for col in self.feature_names_in_
                                if col not in ['tpep_pickup_datetime']]
        else:
            original_features = [col for col in input_features
                                if col not in ['tpep_pickup_datetime']]
        return original_features + ['pickup_dayofweek', 'pickup_hour', 
                                   'pickup_month', 'pickup_day', 'time_sin', 'time_cos']
    
class LocationFrequencyEncoder(BaseEstimator, TransformerMixin):
    # Кодирует частоты локаций и пар локаций
    # Заменяет категориальные PULocationID/DOLocationID на их частоты встречаемости
    def __init__(self):
        self.location_freq_ = None      # Частоты отдельных локаций
        self.pair_freq_ = None          # Частоты пар (PU_DO)

    def fit(self, X, y=None):
    # Вычисляет частоты локаций и пар локаций на тренировочных данных
        # Частота каждой локации (PU и DO вместе)
        all_locations = pd.concat([X['PULocationID'], X['DOLocationID']])
        self.location_freq_ = all_locations.value_counts(normalize=True).to_dict()

        # Частота каждой пары (PU_DO)
        location_pairs = (X['PULocationID'].astype(str) + '_' + 
                         X['DOLocationID'].astype(str))
        self.pair_freq_ = location_pairs.value_counts(normalize=True).to_dict()

        if hasattr(X, 'columns'):
            self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X):
    # Преобразует локации в их частоты
    # Создает признаки: PU_freq, DO_freq, location_pair_freq

        X_new = X.copy()
        X_new['PU_freq'] = X_new['PULocationID'].map(self.location_freq_).fillna(0)
        X_new['DO_freq'] = X_new['DOLocationID'].map(self.location_freq_).fillna(0)
        
        location_pair = (X_new['PULocationID'].astype(str) + '_' + 
                        X_new['DOLocationID'].astype(str))
        X_new['location_pair_freq'] = location_pair.map(self.pair_freq_).fillna(0)
        
        return X_new.drop(columns=['PULocationID', 'DOLocationID'])
    
    def get_feature_names_out(self, input_features=None):
    # Возвращает имена выходных признаков
        if input_features is None:
            original_features = [col for col in self.feature_names_in_
                                if col not in ['PULocationID', 'DOLocationID']]
        else:
            original_features = [col for col in input_features
                                if col not in ['PULocationID', 'DOLocationID']]
        return original_features + ['PU_freq', 'DO_freq', 'location_pair_freq']

class OutlierClipper(BaseEstimator, TransformerMixin):
    # Обрезает выбросы по правилу IQR (1.0 * IQR)
    # Для каждого числового признака: [Q1-1.0*IQR, Q3+1.0*IQR]
    def fit(self, X, y=None):
        # Вычисляет параметры для обрезки выбросов (Q1, Q3, IQR) для каждого признака

        self.q1 = np.nanpercentile(X, 25, axis=0)      # 25-й процентиль
        self.q3 = np.nanpercentile(X, 75, axis=0)      # 75-й процентиль
        self.iqr = self.q3 - self.q1                    # Межквартильный размах
        
        if hasattr(X, 'columns'):
            self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X):
        # Обрезает выбросы в данных по вычисленным границам
        # Пропускает признаки с IQR=0 (константы)

        X_new = X.copy()
        for i, col in enumerate(X_new.columns):
            if self.iqr[i] == 0:  # Пропускаем константные признаки
                continue
            lower = self.q1[i] - 1.0 * self.iqr[i]    # Нижняя граница
            upper = self.q3[i] + 1.0 * self.iqr[i]    # Верхняя граница
            X_new[col] = np.clip(X_new[col], lower, upper)
        return X_new
       
    def inverse_transform(self, X):
        # Возвращает данные без изменений 
        return X
    
    def get_feature_names_out(self, input_features=None):
        # Возвращает те же имена признаков (не создаем новые)
        if input_features is not None:
            return input_features
        return getattr(self, 'feature_names_in_', None)