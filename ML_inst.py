import pandas as pd
import numpy as np
import os, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb
from imblearn.over_sampling import SMOTE

class ml_learn:
    def __init__(self, paths):
        self.path = paths
        df = pd.read_csv(self.path + r"\signals\data.csv")
        print('df_shape', df.shape)
        df = self.rebalance(df)
        print('reb_df_shape', df.shape)
        self.X_fft_row = self.X_to_fft(df)  # Преобразуем временной ряд в ряд фурье
        self.X_stats_fft_row = self.stat_X(self.X_fft_row, 0, 625)  # Считаем статистики по ряду фурье
        self.y_class = df.iloc[:, 0]
        self.y_range = df.iloc[:, 1]
        self.X_time_row = df.iloc[:, 2:1252]
        self.X_stats_row = df.iloc[:, 1252:]
        self.random_state = 42
        self.models = {}

        self.models.update(self.random_forest('X_time_row', self.X_time_row, self.y_class, self.y_range))  # Модели случайного леса по полному временному ряду
        # self.models.update(self.random_forest('X_stats_row', self.X_stats_row, self.y_class, self.y_range))  # Модели случайного леса по статистикам временного ряда
        self.models.update(self.random_forest('X_fft_row', self.X_fft_row, self.y_class, self.y_range))  # Модели случайного леса по рядам фурье
        # self.models.update(self.random_forest('X_stats_fft_row', self.X_stats_fft_row, self.y_class, self.y_range))  # Модели случайного леса по статистикам по рядам фурье

        self.models.update(self.lgb('X_time_row_gb', self.X_time_row, self.y_class, self.y_range))  # Модели гб по полному временному ряду
        # self.models.update(self.lgb('X_stats_row_gb', self.X_stats_row, self.y_class, self.y_range))  # Модели гб по статистикам временного ряда
        # self.models.update(self.lgb('X_fft_row_gb', self.X_fft_row, self.y_class, self.y_range))  # Модели гб по рядам фурье
        # self.models.update(self.lgb('X_stats_fft_row_gb', self.X_stats_fft_row, self.y_class, self.y_range))  # Модели гб по статистикам по рядам фурье

        self.models.update(self.logreg('X_time_row_logreg', self.X_time_row, self.y_class, self.y_range))
        self.models.update(self.logreg('X_fft_row_logreg', self.X_fft_row, self.y_class, self.y_range))
        # df = df.drop(df.columns[1252:], axis=1)
        # self.zero = df.loc[(df["0"] == 0) & (df["1"] == 0), :].mean(axis=0)
        # df =

        self.save_models()  # Сохраняем модели

    def rebalance(self, df):
        oversample = SMOTE()
        y = df.iloc[:, 0]
        X = df.iloc[:, 1:1252]
        X, y = oversample.fit_resample(X, y)
        new_df = pd.concat([X, y], axis=1)
        cols = new_df.columns.tolist()
        new_df = new_df[cols[-1:] + cols[:-1]]
        stat = self.stat_X(new_df,2,1252)
        new_df = pd.concat([new_df, stat], axis=1)
        new_df.to_csv(self.path + r"\signals\data.csv", sep=',', index=False)
        # np.savetxt(self.path + r"\signals\data.csv", new_df, delimiter=',')
        return new_df

    def lgb(self, name: str, X, y_class, y_range):
        model_class = lgb.LGBMClassifier(random_state=self.random_state, n_jobs=-1, n_estimators=200, learning_rate=0.1)
        model_class.fit(X, y_class)

        model_range = lgb.LGBMClassifier(random_state=self.random_state, n_jobs=-1, learning_rate=0.05)
        model_range.fit(X, y_range)
        return {name: (model_class, model_range)}

    def random_forest(self, name: str, X, y_class, y_range):
        model_class = RandomForestClassifier(random_state=self.random_state, n_estimators=200, n_jobs=4)
        model_class.fit(X, y_class)

        model_range = RandomForestClassifier(random_state=self.random_state, n_estimators=200, n_jobs=4)
        model_range.fit(X, y_range)
        return {name: (model_class, model_range)}

    def logreg(self, name: str, X, y_class, y_range):
        model_class = LogisticRegression(random_state=self.random_state, max_iter=300, solver='newton-cg')
        model_class.fit(X, y_class)

        model_range = LogisticRegression(random_state=self.random_state, max_iter=300, solver='newton-cg')
        model_range.fit(X, y_range)
        return {name: (model_class, model_range)}


    def save_models(self):
        if not os.path.exists(self.path + '/models'):
            os.makedirs(self.path + '/models')

        f = open(self.path + '/models/conf.txt', 'a')
        f.truncate(0)
        list_keys = self.models.keys()

        for k in list_keys:
            filename = self.path + '/models/model_' + str(k) + '.pkl'
            joblib.dump(self.models.get(k), filename, compress=9)
            f.write(k + ',' + filename + '\n')
        f.close()

    def X_to_fft(self, df):
        c = []
        for i in df.iloc[:, 2:1252].itertuples(index=False):
            # print(i[1:])
            X_fft = np.fft.fft(i)
            c.append(np.abs(X_fft)[: (1250 // 2)])
        return pd.DataFrame(c)

    def stat_X(self, df, start_column, stop_column):
        stat = pd.DataFrame()
        stat['mean'] = df.iloc[:, start_column:stop_column].mean(axis=1)
        stat['median'] = df.iloc[:, start_column:stop_column].median(axis=1)
        stat['std'] = df.iloc[:, start_column:stop_column].std(axis=1)
        stat['var'] = df.iloc[:, start_column:stop_column].var(axis=1)
        stat['max'] = df.iloc[:, start_column:stop_column].max(axis=1)
        stat['min'] = df.iloc[:, start_column:stop_column].min(axis=1)
        return stat


# path = r'C:\Users\Админ\PycharmProjects\Work_task'
# df = pd.read_csv(path + "\data.csv")
# ml = ml_learn(r"C:\Users\Админ\PycharmProjects\Work_task")
# print(ml.models)
# ml.save_models()


class ml_predict:
    def __init__(self, paths):
        self.path = paths
        self.models = self.load_models()

    def load_models(self):  # Загружаем модели из конфигурационного файла
        f = open(self.path + '/models/conf.txt', 'r')
        models = {}
        for line in f:
            s = line.rstrip('\n').split(',')
            models[s[0]] = joblib.load(s[1])
        f.close()
        return models

    def get_predict(self, x):  # Делаем предсказания для всех загруженных моделей
        models_keys = self.models.keys()
        predict_c = []  # Множество предсказаний класса
        predict_r = []  # Множество предсказаний дальности
        print(models_keys)
        for key in models_keys:
            if key == 'X_time_row':
                _X = x.reshape(1,-1)
                # print('X_time_row',_X)
                self.add_predict(_X, key, predict_c, predict_r)
                # print(predict_c)
            elif key == 'X_stats_row':
                _X = self.stat_x(x)
                # print('X_stats_row',_X)
                self.add_predict(_X, key, predict_c, predict_r)
                # print(predict_c)
            elif key == 'X_fft_row':
                _X = self.fft_X(x)
                # print('X_fft_row',_X)
                self.add_predict(_X, key, predict_c, predict_r)
            elif key == 'X_stats_fft_row':
                _X = self.stat_x(self.fft_X(x))
                # print('X_stats_fft_row',_X)
                self.add_predict(_X, key, predict_c, predict_r)
            elif key == 'X_time_row_gb':
                _X = x.reshape(1,-1)
                # print('X_time_row',_X)
                self.add_predict(_X, key, predict_c, predict_r)
                # print(predict_c)
            elif key == 'X_stats_row_gb':
                _X = self.stat_x(x)
                # print('X_stats_row',_X)
                self.add_predict(_X, key, predict_c, predict_r)
                # print(predict_c)
            elif key == 'X_fft_row_gb':
                _X = self.fft_X(x)
                # print('X_fft_row',_X)
                self.add_predict(_X, key, predict_c, predict_r)
            elif key == 'X_stats_fft_row_gb':
                _X = self.stat_x(self.fft_X(x))
                # print('X_stats_fft_row',_X)
                self.add_predict(_X, key, predict_c, predict_r)
            elif key == 'X_time_row_logreg':
                _X = x.reshape(1,-1)
                # print('X_time_row',_X)
                self.add_predict(_X, key, predict_c, predict_r)
            elif key == 'X_fft_row_logreg':
                _X = self.fft_X(x)
                self.add_predict(_X, key, predict_c, predict_r)

        print('c: ', predict_c)
        print('r: ', predict_r)
        return np.median(predict_c), np.median(predict_r)

    def add_predict(self, x, key, predict_c, predict_r):  # Добавляем предсказания к соответствующему множеству
        predict_c.append(self.models.get(key)[0].predict(x)[0])
        predict_r.append(self.models.get(key)[1].predict(x)[0])

    def stat_x(self, x):  # Считаем статистики
        if type(x) == np.ndarray:
            x = x
        else:
            x = x.reshape(1, -1)
        stat = [np.mean(x), np.median(x), np.std(x), np.var(x), np.max(x), np.min(x)]
        return np.array(stat).reshape(1,-1)

    def fft_X(self, x):  # Делаем преобразование фурье над рядом
        x = x.reshape(1,-1)
        return np.abs(np.fft.fft(x))[:, :(x.shape[1] // 2)]
#
# ml_prd = ml_predict(r'C:\Users\Админ\PycharmProjects\Work_task')
#
# path = r'C:\Users\Админ\PycharmProjects\Work_task'
# df = pd.read_csv(path + "\data.csv")
# X_test = df.loc[(df["0"] == 1) & (df["1"] == 4), :].iloc[0, 2:1252]
#
# print(ml_prd.get_predict(X_test))
