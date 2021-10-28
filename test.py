from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
def stat_X(df, start_column, stop_column):
    stat = pd.DataFrame()
    stat['mean'] = df.iloc[:, start_column:stop_column].mean(axis=1)
    stat['median'] = df.iloc[:, start_column:stop_column].median(axis=1)
    stat['std'] = df.iloc[:, start_column:stop_column].std(axis=1)
    stat['var'] = df.iloc[:, start_column:stop_column].var(axis=1)
    stat['max'] = df.iloc[:, start_column:stop_column].max(axis=1)
    stat['min'] = df.iloc[:, start_column:stop_column].min(axis=1)
    return stat

df = pd.read_csv(r'C:\Users\admin\PycharmProjects\qt-proj\settings\signals\data.csv')
# print(df.head())
oversample = SMOTE()
y = df.iloc[:, 0]
X = df.iloc[:, 1:1252]
X, y = oversample.fit_resample(X, y)
new_df = pd.concat([X, y], axis=1)
cols = new_df.columns.tolist()
new_df = new_df[cols[-1:] + cols[:-1]]
print(new_df)
stat = stat_X(new_df,2,1252)
new_df = pd.concat([new_df, stat], axis=1)
print(new_df)
new_df.to_csv(r'C:\Users\admin\PycharmProjects\qt-proj\settings\signals\data2.csv', sep=',', index=False)
df3 = pd.read_csv(r'C:\Users\admin\PycharmProjects\qt-proj\settings\signals\data2.csv')
print(df3.head())