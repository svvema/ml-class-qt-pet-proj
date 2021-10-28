import numpy as np
# path = r'C:\Users\Админ\PycharmProjects\Work_task'
#
# data = np.zeros(1254)
# insert = np.ones(1254)
# data = np.vstack((data, insert))
# data = np.vstack((data, insert))
# # print(data)
#
#
# import csv
#
# with open(path + '\data.csv', 'a') as f:
#     np.savetxt(f, data, delimiter=",")
#
# import os.path
# print(os.path.isfile(r'C:\Users\Админ\PycharmProjects\Work_task\data.csv'))
def stat_X(x):
    stat = [np.mean(x), np.median(x), np.std(x), np.var(x), np.max(x), np.min(x)]
    return stat
data = np.random.rand(5)
print(stat_X(data))
data = np.append(data, stat_X(data))
print(data)