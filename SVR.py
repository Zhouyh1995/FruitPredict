import tensorflow as tf
from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout
from sklearn.svm import SVR
import math

x_train = pd.read_csv("./data/x/x.csv", encoding='utf-8-sig')
y_train = pd.read_csv("./data/y/y_1.csv", encoding='utf-8-sig')

x_training_set = x_train.iloc[:1300, 2:].values
y_training_set = y_train.iloc[:1300].values
x_value_set = x_train.iloc[1300:1700, 2:].values
y_value_set = y_train.iloc[1300:1700].values
x_test_set = x_train.iloc[1700:, 2:].values
y_test_set = y_train.iloc[1700:].values
y_training_set = y_training_set.ravel()
y_value_set = y_value_set.ravel()
y_test_set = y_test_set.ravel()

sc = MinMaxScaler(feature_range=(0, 1))  # 定义归一化：归一化到(0，1)之间
training_set_scaled = sc.fit_transform(x_training_set)  # 求得训练集的最大值，最小值这些训练集固有的属性，并在训练集上进行归一化
x_value_set = sc.transform(x_value_set)
x_test_set = sc.transform(x_test_set)

svr_rbf = SVR(kernel='rbf', gamma=0.1, epsilon=0.1)
svr_rbf.fit(training_set_scaled, y_training_set)

svr_lin = SVR(kernel='linear', gamma=0.1, epsilon=0.1)
svr_lin.fit(training_set_scaled, y_training_set)

svr_poly = SVR(kernel='poly', gamma=0.1, epsilon=0.1, degree=3)
svr_poly.fit(training_set_scaled, y_training_set)


def ca_pre(x, model):
    X = []
    for ix, x in enumerate(x):
        X.append(model.predict(x.reshape(1, -1)))
    return X


def fun_Mape(pre, actual):
    sum = 0
    n = actual.shape[0]
    for i in range(n):
        sum += abs(pre[i] - actual[i]) / actual[i]

    return (sum / n) * 100


pre_rbf_val = ca_pre(x_value_set, svr_rbf)
pre_lin_val = ca_pre(x_value_set, svr_lin)
pre_poly_val = ca_pre(x_value_set, svr_poly)
pre_rbf_test = ca_pre(x_test_set, svr_rbf)
pre_lin_test = ca_pre(x_test_set, svr_lin)
pre_poly_test = ca_pre(x_test_set, svr_poly)

print(f'rbf_val_MAPE:{fun_Mape(pre_rbf_val, y_value_set)}')
print(f'lin_val_MAPE:{fun_Mape(pre_lin_val, y_value_set)}')
print(f'poly_val_MAPE:{fun_Mape(pre_poly_val, y_value_set)}')
print(f'rbf_test_MAPE:{fun_Mape(pre_rbf_test, y_test_set)}')
print(f'lin_test_MAPE:{fun_Mape(pre_lin_test, y_test_set)}')
print(f'poly_test_MAPE:{fun_Mape(pre_poly_test, y_test_set)}')

print(svr_rbf.score(x_test_set, y_test_set))
print(svr_lin.score(x_test_set, y_test_set))
print(svr_poly.score(x_test_set, y_test_set))



# 取x_test_set的第十行
x_test_10 = x_test_set[9].reshape(1, -1)
x_test_10 = sc.transform(x_test_10)

# 输入网络中，输出结果
y_pred_10 = svr_lin.predict(x_test_10)

# 打印结果和对应的y_test_set
print("预测结果：", y_pred_10)
print("真实结果：", y_test_set[9])