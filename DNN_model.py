import tensorflow as tf
from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout


x_train = pd.read_csv("./data/x/x.csv", encoding='utf-8-sig')
y_train = pd.read_csv("./data/y/y_1.csv", encoding='utf-8-sig')

x_training_set = x_train.iloc[:1300, 2:].values
y_training_set = y_train.iloc[:1300].values
x_value_set = x_train.iloc[1300:1700, 2:].values
y_value_set = y_train.iloc[1300:1700].values
x_test_set = x_train.iloc[1700:, 2:].values
y_test_set = y_train.iloc[1700:].values


sc = MinMaxScaler(feature_range=(0, 1))  # 定义归一化：归一化到(0，1)之间
training_set_scaled = sc.fit_transform(x_training_set)  # 求得训练集的最大值，最小值这些训练集固有的属性，并在训练集上进行归一化
x_value_set = sc.transform(x_value_set)
x_test_set = sc.transform(x_test_set)


model = tf.keras.models.Sequential([
    Dense(32, activation='relu'),
    Dropout(0.1),
    Dense(32, activation='relu'),
    Dropout(0.1),
    Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01),
              loss='mean_squared_error',
              metrics=['mape'])

history = model.fit(training_set_scaled, y_training_set, batch_size=32, epochs=200, validation_data=(x_value_set, y_value_set), validation_freq=1)

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['mape']
val_acc_values = history_dict['val_mape']
epochs = range(1, len(val_loss_values) + 1)

plt.subplot(211)
plt.plot(epochs, loss_values, marker='x', label='Training loss')
plt.plot(epochs, val_loss_values, marker='o', label='Test loss')
plt.title('Training and test loss -- y1')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# plt.subplot(212)
# plt.plot(epochs, acc_values, marker='x', label='Training MAPE')
# plt.plot(epochs, val_acc_values, marker='o', label='Test MAPE')
# plt.title('Training and test MAPE -- y1')
# plt.xlabel('Epochs')
# plt.ylabel('MAPE')
# plt.legend()
# 保存图表为pdf格式
# plt.savefig("y1_2x32.pdf")

plt.show()

loss, Mape = model.evaluate(x_test_set, y_test_set)
print('\ntest loss', loss)
print('Mape', Mape)

#
# # 取x_test_set的第十行
# x_test_10 = x_test_set[9].reshape(1, -1)
# x_test_10 = sc.transform(x_test_10)
#
# # 输入网络中，输出结果
# y_pred_10 = model.predict(x_test_10)
#
# # 打印结果和对应的y_test_set
# print("预测结果：", y_pred_10)
# print("真实结果：", y_test_set[9])