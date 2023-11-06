import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
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


dense_layers = [1, 2, 3, 4, 5, 6, 7]
units = [16, 32, 64, 128, 256]

for the_dense_layers in dense_layers:
    for the_units in units:
        filepath = '.\\models_y1\\{val_mae:.2f}_{epoch:02d}_' + f'dense_{the_dense_layers}_units_{the_units}'
        checkpoint = ModelCheckpoint(filepath=filepath,
                                     save_weights_only=False,
                                     monitor='val_mae',
                                     mode='min',
                                     save_best_only=True)

        # 构建神经网络
        model = Sequential()
        # 第一层

        for i in range(the_dense_layers):
            # 全连接层
            model.add(Dense(the_units, activation='relu'))
            model.add(Dropout(0.1))

        # 输出层
        model.add(Dense(1))

        # 优化器，损失函数，评价函数
        model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01),
                      loss='mse',
                      metrics=['mae'])

        model.fit(training_set_scaled, y_training_set, batch_size=64, epochs=200,
                  validation_data=(x_value_set, y_value_set), validation_freq=1, callbacks=[checkpoint])
