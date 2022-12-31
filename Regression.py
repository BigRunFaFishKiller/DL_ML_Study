from keras.datasets import boston_housing
from keras import models
from keras import layers
import matplotlib.pyplot as plt
import numpy as np

# 回归预测问题，不等价于线性回归.
# 预测波士顿房价问题，每个特征都有不同的取值问题

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# 标准化数据
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

k = 4
num_val_sample = len(train_data) // k
num_epochs = 500
all_mae_histories = []

# k折交叉验证法
for i in range(k):
    print('processing fold ', i)
    # 第k个分区的数据作为验证数据
    val_data = train_data[i * num_val_sample: (i + 1) * num_val_sample]
    val_targets = train_targets[i * num_val_sample: (i + 1) * num_val_sample]

    # 其他分区的数据作为训练数据
    partial_train_data = np.concatenate([train_data[: i * num_val_sample], train_data[(i + 1) * num_val_sample : ]], axis=0)
    partial_train_targets = np.concatenate([train_targets[: i * num_val_sample], train_targets[(i + 1) * num_val_sample : ]], axis=0)

    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=0, validation_data=(val_data, val_targets))
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    mae_history = history.history['mae']
    all_mae_histories.append(mae_history)

# 计算所有轮次中的k折验证分数平均值并作图
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

# 去除一些点后重新绘图
average_mae_history = average_mae_history[10 :]
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

# 根据图像重新训练模型并预测
model = build_model()
model.fit(train_data, train_targets, epochs=80, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)