from keras.datasets import reuters
from keras import models
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
# 多分类（单标签），新闻分类，共有46个类别

# 保留数据集中前10000个最常出现的单词
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

# 将整数序列转变为10000维二进制矩阵
def vectorize_sequences(sequences, dimension = 10000):
    # 生成全0矩阵
    results = np.zeros((len(sequences), dimension))
    # 将矩阵(i, 整数序列第i位数)位置上的元素赋值为1.
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

# 将数据集向量化
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# 将标签使用one-hot方法转化为向量，类似于上述将数据向量化
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

# 标签向量化
one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels) # 或使用keras内置函数 to_categorical(test_labels)  from keras.utils.np_utils import to_categorical

# 构建网络
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
# 输出46个纬度上的概率分布，概率之和为1
model.add(layers.Dense(46, activation='softmax'))

# 编译模型
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# 保留验证集
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

# 训练模型
history = model.fit(partial_x_train, partial_y_train, epochs=6, batch_size=512, validation_data=(x_val, y_val))
result = model.evaluate(x_test, one_hot_test_labels)


# 绘制训练损失和验证损失
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 绘制训练精度和验证精度
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 验证
pre = model.predict(x_test)
np.sum(pre[0])
np.argmax(pre[0])