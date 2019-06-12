## 单词
+ scenario 梗概
+ flatten 平坦的

#### week 2的主要内容
+ 介绍什么是计算机视觉。<br>
对图像内容的分类任务。而对图像的切割任务可以看成是对数据对预处理，是一种特征工程。
+ 加载训练数据的代码
```
fashion_mist = keras.datasets.fashion_mnist
(train_image, train_labels),(test_image, test_labels) = fashion_mnist.load_data()
```
+ + 新观点，对一只靴子用数字来识别类型，比如说9，将有利于将它转换成各种语言形式。而如果直接翻译成英语，就只能让懂英语的人懂了。
+ + https://developers.google.com/machine-learning/fairness-overview/
+ 编码计算机视觉的神经网络
```
model = keras.Sequential([
  keras.layers.Flattern(input_shape=(28,28), # 这里表示输入的图像每个单位的大小是28*28
  keras.layers.Dense(128, activation=tf.nn.relu), # 这里表示中间的算子有128个神经元组成
  keras.layers.Dense(10,activation=tf.nn.softmax) # 这里的10表示输出只有10种种类
])
```
+ 代码整体走一遍
```
model.compile(optimizer = tf.train.AdamOptimizer(),
              loss = 'sparse_categorical_crossentropy')
model.fit(training_image, training_labels, epochs=5)   
model.evaluate(test_image, test_labels)
```
+ 使用callback适时停止训练
```
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
      if(logs.get('loss')<0.4):
        print ("\nLoss is low so canceling training!")
        self.model.stop_training = True
```

```
callbacks = myCallback()
...
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])
```

+ 完整的callback示例
```
import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.6):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True

mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

callbacks = myCallback()

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
```

## 训练结果有差异的原因
+ 数据集不同，较好的结果是用的fashion_mnist,较差的结果是用的minist
+ 训练和预测图片对255.0做了除法，不知何用
+ 构建模型，好的结果在Flatten的位置没有使用参数，而差的结果设置了(28,28)
+ 好的结果隐藏层使用了128个神经元，差的结果使用了512个神经元
+ 编译使用了tf.train.AdamOptimizer()做优化器，而差的结果使用了直接指定‘adam’


## 测试结果差异的原因
+ 神经元改为512个，对结果有提升，说明不是此导致的问题。
+ 增加input_shape参数也对结果有提升，说明不是此导致的问题。
+ 将优化器改为'adam',对结果有提升，说明不是此导致的问题。
+ 取消对训练和预测图片除以255的处理，结果大幅下降，是该问题导致的原因。
