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
