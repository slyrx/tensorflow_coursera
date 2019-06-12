# Weekly Exercise - Your First Neural Network

### Code
```
import tensorflow as tf
import numpy as np
from tensorflow import keras
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
ys = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)
model.fit(xs, ys, epochs=1000)
print(model.predict([7.0]))
```
#### tf.keras.Sequential是什么？
+ Sequential是顺序模型，用于组合神经网络的层结构容器。

#### keras.layers.Dense
+ 全连接层相当于线性回归的公式。
