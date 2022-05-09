## 智能计算系统技术文档
<h5 align="right">1952395沈韬｜Tao Shen(ShaoCHi)</h3>

[TOC]

### 背景介绍

```
开发环境
Python 3.9
Tensorflow 2.0
```
当前古诗句生成任务大多基于单一的循环神经网络（RNN）结构，在生成时需事先给定一个起始字，然后以该起始字为基础进行古诗句的生成，生成过程的可控性较差，往往达不到预期效果。同时，对于NLP（自然语言处理）、情感分析等一般采用RNN结构进行处理，所以这里采用LSTM模型进行实现

基于深度学习的古诗自动生成系统是通过神经网络对数据集进行学习和语义分析后训练出模型，在该模型上对于用户的输入进行响应从而生成对应的古诗。模型可以根据用户的输入生成古诗，例如藏头诗、补全古诗等，生成的古诗格式是保证正确的。

该模型主要分为服务于LSTM神经网络的数据预处理模块、LSTM神经网络模块和GUI模块。数据预处理模块中对于4万多首古诗进行预处理，转化为One-Hot编码，神经网络才能进行矩阵激素爱你、学习，LSTM神经网络模块是最核心的模块，需对激活函数、损失函数进行选取，参数优化等操作。

### 网络结构分析

- 传统RNN

  [循环神经网络](https://www.zhihu.com/search?q=循环神经网络&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A32085405})（Recurrent Neural Network，RNN）是一种用于处理序列数据的神经网络。相比一般的神经网络来说，他能够处理序列变化的数据。比如某个单词的意思会因为上文提到的内容不同而有不同的含义，RNN就能够很好地解决这类问题。

  RNN存在梯度爆炸和梯度消失的问题，所以RNN只能适应于短期神经网络记忆

- LSTM

  [长短期记忆](https://www.zhihu.com/search?q=长短期记忆&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A32085405})（Long short-term memory, LSTM）是一种特殊的RNN，主要是为了解决长序列训练过程中的梯度消失和梯度爆炸问题。简单来说，就是相比普通的RNN，LSTM能够在更长的序列中有更好的表现。

- RNN与LSTM对比

![Screen Shot 2021-12-09 at 10.01.48 PM](/Users/shentao/Library/Application Support/typora-user-images/Screen Shot 2021-12-09 at 10.01.48 PM.png)

​	左边的网络结构为普通RNN网络结构，右边即为LSTM论文结构

````
可以看出RNN网络结构仅有一个状态的输入，即上一节点的h^(t-1),其f函数一般采用tanh和Relu函数
````

````
在LSTM网络结构中，存在两种传输状态，C^(t-1)=>单元状态（cell state）
                               h^(t-1)=>隐蔽状态（hidden state）
````

![Screen Shot 2021-12-09 at 10.07.26 PM](/Users/shentao/Library/Application Support/typora-user-images/Screen Shot 2021-12-09 at 10.07.26 PM.png)

其中h^(t-1)和X^t将拼接得到四个状态，其含义分别如下

![Screen Shot 2021-12-09 at 10.10.01 PM](/Users/shentao/Library/Application Support/typora-user-images/Screen Shot 2021-12-09 at 10.10.01 PM.png)

其使用如下（LSTM内部的三个阶段）

- 忘记阶段。这个阶段主要是对上一个节点传进来的输入进行**选择性**忘记。简单来说就是会 “忘记不重要的，记住重要的”。 具体来说是通过计算得到的 ![[公式]](https://www.zhihu.com/equation?tex=z%5Ef) （f表示forget）来作为忘记门控，来控制上一个状态的 ![[公式]](https://www.zhihu.com/equation?tex=c%5E%7Bt-1%7D) 哪些需要留哪些需要忘。
- 选择记忆阶段。这个阶段将这个阶段的输入有选择性地进行“记忆”。主要是会对输入 ![[公式]](https://www.zhihu.com/equation?tex=x%5Et) 进行选择记忆。哪些重要则着重记录下来哪些不重要，则少记一些。当前的输入内容由前面计算得到的 ![[公式]](https://www.zhihu.com/equation?tex=z+) 表示。而选择的门控信号则是由 ![[公式]](https://www.zhihu.com/equation?tex=z%5Ei) （i代表information)来进行控制。

```
将上面两步得到的结果相加，即可得到传输给下一个状态的C^t。也就是上图中的第一个公式。
```

- 输出阶段。这个阶段将决定哪些将会被当成当前状态的输出。主要是通过 ![[公式]](https://www.zhihu.com/equation?tex=z%5Eo) 来进行控制的。并且还对上一阶段得到的 ![[公式]](https://www.zhihu.com/equation?tex=c%5Eo) 进行了放缩（通过一个[tanh激活函数](https://www.zhihu.com/search?q=tanh激活函数&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A32085405})进行变化)。

其节点状网络结构如图![Screen Shot 2021-12-09 at 10.18.16 PM](/Users/shentao/Library/Application Support/typora-user-images/Screen Shot 2021-12-09 at 10.18.16 PM.png)

其网络结构隐层互相连接，以此来实现长期记忆

#### 激活函数

激活函数选取LSTM常用的tanh函数对数据进行计算，将其数据变为-1到1之间的一个值，然后将其转化为其预测的下一个词的概率

#### 损失函数

因为古诗的生成可以概括为多分类问题，其损失函数选取categorical_crossentropy loss（交叉熵损失函数）

````
-函数表达式
𝐶=−1𝑁∑𝑖∑𝑘[𝑦𝑖𝑘𝑙𝑛(𝑝𝑖𝑘)]
其中，𝑦𝑖𝑘——样本𝑖的label，与正类类别相同为1，否则为0；𝑝𝑖𝑘——样本𝑖i的预测值为第𝑘类的概率。
````

````python
优点
使用逻辑函数得到概率，并结合交叉熵当损失函数时，在模型效果差的时候学习速度比较快，在模型效果好的时候学习速度变慢
缺点
sigmoid(softmax)+cross-entropy loss 擅长于学习类间的信息，因为它采用了类间竞争机制，它只关心对于正确标签预测概率的准确性，忽略了其他非正确标签的差异，导致学习到的特征比较散。基于这个问题的优化有很多，比如对softmax进行改进，如L-Softmax、SM-Softmax、AM-Softmax等。
````

#### 优化器

优化器选择keras.optimizers.Adam()

```
在监督学习中我们使用梯度下降法时，学习率是一个很重要的指标，因为学习率决定了学习进程的快慢（也可以看作步幅的大小）。如果学习率过大，很可能会越过最优值，反而如果学习率过小，优化的效率可能很低，导致过长的运算时间，所以学习率对于算法性能的表现十分重要。而优化器keras.optimizers.Adam()是解决这个问题的一个方案。其大概的思想是开始的学习率设置为一个较大的值，然后根据次数的增多，动态的减小学习率，以实现效率和效果的兼得。
```

#### 参数分析

```python
数据预处理(对数据集进行处理，同时进行One-Hot编码：
# 禁用词，包含如下字符的唐诗将被忽略
DISALLOWED_WORDS = ['（', '）', '(', ')', '__', '《', '》', '【', '】', '[', ']']
# 诗句最大长度
MAX_LEN = 64
# 最小词频（用于过滤低频词）
MIN_WORD_FREQUENCY = 8
```

```python
# 每个epoch训练完成后，随机生成SHOW_NUM首古诗作为展示
SHOW_NUM = 3
```

````python
模型采用随机梯度下降SGD来训练，：将训练集分成多个mini_batch（即常说的batch）,一次迭代训练一个minibatch（即batchsize个样本），根据该batch数据的loss更新权值。这相比于全数据集训练，相当于是在寻找最优时人为增加了一些随机噪声，来修正由局部数据得到的梯度，尽量避免因batchsize过大陷入局部最优。
# 训练
TRAIN_EPOCHS = 20
BATCH_SIZE = 4
# LSTM每层神经元个数
HIDDEN_NUM = 128
````

### 论文研读心得

[基于深度学习的歌词和古诗自动生成系统设计](https://kns.cnki.net/KXReader/Detail?invoice=O2y8QFazN8tqFSWhr2bD8eNFZvGH2mTV1B9g5AVR0xKag9Cz%2B58QYQat34Eb8l9TNT3dujTLM6h%2BhpIT%2BTIpxX32MCeTt3RZef0mHPnx2eCMG9NEmaSCloC89C1cLyWpa41tXyZjh7mDYDG0DuGCdM7i3p7vo0dyWr7gLIhDduw%3D&DBCODE=CJFD&FileName=XDXK202101006&TABLEName=cjfdlast2021&nonce=AE81F1359B434B9FBFE98C5B353D2B60&uid=&TIMESTAMP=1639060869765)

[基于Seq2Seq模型的自定义古诗词生成](https://kns.cnki.net/KXReader/Detail?invoice=HfJZcCo%2BIqcStONzkF9f7jjcWJh3GmzjN3tv1GyHep7E3impu57sU8HeJwN5dWPww1dBTBOXWQtFNfYZZBW3WRwlHg44ae4%2BCNIVpiht7j5ofdycjiYGroFiU8x74E7lOw7kGvm650ZdBqEWc6Mr2Vswxz2CGrBxNEwPf04puVw%3D&DBCODE=CJFD&FileName=DGLG202005010&TABLEName=cjfdlast2020&nonce=D7286FFF793A4465A4CC81E4BB6D39EF&uid=&TIMESTAMP=1639060954593)

对于论文中所提到的系统进行了初步的实现，即该模型；

### 使用（训练和测试）方法
#### 训练模型

- 在setting.py中配置相关路径（现在的路径为相对路径）
- 运行train.py进行训练
- 运行过程中的时间及关键步骤信息等存于 logs/poetry.logs 中
- 所有配置信息都存放于 settings.py
```
现有参数为
epochs：20
batch-size：16
```
#### 测试模型
由于该模型是对于古诗的生成，对于古诗的评价好坏指标没有具体

- 运行eval.py进行模型的查看

- 同时可以进行用户的交互

  ````
  有三种功能
  1、随机生成一首诗
  2、根据用户给定的首句，生成剩下的部分
  3、根据用户的输入，生成一首藏头诗
  ````

### 效果演示

- 模型概览

  ```python
  model = tf.keras.Sequential([
      # 不定长度的输入
      tf.keras.layers.Input((None, )),
  
      # 词嵌入层
      tf.keras.layers.Embedding(input_dim=tokenizer.vocab_size, output_dim=settings.HIDDEN_NUM),
  
      # 第一个LSTM层，返回序列作为下一层的输入
      tf.keras.layers.LSTM(settings.HIDDEN_NUM, dropout=0.5, return_sequences=True),
  
      # 第二个LSTM层，返回序列作为下一层的输入
      tf.keras.layers.LSTM(settings.HIDDEN_NUM, dropout=0.5, return_sequences=True),
  
      # 对每一个时间点的输出都做softmax，预测下一个词的概率
      tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(tokenizer.vocab_size,
                                                            activation='softmax')),
  ])
  ```

![](source/model_summary.png)

```
第一层embedding：将一个特征转换为一个向量。比如最容易理解的one-hot编码。但在实际应用当中，将特征转换为one-hot编码后维度会十分高。所以我们会将one-hot这种稀疏特征转化为稠密特征，通常做法也就是转化为我们常用的embedding。在NLP领域中，我们需要将文本转化为电脑所能读懂的语言，也就是把文本语句转化为向量，也就是embedding，其产生的参数个数为439552
第二层LSTM：其节点个数为128，产生的参数个数为131584，并将其输出作为输入传入下一层的LSTM中
第三层LSTM：其节点个数为128，产生的参数个数为131584
第四层TimeDistri：对每个时间点的输出进行softmax，产生的参数个数为442986

总的参数个数为1M左右，是一个比较小的模型
```

- 参数分析

  ![](source/args.png)

  该模型训练时采用的参数为

  ```
  batch-size:16
  data num:24551
  train epochs:20
  ```

- 训练分析

  ![](source/epoch1.png)

  ![](source/epoch6.png)

  ![](source/epoch20.png)

  ```
  通过对训练过程的部分截图，我们可以看出：
  每一轮迭代的步数为1534
  
  在进行前几轮训练时，产生的古诗格式存在不正确的现象；一句古诗中穿插着一些标点符号，且loss值还比较大
  在经过几轮epoch之后，我们可以看出其
  随着训练的进行，我们可以发现loss逐渐下降，最终趋于稳定3.57左右，产生的古诗格式较为工整，且其意境也大有提高
  
  训练时长：
  在batch-size选择4时，一轮epoch耗时320s左右，整个模型训练花费大概2h
  在batch-size选择16时，一轮epoch耗时207s左右，整个模型训练花费大概1.3h
  ```

- 测试分析

  运行eval.py即可对模型进行测试

  ```python
  # 加载训练好的模型
  logging.info('start test ...')
  model = tf.keras.models.load_model(settings.BEST_MODEL_PATH)
  ```

  对模型进行测试![example1](/Users/shentao/Downloads/example1.png)

  ![example2](/Users/shentao/Downloads/example2.png)

  ![example3](/Users/shentao/Downloads/example3.png)
