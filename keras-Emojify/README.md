## 笔记

只是简单提及代码中的概念，不做长篇大论，具体还需自行查阅资料。 不介绍神经网络基本知识（数学知识和推导），也不介绍Keras基本应用。

### 问题

分类问题。`data.csv`是数据集，第一列是所有训练的句子，第二列是每个句子对应的标签。每个标签表示的是一个表情符号，这里用数字来代替。即用户输入一句话猜测这句话后面比较适合接上什么表情。

输入：一句话；输出：对应标签。

这里为了方便起见，只有5个标签（0-4）

### 词向量

语言（词、句子、篇章等）属于人类认知过程中产生的高层认知抽象实体，而语音和图像属于较为底层的原始输入信号。

人在看到单词或者是句子，会主动将其抽象为已知认知的一些知识（通过先验知识）。而计算机却不能做到。语音和图像是可以根据简单的数字信号来得到，且具有顺序性和关联性，而语言却不是。所以需要一种方式来将语言转换成计算机能认知的事物（信号）。换句话说，就是将语言数字化，词向量是数字化的一种方式。而词向量又有几种方式，最常见的一种就是One-Hot Representation。

```python
# one-hot demo
# 假设有一个词典里面只有三个词"Europe"，"US"，"Asia"
lst = ["Europe"，"US"，"Asia"]
# 那么每个单词对应的词向量就是
d_lst = {
    "Europe": [1, 0, 0],
    "US": [0, 1, 0],
    "Asia": [0, 0, 1],
}
```

还有一种是Distributed Representation，嵌入使用的数据集就是采取的这种方式

```python
# Distributed demo
# 假设有一个词典里面只有三个词"Europe"，"US"，"Asia"
lst = ["Europe"，"US"，"Asia"]
# 现在把每个单词当作一个点映射到一个二维坐标轴上。比如
coordinates_lst = {
    "Europe": (1, 1), 
    "US": (2, 3)，
    "Asia": (5, 3),
}
# 根据点与点之间的距离（可以是欧式距离或者其他），即表示了单词之间的相似度，转换为一个向量。结果如下
d_lst = {
    "Europe": [0, \sqrt(5), 2\sqrt(5)],
    "US": [\sqrt(5), 0, 3],
    "Asia": [2\sqrt(5), 3, 0],
}
```

当然这里只是简单的举个例子，实际上没有这么简单。关于单词该如何当作点映射到坐标轴上，取何种距离计算方式等，在这里不做讲解。`glove.6B.100d.txt`是一种已经转换好的词向量数据集。

### 嵌入

**embedding（嵌入）**：通过一个数值向量代表每个单词在字典中的距离（L2距离或其他距离），这样形成的语义关系几何关系空间向量为嵌入空间。
"食堂"与"午饭"比较接近，即距离更近，"学校"与"蚂蚁"语义不同，即距离更远。
"工作"+"x"="办公室"理解为工作发生在哪？办公室。
使用技术`word2vec`或者矩阵分解。

```python
# 官方博客使用Keras Embedding 示例
from keras.layers.embeddings import Embedding
def official_embedding_layer():
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    # 计算嵌入矩阵
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    # 将嵌入矩阵加载到嵌入层。trainable=False:防止权重更新时，重新训练
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
```

这里读取一个额外的词向量数据集（Global Vectors for Word Representation），来得到嵌入层，进行配合后续神经网络模型训练。

### 定义模型

Keras总共有两种定义模型API，序贯模型API和函数式模型API。这里使用的是后者进行定义。

```python
# 官方demo
from keras.models import Model
from keras.layers import Input, Dense

# 输入为a，维度是32
a = Input(shape=(32,))
# 输出为b，维度是32
b = Dense(32)(a)
# 这样广义的拥有输入和输出模型。
model = Model(inputs=a, outputs=b)
```

```python
# 代码中train_model
from keras.layers import Input, Dense, Dropout, LSTM, Activation
from keras.models import Model

# 定义输入层，输入维度是训练数据集中句子拥有的最大单词个数，根据数据集这里input_shape为10
sentence_indices = Input(shape=input_shape, dtype='int32')
# 定义嵌入层
embedding_layer = pretrain_embedding_layer(word_to_vec_map, word_to_index)
embeddings = embedding_layer(sentence_indices)
# 定义两个LSTM，输出向量维度都是128，第一个拥有10个输出向量，第二个只有一个。在每次LSTM层之后定义Dropout
X = LSTM(128, return_sequences=True)(embeddings)
X = Dropout(0.5)(X)
X = LSTM(128, return_sequences=False)(X)
X = Dropout(0.5)(X)
# 定义一个全连接层，输出维度为5（5个标签）
X = Dense(5, activation='softmax')(X)
# 定义一个激活函数
X = Activation('softmax')(X)
model = Model(sentence_indices, X)
```

### 预训练

这里对输入X进行句子指数转换，对y进行了one-hot。具体解释也在代码中进行了注释。句子指数简单来说，就是将所有的输入句子，将句子拆分，根据每个单词在给定的词向量数据集（`glove.6B.100d.txt`）下的位置，得到输入矩阵。

### 编译模型

关于模型编译的参数已经在代码注释中做了介绍，选择何种参数不做具体讲解。