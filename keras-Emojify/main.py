# coding: utf-8
import pandas as pd
import numpy as np
from keras.layers import Input, Dense, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.models import Model, load_model
# 日志信息输出等级
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def read_data():
    # 一次性读取数据
    df = pd.read_csv('data.csv', header=None)[:180]
    # 随机打乱数据。df.sample(frac=1)不改变索引顺序
    df = df.sample(frac=1).reset_index(drop=True)
    # 划分训练集和测试集
    train, test = df[:120], df[120:]
    # 返回训练集的X，y和测试集的X，y
    return np.array(train[:][0]), np.array(train[:][1]), np.array(test[:][0]), np.array(test[:][1])


def read_glove_vecs():
    # 嵌入层索引，key为word，value为对应向量索引
    embeddings_index = {}
    words = set()
    # 读取文本数据，Global Vectors for Word Representation（全球词向量表示）
    with open(r'glove.6B.100d.txt', encoding='utf-8') as f:
        for line in f:
            # 每行的第一个元素是记录的word，后面的是数值数组
            values = line.strip().split()
            word = values[0]
            words.add(word)
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    # key为word，value为序号
    words_to_index = {key: value for key,
                      value in zip(range(1, len(words) + 1), words)}
    # 跟words_to_index相反，key为序号，value为word
    index_to_words = {v: k for k, v in words_to_index.items()}
    return words_to_index, index_to_words, embeddings_index


def pretrain_embedding_layer(word_to_vec_map, word_to_index):
    # 预训练embedding层
    # 单词个数+1
    vocab_len = len(word_to_index) + 1
    # 嵌入层维度，词向量'cucumber'的长度
    emb_dim = word_to_vec_map['cucumber'].shape[0]
    # 构造全零矩阵，嵌入层矩阵
    emb_matrix = np.zeros((vocab_len, emb_dim))
    # 遍历每个词及对应的词向量
    for word, i in word_to_index.items():
        # 按照顺序，将嵌入层矩阵的数值替换为词向量
        embedding_vector = word_to_vec_map.get(word)
        # 如果对应的词向量不存在，则该行为0
        if embedding_vector is not None:
            emb_matrix[i] = embedding_vector
    # 构成嵌入层，输入数据最大下标+1（单词长度），全连接嵌入的维度（词向量长度），防止在训练过程中权重更新而重新训练
    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)
    # 创建嵌入层的权重，并设置权重大小
    embedding_layer.build((None, ))
    embedding_layer.set_weights([emb_matrix])
    return embedding_layer


def train_model(input_shape, word_to_vec_map, word_to_index):
    # 定义输入层，输入维度为训练集中最长句子的单词个数
    sentence_indices = Input(shape=input_shape, dtype='int32')
    # 预定义嵌入层
    embedding_layer = pretrain_embedding_layer(word_to_vec_map, word_to_index)
    # 定义嵌入层
    embeddings = embedding_layer(sentence_indices)
    # 根据神经网络图，进行构造。
    # LSTM层，输出维度为128。在输出序列中，返回返回全部time step 的 hidden state值。hidden state是根据当前输入得到的输出。将前后LSTM层串在一起，根据输入层input_shape，输出input_shape个128维的词向量
    X = LSTM(128, return_sequences=True)(embeddings)
    # 防止过拟合，以50%概率断开输入神经元
    X = Dropout(0.5)(X)
    # LSTM层，输出维度为128。在输出序列中，返回返回单个time step 的 hidden state值。通常最后一个LSTM中的return_sequences=False,它的默认值为False。由上一层传递过来，由input_shape个128维的词向量，转换后还是input_shape个128维的词向量（输出维度没有变化），但是由于return_sequences=False，所以就只保留最后一个向量结果。
    X = LSTM(128, return_sequences=False)(X)
    X = Dropout(0.5)(X)
    # 全连接层，激活函数为softmax，输出维度为5（有5个标签）
    X = Dense(5, activation='softmax')(X)
    # 对上一个层的输出进行激活
    X = Activation('relu')(X)
    # 使用函数式模型，以sentence_indices为输入，X为输出
    model = Model(sentence_indices, X)
    return model


def sentences_to_indices(X, word_to_index, max_len):
    # 训练集的个数
    m = X.shape[0]
    # 定义训练集词向量矩阵
    X_indices = np.zeros((m, max_len))
    for i in range(m):
        # 对每句话小写，分割出每个单词
        sentence_words = X[i].lower().split()
        # 遍历每句话的每个单词，根据在单词在句子中的位置和在词向量数据集中的位置，得到矩阵
        for j, w in enumerate(sentence_words):
            X_indices[i, j] = word_to_index[w]
    # 返回矩阵
    return X_indices


def convert_to_one_hot(Y, C):
    # 返回标签的one-hot矩阵
    return np.eye(C)[Y.reshape(-1)]


if __name__ == '__main__':
    # 得到训练集和测试集
    train_X, train_y, test_X, test_y = read_data()
    # 在训练集中最长的句子的单词个数
    maxLen = len(max(train_X, key=len).split())
    # 读取词向量数据集
    index_to_word, word_to_index, word_to_vec_map = read_glove_vecs()
    # 训练模型
    model = train_model((maxLen,), word_to_vec_map, word_to_index)

    # 编译模型，定义损失函数，优化器，评价模型的性能指标
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # 将数据进行转换
    X_train_indices = sentences_to_indices(train_X, word_to_index, maxLen)
    # 将标签进行one-hot转换
    Y_train_oh = convert_to_one_hot(train_y, C=5)

    # 训练70轮，每轮32个数据，打乱数据
    model.fit(X_train_indices, Y_train_oh,
              epochs=70, batch_size=32, shuffle=True)

    # model = load_model('Emojify.h5')
    # 同样的对测试数据进行预处理
    X_test_indices = sentences_to_indices(test_X, word_to_index, maxLen)
    # 进行预测
    pred = model.predict(X_test_indices)
    # 根据预测结果与测试数据的标签进行对比，得到准确率。
    print("预测的准确率:", np.count_nonzero(np.argmax(pred, axis=1) == test_y) / 60)
    # 保存训练好的模型，可能是由于词向量数据太大的原因，我本地不能保存。。。
    # model.save('Emojify.h5')
