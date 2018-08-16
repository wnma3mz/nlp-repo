# coding: utf-8
import pandas as pd
import numpy as np
from keras.layers import Input, Dense, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding

"""
note:
embedding（嵌入）：通过一个数值向量代表每个单词在字典中的距离（L2距离或其他距离），这样形成的语义关系几何关系空间向量为嵌入空间。
"食堂"与"午饭"比较接近，即距离更近，"学校"与"蚂蚁"语义不同，即距离更远。
"工作"+"x"="办公室"理解为工作发生在哪？办公室。
使用技术`word2vec`或者矩阵分解
"""


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
