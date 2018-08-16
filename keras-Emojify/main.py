# coding: utf-8
import pandas as pd
import numpy as np
from keras.layers import Input, Dense, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.models import Model, load_model
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
import csv


def read_data():
    # 一次性读取数据
    df = pd.read_csv('data.csv', header=None)[:180]
    # 随机打乱数据。df.sample(frac=1)改变索引顺序
    df = df.sample(frac=1).reset_index(drop=True)
    train, test = df[:120], df[120:]
    return np.array(train[:][0]), np.array(train[:][1]), np.array(test[:][0]), np.array(test[:][1])


def read_glove_vecs():
    embeddings_index = {}
    words = set()
    with open(r'G:\\glove.6B\\glove.6B.100d.txt', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            words.add(word)
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    # key为word，value为数字
    words_to_index = {key: value for key,
                      value in zip(range(1, len(words) + 1), words)}
    index_to_words = {v: k for k, v in words_to_index.items()}
    return words_to_index, index_to_words, embeddings_index


def pretrain_embedding_layer(word_to_vec_map, word_to_index):
    # 预训练embedding层
    vocab_len = len(word_to_index) + 1
    emb_dim = word_to_vec_map['cucumber'].shape[0]
    emb_matrix = np.zeros((vocab_len, emb_dim))
    for word, i in word_to_index.items():
        embedding_vector = word_to_vec_map.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            emb_matrix[i] = embedding_vector
    # 构成嵌入层，输入数据最大下标+1（单词长度），全连接嵌入的维度（），防止在训练过程中权重更新而重新训练
    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)
    # 创建嵌入层的权重，并设置权重大小
    embedding_layer.build((None, ))
    embedding_layer.set_weights([emb_matrix])
    return embedding_layer


def train_model(input_shape, word_to_vec_map, word_to_index):
    # 定义输入层
    sentence_indices = Input(shape=input_shape, dtype='int32')
    # 定义嵌入层
    embedding_layer = pretrain_embedding_layer(word_to_vec_map, word_to_index)
    #
    embeddings = embedding_layer(sentence_indices)
    # 根据神经网络图，进行训练
    X = LSTM(128, return_sequences=True)(embeddings)
    X = Dropout(0.5)(X)
    X = LSTM(128, return_sequences=False)(X)
    X = Dropout(0.5)(X)
    X = Dense(5, activation='softmax')(X)
    X = Activation('softmax')(X)

    model = Model(sentence_indices, X)
    return model


def sentences_to_indices(X, word_to_index, max_len):
    m = X.shape[0]
    X_indices = np.zeros((m, max_len))
    for i in range(m):
        sentence_words = X[i].lower().split()
        for j, w in enumerate(sentence_words):
            X_indices[i, j] = word_to_index[w]
    return X_indices


def convert_to_one_hot(Y, C):
    return np.eye(C)[Y.reshape(-1)]

if __name__ == '__main__':
    train_X, train_y, test_X, test_y = read_data()

    maxLen = len(max(train_X, key=len).split())
    index_to_word, word_to_index, word_to_vec_map = read_glove_vecs()
    model = train_model((maxLen,), word_to_vec_map, word_to_index)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    X_train_indices = sentences_to_indices(train_X, word_to_index, maxLen)
    Y_train_oh = convert_to_one_hot(train_y, C=5)

    model.fit(X_train_indices, Y_train_oh,
    epochs=70, batch_size=32, shuffle=True)

    # model = load_model('Emojify.h5')
    X_test_indices = sentences_to_indices(test_X, word_to_index, maxLen)
    pred = model.predict(X_test_indices)

    print("预测的准确率:", np.count_nonzero(np.argmax(pred, axis=1) == test_y) / 60)

    model.save('Emojify.h5')
