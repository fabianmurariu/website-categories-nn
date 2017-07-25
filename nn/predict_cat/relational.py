# this is based on https://arxiv.org/pdf/1706.01427.pdf
from __future__ import print_function
from functools import partial
# import itertools
# import json
from glob import glob
from keras.preprocessing import image
import numpy as np
import os
from joblib import Parallel, delayed, Memory
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPool2D, Embedding
from keras.layers import Dense, Input, Flatten, Dropout, BatchNormalization
from keras.layers.core import Lambda
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.preprocessing import sequence

# from keras.models import load_model
# from keras.optimizers import Adam
# from keras.preprocessing import sequence
# from os import path

mem = Memory(cachedir='/tmp/relational-images', verbose=0)

from nn.predict_cat import train
import json
from os.path import expanduser

home = expanduser("~")
CLEVER_PATH = home + '/Downloads/CLEVR_CoGenT_v1.0'
valA_questions_path = CLEVER_PATH + "/questions/CLEVR_valA_questions.json.jl.gz"
valB_questions_path = CLEVER_PATH + "/questions/CLEVR_valB_questions.json.jl.gz"
trainA_questions_path = CLEVER_PATH + "/questions/CLEVR_trainA_questions.json.jl.gz"
testB_questions_path = CLEVER_PATH + "/questions/CLEVR_testB_questions.json.jl.gz"
testA_questions_path = CLEVER_PATH + "/questions/CLEVR_testA_questions.json.jl.gz"
GLOVE_PATH = '/Users/murariuf/ml-work/glove.6B'
valA_images_path = CLEVER_PATH + "/images/valA"
valB_images_path = CLEVER_PATH + "/images/valB"
trainA_images_path = CLEVER_PATH + "/images/trainA"
testB_images_path = CLEVER_PATH + "/images/testB"
testA_images_path = CLEVER_PATH + "/images/testA"


def read_json_drop_keys(drop_keys, line):
    if drop_keys is None:
        drop_keys = ["program"]
    d = json.loads(line)
    for k in drop_keys:
        if k in d:
            del d[k]
    return d


def load_clever_questions(path_to_file, infinite=True):
    questions = train.read_lines_paths(partial(read_json_drop_keys, ["program"]), [path_to_file], infinite)
    return questions


def load_image(img_path):
    img = image.load_img(img_path, target_size=(330, 220))
    img_array = image.img_to_array(img)
    return img_array


load_image_mem = mem.cache(load_image)


# TODO: figure out if we still need this
# def read_embeding_matrix(glove_dir=GLOVE_PATH, size=50):
#     import gzip
#     embeddings_index = {}
#     keys = []
#     f = gzip.open(os.path.join(glove_dir, 'glove.6B.%sd.txt.gz' % size))
#     for line in f:
#         values = line.split()
#         word = values[0]
#         coefs = np.asarray(values[1:], dtype='float32')
#         embeddings_index[word] = coefs
#         keys.append(word)
#     f.close()
#     word_index = {keys[i]: i for i in len(keys)}
#     return embeddings_index, word_index


# TODO: figure out if we still need this
# def load_clever_images(path_to_images):
#     image_paths = glob(path_to_images)
#     imgs = Parallel(n_jobs=8)(delayed(load_image)(i) for i in image_paths)
#     img_dict = {i: imgs[i] for i in range(0, len(imgs))}
#     return img_dict


# train_json = load_clever_questions(trainA_questions_path)
# validA_json = load_clever_questions(valA_questions_path)
# validB_json = load_clever_questions(valB_questions_path)


def ohc_word_index_answers(json_path):
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(sparse=False)
    answers = list(set((js['answer'] for js in load_clever_questions(json_path, False))))
    print("answers", answers)
    word_index = {answers[k]: k for k in range(len(answers))}
    enc.fit([[i] for i in range(len(answers))])
    return word_index, enc


def word_index_questions(json_path):
    import re
    word_index = {}
    i = 1
    regx = re.compile("[;? ']+")
    for js in load_clever_questions(json_path, False):
        q = js['question']
        tokens = [s.lower() for s in regx.split(q)]
        for word in tokens:
            if word and word not in word_index:
                word_index[word] = i
                i = i + 1
    return word_index


def prepare_train_x_y_questions(json_path, enc_ans, word_index_qs={}, word_index_ans={}, sentence_length=45,
                                batch_size=16, infinite=False):
    import re
    data1 = train.grouper(load_clever_questions(json_path, infinite), batch_size)
    regx1 = re.compile("[;? ']+")

    def encode_question(js_qs):
        tokens = regx1.split(js_qs)
        return [word_index_qs.get(word, 0) for word in tokens]

    def process_chunk(chunk):
        y1 = enc_ans.transform([[word_index_ans.get(js['answer'])] for js in chunk])
        wtf = [encode_question(js['question']) for js in chunk]
        x1 = sequence.pad_sequences(wtf, maxlen=sentence_length)
        return x1, y1

    return (process_chunk(chunk) for chunk in data1)


def load_x_y_questions(json_path, sentence_length=45, batch_size=16, infinite=False):
    word_index_ans, enc = ohc_word_index_answers(json_path)
    word_index_qs = word_index_questions(json_path)
    questions_gen = prepare_train_x_y_questions(json_path, enc, word_index_qs, word_index_ans, sentence_length,
                                                batch_size,
                                                infinite)
    return word_index_ans, word_index_qs, enc, questions_gen


def prepare_train_x_y_images(json_path, imgs_root, load_img_fn, enc_ans, word_index_ans={}, batch_size=16,
                             infinite=False):
    from os.path import join
    data1 = train.grouper(load_clever_questions(json_path, infinite), batch_size)

    def process_chunk(chunk):
        y1 = enc_ans.transform([[word_index_ans.get(js['answer'])] for js in chunk])
        x1 = np.array([load_img_fn(join(imgs_root, js['image_filename'])) for js in chunk])
        return x1, y1

    return (process_chunk(chunk) for chunk in data1)


def load_x_y_images(json_path, imgs_root, word_index_ans, enc, load_img_fn=load_image_mem, batch_size=16,
                    infinite=False):
    print(type(enc))
    images_gen = prepare_train_x_y_images(json_path, imgs_root, load_img_fn, enc, word_index_ans, batch_size,
                                          infinite)
    return images_gen


def img_model(output_length):
    img_input = Input(shape=(330, 220, 3))
    x = Conv2D(24, (3, 3), strides=(2, 2), activation='relu')(img_input)
    x = BatchNormalization()(x)
    x = Conv2D(24, (3, 3), strides=(2, 2), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    output = Dense(output_length, activation='softmax')(x)
    model = Model(inputs=img_input, outputs=output)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


sentence_length = 45
output_length = 28

# model = img_model(output_length)
# model.summary()
#
# word_index_ans, word_index_qs, enc, questions_gen = load_x_y_questions(valA_questions_path, infinite=True)
# imgs_gen = load_x_y_images(valA_questions_path, valA_images_path, word_index_ans, enc, infinite=True)
# model.fit_generator(imgs_gen, 500, 12)
#
# q_input = Input(shape=(sentence_length,), name='questions_input')
# x = Embedding(85, 128)(q_input)
# x = LSTM(128)(x)
# q_output = Dense(28, activation='softmax')(x)
#
# model = Model(inputs=q_input, outputs=q_output)
#
# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.fit_generator(questions_gen, 10, 10)
