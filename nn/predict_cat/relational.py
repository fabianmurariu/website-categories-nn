# this is based on https://arxiv.org/pdf/1706.01427.pdf
from __future__ import print_function
from functools import partial
import itertools
from keras.preprocessing import image
import numpy as np
from joblib import Parallel, delayed, Memory
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPool2D, Embedding, ZeroPadding2D
from keras.layers import Dense, Input, Flatten, Dropout, BatchNormalization, Reshape
from keras.optimizers import Adam, RMSprop
from keras.layers.core import Lambda
from keras.layers.recurrent import LSTM
from keras.models import Model
import keras.backend as K
import keras
from keras.preprocessing import sequence
from keras.layers import concatenate, add

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

image_size = (128, 128)
image_size_ch = (128, 128, 3)
BATCH_SIZE = 64


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
    img = image.load_img(img_path, target_size=image_size)

    img_array = image.img_to_array(img)
    return img_array


load_image_mem = mem.cache(load_image)


def ohc_word_index_answers(json_path):
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(sparse=False)
    answers = list(set((js['answer'] for js in load_clever_questions(json_path, False))))
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
                                batch_size=BATCH_SIZE, infinite=False):
    import re
    data1 = train.grouper(load_clever_questions(json_path, infinite), batch_size)
    data2 = train.grouper(load_clever_questions(json_path, infinite), batch_size)
    regx1 = re.compile("[;? ']+")

    def encode_question(js_qs):
        tokens = regx1.split(js_qs)
        return [word_index_qs.get(word, 0) for word in tokens]

    def process_chunk_x(chunk):
        encoded_questions = [encode_question(js['question']) for js in chunk]
        x1 = sequence.pad_sequences(encoded_questions, maxlen=sentence_length)
        return x1

    def process_chunk_y(chunk):
        return enc_ans.transform([[word_index_ans.get(js['answer'])] for js in chunk])

    x_input = (process_chunk_x(chunk) for chunk in data1)
    y_output = (process_chunk_y(chunk) for chunk in data2)
    return x_input, y_output


def load_x_y_questions(json_path, sentence_length=45, batch_size=BATCH_SIZE, infinite=False):
    word_index_ans, enc = ohc_word_index_answers(json_path)
    word_index_qs = word_index_questions(json_path)
    x_input, y_output = prepare_train_x_y_questions(json_path, enc, word_index_qs, word_index_ans, sentence_length,
                                                    batch_size,
                                                    infinite)
    return word_index_ans, word_index_qs, enc, x_input, y_output


def prepare_train_x_y_images(json_path, imgs_root, load_img_fn, enc_ans, word_index_ans={}, batch_size=BATCH_SIZE,
                             infinite=False):
    from os.path import join
    data1 = train.grouper(load_clever_questions(json_path, infinite), batch_size)

    def process_chunk(chunk):
        x1 = np.array([load_img_fn(join(imgs_root, js['image_filename'])) for js in chunk])
        return x1

    return (process_chunk(chunk) for chunk in data1)


def load_x_y_images(json_path, imgs_root, word_index_ans, enc, load_img_fn=load_image_mem, batch_size=BATCH_SIZE,
                    infinite=False):
    images_gen = prepare_train_x_y_images(json_path, imgs_root, load_img_fn, enc, word_index_ans, batch_size,
                                          infinite)
    return images_gen


def relational_model(sentence_length=45, output_length=28, opt=Adam(), image_size=image_size_ch,
                     relational_mlp_size=128):
    def PairMLP(source_layer, obj_i, obj_j, qs_lstm_output):
        def concat_objects(t):
            return K.concatenate([t[:, obj_i, :], t[:, obj_j, :]], axis=1)

        k = Lambda(concat_objects)(source_layer)
        k = concatenate([k, qs_lstm_output], axis=-1)
        k = Dense(relational_mlp_size, activation='relu')(k)
        k = Dense(relational_mlp_size, activation='relu')(k)
        k = Dense(relational_mlp_size, activation='relu')(k)
        k = Dense(relational_mlp_size, activation='relu')(k)
        return k

    def Conv2DRelational(input_shape, img_conv_layer, qs_lstm_output):
        _, d, z = input_shape
        a = add([PairMLP(img_conv_layer, i, j, qs_lstm_output) for i in range(d) for j in range(d) if i != j])
        a = Dense(relational_mlp_size, activation='relu')(a)
        a = Dense(relational_mlp_size, activation='relu')(a)
        a = Dropout(0.5)(a)
        a = Dense(29, activation='relu')(a)
        return a

    # then questions
    q_input = Input(shape=(sentence_length,), name='questions_input')
    x2 = Embedding(85, 32)(q_input)
    x2 = LSTM(128)(x2)

    # images
    img_input = Input(shape=image_size, name='img_input')
    x1 = ZeroPadding2D()(img_input)
    x1 = Conv2D(24, (3, 3), strides=(2, 2), activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPool2D()(x1)
    x1 = Conv2D(24, (3, 3), strides=(2, 2), activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = ZeroPadding2D()(x1)
    x1 = Conv2D(24, (3, 3), strides=(2, 2), activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = ZeroPadding2D()(x1)
    x1 = Conv2D(24, (3, 3), strides=(2, 2), activation='relu')(x1)
    img_last_conv = Reshape(input_shape=(4, 4, 24), target_shape=(16, 24))
    x1 = img_last_conv(x1)
    print(img_last_conv.output_shape)
    # relational
    x1 = Conv2DRelational(img_last_conv.output_shape, x1, x2)
    # final output
    output = Dense(output_length, activation='softmax', name='main_output')(x1)
    model = Model(inputs=[img_input, q_input], output=output)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def combine_generators(questions_gen, imgs_gen, answers_gen):
    return (({'questions_input': qs, 'img_input': img}, {'main_output': out1}) for qs, img, out1 in
            itertools.izip(questions_gen, imgs_gen, answers_gen))


def generate_data(json_path, image_root, batch_size=BATCH_SIZE, infinite=False):
    word_index_ans, word_index_qs, enc, questions_gen, answers_gen = load_x_y_questions(json_path,
                                                                                        infinite=infinite,
                                                                                        batch_size=batch_size)
    imgs_gen = load_x_y_images(json_path, image_root, word_index_ans, enc, infinite=infinite, batch_size=batch_size)
    return combine_generators(questions_gen, imgs_gen, answers_gen)


def generate_data2(json_path, image_root, load_img_fn=load_image_mem, batch_size=BATCH_SIZE, infinite=False):
    import re
    from os.path import join

    word_index_ans, enc = ohc_word_index_answers(json_path)
    word_index_qs = word_index_questions(json_path)
    regx1 = re.compile("[;? ']+")

    def encode_question(js_qs):
        tokens = regx1.split(js_qs)
        return [word_index_qs.get(word, 0) for word in tokens]

    def process_chunk_question(chunk):
        encoded_questions = [encode_question(js['question']) for js in chunk]
        x1 = sequence.pad_sequences(encoded_questions, maxlen=sentence_length)
        return x1

    def process_chunk_answer(chunk):
        return enc.transform([[word_index_ans.get(js['answer'])] for js in chunk])

    def process_chunk_image(chunk):
        x1 = np.array([load_img_fn(join(image_root, js['image_filename'])) for js in chunk])
        return x1

    def process_chunk(chunk):
        qs = process_chunk_question(chunk)
        answer = process_chunk_answer(chunk)
        img = process_chunk_image(chunk)
        return {'questions_input': qs, 'img_input': img}, {'main_output': answer}

    chunks = train.grouper(load_clever_questions(json_path, infinite), batch_size)
    return (process_chunk(chunk) for chunk in chunks)


sentence_length = 45
output_length = 28

opt = Adam(lr=0.00025)
model = relational_model(sentence_length, output_length, opt)
model.summary()

val_gen = generate_data2(valA_questions_path, valA_images_path, infinite=True)
train_gen = generate_data2(trainA_questions_path, trainA_images_path, infinite=True)

tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
model.fit_generator(generator=train_gen,
                    validation_data=val_gen,
                    validation_steps=15,
                    callbacks=[tbCallBack],
                    epochs=10,
                    steps_per_epoch=100)
