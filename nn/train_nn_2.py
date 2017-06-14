'''This script loads pre-trained word embeddings (GloVe embeddings)
into a frozen Keras Embedding layer, and uses it to
train a text classification model on the 20 Newsgroup dataset
(classication of newsgroup messages into 20 different categories).

GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)

20 Newsgroup data can be found at:
http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html
'''

from __future__ import print_function

import json
import glob
import itertools
import numpy as np
from random import randint
from keras.layers import Dense, Input, Flatten, Dropout, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.optimizers import Adam
from os.path import expanduser

home = expanduser("~")
BASE_DIR = home + '/ml-work'
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.2

# first, build index mapping words in the embeddings set
# to their embedding vector
pre_nn_output = home + '/ml-work/dmoz/websites-features'
embeddings_path = pre_nn_output + '/embeddings'
labels_path = pre_nn_output + '/labels'
features_path = pre_nn_output + '/features'
class_weights_path = pre_nn_output + '/class-weights'

print('Indexing word vectors.')

embeddings_files = glob.glob(embeddings_path + '/*')
label_files = glob.glob(labels_path + '/*')
features_files = glob.glob(features_path + '/*')
class_weights_files = glob.glob(features_path + '/*')


def read_file_lines(file_path, fn):
    fh = open(file_path, 'r')
    while True:
        line = fh.readline()
        if not line:
            break
        yield fn(line)


embedding_matrix = np.array(
    list((item['vector'] for item in
          itertools.chain(*[read_file_lines(f, lambda line: json.loads(line)) for f in embeddings_files]))))

labels = list(itertools.chain(*[read_file_lines(f, lambda line: line.strip()) for f in label_files]))

num_words = min(MAX_NB_WORDS, len(embedding_matrix))

class_weights = json.loads(list(itertools.chain(*[read_file_lines(f, lambda line: line.strip()) for f in class_weights_files]))[0])

print('Found %s words' % num_words)

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)


def data_point(line):
    point = json.loads(line)
    x1 = np.array(point['paddedWords'])
    y1 = np.array(point['category'])
    return x1, y1


# python iterators are.. terrible
# This looks like this because I want to stream

ds1 = itertools.chain(*[read_file_lines(f, data_point) for f in features_files])
ds2 = itertools.chain(*[read_file_lines(f, data_point) for f in features_files])

rand_block = [False if randint(1, 10) <= 2 else True for x in range(0, 1000)]
# endless iterators of the same block
tv1 = itertools.cycle(rand_block)
tv2 = itertools.cycle(rand_block)

train = (item for item, mark in itertools.izip(ds1, tv1) if mark)
valid = (item for item, mark in itertools.izip(ds2, tv1) if not mark)

print('Training model.')


def to_numpy_fixed(ts):
    l = list(ts)
    x_list = list((x for x, y in l))
    y_list = list((y for x, y in l))
    return np.array(x_list), np.array(y_list)

x_train, y_train = to_numpy_fixed(train)
x_valid, y_valid = to_numpy_fixed(valid)

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = BatchNormalization()(x)
x = MaxPooling1D(5)(x)
x = BatchNormalization()(x)
x = Conv1D(128, 5, activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(5)(x)
x = BatchNormalization()(x)
x = Conv1D(128, 5, activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(35)(x)
x = BatchNormalization()(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(.1)(x)
x = BatchNormalization()(x)
x = Dense(len(labels), activation='softmax')(x)

opt = Adam()
model = Model(sequence_input, x)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=200,
          validation_data=(x_valid, y_valid))

from tensorflow.python.saved_model import builder as saved_model_builder

export_path = home + '/dmoz/model-tf-serve'
builder = saved_model_builder.SavedModelBuilder(export_path)

# model.fit_generator(train,
#                     steps_per_epoch=3000,
#                     validation_data=valid,
#                     validation_steps=300,
#                     epochs=100)
