from __future__ import print_function

import itertools
import json
from glob import glob

import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

import numpy as np
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import Dense, Input, Flatten, Dropout, BatchNormalization
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing import sequence
from os import path
import gzip


def grouper(iterable, chunk_size):
    a = iter(iterable)
    while True:
        chunk = list(itertools.islice(a, chunk_size))
        if len(chunk) > 0:
            yield chunk
        else:
            break


def read_file_lines(file_path, fn):
    fh = gzip.open(file_path, 'r') if str(file_path).endswith(".gz") else open(file_path, 'r', encoding='UTF-8')
    while True:
        line = fh.readline()
        if not line:
            break
        yield fn(line)


def read_lines(fn, *file_paths):
    return itertools.chain(*map(lambda fp: read_file_lines(fp, fn), file_paths))


def flat_map(tss):
    for ts in tss:
        for t in ts:
            yield t


def read_lines_paths(fn, file_paths, infinite=False):
    if infinite:
        return flat_map((read_file_lines(fp, fn) for fp in itertools.cycle(file_paths)))
    else:
        return itertools.chain(*(read_file_lines(fp, fn) for fp in file_paths))


def json_lines(path):
    return read_lines(json.loads, *glob(path))


def text_lines(path):
    return read_lines(lambda line: line.strip(), *glob(path))


def load_class_weights(path, labels):
    lines = list(json_lines(path))
    labels_idx = {label: idx for label, idx in zip(labels, range(len(labels)))}
    weights = lines[0]['weights']
    return {labels_idx[label]: weight for label, weight in weights.items() if label in labels_idx}


def data_point(line):
    point = json.loads(line if isinstance(line, str) else line.decode('utf-8'))
    x1 = np.array(point['paddedWords'])
    y1 = np.array(point['category'])
    return x1, y1


def create_batch(b, max_seq_length=None):
    x = sequence.pad_sequences(np.array([x for x, _ in b]), maxlen=max_seq_length) if max_seq_length else np.array(
        [x for x, _ in b])
    y = np.array([y for _, y in b])
    return x, y


def get_data(features_path, batch_size, max_seq_length):
    train_path = features_path + "/train"
    valid_path = features_path + "/valid"
    test_path = features_path + "/test"
    # valid
    ds1 = read_lines_paths(data_point, glob(valid_path + '/part*'), True)
    valid_ds = (create_batch(b, max_seq_length) for b in grouper(ds1, batch_size))

    # train
    ds2 = read_lines_paths(data_point, glob(train_path + '/part*'), True)
    train_ds = (create_batch(b, max_seq_length) for b in grouper(ds2, batch_size))

    ds3 = read_lines_paths(data_point, glob(test_path + '/part*'), True)
    test_ds = (create_batch(b, max_seq_length) for b in grouper(ds3, batch_size))

    return train_ds, valid_ds, test_ds


def build_model(embeddings_path, labels, max_nb_words, embedding_dim=50, max_seq_length=1000):
    embedding_matrix = np.array([item['vector'] for item in json_lines(embeddings_path + '/*')])
    num_words = min(max_nb_words, len(embedding_matrix))
    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=max_seq_length,
                                trainable=True)

    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(max_seq_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 3, activation='relu')(embedded_sequences)
    x = BatchNormalization()(x)
    x = MaxPooling1D(3)(x)
    x = BatchNormalization()(x)
    x = Conv1D(128, 3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(3)(x)
    x = BatchNormalization()(x)
    # x = Conv1D(128, 5, activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = MaxPooling1D(35)(x)
    # x = BatchNormalization()(x)
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
    return model


def fit_model(model, valid_ds, train_ds, class_weights, epochs):
    steps_per_epoch = 5000
    model.fit_generator(train_ds, validation_data=valid_ds, steps_per_epoch=steps_per_epoch,
                        validation_steps=steps_per_epoch / 10,
                        epochs=epochs, class_weight=class_weights)


def save_model(model, path):
    model.save(path, overwrite=True)


if __name__ == "__main__":
    import sys, getopt
    from os import path

    print("ARGS", sys.argv)
    try:
        opts, args = getopt.getopt(sys.argv[1:], "f", ["features_path=", "epochs=", "model_out="])
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)

    opts_dict = {k: v for k, v in opts}

    pre_nn_output = path.abspath(path.expanduser(opts_dict['--features_path']))
    model_path = path.abspath(path.expanduser(opts_dict['--model_out']))
    epochs = int(opts_dict.get('--epochs', 1000))

    print("Loading features from %s" % pre_nn_output)
    print("Training for %s epochs" % epochs)
    print("Saving the model at %s" % model_path)

    MAX_SEQUENCE_LENGTH = 128
    MAX_NB_WORDS = 50000
    EMBEDDING_DIM = 50
    batch_size = 64
    # first, build index mapping words in the embeddings set
    # to their embedding vector
    embeddings_path = pre_nn_output + '/embeddings'
    labels_path = pre_nn_output + '/labels'
    features_path = pre_nn_output + '/features'
    class_weights_path = pre_nn_output + '/class-weights'
    # do stuff
    labels = list(text_lines(labels_path + '/*'))
    print("labels count %s" % len(labels))
    class_weights = load_class_weights(class_weights_path + '/*', labels)
    print("loading data")
    train_ds, valid_ds, test_ds = get_data(features_path, batch_size, MAX_SEQUENCE_LENGTH)
    print("building model")
    model = load_model(model_path) if path.exists(model_path) else build_model(embeddings_path, labels, MAX_NB_WORDS,
                                                                               EMBEDDING_DIM, MAX_SEQUENCE_LENGTH)
    print("training...")
    fit_model(model, valid_ds, train_ds, class_weights, epochs)
    print("done training saving model")
    save_model(model, model_path)
