import sys, getopt
from os import path
from glob import glob
import numpy as np
import json
from os.path import join
from functools import partial

from sklearn import preprocessing as prep
from keras.preprocessing import sequence
from keras.layers import Embedding, Dense, Input, Flatten, Dropout, BatchNormalization
from keras.layers import Bidirectional, LSTM
from nn.predict_cat.train import read_lines_paths, grouper, text_lines, load_class_weights, json_lines
from keras.optimizers import Adam
from keras.models import Model


def category_encoders(features_path_root, missing_label='X'):
    labels = join(features_path_root, 'labels')
    labels3 = list(read_lines_paths(lambda line: line.split("/"), glob(labels + '/part*'), infinite=False))
    encoders = []
    for i in range(0, 3):
        label_values = [l[i] if len(l) > i else missing_label for l in labels3]
        l_enc = prep.LabelEncoder()
        l_enc.fit(label_values)
        ohe = prep.OneHotEncoder()
        ohe.fit(l_enc.transform(label_values).reshape(-1, 1))
        encoders += [(l_enc, ohe)]
    return encoders


def transform_label(labels, l_enc, ohe):
    # type: (list, prep.LabelEncoder, prep.OneHotEncoder) -> np.ndarray
    return ohe.transform(l_enc.transform(labels).reshape(-1, 1)).toarray()


def data_point(line):
    point = json.loads(line)
    x1 = np.array(point['paddedWords'])
    category_name = point['categoryName']
    categories = category_name.split("/")
    return x1, categories[0], categories[1]


def create_batch(encoders, b, max_seq_length=None):
    (l_enc1, ohe1), (l_enc2, ohe2), _, = encoders
    x = sequence.pad_sequences(np.array([x for x, _, _ in b]), maxlen=max_seq_length) if max_seq_length else np.array(
        [x for x, _, _ in b])
    y1 = transform_label([y for _, y, _ in b], l_enc1, ohe1)
    y2 = transform_label(np.array([y for _, _, y in b]), l_enc2, ohe2)
    return x, {'cat1_output': y1}


def batch_x(b, max_seq_length=None):
    return sequence.pad_sequences(np.array([x for x, _, _ in b]),
                                  maxlen=max_seq_length) if max_seq_length else np.array(
        [x for x, _, _ in b])


def batch_y1(l_enc1, ohe1, b, max_seq_length=None):
    return transform_label([y for _, y, _ in b], l_enc1, ohe1)


def batch_y2(l_enc2, ohe2, b, max_seq_length=None):
    return transform_label([y for _, _, y in b], l_enc2, ohe2)


def get_data(features_path_root, batch_size, max_seq_length):
    encoders = category_encoders(features_path_root)

    features_path = features_path_root + '/features'
    train_path = features_path + "/train"
    valid_path = features_path + "/valid"
    test_path = features_path + "/test"
    create_batch_enc = partial(create_batch, encoders)
    # valid
    ds1_1 = read_lines_paths(data_point, glob(valid_path + '/part*'), True)
    valid = (create_batch_enc(b, max_seq_length) for b in grouper(ds1_1, batch_size))

    # train
    ds2_1 = read_lines_paths(data_point, glob(train_path + '/part*'), True)
    train = (create_batch_enc(b, max_seq_length) for b in grouper(ds2_1, batch_size))

    # ds3 = read_lines_paths(data_point, glob(test_path + '/part*'), True)
    # test_ds = (create_batch_enc(b, max_seq_length) for b in grouper(ds3, batch_size))

    return train, valid


def embeddings(embeddings_path):
    return np.array([item['vector'] for item in json_lines(embeddings_path + '/*')])


def vocabularies(vocabulary_path):
    num_to_word = {int(item['id']): item['word'] for item in json_lines(vocabulary_path + '/*')}
    reverse = {v: k for k, v in num_to_word.items()}
    return num_to_word, reverse


def build_model(embeddings_path, embedding_dim=50, num_words=50000, max_seq_length=128):
    embedding_matrix = embeddings(embeddings_path)
    sentence_input = Input(shape=(128,), name='sentence_input')
    x = Embedding(num_words,
                  embedding_dim,
                  weights=[embedding_matrix],
                  input_length=max_seq_length,
                  trainable=True)(sentence_input)
    x = Bidirectional(LSTM(32, return_sequences=True))(x)
    x = Bidirectional(LSTM(32, return_sequences=True))(x)
    x = LSTM(32)(x)
    x = Dense(32, activation='sigmoid', name='cat1_output')(x)
    opt = Adam()
    model = Model(inputs=[sentence_input], outputs=[x])

    model.compile(loss={'cat1_output': 'categorical_crossentropy'},
                  optimizer=opt,
                  metrics=['acc'])
    model.summary()

    return model


def main(features_path_root, model_path, epochs):
    MAX_SEQUENCE_LENGTH = 128
    MAX_NB_WORDS = 50000
    EMBEDDING_DIM = 50
    batch_size = 16
    # first, build index mapping words in the embeddings set
    # to their embedding vector
    embeddings_path = join(features_path_root, 'embeddings')
    labels_path = join(features_path_root, 'labels')
    features_path = join(features_path_root, 'features')
    class_weights_path = join(features_path_root, 'class-weights')
    vocabulary_path = join(features_path_root, 'vocabulary')
    num_2_word, word_2_num = vocabularies(vocabulary_path)

    labels = list(text_lines(labels_path + '/*'))
    print("labels count %s" % len(labels))
    class_weights = load_class_weights(class_weights_path + '/*', labels)
    print("loading data")
    train, valid = get_data(
        features_path_root, batch_size, MAX_SEQUENCE_LENGTH)

    model = build_model(embeddings_path=embeddings_path,
                        embedding_dim=EMBEDDING_DIM,
                        num_words=MAX_NB_WORDS,
                        max_seq_length=MAX_SEQUENCE_LENGTH)

    model.fit_generator(train,
                        epochs=200,
                        shuffle=False,
                        steps_per_epoch=50)


if __name__ == "__main__":
    print("ARGS", sys.argv)
    try:
        opts, args = getopt.getopt(sys.argv[1:], "f", ["features_path=", "epochs=", "model_out="])
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)

    opts_dict = {k: v for k, v in opts}

    features_path = path.abspath(path.expanduser(opts_dict['--features_path']))
    model_path = path.abspath(path.expanduser(opts_dict['--model_out']))
    epochs = int(opts_dict.get('--epochs', 50))

    oh1, oh2, oh3 = category_encoders(features_path)
    (l_enc, ohe) = oh1
    main(features_path_root=features_path, model_path=model_path, epochs=epochs)
