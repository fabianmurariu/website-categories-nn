from keras.models import load_model
from src.nn.predict_cat.train2 import get_data, read_lines_paths
import numpy as np
from glob import glob
from itertools import islice
import json

model = load_model('cat2v7')

model.summary()

features_path = '/home/murariuf/dmoz/websites-features-v7/features'
batch_size = 256
MAX_SEQUENCE_LENGTH = 128
batches = 2


def get_text_data(path):
    jsons = read_lines_paths(lambda line: (json.loads(line if isinstance(line, str) else line.decode('utf-8'))),
                             glob(path + '/part*'), True)
    return ((j['uri'], j['categoryName'], j['text'][:75], j['category']['indices'][0]) for j in jsons)


_, _, test_ds = get_data(features_path, batch_size, MAX_SEQUENCE_LENGTH)
text_ds = get_text_data(features_path + '/test')

expected = list(islice(text_ds, batches * batch_size))

predictions = model.predict_generator(test_ds, batches)

actual = np.argmax(predictions, axis=1)

compare = list(zip(expected, actual))

for o in compare[:512]:
    print(o)

