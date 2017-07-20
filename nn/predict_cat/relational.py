# this is based on https://arxiv.org/pdf/1706.01427.pdf
from __future__ import print_function
from functools import partial
# import itertools
# import json
from glob import glob
from keras.preprocessing import image
import numpy as np
from joblib import Parallel, delayed, Memory
from keras.layers import Conv1D, MaxPooling1D, Embedding
# from keras.layers import Dense, Input, Flatten, Dropout, BatchNormalization
from keras.layers.core import Lambda
# from keras.layers.recurrent import LSTM
from keras.models import Model
# from keras.models import load_model
# from keras.optimizers import Adam
# from keras.preprocessing import sequence
# from os import path

mem = Memory(cachedir='/tmp/relational-images')

from nn.predict_cat import train
import json
# load CLEVER images and questions
from os.path import expanduser

home = expanduser("~")
CLEVER_PATH = home + '/Downloads/CLEVR_CoGenT_v1.0'
valA_questions_path = CLEVER_PATH + "/questions/CLEVR_valA_questions.json.jl.gz"
valB_questions_path = CLEVER_PATH + "/questions/CLEVR_valB_questions.json.jl.gz"
trainA_questions_path = CLEVER_PATH + "/questions/CLEVR_trainA_questions.json.jl.gz"
testB_questions_path = CLEVER_PATH + "/questions/CLEVR_testB_questions.json.jl.gz"
testA_questions_path = CLEVER_PATH + "/questions/CLEVR_testA_questions.json.jl.gz"

valA_images_path = CLEVER_PATH + "/images/valA"
valB_images_path = CLEVER_PATH + "/images/valB"
trainA_images_path = CLEVER_PATH + "/images/trainA"
testB_images_path = CLEVER_PATH + "/images/testB"
testA_images_path = CLEVER_PATH + "/images/testA"


# {"question_index": 0, "question_family_index": 46, "image_index": 0,
#  "question": "Are there any gray things made of the same material as the big cyan cylinder?", "answer": "no",
#  "image_filename": "CLEVR_trainA_000000.png", "split": "trainA",
#  "program": [{"value_inputs": [], "inputs": [], "function": "scene"},
#              {"value_inputs": ["large"], "inputs": [0], "function": "filter_size"},
#              {"value_inputs": ["cyan"], "inputs": [1], "function": "filter_color"},
#              {"value_inputs": ["cylinder"], "inputs": [2], "function": "filter_shape"},
#              {"value_inputs": [], "inputs": [3], "function": "unique"},
#              {"value_inputs": [], "inputs": [4], "function": "same_material"},
#              {"value_inputs": ["gray"], "inputs": [5], "function": "filter_color"},
#              {"value_inputs": [], "inputs": [6], "function": "exist"}]}

def read_json_drop_keys(drop_keys, line):
    if drop_keys is None:
        drop_keys = ["program"]
    d = json.loads(line)
    for k in drop_keys:
        if k in d:
            del d[k]
    return d


def load_clever_questions(path_to_file):
    questions = train.read_lines_paths(partial(read_json_drop_keys, ["program"]), [path_to_file], True)
    return questions


def load_image(img_path):
    img = image.load_img(img_path, target_size=(330, 220))
    img_array = image.img_to_array(img)
    return np.expand_dims(img_array, axis=0)


# def load_clever_images(path_to_images):
#     image_paths = glob(path_to_images)
#     imgs = Parallel(n_jobs=8)(delayed(load_image)(i) for i in image_paths)
#     img_dict = {i: imgs[i] for i in range(0, len(imgs))}
#     return img_dict



train_json = load_clever_questions(trainA_questions_path)
validA_json = load_clever_questions(valA_questions_path)
validB_json = load_clever_questions(valB_questions_path)
# takes too much time, need alternative
# train_images = load_clever_images(trainA_images_path)
# validA_images = load_clever_images(valA_images_path)
# validB_images = load_clever_images(valB_images_path)
num_words = 45
embedding_dim = 50
embedding_layer = Embedding(num_words,
                            embedding_dim)
