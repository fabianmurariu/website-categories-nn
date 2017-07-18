from __future__ import print_function

# import itertools
# import json
# from glob import glob
#
# import numpy as np
# from keras.layers import Conv1D, MaxPooling1D, Embedding
# from keras.layers import Dense, Input, Flatten, Dropout, BatchNormalization
# from keras.layers.recurrent import LSTM
# from keras.models import Model
# from keras.models import load_model
# from keras.optimizers import Adam
# from keras.preprocessing import sequence
# from os import path

import ijson

# load CLEVER images and questions
from os.path import expanduser

home = expanduser("~")
CLEVER_PATH = home + '/Downloads/CLEVR_CoGenT_v1.0'
sample = CLEVER_PATH + "/questions/CLEVR_valA_questions.json.jl"
