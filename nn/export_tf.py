from keras.models import load_model
from keras import backend as K
import tensorflow as tf
from os.path import expanduser
from keras.preprocessing.sequence import pad_sequences

# this is key otherwise the model will not run once in the JVM
K.set_learning_phase(0)
# init tensorflow
server = tf.train.Server.create_local_server()
sess = tf.Session(server.target)
K.set_session(sess)
home = expanduser("~")
root = home + '/ml-work/dmoz'
# load the keras model
model = load_model('%s/tf-model' % root)
# export path and version
export_path = '%s/model-tf-serve' % root
export_version = 1
model.summary()
# I don't care about the results here, I just need to init all the internal tensorflow variables and
# make sure the model can predict after it has been loaded
x = pad_sequences([[127, 98, 112]], 1000, padding='post')
model.predict(x)

builder = tf.saved_model.builder.SavedModelBuilder(root + '/model-tf-serve4')
builder.add_meta_graph_and_variables(sess,
                                     [tf.saved_model.tag_constants.SERVING])
builder.save(True)
