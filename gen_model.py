import numpy as np
import tensorflow as tf
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

EMBEDDING_DIM = 256

lstm_layers = 2
dropout_rate = 0.2
learning_rate = 0.001

def to_words(specifications):
  all_spec = list()
  for key in specifications.keys():
    [all_spec.append(d) for d in specifications[key]]
  return all_spec

def create_tokenizer(specifications):
  words = to_words(specifications)
  tokening = Tokenizer()
  tokening.fit_on_texts(words)
  return tokening


def max_length(specifications):
  words = to_words(specifications)
  return max(len(d.split()) for d in words)

def create_sequences(tokening, len_upper, spec_array, photo):
  vocab_size = len(tokening.word_index) + 1

  X1, X2, y = [], [], []
  for desc in spec_array:
    seq = tokening.texts_to_sequences([desc])[0]
    for i in range(1, len(seq)):
      in_seq, out_seq = seq[:i], seq[i]
      in_seq = pad_sequences([in_seq], maxlen=len_upper)[0]
      out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
      X1.append(photo)
      X2.append(in_seq)
      y.append(out_seq)
  return np.array(X1), np.array(X2), np.array(y)

def data_generator(specifications, pictures, tokening, len_upper, allLayer = 1):
  while 1:
    keys = list(specifications.keys())
    for i in range(0, len(keys), allLayer):
      picsX, orderY, z = list(), list(),list()
      for j in range(i, min(len(keys), i+allLayer)):
        picture_id = keys[j]
        picture = pictures[picture_id][0]
        spec_array = specifications[picture_id]
        in_img, in_seq, out_word = create_sequences(tokening, len_upper, spec_array, picture)
        for k in range(len(in_img)):
          picsX.append(in_img[k])
          orderY.append(in_seq[k])
          z.append(out_word[k])
      yield [[np.array(picsX), np.array(orderY)], np.array(z)]

def categorical_crossentropy_from_logits(left_1, predict_y):
  left_1 = left_1[:, :-1, :]  
  predict_y = predict_y[:, :-1, :]  
  loss = tf.nn.softmax_cross_entropy_with_logits(labels=left_1,
                                                 logits=predict_y)
  return loss

def categorical_accuracy_with_variable_timestep(left_1, predict_y):
  left_1 = left_1[:, :-1, :]  
  predict_y = predict_y[:, :-1, :]  

  shape = tf.shape(left_1)
  left_1 = tf.reshape(left_1, [-1, shape[-1]])
  predict_y = tf.reshape(predict_y, [-1, shape[-1]])

  is_zero_y_true = tf.equal(left_1, 0)
  is_zero_row_y_true = tf.reduce_all(is_zero_y_true, axis=-1)
  left_1 = tf.boolean_mask(left_1, ~is_zero_row_y_true)
  predict_y = tf.boolean_mask(predict_y, ~is_zero_row_y_true)

  accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(left_1, axis=1),
                                              tf.argmax(predict_y, axis=1)),
                                    dtype=tf.float32))
  return accuracy

def define_model(spec_size, len_up):
  x = Input(shape=(4096,))
  u1 = Dropout(0.5)(x)
  u2 = Dense(EMBEDDING_DIM, activation='relu')(u1)
  u3 = RepeatVector(len_up)(u2)

  y = Input(shape=(len_up,))
  emb2 = Embedding(spec_size, EMBEDDING_DIM, mask_zero=True)(y)

  merged = concatenate([u3, emb2])
  lm2 = LSTM(500, return_sequences=False)(merged)
  outputs = Dense(spec_size, activation='softmax')(lm2)

  model = Model(inputs=[x, y], outputs=outputs)
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  print(model.summary())
  plot_model(model, show_shapes=True, to_file='model.png')
  return model
