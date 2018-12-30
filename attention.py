from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Reshape, Concatenate, Activation, Dense
import keras
import numpy as np
import string

from keras.models import Model
import tensorflow as tf

class AttentionModel:

    def __init__(self):

       
        model = VGG16()
        model.layers.pop()
       
        final_conv = Reshape([49,512])(model.layers[-4].output)
        self.model = Model(inputs=model.inputs, outputs=final_conv)
        print(self.model.summary())

    
        self.dimension_c = 512
        self.var_n_c = 49
        self.lstm_dimension_cell = 128
        self.lstm_dimension_hidden = 128

      
        self.multiple_hidden_layer = 256

        self.var_input = Input(shape=(self.dimenb jksion_c,))
        func_c_input = Dense(self.multiple_hidden_layer,activation="relu")(self.var_input)
        self.func_c_input = Dense(self.lstm_dimension_cell,activation=None)(func_c_input)

        self.helper_multi_hidden = 256

        self.inputs_h = Input(shape=(self.dimension_c,))
        temp = Dense(self.helper_multi_hidden,activation="relu")(self.inputs_h)
        self.temp = Dense(self.lstm_dimension_hidden,activation=None)(temp)

       
        self.att_mlp_hidden = 256

        self.attributes_inputs = Input(shape=(self.dimension_c+self.lstm_dimension_hidden,))
        x = Dense(self.att_mlp_hidden,activation="relu")(self.attributes_inputs)
        x = Dense(1,activation=None)(x)
        self.betas = Activation("softmax")(x)

        self.sess = tf.Session()

   
    def start_states_of_lstm(self,contexts):
        cell_state = self.sess.run(self.func_c_input,feed_dict={self.var_input:contexts})
        hidden_state = self.sess.run(self.temp,feed_dict={self.inputs_h:contexts})
        return cell_state,hidden_state

 
  
    def generate_betas(self,contexts,hidden_state):
        size_of_batch = contexts.shape[0]
        hiddenstate_title = tf.tile([[hidden_state]],[size_of_batch,var_n_c,1])
        concat_input = Concatenate(axis=-1)((contexts,hiddenstate_title))
        return self.sess.run(self.betas,feed_dict={self.attributes_inputs:concat_input})

    
    def vector_make_soft(contexts,betas):
        return contexts*tf.reshape(betas,[1,-1,1])

   
    def derive_features(images):
        return self.sess.run(self.model.output,feed_dict={})

        


