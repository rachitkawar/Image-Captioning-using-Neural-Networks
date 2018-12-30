from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Reshape, Concatenate
import numpy as np
import string
from progressbar import progressbar
from keras.models import Model

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

class OrganizeData():
    
    def __init__(self):
        self.ImageDirectory = 'Flickr8k_Dataset'
        self.CaptionFile = 'Flickr8k_text/Flickr8k.token.txt'

    def load_image(self,path):
        
        img = load_img(path, target_size=(224,224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return np.asarray(img)

    
    def getthelist_of_features(self,is_attention=False):
    
      if is_attention:
        model = VGG16()
        model.layers.pop()
    
        final_conv = Reshape([49,512])(model.layers[-4].output)
        model = Model(inputs=model.inputs, outputs=final_conv)
        print(model.summary())
        list_of_features = dict()
      else:
        model = VGG16()
    
        model.layers.pop()
        model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
        print(model.summary())
    
        list_of_features = dict()

      for name in progressbar(listdir(self.ImageDirectory)):
    
        if name == 'README.md':
          continue
        fname = self.ImageDirectory + '/' + name
        image = self.load_image(fname)
        
        feature = model.predict(image, verbose=0)
        
        image_id = name.split('.')[0]
        
        list_of_features[image_id] = feature
        print('>%s' % name)
      print('Extracted list_of_features: %d' %len(list_of_features))
      
      dump(list_of_features, open('models/list_of_features.pkl', 'wb'))

    
    def acquire_doc(self):
    
      file = open(self.CaptionFile, 'r')
    
      text = file.read()
    
      file.close()
      return text

    
    def loading_descriptions(self,doc):
      mapping = dict()
    
      for line in doc.split('\n'):
    
        tokens = line.split()
        if len(line) < 2:
          continue
    
        image_id, image_desc = tokens[0], tokens[1:]
    
        image_id = image_id.split('.')[0]
    
        image_desc = ' '.join(image_desc)
    
        if image_id not in mapping:
          mapping[image_id] = list()
    
        mapping[image_id].append(image_desc)
      return mapping

    def cleandesc(self,descriptions):
    
      table = str.maketrans('', '', string.punctuation)
      for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
          desc = desc_list[i]
          
          desc = desc.split()
          
          desc = [word.lower() for word in desc]
          
          desc = [w.translate(table) for w in desc]
          
          desc = [word for word in desc if len(word)>1]
          
          desc = [word for word in desc if word.isalpha()]
          
          desc_list[i] =  ' '.join(desc)

    
    def to_vocabulary(self,descriptions):
      
      all_desc = set()
      for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
      vocabulary = all_desc
      print('Vocabulary Size: %d' % len(vocabulary))

    
    def save_descriptions(self,descriptions, fname):
      lines = list()
      for key, desc_list in descriptions.items():
        for desc in desc_list:
          lines.append(key + ' ' + desc)
      data = '\n'.join(lines)
      file = open(fname, 'w')
      file.write(data)
      file.close()

    def preparedescriptions(self,text):
          
      descriptions = self.loading_descriptions(text)
      print('Loaded: %d ' % len(descriptions))
      self.cleandesc(descriptions)
      vocab = self.to_vocabulary(descriptions)
      print('Vocabulary size: %d' % len(vocab))
      

if __name__ == '__main__':
  pd = OrganizeData()
  pd.getthelist_of_features()
  text = pd.acquire_doc()
  pd.preparedescriptions(text)

