from pickle import load
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.my_image import load_image
from keras.preprocessing.my_image import convert_img_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu

import load_data as ld
import generate_model as gen
import argparse
import warnings


def ext_features(my_file):
 
  model = VGG16()

  model.layers.pop()
  model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

  my_image = load_image(my_file, target_size=(224, 224))

  my_image = convert_img_array(my_image)

  my_image = my_image.reshape((1, my_image.shape[0], my_image.shape[1], my_image.shape[2]))

  my_image = preprocess_input(my_image)

  feature = model.predict(my_image, verbose=0)
  return feature


def gen_description(model, tokenizer, extracted_features_pic, index_word, max_length, beam_size=5):

  caps = [['startseq', 0.0]]

  start_key_word = 'startseq'

  for i in range(max_length):
    all_captions = []

    for cap in caps:
      sentence, score = cap

      if sentence.split()[-1] == 'endseq':
        all_captions.append(cap)
        continue

      sentenceseq = tokenizer.texts_to_sequences([sentence])[0]

      sentenceseq = pad_sequences([sentenceseq], maxlen=max_length)

      y_pred = model.predict([extracted_features_pic,sentenceseq], verbose=0)[0]
 
      out = np.argsort(y_pred)[-beam_size:]

      for o in out:

        word = index_word.get(j)

        if word is None:
          continue

        caption = [sentence + ' ' + word, score + np.log(y_pred[o])]
        all_captions.append(caption)

  
    ordered = sorted(all_captions, key=lambda tup:tup[1], reverse=True)
    caps = ordered[:beam_size]

  return caps


def eval_my_model(model, descriptions, extracted_features_pics, tokenizer, index_word, max_length):
  actual, predicted = list(), list()

  for key, desc_list in descriptions.items():

    yout = gen_description(model, tokenizer, extracted_features_pics[key], index_word, max_length)[0]

    my_refs = [d.split() for d in desc_list]
    actual.append(my_refs)

    predicted.append(yout[0].split())


  print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

def evaluate_my_set(model, descriptions, extracted_features_pics, tokenizer, index_word, max_length):
  actual, predicted = list(), list()
 
  for key, desc_list in descriptions.items():

    yout = gen_description(model, tokenizer, extracted_features_pics[key], index_word, max_length)[0]
 
    my_refs = [d.split() for d in desc_list]
    actual.append(my_refs)

    predicted.append(yout[0].split())
  predicted = sorted(predicted)
  actual = [x for _,x in sorted(zip(actual,predicted))]

if __name__ == '__main__':
  print("insid main")
  parser = argparse.ArgumentParser(description='Find captions for my iamges')

  args = parser.parse_args()


  tokenizer = load(open('models/tokenizer.pkl', 'rb'))
  index_word = load(open('models/index_word.pkl', 'rb'))

  max_length = 34


  if args.model:
    my_file = args.model
  else:
    my_file = 'models/model_weight.h5'
  model = load_model(my_file)

  if args.my_image:
   
    extracted_features_pic = ext_features(args.my_image)
   
    caps = gen_description(model, tokenizer, extracted_features_pic, index_word, max_length)
    for cap in caps:
      
      seq = cap[0].split()[1:-1]
      desc = ' '.join(seq)
      print('{} '.format(desc))
      break
  else:
    
    t_feat, t_desc = ld.prepare_dataset('test')[1]

   
    eval_my_model(model, t_desc, t_feat, tokenizer, index_word, max_length)
      
