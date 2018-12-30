import load_data as ld
import generate_model as gen
from keras.callbacks import ModelCheckpoint
from pickle import dump

def model_training(weight = None, epochs = 10):
  
  train_d = ld.prepare_dataset('train')
  train_features, train_descriptions = train_d[0]
  test_features, test_descriptions = train_d[1]

  
  split_sentences = gen.create_tokenizer(train_descriptions)
  
  dump(split_sentences, open('models/tokenizer.pkl', 'wb'))
  
  index_word = {value: key for key, value in split_sentences.word_index.items()}
  
  dump(index_word, open('models/index_word.pkl', 'wb'))

  vocab_size = len(split_sentences.word_index) + 1
  print('Size of the Vocabulary: %d' % vocab_size)

  
  max_length = gen.max_length(train_descriptions)
  print('Length of the Descriptions: %d' % max_length)

  
  model = gen.define_model(vocab_size, max_length)

  
  if weight != None:
    model.load_weights(weight)

  
  filepath = 'models/model.h5'
  checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                save_best_only=True, mode='min')

  steps = len(train_descriptions)
  val_steps = len(test_descriptions)
  
  tg = gen.data_generator(train_descriptions, train_features, split_sentences, max_length)
  vg = gen.data_generator(test_descriptions, test_features, split_sentences, max_length)

  
  model.fit_generator(tg, epochs=epochs, steps_per_epoch=steps, verbose=1,
        callbacks=[checkpoint], validation_data=vg, validation_steps=val_steps)

  try:
      model.save('models/wholeModel.h5', overwrite=True)
      model.save_weights('models/weights.h5',overwrite=True)
  except:
      print("Error")
  print("Training has been completed broooooooooooo...\n")

if __name__ == '__main__':
    model_training(epochs=20)
