from pickle import load
import argparse

def get_document_to_python(inputImage):
  file = open(inputImage, 'r')
  
  text = file.read()
  
  file.close()
  return text
def load_set(inputImage):
  doc = get_document_to_python(inputImage)
  imageItems = list()
  
  for line in doc.split('\n'):
    
    if len(line) < 1:
      continue
    
    uniqueKey = line.split('.')[0]
    imageItems.append(uniqueKey)
  return set(imageItems)


def train_test_split(imageItems):

  ordered = sorted(imageItems)

  return set(ordered[:100]), set(ordered[100:200])


def cleansing_Image_details(inputImage, imageItems):

  doc = get_document_to_python(inputImage)
  details = dict()
  for line in doc.split('\n'):

    tokens = line.split()

    photo_key, photo_details = tokens[0], tokens[1:]

    if photo_key in imageItems:

      if photo_key not in details:
        details[photo_key] = list()

      desc = 'startseq ' + ' '.join(photo_details) + ' endseq'

      details[photo_key].append(desc)
  return details


def load_photo_features(inputImage, imageItems):
  
  all_features = load(open(inputImage, 'rb'))
  
  features = {k: all_features[k] for k in imageItems}
  return features

def prepare_imageItems(data='dev'):

  assert data in ['dev', 'train', 'test']

  train_features = None
  train_descriptions = None

  if data == 'dev':
  
    inputImage = 'Flickr8k_text/Flickr_8k.devImages.txt'
    imageItems = load_set(inputImage)
    print('imageItems: %d' % len(imageItems))

  
    train, test = train_test_split(imageItems)
  

  
    train_descriptions = cleansing_Image_details('models/descriptions.txt', train)
    test_descriptions = cleansing_Image_details('models/descriptions.txt', test)
    print('Descriptions: train=%d, test=%d' % (len(train_descriptions), len(test_descriptions)))

  
    train_features = load_photo_features('models/features.pkl', train)
    test_features = load_photo_features('models/features.pkl', test)
    print('Photos: train=%d, test=%d' % (len(train_features), len(test_features)))

  elif data == 'train':
  
    inputImage = 'Flickr8k_text/Flickr_8k.trainImages.txt'
    train = load_set(inputImage)

    inputImage = 'Flickr8k_text/Flickr_8k.devImages.txt'
    test = load_set(inputImage)
    print('imageItems: %d' % len(train))

  
    train_descriptions = cleansing_Image_details('models/descriptions.txt', train)
    test_descriptions = cleansing_Image_details('models/descriptions.txt', test)
    print('Descriptions: train=%d, test=%d' % (len(train_descriptions), len(test_descriptions)))

  
    train_features = load_photo_features('models/features.pkl', train)
    test_features = load_photo_features('models/features.pkl', test)
    print('Photos: train=%d, test=%d' % (len(train_features), len(test_features)))

  elif data == 'test':
  
    inputImage = 'Flickr8k_text/Flickr_8k.testImages.txt'
    test = load_set(inputImage)
    print('imageItems: %d' % len(test))
  
    test_descriptions = cleansing_Image_details('models/descriptions.txt', test)
    print('Descriptions: test=%d' % len(test_descriptions))
  
    test_features = load_photo_features('models/features.pkl', test)
    print('Photos: test=%d' % len(test_features))

  return (train_features, train_descriptions), (test_features, test_descriptions)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Generate imageItems features')
  parser.add_argument("-t", "--train", action='store_const', const='train',
    default = 'dev', help="Use large 6K training set")
  args = parser.parse_args()
  prepare_imageItems(args.train)
