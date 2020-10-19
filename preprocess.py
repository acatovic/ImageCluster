from keras.applications.vgg16 import preprocess_input

class VggPreprocess(object):
    def __call__(self, sample):
        sample['data'] = preprocess_input(sample['data'])
        return sample