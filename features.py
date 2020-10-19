import numpy as np

class VggFeaturize(object):
    def __init__(self, model):
        assert model, 'No model supplied'
        self.model = model
    
    def __call__(self, sample):
        data = np.expand_dims(sample['data'], axis=0)
        features = self.model.predict(data)
        features = np.array(features).flatten()
        return features