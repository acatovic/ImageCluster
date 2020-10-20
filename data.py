import glob

from keras.preprocessing import image

import matplotlib.pyplot as plt
import os

image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp']

class ImageDataset(object):
    """Helper class with memory friendly handling of image data."""
    
    def __init__(self, root_dir, image_size=(224, 224), preprocess=None):
        self.filenames = []
        for ext in image_extensions:
            self.filenames.extend(glob.glob(os.path.join(root_dir, ext)))
        assert self.filenames, 'No images found in {}'.format(root_dir)
        self.image_size = image_size
        self.preprocess = preprocess
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, ix):
        filename = self.filenames[ix]
        data = image.load_img(filename, target_size=self.image_size)
        data = image.img_to_array(data, dtype=int)
        sample = {'filename': filename, 'data': data}
        
        if self.preprocess:
            sample = self.preprocess(sample)
        
        return sample
    
    def show_image(self, i, title='', ax=None):
        if ax is None:
            _, ax = plt.subplots()
        
        ax.set_title(title)
        ax.axis('off')
        ax.imshow(image.load_img(self.filenames[i]))