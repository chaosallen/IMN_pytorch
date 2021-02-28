import numpy as np
import os
import random
import scipy.misc as misc
import torch
import imageio
import matplotlib.pyplot as plt

class Batchreader:

    def __init__(self, path):
        print("Initializing Batch Dataset Reader...")
        self.namelist = {};
        self.path = path
        self.namelist = os.listdir(os.path.join(path,'img'))

    def next_batch(self, batch_size,image_size,input_size):
        image = np.arange(batch_size * 1 * input_size[0] * input_size[1]).reshape(batch_size, 1, input_size[0],input_size[1])
        label = np.arange(batch_size * 1 * input_size[0] * input_size[1]).reshape(batch_size, input_size[0],input_size[1])
        for batch in range(0,batch_size):
            id = random.randint(0, len(self.namelist) - 1)
            name = self.namelist[id]
            nx=random.randint(input_size[0]/2,image_size[0]-input_size[0]/2)
            ny = random.randint(input_size[1] / 2, image_size[1] - input_size[1] / 2)
            startx = nx-int(input_size[0]/2)
            endx = nx+int(input_size[0]/2)
            starty= ny-int(input_size[1]/2)
            endy = ny + int(input_size[1]/2)
            input_image=np.array(imageio.imread(os.path.join(self.path,'img',name)))
            input_label=np.array(imageio.imread(os.path.join(self.path,'gt',name)))/255
            image[batch, 0, 0:input_size[0], 0:input_size[1]] = input_image[startx:endx,starty:endy]
            label[batch, 0:input_size[0], 0:input_size[1]] = input_label[startx:endx,starty:endy]
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        return image, label


