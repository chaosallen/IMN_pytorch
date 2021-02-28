import natsort
import numpy as np
import os
import skimage.morphology as morphology
import cv2

def get_skeleton():
    load_path = '/home/limingchao/PycharmProjects/untitled/IMN_pytorch/logs/test_results'
    save_path = '/home/limingchao/PycharmProjects/untitled/IMN_pytorch/logs/test_results_line'
    names = natsort.natsorted(os.listdir(load_path))

    for name in names:
        img2 = cv2.imread(os.path.join(load_path,name),cv2.IMREAD_GRAYSCALE)
        img2 = np.where(img2==255,np.ones_like(img2),np.zeros_like(img2))
        img2 = morphology.skeletonize(img2).astype('int')
        cv2.imwrite(os.path.join(save_path,name),img2*255) #could delete *255 if needn't visualization

get_skeleton()