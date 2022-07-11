import numpy as np
from PIL import Image
import os
import time

IMAGE_PATH = '/Users/xinhuang/Documents/Lab/WORK/drought/SPEI-sub4/'

WIDTH = 71
HEIGHT = 31
SEQUENCE = np.array([])
BASIC_SEQUENCE = np.array([])
NEXT_SEQUENCE = np.array([])
NUMBER = 1380
flist=os.listdir(IMAGE_PATH)
flist.sort()

for file in flist:
    image_array = np.loadtxt(os.path.join(IMAGE_PATH, file))
    SEQUENCE = np.append(SEQUENCE, image_array)
    print(file, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

SEQUENCE = SEQUENCE.reshape(NUMBER, WIDTH * HEIGHT)

np.savez('sequence_array.npz', sequence_array=SEQUENCE)
print('Data saved.')
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

