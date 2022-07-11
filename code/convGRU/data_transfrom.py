import time
import numpy as np
from scipy import misc

IMAGE_PATH = '/Users/xinhuang/Documents/Lab/WORK/drought/convGRU/dataAll_clip/'

def imgmaps_tonumpy_crop(imgList,h,w):
	nSeq=len(imgList)
	SEQUENCE=np.zeros((nSeq,h,w))
	for i in range(nSeq):
		image_array = misc.imread(os.path.join(IMAGE_PATH, imgList[i]))
    		SEQUENCE[i] = image_array
    		print(file, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
	SEQUENCE=SEQUENCE/255.
	return SEQUENCE

def loadData(fpath):
	flist=os.listDir(fpath)
	flist.sort()
	flist.reshape()
	tList=

		

	
