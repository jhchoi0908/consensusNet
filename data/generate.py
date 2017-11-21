import argparse, os, random
import numpy		as np
from scipy.ndimage	import imread
from scipy.misc		import toimage
from scipy.misc		import imsave
from skimage.util	import random_noise

parser	= argparse.ArgumentParser(description='')
parser.add_argument('--sigma', dest='sigma', type=int, default=20, help='noise level (sigma)')
sigma	= parser.parse_args().sigma

dir_in	= "./groundtruth"
dir_out	= "./inputs"
if not os.path.exists(dir_out):
	os.makedirs(dir_out)

np.random.seed(4321)
files	= [f for f in os.listdir(dir_in) if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png") or f.endswith(".JPEG")]
files.sort()
Sigma	= np.random.randint(10, 50, len(files))
for i, fname in enumerate(files):
	image	= imread(os.path.join(dir_in, fname), flatten=True)/255.0
	if sigma<0:
		noisy	= random_noise(image, var=(Sigma[i]/255.0)**2)
		print("%s %d" % (fname, Sigma[i]))
	else:
		noisy	= random_noise(image, var=(sigma/255.0)**2)
		print("%s %d" % (fname, sigma))
	toimage(noisy, cmin=0.0, cmax=1.0).save(os.path.join(dir_out, fname))

