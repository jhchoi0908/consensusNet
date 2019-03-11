import os, sys, time, argparse, random
import tensorflow	as tf
import numpy		as np
sys.path.append('/home/choi240/CSNet')
from utils		import *
from network		import *

parser	= argparse.ArgumentParser(description='')
parser.add_argument('--patch_size',	dest='patch_size',	type=int,	default=64,						help='patch size')
parser.add_argument('--test_dir',	dest='test_dir',			default='/depot/chan129/data/CSNet/results/different_type',	help='the directory for validation data')
parser.add_argument('--ckpt_dir',	dest='ckpt_dir',			default='/home/choi240/CSNet/trained_model',		help='the directory for meta file')
args	= parser.parse_args()


def main():
	
	patch_size	= args.patch_size
	ckpt_dir	= args.ckpt_dir
	sigmaSet	= range(10, 51, 10)
	
	for i, sigma in enumerate(sigmaSet):
		
		test_dir	= os.path.join(args.test_dir, str(sigma))
		path_gt		= os.path.join(test_dir, 'groundtruth')
		path_in		= os.path.join(test_dir, 'inputs')
		path_rednet	= os.path.join(test_dir, 'rednet')
		
		if not os.path.exists(path_rednet):
			os.makedirs(path_rednet)
		
		model_rednet	= network0(os.path.join(args.ckpt_dir, 'rednet'+str(sigma)))
		
		files		= os.listdir(path_gt)
		sorted(files)
		for fname in files:
			img_gt	= load_image(os.path.join(path_gt, fname))
			img_no	= np.loadtxt(os.path.join(path_in, fname[:-4]+'.txt'), delimiter=' ')
			
			img_no4	= np.reshape(img_no, (1, img_no.shape[0], img_no.shape[1], 1))
			img_red	= model_rednet.run(img_no4)[0,:,:,0]
			save_image(img_red, os.path.join(path_rednet, fname))


if __name__ == "__main__":
	main()

