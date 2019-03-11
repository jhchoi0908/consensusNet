import os, sys, time, argparse, random
import tensorflow	as tf
import numpy		as np
from sklearn.utils	import shuffle
from utils		import *
from network		import *

parser	= argparse.ArgumentParser(description='')
parser.add_argument('--patch_size',	dest='patch_size',	type=int,	default=64,					help='patch size')
parser.add_argument('--num_patches',	dest='num_patches',	type=int,	default=64,					help='number of patches')
parser.add_argument('--num_aug',	dest='num_aug',		type=int,	default=6,					help='1 - 6')
parser.add_argument('--directory',	dest='directory',			default='/depot/chan129/data/CSNet/BSD300',	help='the directory for training data')
parser.add_argument('--data_gt',	dest='data_gt',				default='../../BSD300_gt',			help='the directory for training data')
parser.add_argument('--data_no',	dest='data_no',				default='../../BSD300_no',			help='the directory for training data')
parser.add_argument('--data_den',	dest='data_den',			default='../../BSD300_den',			help='the directory for training data')
parser.add_argument('--ckpt_dir',	dest='ckpt_dir',			default='../../trained_model',			help='the directory for meta file')
args	= parser.parse_args()


def main():
	
	patch_size	= args.patch_size
	num_patches	= args.num_patches
	num_aug		= args.num_aug
	path		= args.directory
	
	init_denoisers	= ["rednet10", "rednet20", "rednet30", "rednet40", "rednet50"]
	model_0		= [network0(os.path.join(args.ckpt_dir, d)) for d in init_denoisers]
	
	print("[*] Load data")
	start_time_tot	= time.time()
	
	files		= os.listdir(path)
	data_gt		= np.empty([num_aug*len(files)*num_patches*len(model_0), patch_size, patch_size, 1])
	data_no		= np.empty([num_aug*len(files)*num_patches*len(model_0), patch_size, patch_size, 1])
	data_den	= np.empty([num_aug*len(files)*num_patches*len(model_0), patch_size, patch_size, 1])
	cnt		= 0
	for fname in files:
		img	= load_image(os.path.join(path, fname))
		for j in range(num_aug):
			img_gt	= augmentation(img, j)
			tmp_gt	= extract_patches_random(img_gt, patch_size, num_patches)
			for i in range(tmp_gt.shape[0]):
				sigma	= np.random.randint(1, 71)
				tmp_no	= tmp_gt[i:i+1] + (sigma/255.0)*np.random.normal(size=tmp_gt[i:i+1].shape)
				for model in model_0:
					data_gt[cnt,:,:,0]	= tmp_gt[i,:,:,0]
					data_no[cnt,:,:,0]	= tmp_no[0,:,:,0]
					data_den[cnt,:,:,0]	= model.run(tmp_no)[0,:,:,0]
					cnt	+= 1
	data_gt, data_no, data_den	= shuffle(data_gt, data_no, data_den)
	
	np.save(args.data_gt, data_gt)
	np.save(args.data_no, data_no)
	np.save(args.data_den, data_den)
	print(data_gt.shape, data_no.shape, data_den.shape)
	print("%f secs" % (time.time() - start_time_tot))
	

if __name__ == "__main__":
	main()

