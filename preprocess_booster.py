import os, sys, time, argparse, random
import tensorflow	as tf
import numpy		as np
from sklearn.utils	import shuffle
from utils		import *
from network		import *

parser	= argparse.ArgumentParser(description='')
parser.add_argument('--patch_size',	dest='patch_size',	type=int,	default=64,					help='patch size')
parser.add_argument('--num_patches',	dest='num_patches',	type=int,	default=128,					help='number of patches')
parser.add_argument('--num_aug',	dest='num_aug',		type=int,	default=6,					help='1 - 6')
parser.add_argument('--directory',	dest='directory',			default='/depot/chan129/data/CSNet/BSD300',	help='the directory for training data')
parser.add_argument('--data_gt',	dest='data_gt',				default='../../BSD300_gt',			help='the directory for training data')
parser.add_argument('--data_no',	dest='data_no',				default='../../BSD300_no',			help='the directory for training data')
parser.add_argument('--data_com',	dest='data_com',			default='../../BSD300_com',			help='the directory for training data')
parser.add_argument('--ckpt_dir',	dest='ckpt_dir',			default='../../trained_model',			help='the directory for meta file')
parser.add_argument('--meta1',		dest='meta1',				default='mse_estimator',			help='the file for mse estimator')
args	= parser.parse_args()


def main():
	
	patch_size	= args.patch_size
	num_patches	= args.num_patches
	num_aug		= args.num_aug
	path		= args.directory
	
	init_denoisers	= ["rednet10", "rednet20", "rednet30", "rednet40", "rednet50"]
	model_0		= [network0(os.path.join(args.ckpt_dir, d)) for d in init_denoisers]
	model_1		= network1(os.path.join(args.ckpt_dir, args.meta1))
	
	print("[*] Load data")
	start_time_tot	= time.time()
	
	files		= os.listdir(path)
	data_gt		= np.empty([num_aug*len(files)*num_patches*len(model_0), patch_size, patch_size, 1])
	data_no		= np.empty([num_aug*len(files)*num_patches*len(model_0), patch_size, patch_size, 1])
	data_com	= np.empty([num_aug*len(files)*num_patches*len(model_0), patch_size, patch_size, 1])
	cnt		= 0
	for fname in files:
		img	= load_image(os.path.join(path, fname))
		for j in range(num_aug):
			img_gt	= augmentation(img, j)
			tmp_gt	= extract_patches_random(img_gt, patch_size, num_patches)
			for i in range(tmp_gt.shape[0]):
				sigma	= np.random.randint(1, 71)
				tmp_no	= tmp_gt[i:i+1] + (sigma/255.0)*np.random.normal(size=tmp_gt[i:i+1].shape)
				tmp_den	= [model.run(tmp_no) for model in model_0]
				tmp_mse	= [model_1.run(tmp_no, den) for den in tmp_den]
				tmp_den	= [den[0,:,:,0] for den in tmp_den]
				tmp_com, weights	= combine(tmp_den, tmp_mse)
				
				data_gt[cnt,:,:,0]	= tmp_gt[i,:,:,0]
				data_no[cnt,:,:,0]	= tmp_no[0,:,:,0]
				data_com[cnt,:,:,0]	= tmp_com
				cnt	+= 1
	data_gt, data_no, data_com	= shuffle(data_gt, data_no, data_com)
	
	np.save(args.data_gt, data_gt)
	np.save(args.data_no, data_no)
	np.save(args.data_com, data_com)
	print("%f secs" % (time.time() - start_time_tot))
	

if __name__ == "__main__":
	main()

