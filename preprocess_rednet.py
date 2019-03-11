import os, sys, time, argparse
import tensorflow	as tf
import numpy		as np
from utils		import *


parser	= argparse.ArgumentParser(description='')
parser.add_argument('--patch_size',	dest='patch_size',	type=int,	default=64,					help='patch size')
parser.add_argument('--num_patches',	dest='num_patches',	type=int,	default=64,					help='number of patches')
parser.add_argument('--num_aug',	dest='num_aug',		type=int,	default=6,					help='1 - 6')
parser.add_argument('--directory',	dest='directory',			default='/depot/chan129/data/CSNet/BSD300',	help='the directory for training data')
parser.add_argument('--new_file',	dest='new_file',			default='../../BSD300',				help='the directory for training data')
args	= parser.parse_args()


def main():
	
	patch_size	= args.patch_size
	num_patches	= args.num_patches
	num_aug		= args.num_aug
	path		= args.directory
	new_file	= args.new_file
	
	print("[*] Load data")
	start_time_tot	= time.time()
	
	files		= os.listdir(path)
	data_gt		= np.empty([num_aug*len(files)*num_patches, patch_size, patch_size, 1])
	cnt		= 0
	for fname in files:
		img	= load_image(os.path.join(path, fname))
		for j in range(num_aug):
			img_gt	= augmentation(img, j)
			data_gt[cnt:cnt+num_patches]	= extract_patches_random(img_gt, patch_size, num_patches)
			cnt	+= num_patches
	np.random.shuffle(data_gt)
	
	np.save(new_file, data_gt)
	print("%f secs" % (time.time() - start_time_tot))


if __name__ == "__main__":
	main()

