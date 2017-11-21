import argparse, os
from network	import *

parser	= argparse.ArgumentParser(description='')
parser.add_argument('--maximages',	dest='max_images',	type=int,	default=2000,					help='number of images for training')
parser.add_argument('--numpatches',	dest='num_patches',	type=int,	default=50,					help='number of patches')
parser.add_argument('--patchsize',	dest='patch_sz',	type=int,	default=50,					help='patch size')
parser.add_argument('--batchsize',	dest='batch_sz',	type=int,	default=50,					help='batch size')
parser.add_argument('--lr',		dest='lr',		type=float,	default=1e-4,					help='initial learning rate')
parser.add_argument('--epochs',		dest='epochs',		type=int,	default=50,					help='number of epochs')
parser.add_argument('--sigma',		dest='sigma',		type=int,	default=20,					help='noise level (integer)')

parser.add_argument('--ckpt_dir',	dest='ckpt_dir',			default='../../meta/different_noise',		help='the directory where models are saved')
parser.add_argument('--meta',		dest='meta',				default='rednet_generic_10',			help='the meta file name')
parser.add_argument('--train_dir',	dest='train_dir',			default='../../data',				help='the directory of training images')

parser.add_argument('--seed',		dest='seed',		type=int,	default=1234,					help='random seed number')
args	= parser.parse_args()


def main():
	
	if not os.path.exists(args.ckpt_dir):
		os.makedirs(args.ckpt_dir)
	
	options			= {}
	options['num_patches']	= args.num_patches
	options['patch_sz']	= args.patch_sz
	options['batch_sz']	= args.batch_sz
	options['lr']		= args.lr
	options['epochs']	= args.epochs
	options['sigma']	= args.sigma
	
	options['ckpt_dir']	= args.ckpt_dir
	options['meta']		= args.meta
	options['train_dir']	= os.path.join(args.train_dir, "groundtruth")
	
	options['model']	= 'rednet'
	
	options['seed']		= args.seed
	
	with tf.Session() as sess:
		model_	= network(sess, options)
		model_.train(args.max_images)


if __name__ == "__main__":
	main()

