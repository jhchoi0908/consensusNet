import argparse, os
from network	import *

parser	= argparse.ArgumentParser(description='')
parser.add_argument('--lr',		dest='lr',		type=float,	default=1e-4,				help='learning rate')
parser.add_argument('--epochs',		dest='epochs',		type=int,	default=50,				help='number of epochs')

parser.add_argument('--meta',		dest='meta',				default='noiselevel_booster',		help='meta file name')
parser.add_argument('--ckpt_dir',	dest='ckpt_dir',			default='../../meta/different_noise',	help='the directory where models are saved')
parser.add_argument('--train_dir',	dest='train_dir',			default='../../data',			help='the directory of training images')

parser.add_argument('--seed',		dest='seed',		type=int,	default=1234,				help='random seed number')
args	= parser.parse_args()


def main():
	
	if not os.path.exists(args.ckpt_dir):
		os.makedirs(args.ckpt_dir)
	
	options			= {}
	options['lr']		= args.lr
	options['epochs']	= args.epochs
	
	options['meta']		= args.meta
	options['ckpt_dir']	= args.ckpt_dir
	options['input_dir']	= os.path.join(args.train_dir, 'combined')
	options['gt_dir']	= os.path.join(args.train_dir, 'groundtruth')
	
	options['model']	= 'rednet'
	options['seed']		= args.seed
	
	with tf.Session() as sess:
		model_	= network(sess, options)
		model_.train()


if __name__ == "__main__":
	main()

