import argparse, os, sys
import tensorflow	as tf
import numpy		as np
sys.path.append('../utilities')
from utils		import *


parser	= argparse.ArgumentParser(description='')
parser.add_argument('--ckpt_dir',	dest='ckpt_dir',			default='../meta/different_denoiser',		help='the directory where models are saved')
parser.add_argument('--test_dir',	dest='data_dir',			default='../data',				help='the directory where test images are saved')
parser.add_argument('--meta_file',	dest='meta_file',			default='rednet_generic_10',			help='meta file')
parser.add_argument('--sigma',		dest='sigma',		type=int,	default=10,					help='noise level')
args	= parser.parse_args()


class import_graph():
	
	def __init__(self, loc):
		self.graph	= tf.Graph()
		self.sess	= tf.Session(graph=self.graph)
		with self.graph.as_default():
			saver	= tf.train.import_meta_graph(loc + '.meta', clear_devices=True)
			saver.restore(self.sess, loc)
			self.activation	= tf.get_collection('activation')[0]
	
	def run(self, data):
		return self.sess.run(self.activation, feed_dict={"y_test:0": data})


def main():
	
	ckpt_dir	= args.ckpt_dir
	meta_file	= args.meta_file
	sigma		= args.sigma
	
	if not os.path.exists(args.ckpt_dir):
		sys.exit('No trained model')
	
	path_gt		= os.path.join(args.data_dir, "groundtruth")
	path_in		= os.path.join(args.data_dir, "inputs")
	path_out	= os.path.join(path_in, "rednet")
	model		= import_graph(os.path.join(ckpt_dir, meta_file))
	
	np.random.seed(1337)
	files		= get_common_files(path_in, path_gt)
	
	thefile		= open(os.path.join(path_out, 'mse.txt'), 'w')
	for fname in files:
		groundtruth	= load_image(path_gt, fname)
		input_img	= load_image(path_in, fname)
		input_img	= np.reshape(input_img, (1, input_img.shape[0], input_img.shape[1], 1))
		
		MSE, y_		= cal_estimated_mse(input_img, model, sigma)
		
		save_image(y_[0,:,:,0], os.path.join(path_out, fname))
		thefile.write("%s %f\n" % (fname, MSE))
		print("%s %f" % (fname, MSE))


if __name__ == "__main__":
	main()

