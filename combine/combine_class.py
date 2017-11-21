import os, sys, argparse, random, time
import tensorflow	as tf
import numpy		as np
sys.path.append('../utilities')
from utils		import *

parser	= argparse.ArgumentParser(description='')
parser.add_argument('--ckpt_dir',	dest='ckpt_dir',			default='../meta/different_class',		help='the directory where models are saved')
parser.add_argument('--data_dir',	dest='data_dir',			default='../data',				help='the directory where images are saved')
parser.add_argument('--sigma',		dest='sigma',		type=int,	default=20,					help='noise level (integer)')
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
	
	sigma		= args.sigma
	trained_sigmas	= [sigma]*3
	trained_class	= ['building', 'face', 'flower']
	metafiles	= ['rednet_'+c+'_'+str(s) for s, c in zip(trained_sigmas, trained_class)]
	
	ckpt_dir	= args.ckpt_dir
	if not os.path.exists(ckpt_dir):
		sys.exit('No trained model')
	
	path_gt		= os.path.join(args.data_dir, "groundtruth")
	path_in		= os.path.join(args.data_dir, "inputs")
	path_out	= os.path.join(args.data_dir, "combined")
	models		= [import_graph(os.path.join(ckpt_dir, f)) for f in metafiles]
	
	np.random.seed(1234)
	files		= get_common_files(path_in, path_gt)
	
	for cnt, fname in enumerate(files):
		
		groundtruth	= load_image(path_gt, fname)
		
		input_img	= load_image(path_in, fname)
		input_img	= np.reshape(input_img, (1, input_img.shape[0], input_img.shape[1], 1))
		
		output_img	= weighted_average(input_img, models, trained_sigmas, sigma)
		save_image(output_img[0,:,:,0], os.path.join(path_out, fname))


if __name__ == "__main__":
	main()

