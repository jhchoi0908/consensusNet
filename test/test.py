import argparse, os, sys
import tensorflow	as tf
import numpy		as np
sys.path.append('../utilities')
from utils		import *


parser	= argparse.ArgumentParser(description='')
parser.add_argument('--ckpt_dir',	dest='ckpt_dir',			default='../meta/different_class',		help='the directory where models are saved')
parser.add_argument('--test_dir',	dest='test_dir',			default='../data',				help='the directory where test images are saved')
parser.add_argument('--meta_file',	dest='meta_file',			default='class_booster',			help='meta file name')
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
	test_dir	= args.test_dir
	meta_file	= args.meta_file
	
	model		= import_graph(os.path.join(ckpt_dir, meta_file))
	
	path_gt		= os.path.join(test_dir, "groundtruth")
	path_in		= os.path.join(test_dir, "combined")
	path_out	= os.path.join(test_dir, "outputs")
	files		= get_common_files(path_in, path_gt)
	files.sort()
	
	for fname in files:
		x_test		= load_image(path_gt, fname)
		y_test		= load_image(path_in, fname)
		y_test		= np.reshape(y_test, (1, y_test.shape[0], y_test.shape[1], 1))
		y_		= model.run(y_test)
		print("%s %.4f"%(fname, cal_psnr(x_test, y_[0,:,:,0])))
		save_image(y_[0,:,:,0], os.path.join(path_out, fname))

if __name__ == "__main__":
	main()

