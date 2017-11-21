import os, sys, random, time
import tensorflow	as tf
import numpy		as np
sys.path.append('../../utilities')
sys.path.append('../model')
from utils		import *


class network(object):
	
	def __init__(self, sess, options):
		
		self.sess	= sess
		
		self.lr		= options['lr']
		self.epochs	= options['epochs']
		
		self.meta	= options['meta']
		self.ckpt_dir	= options['ckpt_dir']
		self.path_in	= options['input_dir']
		self.path_gt	= options['gt_dir']
		
		self.model	= getattr(__import__(options['model'], fromlist=['model']), 'model')
		
		self.print_iter	= 100
		seed_num	= options['seed']
		
		random.seed(seed_num)
		np.random.seed(seed_num)
		tf.set_random_seed(seed_num)
		self.build_model()
	
	
	def build_model(self):
		
		self.x		= tf.placeholder(tf.float32,	shape=[None, None, None, 1],	name="x")	# Groundtruth
		self.y		= tf.placeholder(tf.float32,	shape=[None, None, None, 1],	name="y")	# Input
		self.y_test	= tf.placeholder(tf.float32,	shape=[None, None, None, 1],	name="y_test")	# Test Input
		
		self.x_		= self.model(self.y,		scope=self.meta,	reUse=False)
		self.x_test	= self.model(self.y_test,	scope=self.meta,	reUse=True)
		
		self.loss	= tf.nn.l2_loss(tf.subtract(self.x, self.x_))
		self.train_step	= tf.train.AdamOptimizer(self.lr).minimize(self.loss)
		
		self.init	= tf.global_variables_initializer()
		self.saver	= tf.train.Saver()
		print("[*] Initialize model successfully...")
	
	
	def train(self):
		
		self.sess.run(self.init)
		
		print("[*] Loading training data ...")
		files		= get_common_files(self.path_in, self.path_gt)
		random.shuffle(files)
		x_train		= [load_image(self.path_gt, fname) for fname in files]
		y_train		= [load_image(self.path_in, fname) for fname in files]
		
		print("[*] Start training")
		start_time	= time.time()
		for e in range(self.epochs):
			cnt	= 0
			for i in range(len(x_train)):
				batch_image	= x_train[i]
				train_image	= y_train[i]
				batch_image	= np.reshape(batch_image, (1, batch_image.shape[0], batch_image.shape[1], 1))
				train_image	= np.reshape(train_image, (1, train_image.shape[0], train_image.shape[1], 1))
				_, ls		= self.sess.run([self.train_step, self.loss], feed_dict={self.x:batch_image, self.y:train_image})
				cnt += 1
				if cnt%self.print_iter==0:
					print("Epoch: [%2d] [%4d] time: %4.4f, average loss: %.6f" % (e+1, cnt, time.time()-start_time, ls))
			print("Epoch: [%2d] [%4d] time: %4.4f, average loss: %.6f" % (e+1, cnt, time.time()-start_time, ls))
			
			if (e+1)%10==0 or e==self.epochs-1:
				print("[*] Saving model...")
				tf.add_to_collection("activation", self.x_test)
				self.saver.save(self.sess, os.path.join(self.ckpt_dir, self.meta))
		
		print("[*] Finish training")

