import os, sys, random, time
import tensorflow	as tf
import numpy		as np
sys.path.append('../../utilities')
sys.path.append('../model')
from utils		import *


class network(object):
	
	def __init__(self, sess, options):
		
		self.sess	= sess
		
		self.numpatches	= options['num_patches']
		self.patch_sz	= options['patch_sz']
		self.batch_sz	= options['batch_sz']
		self.lr		= options['lr']
		self.epochs	= options['epochs']
		self.sigma	= options['sigma']
		
		self.ckpt_dir	= options['ckpt_dir']
		self.meta	= options['meta']
		self.train_dir	= options['train_dir']
		
		self.model	= getattr(__import__(options['model'], fromlist=['model']), 'model')
		
		self.print_iter	= 100
		seed_num	= options['seed']
		
		random.seed(seed_num)
		np.random.seed(seed_num)
		tf.set_random_seed(seed_num)
		self.build_model()
	
	
	def build_model(self):
		
		self.x		= tf.placeholder(tf.float32,	shape=[None, self.patch_sz, self.patch_sz, 1],	name="x")	# Groundtruth
		self.y		= tf.placeholder(tf.float32,	shape=[None, self.patch_sz, self.patch_sz, 1],	name="y")	# Input
		self.y_test	= tf.placeholder(tf.float32,	shape=[None, None, None, 1],			name="y_test")	# Test Input
		
		self.x_		= self.model(self.y,		scope=self.meta,	reUse=False)
		self.x_test	= self.model(self.y_test,	scope=self.meta,	reUse=True)
		
		self.loss	= tf.nn.l2_loss(tf.subtract(self.x, self.x_)) / self.batch_sz
		self.train_step	= tf.train.AdamOptimizer(self.lr).minimize(self.loss)
		
		self.init	= tf.global_variables_initializer()
		self.saver	= tf.train.Saver()
		print("[*] Initialize model successfully...")
	
	
	def train(self, max_images):
		
		self.sess.run(self.init)
		
		print("[*] Loading training data ...")
		files		= get_files(self.train_dir, israndom=True)
		files		= files[:max_images]
		
		self.y_train	= load_data(self.train_dir, files, self.patch_sz, self.numpatches)
		print("Input size: %d*%d*%d*%d"%(self.y_train.shape[0], self.y_train.shape[1], self.y_train.shape[2], self.y_train.shape[3]))
		
		print("[*] Start training")
		start_time	= time.time()
		for e in range(self.epochs):
			cnt	= 0
			for i in range(0, self.y_train.shape[0], self.batch_sz):
				batch_images	= self.y_train[i:i+self.batch_sz]
				train_images	= add_noise(batch_images, self.sigma)
				_, ls		= self.sess.run([self.train_step, self.loss], feed_dict={self.x:batch_images, self.y:train_images})
				cnt += 1
				if cnt%self.print_iter==0:
					print("Epoch: [%2d] [%4d] time: %4.4f, average loss: %.6f" % (e+1, cnt, time.time()-start_time, ls))
			if cnt>0:
				print("Epoch: [%2d] [%4d] time: %4.4f, average loss: %.6f" % (e+1, cnt, time.time()-start_time, ls))
			
				if (e+1)%10==0 or e==self.epochs-1:
					print("[*] Saving model...")
					tf.add_to_collection("activation", self.x_test)
					self.saver.save(self.sess, os.path.join(self.ckpt_dir, self.meta))
		
		print("[*] Finish training")

