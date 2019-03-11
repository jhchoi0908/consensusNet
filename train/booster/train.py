import os, sys, time, argparse, random, collections
import tensorflow	as tf
import numpy		as np
from sklearn.utils	import shuffle
sys.path.append('../../')
from utils		import *
from network		import *

parser	= argparse.ArgumentParser(description='')
parser.add_argument('--lr',		dest='lr',		type=float,	default=1e-3,				help='learning rate')
parser.add_argument('--epochs',		dest='epochs',		type=int,	default=100,				help='number of epochs')
parser.add_argument('--num_patches',	dest='num_patches',	type=int,	default=128,				help='the number of extracted patches')
parser.add_argument('--patch_size',	dest='patch_size',	type=int,	default=64,				help='patch size')
parser.add_argument('--batch_size',	dest='batch_size',	type=int,	default=128,				help='batch size')
parser.add_argument('--train_dir',	dest='train_dir',			default='../../train_data',		help='the directory for training data')
parser.add_argument('--valid_dir',	dest='valid_dir',			default='../../valid',			help='the directory for validation data')
parser.add_argument('--ckpt_dir',	dest='ckpt_dir',			default='../../trained_model',		help='the directory for meta file')
parser.add_argument('--meta1',		dest='meta1',				default='mse_estimator',		help='the file for mse estimator')
parser.add_argument('--meta2',		dest='meta2',				default='booster_T3',			help='the file for booster')
args	= parser.parse_args()


def block(z0, y, reUse, Scope):
	
	n_ch	= 64
	with tf.variable_scope(Scope, reuse=reUse):
		
		h0	= tf.concat([z0, y], -1)
		h1	= tf.contrib.layers.conv2d		(h0,	n_ch,	[3,3],	activation_fn=tf.nn.relu,	scope='conv1')
		h2	= tf.contrib.layers.conv2d		(h1,	n_ch,	[3,3],	activation_fn=tf.nn.relu,	scope='conv2')
		h3	= tf.contrib.layers.conv2d		(h2,	n_ch,	[3,3],	activation_fn=tf.nn.relu,	scope='conv3')
		h4	= tf.contrib.layers.conv2d		(h3,	n_ch,	[3,3],	activation_fn=tf.nn.relu,	scope='conv4')
		h5	= tf.contrib.layers.conv2d		(h4,	n_ch,	[3,3],	activation_fn=tf.nn.relu,	scope='conv5')
		h6	= tf.contrib.layers.conv2d_transpose	(h5,	n_ch,	[3,3],	activation_fn=None,		scope='deconv1')
		h6	= tf.nn.relu(tf.add(h4, h6))
		h7	= tf.contrib.layers.conv2d_transpose	(h6,	n_ch,	[3,3],	activation_fn=None,		scope='deconv2')
		h7	= tf.nn.relu(tf.add(h3, h7))
		h8	= tf.contrib.layers.conv2d_transpose	(h7,	n_ch,	[3,3],	activation_fn=None,		scope='deconv3')
		h8	= tf.nn.relu(tf.add(h2, h8))
		h9	= tf.contrib.layers.conv2d_transpose	(h8,	n_ch,	[3,3],	activation_fn=None,		scope='deconv4')
		h9	= tf.nn.relu(tf.add(h1, h9))
		h10	= tf.contrib.layers.conv2d_transpose	(h9,	1,	[3,3],	activation_fn=None,		scope='deconv5')
		z	= tf.add(z0, h10)
		return z


def booster(z0, y, reUse):
	
	n_layer	= 5
	z	= z0
	for i in range(n_layer):
		z	= block(z, y, reUse, 'block%d'%(i+1))
	return z


def main():
	
	start_time_tot	= time.time()
	
	init_denoisers	= ["rednet10", "rednet20", "rednet30", "rednet40", "rednet50"]
	model_0		= [network0(os.path.join(args.ckpt_dir, d)) for d in init_denoisers]
	model_1		= network1(os.path.join(args.ckpt_dir, args.meta1))
	
	lr		= args.lr
	epochs		= args.epochs
	num_patches	= args.num_patches
	patch_size	= args.patch_size
	batch_size	= args.batch_size
	
	path_tr		= args.train_dir
	path_val	= args.valid_dir
	ckpt_dir	= args.ckpt_dir
	meta		= args.meta2
	save_iter	= 5
	
	
	# Initialize model
	x		= tf.placeholder(tf.float32,	shape=[batch_size, patch_size, patch_size, 1],	name="x")		# Groundtruth
	y		= tf.placeholder(tf.float32,	shape=[batch_size, patch_size, patch_size, 1],	name="y")		# Noisy
	xhat		= tf.placeholder(tf.float32,	shape=[batch_size, patch_size, patch_size, 1],	name="xhat")		# Denoised
	y_test		= tf.placeholder(tf.float32,	shape=[None, None, None, 1],			name="y_test")		# Test Noisy
	xhat_test	= tf.placeholder(tf.float32,	shape=[None, None, None, 1],			name="xhat_test")	# Test Denoised
	
	out		= booster(xhat,		y,	False)
	out_test	= booster(xhat_test,	y_test,	True)
	
	loss		= tf.reduce_mean(tf.reduce_sum(tf.squared_difference(out, x), [1,2,3]))
	train_step	= tf.train.AdamOptimizer(lr).minimize(loss)
	saver		= tf.train.Saver()
	print("[*] Initialize model successfully...")
	
	sess		= tf.Session()
	sess.run(tf.global_variables_initializer())
	
	if tf.gfile.Exists(os.path.join(ckpt_dir, meta+'.meta')):
		saver.restore(sess, os.path.join(ckpt_dir, meta))
		print("[*] Model Restored")
	
	
	# Load data
	start_time_load	= time.time()
	images_tr	= [load_image(os.path.join(path_tr, fname)) for fname in os.listdir(path_tr)]
	sigmaSet_tr	= range(1, 71)
	
	images_val	= [load_image(os.path.join(path_val, fname)) for fname in sorted(os.listdir(path_val))]
	sigmaSet_val	= range(10, 51, 10)
	data_no_val	= [[None for j in xrange(len(images_val))] for i in xrange(len(sigmaSet_val))]
	data_com_val	= [[None for j in xrange(len(images_val))] for i in xrange(len(sigmaSet_val))]
	for i in xrange(len(sigmaSet_val)):
		for j in xrange(len(images_val)):
			data_no_val[i][j], data_com_val[i][j]	= load_data_test(images_val[j], sigmaSet_val[i], patch_size, model_0, model_1)
	print("[*] Loaded images successfully... %.4f secs" % (time.time()-start_time_load))
	
	
	print("[*] Start training")
	start_time	= time.time()
	for e in range(epochs):
		
		# Training
		data_gt, data_no, data_com	= load_data(images_tr, patch_size, num_patches, sigmaSet_tr, model_0, model_1)
		for i in range(0, data_gt.shape[0], batch_size):
			_	= sess.run(train_step,	feed_dict={x:data_gt[i:i+batch_size], y:data_no[i:i+batch_size], xhat:data_com[i:i+batch_size]})
		
		# Calculate validation loss
		PSNR_before	= [0.0 for i in xrange(len(sigmaSet_val))]
		PSNR_after	= [0.0 for i in xrange(len(sigmaSet_val))]
		for i in xrange(len(data_no_val)):
			for j in xrange(len(data_no_val[0])):
				img		= images_val[j]
				data_no		= data_no_val[i][j]
				data_com	= data_com_val[i][j]
				out_val		= sess.run(out_test,	feed_dict={y_test:data_no, xhat_test:data_com})
				PSNR_before[i]	+= cal_psnr(data_com[0,:,:,0], img)
				PSNR_after [i]	+= cal_psnr(out_val[0,:,:,0], img)
			PSNR_before[i]	/= len(images_val)
			PSNR_after [i]	/= len(images_val)
		
		print(' '.join('{0:0.4f}'.format(PSNR) for PSNR in PSNR_before))
		print(' '.join('{0:0.4f}'.format(PSNR) for PSNR in PSNR_after))
		print("Epoch: [%2d] time: %11.4f" % (e+1, time.time()-start_time))
		
		if (e+1)%save_iter==0 or e+1==epochs:
			print("[*] Saving model...")
			tf.add_to_collection("activation", out_test)
			saver.save(sess, os.path.join(ckpt_dir, meta))
	
	print("[*] Finish training")
	print("[*] Toal Time: %.4f" % (time.time()-start_time_tot))


if __name__ == "__main__":
	main()

