import os, sys, time, argparse, random
import tensorflow	as tf
import numpy		as np
sys.path.append('../../')
from utils		import *
from network		import *

parser	= argparse.ArgumentParser(description='')
parser.add_argument('--lr',		dest='lr',		type=float,	default=1e-4,				help='learning rate')
parser.add_argument('--epochs',		dest='epochs',		type=int,	default=50,				help='number of epochs')
parser.add_argument('--patch_size',	dest='patch_size',	type=int,	default=64,				help='patch size')
parser.add_argument('--batch_size',	dest='batch_size',	type=int,	default=128,				help='batch size')
parser.add_argument('--data_gt',	dest='data_gt',				default='../../BSD300_gt.npy',		help='the directory for patches from training groundtruth images')
parser.add_argument('--data_no',	dest='data_no',				default='../../BSD300_no.npy',		help='the directory for patches from training noisy images')
parser.add_argument('--data_den',	dest='data_den',			default='../../BSD300_den.npy',		help='the directory for patches from training denoiserd images')
parser.add_argument('--valid_dir',	dest='valid_dir',			default='../../data',			help='the directory for validation groundtruth images')
parser.add_argument('--ckpt_dir',	dest='ckpt_dir',			default='../../trained_model',		help='the directory for meta file')
parser.add_argument('--meta1',		dest='meta1',				default='mse_estimator',		help='the file for initial denoiser')
args	= parser.parse_args()


def mse_estimator(y, xhat, training):
	
	x1	= tf.contrib.layers.conv2d		(xhat,	32,	[3,3],	activation_fn=None,	scope='conv1_1')	# Output: batx64x64x32,		Weights: 3x3x1x32
	x1	= tf.contrib.layers.batch_norm		(x1,	is_training=training,	activation_fn=tf.nn.relu)
	x2	= tf.contrib.layers.conv2d		(x1,	32,	[3,3],	activation_fn=None,	scope='conv2_1')	# Output: batx64x64x32,		Weights: 3x3x32x32
	x2	= tf.contrib.layers.batch_norm		(x2,	is_training=training,	activation_fn=tf.nn.relu)
	x2	= tf.contrib.layers.max_pool2d		(x2,	[2, 2])
	
	y1	= tf.contrib.layers.conv2d		(y,	32,	[3,3],	activation_fn=None,	scope='conv1_2')	# Output: batx64x64x32,		Weights: 3x3x1x32
	y1	= tf.contrib.layers.batch_norm		(y1,	is_training=training,	activation_fn=tf.nn.relu)
	y2	= tf.contrib.layers.conv2d		(y1,	32,	[3,3],	activation_fn=None,	scope='conv2_2')	# Output: batx64x64x32,		Weights: 3x3x32x32
	y2	= tf.contrib.layers.batch_norm		(y2,	is_training=training,	activation_fn=tf.nn.relu)
	y2	= tf.contrib.layers.max_pool2d		(y2,	[2, 2])
	
	h2	= tf.concat([x2, y2], -1)
	h3	= tf.contrib.layers.conv2d		(h2,	64,	[3,3],	activation_fn=None,	scope='conv3')		# Output: batx32x32x64,		Weights: 3x3x64x64
	h3	= tf.contrib.layers.batch_norm		(h3,	is_training=training,	activation_fn=tf.nn.relu)
	h4	= tf.contrib.layers.conv2d		(h3,	64,	[3,3],	activation_fn=None,	scope='conv4')		# Output: batx32x32x64,		Weights: 3x3x64x64
	h4	= tf.contrib.layers.batch_norm		(h4,	is_training=training,	activation_fn=tf.nn.relu)
	h4	= tf.contrib.layers.max_pool2d		(h4,	[2, 2])
	
	h5	= tf.contrib.layers.conv2d		(h4,	64,	[3,3],	activation_fn=None,	scope='conv5')		# Output: batx16x16x64,		Weights: 3x3x64x64
	h5	= tf.contrib.layers.batch_norm		(h5,	is_training=training,	activation_fn=tf.nn.relu)
	h6	= tf.contrib.layers.conv2d		(h5,	1,	[3,3],	activation_fn=None,	scope='conv6')		# Output: batx16x16x1,		Weights: 3x3x64x1
	h6	= tf.contrib.layers.batch_norm		(h6,	is_training=training,	activation_fn=tf.nn.relu)
	h7	= tf.contrib.layers.flatten		(h6)									# Output: batx256
	
	h8	= tf.contrib.layers.fully_connected	(h7,	512,		activation_fn=tf.nn.relu,	scope='fc1')	# Output: batx512,		Weights: 256x512
	h8	= tf.contrib.layers.dropout		(h8,	keep_prob=0.5,	is_training=training)
	output	= tf.contrib.layers.fully_connected	(h8,	1,		activation_fn=tf.nn.sigmoid,	scope='fc2')	# Output: batx1,		Weights: 512x1
	
	return output


def main():
	
	start_time_tot	= time.time()
	
	init_denoisers	= ["rednet10", "rednet20", "rednet30", "rednet40", "rednet50"]
	model_0		= [network0(os.path.join(args.ckpt_dir, d)) for d in init_denoisers]
	
	lr		= args.lr
	epochs		= args.epochs
	patch_size	= args.patch_size
	batch_size	= args.batch_size
	
	path_val	= args.valid_dir
	ckpt_dir	= args.ckpt_dir
	meta		= args.meta1
	save_iter	= 5
	
	# Initialize model
	y		= tf.placeholder(tf.float32,	shape=[None, patch_size, patch_size, 1],	name="y")	# Noisy
	xhat		= tf.placeholder(tf.float32,	shape=[None, patch_size, patch_size, 1],	name="xhat")	# Denoised
	mse		= tf.placeholder(tf.float32,	shape=[None, 1],				name="mse")	# True MSE
	training	= tf.placeholder(tf.bool,	name='training')
	
	mse_hat		= mse_estimator(y, xhat, training)
	loss		= tf.reduce_mean(tf.abs(tf.subtract(mse, mse_hat)))
	update_ops	= tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
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
	data_gt		= np.load(args.data_gt)
	data_no		= np.load(args.data_no)
	data_den	= np.load(args.data_den)
	labels		= np.empty([data_gt.shape[0], 1])
	for i in range(data_gt.shape[0]):
		labels[i,0]	= cal_mse(data_den[i:i+1], data_gt[i:i+1])
	print(labels.shape, data_gt.shape, data_no.shape, data_den.shape)
	print("[*] Load training data successfully... %.4f secs" % (time.time()-start_time_load))
	
	
	start_time_load	= time.time()
	files_val	= os.listdir(path_val)
	files_val.sort()
	sigmaSet	= range(10, 51, 10)
	data_no_val	= {sigma:{d:{fname:None for fname in files_val} for d in init_denoisers} for sigma in sigmaSet}
	data_den_val	= {sigma:{d:{fname:None for fname in files_val} for d in init_denoisers} for sigma in sigmaSet}
	labels_val	= {sigma:{d:{fname:None for fname in files_val} for d in init_denoisers} for sigma in sigmaSet}
	for fname in files_val:
		img_gt	= load_image(os.path.join(path_val, fname))
		for sigma in sigmaSet:
			img_no	= img_gt + sigma/255.0*np.random.normal(size=img_gt.shape)
			img_no4	= np.reshape(img_no, (1, img_no.shape[0], img_no.shape[1], 1))
			for i, den in enumerate(init_denoisers):
				img_den		= model_0[i].run(img_no4)[0,:,:,0]
				data_no_val[sigma][den][fname]	= extract_patches(img_no, patch_size)
				data_den_val[sigma][den][fname]	= extract_patches(img_den, patch_size)
				labels_val[sigma][den][fname]	= cal_mse(img_gt, img_den)
	print("[*] Load test data successfully... %.4f secs" % (time.time()-start_time_load))
	
	
	# Training
	print("[*] Start training")
	start_time	= time.time()
	
	for e in range(epochs):
		
		# Training
		for i in range(0, labels.shape[0], batch_size):
			sess.run(train_step,	feed_dict={mse:labels[i:i+batch_size], y:data_no[i:i+batch_size], xhat:data_den[i:i+batch_size], training:True})
		
		# Calculate training loss
		loss_tr	= 0.0
		for i in range(0, labels.shape[0], batch_size):
			loss_tmp	= sess.run(loss,	feed_dict={mse:labels[i:i+batch_size], y:data_no[i:i+batch_size], xhat:data_den[i:i+batch_size], training:False})
			loss_tr		+= loss_tmp*labels[i:i+batch_size].shape[0]
		loss_tr	/= labels.shape[0]
		
		# Calculate validation loss
		chk	= [0]*len(sigmaSet)
		for j, sigma in enumerate(sigmaSet):
			tot_tr	= [0.0]*len(init_denoisers)
			tot_est	= [0.0]*len(init_denoisers)
			for i, den in enumerate(init_denoisers):
				for fname in files_val:
					msehats	= sess.run(mse_hat,	feed_dict={y:data_no_val[sigma][den][fname], xhat:data_den_val[sigma][den][fname], training:False})
					t1	= labels_val[sigma][den][fname]
					t2	= np.average(msehats)
					tot_tr[i]	+= t1
					tot_est[i]	+= t2
				tot_tr[i]	/= len(files_val)
				tot_est[i]	/= len(files_val)
			print("  %d true: %.6f %.6f %.6f %.6f %.6f" % (sigma, tot_tr[0], tot_tr[1], tot_tr[2], tot_tr[3], tot_tr[4]))
			print("  %d est : %.6f %.6f %.6f %.6f %.6f" % (sigma, tot_est[0], tot_est[1], tot_est[2], tot_est[3], tot_est[4]))
			err	= [np.absolute(t1-t2)/t1 for t1, t2 in zip(tot_tr, tot_est)]
			if all(ee<0.001 for ee in err):
				chk[j]	= 1
		
		# Save the network
		print("[*] Saving model...")
		tf.add_to_collection("activation", mse_hat)
		saver.save(sess, os.path.join(ckpt_dir, meta))
		
		# Print out results
		print("Epoch: [%2d] time: %11.4f, training loss: %.6f" % (e+1, time.time()-start_time, loss_tr))
		
		if all(c==1 for c in chk):
			print("[*] Finish training")
			print("[*] Toal Time: %.4f" % (time.time()-start_time_tot))
			sys.exit()
	
	print("[*] Finish training")
	print("[*] Toal Time: %.4f" % (time.time()-start_time_tot))


if __name__ == "__main__":
	main()

