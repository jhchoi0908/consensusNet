import os, sys, time, argparse
import tensorflow	as tf
import numpy		as np
sys.path.append('/home/choi240/ConsensusNet')
from utils		import *

parser	= argparse.ArgumentParser(description='')
parser.add_argument('--lr',		dest='lr',		type=float,	default=1e-4,						help='learning rate')
parser.add_argument('--epochs',		dest='epochs',		type=int,	default=50,						help='number of epochs')
parser.add_argument('--patch_size',	dest='patch_size',	type=int,	default=64,						help='patch size')
parser.add_argument('--batch_size',	dest='batch_size',	type=int,	default=128,						help='batch size')
parser.add_argument('--sigma',		dest='sigma',		type=int,	default=10,						help='noise level')
parser.add_argument('--train_data',	dest='train_data',			default='/depot/chan129/data/CSNet/BSD300.npy',		help='the directory for training data')
parser.add_argument('--valid_dir',	dest='valid_dir',			default='/depot/chan129/data/CSNet/Kodak',		help='the directory for validation data')
parser.add_argument('--ckpt_dir',	dest='ckpt_dir',			default='/home/choi240/CSNet/trained_model',		help='the directory for meta file')
parser.add_argument('--meta0',		dest='meta0',				default='rednet',					help='the file for initial denoiser')
args	= parser.parse_args()


def rednet(y, reUse):
	
	with tf.variable_scope('rednet', reuse=reUse):
		
		x1	= tf.contrib.layers.conv2d		(y,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv1')
		x2	= tf.contrib.layers.conv2d		(x1,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv2')
		x3	= tf.contrib.layers.conv2d		(x2,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv3')
		x4	= tf.contrib.layers.conv2d		(x3,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv4')
		x5	= tf.contrib.layers.conv2d		(x4,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv5')
		x6	= tf.contrib.layers.conv2d		(x5,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv6')
		x7	= tf.contrib.layers.conv2d		(x6,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv7')
		x8	= tf.contrib.layers.conv2d		(x7,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv8')
		x9	= tf.contrib.layers.conv2d		(x8,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv9')
		x10	= tf.contrib.layers.conv2d		(x9,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv10')
		x11	= tf.contrib.layers.conv2d		(x10,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv11')
		x12	= tf.contrib.layers.conv2d		(x11,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv12')
		x13	= tf.contrib.layers.conv2d		(x12,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv13')
		x14	= tf.contrib.layers.conv2d		(x13,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv14')
		x15	= tf.contrib.layers.conv2d		(x14,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv15')
		
		x16	= tf.contrib.layers.conv2d_transpose	(x15,	64,	[3,3],	activation_fn=None,		scope='conv16')
		x16	= tf.nn.relu(tf.add(x16, x14))
		x17	= tf.contrib.layers.conv2d_transpose	(x16,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv17')
		x18	= tf.contrib.layers.conv2d_transpose	(x17,	64,	[3,3],	activation_fn=None,		scope='conv18')
		x18	= tf.nn.relu(tf.add(x18, x12))
		x19	= tf.contrib.layers.conv2d_transpose	(x18,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv19')
		x20	= tf.contrib.layers.conv2d_transpose	(x19,	64,	[3,3],	activation_fn=None,		scope='conv20')
		x20	= tf.nn.relu(tf.add(x20, x10))
		x21	= tf.contrib.layers.conv2d_transpose	(x20,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv21')
		x22	= tf.contrib.layers.conv2d_transpose	(x21,	64,	[3,3],	activation_fn=None,		scope='conv22')
		x22	= tf.nn.relu(tf.add(x22, x8))
		x23	= tf.contrib.layers.conv2d_transpose	(x22,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv23')
		x24	= tf.contrib.layers.conv2d_transpose	(x23,	64,	[3,3],	activation_fn=None,		scope='conv24')
		x24	= tf.nn.relu(tf.add(x24, x6))
		x25	= tf.contrib.layers.conv2d_transpose	(x24,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv25')
		x26	= tf.contrib.layers.conv2d_transpose	(x25,	64,	[3,3],	activation_fn=None,		scope='conv26')
		x26	= tf.nn.relu(tf.add(x26, x4))
		x27	= tf.contrib.layers.conv2d_transpose	(x26,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv27')
		x28	= tf.contrib.layers.conv2d_transpose	(x27,	64,	[3,3],	activation_fn=None,		scope='conv28')
		x28	= tf.nn.relu(tf.add(x28, x2))
		x29	= tf.contrib.layers.conv2d_transpose	(x28,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv29')
		x30	= tf.contrib.layers.conv2d_transpose	(x29,	1,	[3,3],	activation_fn=None,		scope='conv30')
		output	= tf.nn.relu(tf.add(x30, y))
		
		return output


def main():
	
	start_time_tot	= time.time()
	
	lr		= args.lr
	epochs		= args.epochs
	patch_size	= args.patch_size
	batch_size	= args.batch_size
	sigma		= args.sigma
	
	meta		= "rednet"+str(sigma)
	ckpt_dir	= args.ckpt_dir
	train_data	= args.train_data
	path_val	= args.valid_dir
	save_iter	= 5
	
	x		= tf.placeholder(tf.float32,	shape=[None, patch_size, patch_size, 1],	name="x")	# Groundtruth
	y		= tf.placeholder(tf.float32,	shape=[None, patch_size, patch_size, 1],	name="y")	# Noisy
	x_gen		= tf.placeholder(tf.float32,	shape=[None, None, None, 1],			name="x_gen")	# Groundtruth
	y_gen		= tf.placeholder(tf.float32,	shape=[None, None, None, 1],			name="y_gen")	# Noisy
	
	xhat		= rednet(y, False)
	xhat_gen	= rednet(y_gen, True)
	loss		= tf.reduce_mean(tf.square(tf.subtract(xhat, x)))
	train_step	= tf.train.AdamOptimizer(lr).minimize(loss)
	saver		= tf.train.Saver()
	print("[*] Initialize model successfully...")
	
	sess		= tf.Session()
	sess.run(tf.global_variables_initializer())
	
	if tf.gfile.Exists(os.path.join(ckpt_dir, meta+'.meta')):
		saver.restore(sess, os.path.join(ckpt_dir, meta))
		print("[*] Model Restored")
	
	print("[*] Load data")
	start_time_load	= time.time()
	data_gt		= np.load(train_data)
	images_gt	= [load_image(os.path.join(path_val, f)) for f in os.listdir(path_val)]
	images_gt	= [np.reshape(img, (1, img.shape[0], img.shape[1], 1)) for img in images_gt]
	images_no	= [img + (sigma/255.0)*np.random.normal(size=img.shape) for img in images_gt]
	print data_gt.shape
	print("[*] Load data successfully... %.4f secs" % (time.time()-start_time_load))
	
	
	print("[*] Start training")
	start_time	= time.time()
	for e in range(epochs):
		
		# Training
		for i in range(0, data_gt.shape[0], batch_size):
			data	= data_gt[i:i+batch_size]
			data_no	= data + (sigma/255.0)*np.random.normal(size=data.shape)
			_	= sess.run(train_step,	feed_dict={x:data, y:data_no})
		
		# Calculate training loss
		ls	= 0.0
		for i in range(0, data_gt.shape[0], batch_size):
			data	= data_gt[i:i+batch_size]
			data_no	= data + (sigma/255.0)*np.random.normal(size=data.shape)
			tmp	= sess.run(loss,	feed_dict={x:data, y:data_no})
			ls	+= tmp*data.shape[0]
		ls	/= data_gt.shape[0]
		
		# Calculate PSNR of validation
		images_den	= [sess.run(xhat_gen, feed_dict={y_gen:img}) for img in images_no]
		PSNR		= [cal_psnr(img1, img2) for img1, img2 in zip(images_gt, images_den)]
		print("Epoch: [%3d] time: %11.4f, training loss: %.6f, PSNR average: %.6f" % (e+1, time.time()-start_time, ls, sum(PSNR)/len(PSNR)))
		
		# Save the model
		if (e+1)%save_iter==0 or e+1==epochs:
			print("[*] Saving model...")
			tf.add_to_collection("activation", xhat_gen)
			saver.save(sess, os.path.join(ckpt_dir, meta))
	
	print("[*] Finish training")
	print("[*] Toal Time: %.4f" % (time.time()-start_time_tot))


if __name__ == "__main__":
	main()

