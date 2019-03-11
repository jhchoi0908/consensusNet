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
parser.add_argument('--train_data',	dest='train_data',			default='/depot/chan129/data/CSNet/BSD300.npy',		help='the directory for training data')
parser.add_argument('--valid_dir',	dest='valid_dir',			default='/depot/chan129/data/CSNet/Kodak',		help='the directory for validation data')
parser.add_argument('--ckpt_dir',	dest='ckpt_dir',			default='/home/choi240/CSNet/trained_model',		help='the directory for meta file')
args	= parser.parse_args()


def rednet(x0, reUse):
	
	with tf.variable_scope('rednet', reuse=reUse):
		
		x1	= tf.contrib.layers.conv2d		(x0,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv1')
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
		
		y1	= tf.contrib.layers.conv2d_transpose	(x15,	64,	[3,3],	activation_fn=None,		scope='deconv1')
		y1	= tf.nn.relu(tf.add(y1, x14))
		y2	= tf.contrib.layers.conv2d_transpose	(y1,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv2')
		y3	= tf.contrib.layers.conv2d_transpose	(y2,	64,	[3,3],	activation_fn=None,		scope='deconv3')
		y3	= tf.nn.relu(tf.add(y3, x12))
		y4	= tf.contrib.layers.conv2d_transpose	(y3,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv4')
		y5	= tf.contrib.layers.conv2d_transpose	(y4,	64,	[3,3],	activation_fn=None,		scope='deconv5')
		y5	= tf.nn.relu(tf.add(y5, x10))
		y6	= tf.contrib.layers.conv2d_transpose	(y5,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv6')
		y7	= tf.contrib.layers.conv2d_transpose	(y6,	64,	[3,3],	activation_fn=None,		scope='deconv7')
		y7	= tf.nn.relu(tf.add(y7, x8))
		y8	= tf.contrib.layers.conv2d_transpose	(y7,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv8')
		y9	= tf.contrib.layers.conv2d_transpose	(y8,	64,	[3,3],	activation_fn=None,		scope='deconv9')
		y9	= tf.nn.relu(tf.add(y9, x6))
		y10	= tf.contrib.layers.conv2d_transpose	(y9,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv10')
		y11	= tf.contrib.layers.conv2d_transpose	(y10,	64,	[3,3],	activation_fn=None,		scope='deconv11')
		y11	= tf.nn.relu(tf.add(y11, x4))
		y12	= tf.contrib.layers.conv2d_transpose	(y11,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv12')
		y13	= tf.contrib.layers.conv2d_transpose	(y12,	64,	[3,3],	activation_fn=None,		scope='deconv13')
		y13	= tf.nn.relu(tf.add(y13, x2))
		y14	= tf.contrib.layers.conv2d_transpose	(y13,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv14')
		y15	= tf.contrib.layers.conv2d_transpose	(y14,	1,	[3,3],	activation_fn=None,		scope='deconv15')
		output	= tf.nn.relu(tf.add(y15, x0))
		
		return output


def main():
	
	start_time_tot	= time.time()
	
	lr		= args.lr
	epochs		= args.epochs
	patch_size	= args.patch_size
	batch_size	= args.batch_size
	
	meta		= "rednet_blind30"
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
	sigmaSet	= range(10, 51, 10)
	data_gt_val	= [load_image(os.path.join(path_val, fname)) for fname in os.listdir(path_val)]
	data_gt_val	= [np.reshape(img, (1, img.shape[0], img.shape[1], 1)) for img in data_gt_val]
	data_no_val	= [[img + sigma/255.0*np.random.normal(size=img.shape) for img in data_gt_val] for sigma in sigmaSet]
	print data_gt.shape
	print("[*] Load data successfully... %.4f secs" % (time.time()-start_time_load))
	
	
	print("[*] Start training")
	start_time	= time.time()
	for e in range(epochs):
		
		# Training
		for i in range(0, data_gt.shape[0], batch_size):
			data	= data_gt[i:i+batch_size]
			noise	= np.random.randint(1, 5, size=(batch_size, 1, 1, 1))*10.0/255.0
			data_no	= data + np.multiply(noise, np.random.normal(size=data.shape))
			sess.run(train_step,	feed_dict={x:data, y:data_no})
		
		# Calculate test PSNR
		PSNR	= [sum([cal_psnr(sess.run(xhat_gen, feed_dict={y_gen:data[j]}), data_gt_val[j]) for j in range(len(data))])/len(data) for data in data_no_val]
		print("Epoch: [%3d/%3d] time: %11.4f, test PSNR: %.4f %.4f %.4f %.4f %.4f" % (e+1, epochs, time.time()-start_time, PSNR[0], PSNR[1], PSNR[2], PSNR[3], PSNR[4]))
		
		# Save the model
		print("[*] Saving model...")
		tf.add_to_collection("activation", xhat_gen)
		saver.save(sess, os.path.join(ckpt_dir, meta))
	
	print("[*] Finish training")
	print("[*] Toal Time: %.4f" % (time.time()-start_time_tot))


if __name__ == "__main__":
	main()

