import tensorflow	as tf
import math


def conv_layer(x, weight_shape):
	W	= tf.get_variable('weights',	weight_shape,		initializer=tf.truncated_normal_initializer(stddev=0.05))
	b	= tf.get_variable('biases',	[weight_shape[-1]],	initializer=tf.constant_initializer(0.0))
	var	= tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
	var	= tf.nn.bias_add(var, b)
	return var


def deconv_layer( x, weight_shape, output_shape):
	W	= tf.get_variable('weights',	weight_shape,		initializer=tf.truncated_normal_initializer(stddev=0.05))
	b	= tf.get_variable('biases',	[weight_shape[-2]],	initializer=tf.constant_initializer(0.0))
	var	= tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=[1,1,1,1], padding="SAME")
	var	= tf.nn.bias_add(var, b)
	return var


def bn_layer(x, output_dim):
	alpha	= tf.get_variable('alpha',	[1, output_dim],	initializer=tf.truncated_normal_initializer(stddev=0.01))
	beta	= tf.get_variable('beta' ,	[1, output_dim],	initializer=tf.constant_initializer(0.0))
	mean, var	= tf.nn.moments(x,	[0, 1, 2])
	output	= tf.nn.batch_normalization(x, mean, var, beta, alpha, variance_epsilon=1e-5)
	return output
