import tensorflow	as tf
from ops		import *


def model(data, scope, reUse):

	with tf.variable_scope(scope, reuse=reUse):

		with tf.variable_scope('conv1', reuse=reUse):
			h_conv1		= conv_layer	(data,		[3,3, 1, 128])
			h_conv1		= tf.nn.relu(h_conv1)
		
		with tf.variable_scope('conv2', reuse=reUse):
			h_conv2		= conv_layer	(h_conv1,	[3,3,128,128])
			h_conv2		= tf.nn.relu(h_conv2)
		
		with tf.variable_scope('conv3', reuse=reUse):
			h_conv3		= conv_layer	(h_conv2,	[3,3,128,128])
			h_conv3		= tf.nn.relu(h_conv3)
		
		with tf.variable_scope('conv4', reuse=reUse):
			h_conv4		= conv_layer	(h_conv3,	[3,3,128,128])
			h_conv4		= tf.nn.relu(h_conv4)
		
		with tf.variable_scope('conv5', reuse=reUse):
			h_conv5		= conv_layer	(h_conv4,	[3,3,128,128])
			h_conv5		= tf.nn.relu(h_conv5)
		
		with tf.variable_scope('conv6', reuse=reUse):
			h_conv6		= conv_layer	(h_conv5,	[3,3,128,128])
			h_conv6		= tf.nn.relu(h_conv6)
		
		with tf.variable_scope('conv7', reuse=reUse):
			h_conv7		= conv_layer	(h_conv6,	[3,3,128,128])
			h_conv7		= tf.nn.relu(h_conv7)
		
		with tf.variable_scope('conv8', reuse=reUse):
			h_conv8		= conv_layer	(h_conv7,	[3,3,128,128])
			h_conv8		= tf.nn.relu(h_conv8)
		
		with tf.variable_scope('conv9', reuse=reUse):
			h_conv9		= conv_layer	(h_conv8,	[3,3,128,128])
			h_conv9		= tf.nn.relu(h_conv9)
		
		with tf.variable_scope('conv10', reuse=reUse):
			h_conv10	= conv_layer	(h_conv9,	[3,3,128,128])
			h_conv10	= tf.nn.relu(h_conv10)
		
		with tf.variable_scope('conv11', reuse=reUse):
			h_conv11	= conv_layer	(h_conv10,	[3,3,128,128])
			h_conv11	= tf.nn.relu(h_conv11)
		
		with tf.variable_scope('conv12', reuse=reUse):
			h_conv12	= conv_layer	(h_conv11,	[3,3,128,128])
			h_conv12	= tf.nn.relu(h_conv12)
		
		with tf.variable_scope('conv13', reuse=reUse):
			h_conv13	= conv_layer	(h_conv12,	[3,3,128,128])
			h_conv13	= tf.nn.relu(h_conv13)
		
		with tf.variable_scope('conv14', reuse=reUse):
			h_conv14	= conv_layer	(h_conv13,	[3,3,128,128])
			h_conv14	= tf.nn.relu(h_conv14)
		
		with tf.variable_scope('conv15', reuse=reUse):
			h_conv15	= conv_layer	(h_conv14,	[3,3,128,128])
			h_conv15	= tf.nn.relu(h_conv15)
		
		with tf.variable_scope('deconv1', reuse=reUse):
			h_deconv1	= deconv_layer	(h_conv15,	[3,3,128,128],	tf.shape(h_conv14))
			h_deconv1	= tf.nn.relu(h_deconv1)
			h_deconv1	= tf.add(h_deconv1, h_conv14)
			h_deconv1	= tf.nn.relu(h_deconv1)
		
		with tf.variable_scope('deconv2', reuse=reUse):
			h_deconv2	= deconv_layer	(h_deconv1,	[3,3,128,128],	tf.shape(h_conv13))
			h_deconv2	= tf.nn.relu(h_deconv2)
		
		with tf.variable_scope('deconv3', reuse=reUse):
			h_deconv3	= deconv_layer	(h_deconv2,	[3,3,128,128],	tf.shape(h_conv12))
			h_deconv3	= tf.nn.relu(h_deconv3)
			h_deconv3	= tf.add(h_deconv3, h_conv12)
			h_deconv3	= tf.nn.relu(h_deconv3)
		
		with tf.variable_scope('deconv4', reuse=reUse):
			h_deconv4	= deconv_layer	(h_deconv3,	[3,3,128,128],	tf.shape(h_conv11))
			h_deconv4	= tf.nn.relu(h_deconv4)
		
		with tf.variable_scope('deconv5', reuse=reUse):
			h_deconv5	= deconv_layer	(h_deconv4,	[3,3,128,128],	tf.shape(h_conv10))
			h_deconv5	= tf.nn.relu(h_deconv5)
			h_deconv5	= tf.add(h_deconv5, h_conv10)
			h_deconv5	= tf.nn.relu(h_deconv5)
		
		with tf.variable_scope('deconv6', reuse=reUse):
			h_deconv6	= deconv_layer	(h_deconv5,	[3,3,128,128],	tf.shape(h_conv9))
			h_deconv6	= tf.nn.relu(h_deconv6)
		
		with tf.variable_scope('deconv7', reuse=reUse):
			h_deconv7	= deconv_layer	(h_deconv6,	[3,3,128,128],	tf.shape(h_conv8))
			h_deconv7	= tf.nn.relu(h_deconv7)
			h_deconv7	= tf.add(h_deconv7, h_conv8)
			h_deconv7	= tf.nn.relu(h_deconv7)
		
		with tf.variable_scope('deconv8', reuse=reUse):
			h_deconv8	= deconv_layer	(h_deconv7,	[3,3,128,128],	tf.shape(h_conv7))
			h_deconv8	= tf.nn.relu(h_deconv8)
		
		with tf.variable_scope('deconv9', reuse=reUse):
			h_deconv9	= deconv_layer	(h_deconv8,	[3,3,128,128],	tf.shape(h_conv6))
			h_deconv9	= tf.nn.relu(h_deconv9)
			h_deconv9	= tf.add(h_deconv9, h_conv6)
			h_deconv9	= tf.nn.relu(h_deconv9)
		
		with tf.variable_scope('deconv10', reuse=reUse):
			h_deconv10	= deconv_layer	(h_deconv9,	[3,3,128,128],	tf.shape(h_conv5))
			h_deconv10	= tf.nn.relu(h_deconv10)
		
		with tf.variable_scope('deconv11', reuse=reUse):
			h_deconv11	= deconv_layer	(h_deconv10,	[3,3,128,128],	tf.shape(h_conv4))
			h_deconv11	= tf.nn.relu(h_deconv11)
			h_deconv11	= tf.add(h_deconv11, h_conv4)
			h_deconv11	= tf.nn.relu(h_deconv11)
		
		with tf.variable_scope('deconv12', reuse=reUse):
			h_deconv12	= deconv_layer	(h_deconv11,	[3,3,128,128],	tf.shape(h_conv3))
			h_deconv12	= tf.nn.relu(h_deconv12)
		
		with tf.variable_scope('deconv13', reuse=reUse):
			h_deconv13	= deconv_layer	(h_deconv12,	[3,3,128,128],	tf.shape(h_conv2))
			h_deconv13	= tf.nn.relu(h_deconv13)
			h_deconv13	= tf.add(h_deconv13, h_conv2)
			h_deconv13	= tf.nn.relu(h_deconv13)
		
		with tf.variable_scope('deconv14', reuse=reUse):
			h_deconv14	= deconv_layer	(h_deconv13,	[3,3,128,128],	tf.shape(h_conv1))
			h_deconv14	= tf.nn.relu(h_deconv14)
		
		with tf.variable_scope('deconv15', reuse=reUse):
			h		= deconv_layer	(h_deconv14,	[3,3, 1,128],	tf.shape(data))
			h		= tf.add(h, data)
		
		return h

