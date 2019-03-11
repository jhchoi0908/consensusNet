import os, sys, time, argparse
import tensorflow	as tf
import numpy		as np
sys.path.append('/home/choi240/ConsensusNet')
from utils		import *

parser	= argparse.ArgumentParser(description='')
parser.add_argument('--lr',		dest='lr',		type=float,	default=1e-4,						help='learning rate')
parser.add_argument('--epochs',		dest='epochs',		type=int,	default=50,						help='number of epochs')
parser.add_argument('--patch_size',	dest='patch_size',	type=int,	default=64,						help='patch size')
parser.add_argument('--batch_size',	dest='batch_size',	type=int,	default=64,						help='batch size')
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
		
		x16	= tf.contrib.layers.conv2d		(x15,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv16')
		x17	= tf.contrib.layers.conv2d		(x16,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv17')
		x18	= tf.contrib.layers.conv2d		(x17,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv18')
		x19	= tf.contrib.layers.conv2d		(x18,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv19')
		x20	= tf.contrib.layers.conv2d		(x19,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv20')
		x21	= tf.contrib.layers.conv2d		(x20,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv21')
		x22	= tf.contrib.layers.conv2d		(x21,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv22')
		x23	= tf.contrib.layers.conv2d		(x22,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv23')
		x24	= tf.contrib.layers.conv2d		(x23,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv24')
		x25	= tf.contrib.layers.conv2d		(x24,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv25')
		x26	= tf.contrib.layers.conv2d		(x25,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv26')
		x27	= tf.contrib.layers.conv2d		(x26,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv27')
		x28	= tf.contrib.layers.conv2d		(x27,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv28')
		x29	= tf.contrib.layers.conv2d		(x28,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv29')
		x30	= tf.contrib.layers.conv2d		(x29,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv30')
		
		x31	= tf.contrib.layers.conv2d		(x30,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv31')
		x32	= tf.contrib.layers.conv2d		(x31,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv32')
		x33	= tf.contrib.layers.conv2d		(x32,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv33')
		x34	= tf.contrib.layers.conv2d		(x33,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv34')
		x35	= tf.contrib.layers.conv2d		(x34,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv35')
		x36	= tf.contrib.layers.conv2d		(x35,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv36')
		x37	= tf.contrib.layers.conv2d		(x36,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv37')
		x38	= tf.contrib.layers.conv2d		(x37,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv38')
		x39	= tf.contrib.layers.conv2d		(x38,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv39')
		x40	= tf.contrib.layers.conv2d		(x39,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv40')
		x41	= tf.contrib.layers.conv2d		(x40,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv41')
		x42	= tf.contrib.layers.conv2d		(x41,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv42')
		x43	= tf.contrib.layers.conv2d		(x42,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv43')
		x44	= tf.contrib.layers.conv2d		(x43,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv44')
		x45	= tf.contrib.layers.conv2d		(x44,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv45')
		
		x46	= tf.contrib.layers.conv2d		(x45,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv46')
		x47	= tf.contrib.layers.conv2d		(x46,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv47')
		x48	= tf.contrib.layers.conv2d		(x47,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv48')
		x49	= tf.contrib.layers.conv2d		(x48,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv49')
		x50	= tf.contrib.layers.conv2d		(x49,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv50')
		x51	= tf.contrib.layers.conv2d		(x50,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv51')
		x52	= tf.contrib.layers.conv2d		(x51,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv52')
		x53	= tf.contrib.layers.conv2d		(x52,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv53')
		x54	= tf.contrib.layers.conv2d		(x53,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv54')
		x55	= tf.contrib.layers.conv2d		(x54,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv55')
		x56	= tf.contrib.layers.conv2d		(x55,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv56')
		x57	= tf.contrib.layers.conv2d		(x56,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv57')
		x58	= tf.contrib.layers.conv2d		(x57,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv58')
		x59	= tf.contrib.layers.conv2d		(x58,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv59')
		x60	= tf.contrib.layers.conv2d		(x59,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv60')
		
		x61	= tf.contrib.layers.conv2d		(x60,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv61')
		x62	= tf.contrib.layers.conv2d		(x61,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv62')
		x63	= tf.contrib.layers.conv2d		(x62,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv63')
		x64	= tf.contrib.layers.conv2d		(x63,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv64')
		x65	= tf.contrib.layers.conv2d		(x64,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv65')
		x66	= tf.contrib.layers.conv2d		(x65,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv66')
		x67	= tf.contrib.layers.conv2d		(x66,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv67')
		x68	= tf.contrib.layers.conv2d		(x67,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv68')
		x69	= tf.contrib.layers.conv2d		(x68,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv69')
		x70	= tf.contrib.layers.conv2d		(x69,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv70')
		x71	= tf.contrib.layers.conv2d		(x70,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv71')
		x72	= tf.contrib.layers.conv2d		(x71,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv72')
		x73	= tf.contrib.layers.conv2d		(x72,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv73')
		x74	= tf.contrib.layers.conv2d		(x73,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv74')
		x75	= tf.contrib.layers.conv2d		(x74,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='conv75')

		y1	= tf.contrib.layers.conv2d_transpose	(x75,	64,	[3,3],	activation_fn=None,		scope='deconv1')
		y1	= tf.nn.relu(tf.add(y1, x74))
		y2	= tf.contrib.layers.conv2d_transpose	(y1,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv2')
		y3	= tf.contrib.layers.conv2d_transpose	(y2,	64,	[3,3],	activation_fn=None,		scope='deconv3')
		y3	= tf.nn.relu(tf.add(y3, x72))
		y4	= tf.contrib.layers.conv2d_transpose	(y3,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv4')
		y5	= tf.contrib.layers.conv2d_transpose	(y4,	64,	[3,3],	activation_fn=None,		scope='deconv5')
		y5	= tf.nn.relu(tf.add(y5, x70))
		y6	= tf.contrib.layers.conv2d_transpose	(y5,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv6')
		y7	= tf.contrib.layers.conv2d_transpose	(y6,	64,	[3,3],	activation_fn=None,		scope='deconv7')
		y7	= tf.nn.relu(tf.add(y7, x68))
		y8	= tf.contrib.layers.conv2d_transpose	(y7,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv8')
		y9	= tf.contrib.layers.conv2d_transpose	(y8,	64,	[3,3],	activation_fn=None,		scope='deconv9')
		y9	= tf.nn.relu(tf.add(y9, x66))
		y10	= tf.contrib.layers.conv2d_transpose	(y9,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv10')
		
		y11	= tf.contrib.layers.conv2d_transpose	(y10,	64,	[3,3],	activation_fn=None,		scope='deconv11')
		y11	= tf.nn.relu(tf.add(y11, x64))
		y12	= tf.contrib.layers.conv2d_transpose	(y11,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv12')
		y13	= tf.contrib.layers.conv2d_transpose	(y12,	64,	[3,3],	activation_fn=None,		scope='deconv13')
		y13	= tf.nn.relu(tf.add(y13, x62))
		y14	= tf.contrib.layers.conv2d_transpose	(y13,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv14')
		y15	= tf.contrib.layers.conv2d_transpose	(y14,	64,	[3,3],	activation_fn=None,		scope='deconv15')
		y15	= tf.nn.relu(tf.add(y15, x60))
		y16	= tf.contrib.layers.conv2d_transpose	(y15,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv16')
		y17	= tf.contrib.layers.conv2d_transpose	(y16,	64,	[3,3],	activation_fn=None,		scope='deconv17')
		y17	= tf.nn.relu(tf.add(y17, x58))
		y18	= tf.contrib.layers.conv2d_transpose	(y17,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv18')
		y19	= tf.contrib.layers.conv2d_transpose	(y18,	64,	[3,3],	activation_fn=None,		scope='deconv19')
		y19	= tf.nn.relu(tf.add(y19, x56))
		y20	= tf.contrib.layers.conv2d_transpose	(y19,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv20')
		
		y21	= tf.contrib.layers.conv2d_transpose	(y20,	64,	[3,3],	activation_fn=None,		scope='deconv21')
		y21	= tf.nn.relu(tf.add(y21, x54))
		y22	= tf.contrib.layers.conv2d_transpose	(y21,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv22')
		y23	= tf.contrib.layers.conv2d_transpose	(y22,	64,	[3,3],	activation_fn=None,		scope='deconv23')
		y23	= tf.nn.relu(tf.add(y23, x52))
		y24	= tf.contrib.layers.conv2d_transpose	(y23,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv24')
		y25	= tf.contrib.layers.conv2d_transpose	(y24,	64,	[3,3],	activation_fn=None,		scope='deconv25')
		y25	= tf.nn.relu(tf.add(y25, x50))
		y26	= tf.contrib.layers.conv2d_transpose	(y25,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv26')
		y27	= tf.contrib.layers.conv2d_transpose	(y26,	64,	[3,3],	activation_fn=None,		scope='deconv27')
		y27	= tf.nn.relu(tf.add(y27, x48))
		y28	= tf.contrib.layers.conv2d_transpose	(y27,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv28')
		y29	= tf.contrib.layers.conv2d_transpose	(y28,	64,	[3,3],	activation_fn=None,		scope='deconv29')
		y29	= tf.nn.relu(tf.add(y29, x46))
		y30	= tf.contrib.layers.conv2d_transpose	(y29,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv30')
		
		y31	= tf.contrib.layers.conv2d_transpose	(y30,	64,	[3,3],	activation_fn=None,		scope='deconv31')
		y31	= tf.nn.relu(tf.add(y31, x44))
		y32	= tf.contrib.layers.conv2d_transpose	(y31,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv32')
		y33	= tf.contrib.layers.conv2d_transpose	(y32,	64,	[3,3],	activation_fn=None,		scope='deconv33')
		y33	= tf.nn.relu(tf.add(y33, x42))
		y34	= tf.contrib.layers.conv2d_transpose	(y33,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv34')
		y35	= tf.contrib.layers.conv2d_transpose	(y34,	64,	[3,3],	activation_fn=None,		scope='deconv35')
		y35	= tf.nn.relu(tf.add(y35, x40))
		y36	= tf.contrib.layers.conv2d_transpose	(y35,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv36')
		y37	= tf.contrib.layers.conv2d_transpose	(y36,	64,	[3,3],	activation_fn=None,		scope='deconv37')
		y37	= tf.nn.relu(tf.add(y37, x38))
		y38	= tf.contrib.layers.conv2d_transpose	(y37,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv38')
		y39	= tf.contrib.layers.conv2d_transpose	(y38,	64,	[3,3],	activation_fn=None,		scope='deconv39')
		y39	= tf.nn.relu(tf.add(y39, x36))
		y40	= tf.contrib.layers.conv2d_transpose	(y39,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv40')
		
		y41	= tf.contrib.layers.conv2d_transpose	(y40,	64,	[3,3],	activation_fn=None,		scope='deconv41')
		y41	= tf.nn.relu(tf.add(y41, x34))
		y42	= tf.contrib.layers.conv2d_transpose	(y41,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv42')
		y43	= tf.contrib.layers.conv2d_transpose	(y42,	64,	[3,3],	activation_fn=None,		scope='deconv43')
		y43	= tf.nn.relu(tf.add(y43, x32))
		y44	= tf.contrib.layers.conv2d_transpose	(y43,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv44')
		y45	= tf.contrib.layers.conv2d_transpose	(y44,	64,	[3,3],	activation_fn=None,		scope='deconv45')
		y45	= tf.nn.relu(tf.add(y45, x30))
		y46	= tf.contrib.layers.conv2d_transpose	(y45,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv46')
		y47	= tf.contrib.layers.conv2d_transpose	(y46,	64,	[3,3],	activation_fn=None,		scope='deconv47')
		y47	= tf.nn.relu(tf.add(y47, x28))
		y48	= tf.contrib.layers.conv2d_transpose	(y47,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv48')
		y49	= tf.contrib.layers.conv2d_transpose	(y48,	64,	[3,3],	activation_fn=None,		scope='deconv49')
		y49	= tf.nn.relu(tf.add(y49, x26))
		y50	= tf.contrib.layers.conv2d_transpose	(y49,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv50')
		
		y51	= tf.contrib.layers.conv2d_transpose	(y50,	64,	[3,3],	activation_fn=None,		scope='deconv51')
		y51	= tf.nn.relu(tf.add(y51, x24))
		y52	= tf.contrib.layers.conv2d_transpose	(y51,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv52')
		y53	= tf.contrib.layers.conv2d_transpose	(y52,	64,	[3,3],	activation_fn=None,		scope='deconv53')
		y53	= tf.nn.relu(tf.add(y53, x22))
		y54	= tf.contrib.layers.conv2d_transpose	(y53,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv54')
		y55	= tf.contrib.layers.conv2d_transpose	(y54,	64,	[3,3],	activation_fn=None,		scope='deconv55')
		y55	= tf.nn.relu(tf.add(y55, x20))
		y56	= tf.contrib.layers.conv2d_transpose	(y55,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv56')
		y57	= tf.contrib.layers.conv2d_transpose	(y56,	64,	[3,3],	activation_fn=None,		scope='deconv57')
		y57	= tf.nn.relu(tf.add(y57, x18))
		y58	= tf.contrib.layers.conv2d_transpose	(y57,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv58')
		y59	= tf.contrib.layers.conv2d_transpose	(y58,	64,	[3,3],	activation_fn=None,		scope='deconv59')
		y59	= tf.nn.relu(tf.add(y59, x16))
		y60	= tf.contrib.layers.conv2d_transpose	(y59,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv60')
		
		y61	= tf.contrib.layers.conv2d_transpose	(y60,	64,	[3,3],	activation_fn=None,		scope='deconv61')
		y61	= tf.nn.relu(tf.add(y61, x14))
		y62	= tf.contrib.layers.conv2d_transpose	(y61,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv62')
		y63	= tf.contrib.layers.conv2d_transpose	(y62,	64,	[3,3],	activation_fn=None,		scope='deconv63')
		y63	= tf.nn.relu(tf.add(y63, x12))
		y64	= tf.contrib.layers.conv2d_transpose	(y63,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv64')
		y65	= tf.contrib.layers.conv2d_transpose	(y64,	64,	[3,3],	activation_fn=None,		scope='deconv65')
		y65	= tf.nn.relu(tf.add(y65, x10))
		y66	= tf.contrib.layers.conv2d_transpose	(y65,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv66')
		y67	= tf.contrib.layers.conv2d_transpose	(y66,	64,	[3,3],	activation_fn=None,		scope='deconv67')
		y67	= tf.nn.relu(tf.add(y67, x8))
		y68	= tf.contrib.layers.conv2d_transpose	(y67,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv68')
		y69	= tf.contrib.layers.conv2d_transpose	(y68,	64,	[3,3],	activation_fn=None,		scope='deconv69')
		y69	= tf.nn.relu(tf.add(y69, x6))
		y70	= tf.contrib.layers.conv2d_transpose	(y69,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv70')
		
		y71	= tf.contrib.layers.conv2d_transpose	(y70,	64,	[3,3],	activation_fn=None,		scope='deconv71')
		y71	= tf.nn.relu(tf.add(y71, x4))
		y72	= tf.contrib.layers.conv2d_transpose	(y71,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv72')
		y73	= tf.contrib.layers.conv2d_transpose	(y72,	64,	[3,3],	activation_fn=None,		scope='deconv73')
		y73	= tf.nn.relu(tf.add(y73, x2))
		y74	= tf.contrib.layers.conv2d_transpose	(y73,	64,	[3,3],	activation_fn=tf.nn.relu,	scope='deconv74')
		y75	= tf.contrib.layers.conv2d_transpose	(y74,	1,	[3,3],	activation_fn=None,		scope='deconv75')
		output	= tf.nn.relu(tf.add(y75, x0))
		
		return output


def main():
	
	start_time_tot	= time.time()
	
	lr		= args.lr
	epochs		= args.epochs
	patch_size	= args.patch_size
	batch_size	= args.batch_size
	
	meta		= "rednet_blind150"
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

