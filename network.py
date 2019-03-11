import tensorflow	as tf
import numpy		as np

class network0(object):
	
	def __init__(self, loc):
		
		self.graph	= tf.Graph()
		self.sess	= tf.Session(graph=self.graph)
		with self.graph.as_default():
			saver	= tf.train.import_meta_graph(loc + '.meta', clear_devices=True)
			saver.restore(self.sess, loc)
			self.activation	= tf.get_collection('activation')[0]
	
	def run(self, data):
		return self.sess.run(self.activation, feed_dict={"y_gen:0": data})


class network_class(object):
        
        def __init__(self, meta):
                self.graph      = tf.Graph()
                self.sess       = tf.Session(graph=self.graph)
                with self.graph.as_default():
                        saver   = tf.train.import_meta_graph(meta + '.meta', clear_devices=True)
                        saver.restore(self.sess, meta)
                        self.activation = tf.get_collection('activation')[0]

        def run(self, data):
                return self.sess.run(self.activation, feed_dict={"y_test:0": data})


class network1(object):
	
	def __init__(self, loc):
		
		self.graph	= tf.Graph()
		self.sess	= tf.Session(graph=self.graph)
		with self.graph.as_default():
			saver	= tf.train.import_meta_graph(loc + '.meta', clear_devices=True)
			saver.restore(self.sess, loc)
			self.activation	= tf.get_collection('activation')[0]
	
	def run(self, y, xhat):
		msehats	= self.sess.run(self.activation, feed_dict={"y:0": y, "xhat:0": xhat, "training:0": False})
		return np.average(msehats).tolist()


class network2(object):
	
	def __init__(self, loc):
		
		self.graph	= tf.Graph()
		self.sess	= tf.Session(graph=self.graph)
		with self.graph.as_default():
			saver	= tf.train.import_meta_graph(loc + '.meta', clear_devices=True)
			saver.restore(self.sess, loc)
			self.activation	= tf.get_collection('activation')[0]
	
	def run(self, y, xhat):
		return self.sess.run(self.activation, feed_dict={"y_test:0": y, "xhat_test:0": xhat})

