# Pytorch tensorflow summary for using tensorboard.
# This utility is motivated by https://github.com/lanpa/tensorboard-pytorch.

import tensorflow as tf

class PytorchSummary(object):
	def __init__(self, log_dir):
		self.sess = tf.Session()
		self.writer = tf.summary.FileWriter(log_dir)
		self.summary_dict = {}
		self.input_dict = {}

	def add_summary(self, summary, feed_dict={}, global_step=0):
		sess = self.sess
		writer = self.writer
		writer.add_summary(sess.run(summary, feed_dict=feed_dict),
			global_step=global_step)

	def add_scalar(self, name, value, global_step=0):
		summary_dict = self.summary_dict
		input_dict = self.input_dict
		if name not in summary_dict:
			input_dict[name] = tf.placeholder(tf.float32, name=name)
			summary_dict[name] = tf.summary.scalar(name, input_dict[name])
		self.add_summary(summary_dict[name], feed_dict={input_dict[name]: value},
			global_step=global_step)

	def add_histogram(self, name, values, global_step=0):
		summary_dict = self.summary_dict
		input_dict = self.input_dict
		if name not in summary_dict:
			input_dict[name] = tf.placeholder(tf.float32, name=name)
			summary_dict[name] = tf.summary.histogram(name, input_dict[name])
		self.add_summary(summary_dict[name], feed_dict={input_dict[name]: values},
			global_step=global_step)
