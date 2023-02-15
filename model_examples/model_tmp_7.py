
import tensorflow as tf
import numpy as np
import math
maximum_elemts = 300000000
locally_connected_implementation = 2

def get_new_weigths(layer, new_raw_weigths):
	new_raw_weigths = new_raw_weigths
	weights = layer.get_weights()
	if type(weights) == list:
		weights = weights[0]
	original_shape = np.shape(weights)
	flatten_weights = np.reshape(weights, (-1))
	multiplier = math.ceil( len(flatten_weights)/len(new_raw_weigths) )
	tmp_new_weigths =  np.tile(new_raw_weigths, multiplier) 
	new_weigths_unshaped = tmp_new_weigths[:len(flatten_weights)]
	new_weighs = np.reshape(new_weigths_unshaped, original_shape)
	return new_weighs

class L_none(tf.keras.layers.Layer):
	def __init__(self):
		super(L_none, self).__init__()
	def call(self, inputs):
		return inputs
layer_none = L_none()

	
class L6(tf.keras.layers.Layer):
	def __init__(self):
		super(L6, self).__init__()
	def build(self, input_shape):
		inp_lsit = input_shape.as_list()
		n = len(inp_lsit)
		self.check_3 = False
		self.check_2 = False
		self.kernel_size = 0
		self.strides = 0
		self.filters_par = 0
		if maximum_elemts < self.filters_par:
			self.filters_par = maximum_elemts
		if n == 3 and False:
			self.check_3 = True
			if inp_lsit[-2] < self.kernel_size:
				self.kernel_size = inp_lsit[-2]
			if inp_lsit[-2] < self.strides:
				self.strides = inp_lsit[-2]
		elif n == 2 and False:
			self.check_2 = True
			if inp_lsit[-1] < self.kernel_size:
				self.kernel_size = inp_lsit[-1]
			if inp_lsit[-1] < self.strides:
				self.strides = inp_lsit[-1]
		if self.check_3 == True:
			self.i_layer = tf.keras.layers.Conv1D(filters=self.filters_par, kernel_size=self.kernel_size, strides=self.strides, padding='valid')
		elif self.check_2 == True:
			self.i_layer = tf.keras.layers.Conv1D(filters=self.filters_par, kernel_size=self.kernel_size, strides=self.strides, padding='valid')
	def call(self, inputs):
		if self.check_3 == True:
			return self.i_layer(inputs)
		elif self.check_2 == True:
			return self.i_layer( tf.expand_dims(inputs, -1) )
		return inputs
	
class L1(tf.keras.layers.Layer):
	def __init__(self):
		super(L1, self).__init__()
	def call(self, inputs):
		return inputs
			
class L7(tf.keras.layers.Layer):
	def __init__(self):
		super(L7, self).__init__()
	def call(self, inputs):
		return inputs
			
class L5(tf.keras.layers.Layer):
	def __init__(self):
		super(L5, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.002)
		constrain = None

		
		self.w = self.add_weight(
			shape=(input_shape[-1], 1),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		if False:
			self.w = tf.broadcast_to(self.w, shape=input_shape)
		

	def call(self, inputs):
		return (tf.matmul(inputs, self.w))
			
class Strong_concat(tf.keras.layers.Layer):
	def __init__(self):
		super(Strong_concat, self).__init__()
	def build(self, input_shape):
		k_max = 1
		maximum_shape = []
		for layer_raw in input_shape:
			shape_i = layer_raw.as_list()[1:]
			if len(shape_i)>k_max:
				k_max = len(shape_i)
			for ax_ind, axi in enumerate(reversed(shape_i)):
				if len(maximum_shape) >= ax_ind+1:
					if axi > maximum_shape[ax_ind]:
						 maximum_shape[ax_ind] = axi
				else:
					maximum_shape.append(axi)
		

		self.batch_size = input_shape[0].as_list()[0]
		self.maximum_shape = list(reversed(maximum_shape))
		self.maximum_shape_with_bacth = [self.batch_size] + self.maximum_shape
		self.paddings = []
		self.shapes = []
		self.max_shape_len = len(self.maximum_shape_with_bacth)
		self.shape_len=[]
		self.add_shapes = []
		

		for layer_raw in input_shape:
			shape_i = layer_raw.as_list()[1:]
			self.shapes.append(layer_raw.as_list())
			self.paddings.append([])

			self.shape_len.append(len(layer_raw.as_list()))
			self.add_shapes.append(self.max_shape_len-self.shape_len[-1])

			maximum_shape_i = self.maximum_shape[-len(shape_i):]
			for axis_indx, axi in enumerate(shape_i):
				self.paddings[-1].append([0, maximum_shape_i[axis_indx]-axi])
			self.paddings[-1].insert(0, [0, 0])

	def call(self, inputs):
		new_concat_input = []
		for layer_raw, pad, shp, shp_l, ad_shp in zip(inputs, self.paddings, self.shapes, self.shape_len, self.add_shapes):
			layer_pad_out = tf.pad(layer_raw, pad)
			if shp_l == self.max_shape_len:
				new_concat_input.append( layer_pad_out )
			else:
				for i in range(ad_shp):
					layer_pad_out = tf.expand_dims(layer_pad_out, axis=1)
					layer_pad_out = tf.repeat(layer_pad_out, repeats=self.maximum_shape_with_bacth[-i-2], axis=1)
				new_concat_input.append( layer_pad_out)
		layer_strong_concat = new_concat_input[0]
		for add_inp in new_concat_input[1:]:
			layer_strong_concat = tf.add(layer_strong_concat, add_inp)
		return layer_strong_concat
	
set_weigths = {}
	
layer_1_out = tf.keras.Input(shape=[28, 28], name="1, Input", batch_size=32)
		

layer_7 = L7()
layer_7_out = layer_7(layer_1_out)

layer_5 = L5()
try:
	try:
		layer_5_concat = tf.keras.layers.concatenate([layer_1_out, layer_7_out])
		layer_5_out = layer_5(layer_5_concat)
	except ValueError:
		layer_5_concat = tf.keras.layers.concatenate([layer_1_out, layer_7_out], axis=-2)
		layer_5_out = layer_5(layer_5_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_5 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_7_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_5_concat = strong_concat_5(new_list_of_inputs)
	layer_5_out = layer_5(layer_5_concat)
if layer_5.count_params() != 0:
	if 5 in set_weigths:
		if len(layer_5.get_weights()) > 1:
			new_w = [get_new_weigths(layer_5, set_weigths[5])]+layer_5.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_5, set_weigths[5])]
		layer_5.set_weights(new_w)


try:
	count = 1
	for s in layer_5_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #5:',count)
except AttributeError:
	for i in layer_5_out:
		count = 1
		for s in layer_5_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #5:',count)



layer_6 = L6()
try:
	try:
		layer_6_concat = tf.keras.layers.concatenate([layer_1_out, layer_7_out])
		layer_6_out = layer_6(layer_6_concat)
	except ValueError:
		layer_6_concat = tf.keras.layers.concatenate([layer_1_out, layer_7_out], axis=-2)
		layer_6_out = layer_6(layer_6_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_6 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_7_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_6_concat = strong_concat_6(new_list_of_inputs)
	layer_6_out = layer_6(layer_6_concat)
if layer_6.count_params() != 0:
	if 6 in set_weigths:
		if len(layer_6.get_weights()) > 1:
			new_w = [get_new_weigths(layer_6, set_weigths[6])]+layer_6.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_6, set_weigths[6])]
		layer_6.set_weights(new_w)


try:
	count = 1
	for s in layer_6_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #6:',count)
except AttributeError:
	for i in layer_6_out:
		count = 1
		for s in layer_6_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #6:',count)



layer_5_out = tf.keras.layers.Flatten()(layer_5_out)
layer_5_out = tf.keras.layers.Dense(10, activation='sigmoid')(layer_5_out)
model = tf.keras.Model(
    inputs=[layer_1_out],
    outputs=[layer_5_out],
)

'''
model.summary()
tf.keras.utils.plot_model(model, "random_model_99.png",show_shapes=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer='adam',
			              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
			              metrics=['accuracy'])
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#a = model(train_images[0:32])
try:
	tf.config.run_functions_eagerly(False)
	model.fit(train_images, train_labels, epochs=1, batch_size=32)
except ValueError:
	tf.config.run_functions_eagerly(True)
	model.fit(train_images, train_labels, epochs=1, batch_size=32)
'''
	