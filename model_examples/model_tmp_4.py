
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

	
class L29(tf.keras.layers.Layer):
	def __init__(self):
		super(L29, self).__init__()
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
	
class L5(tf.keras.layers.Layer):
	def __init__(self):
		super(L5, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L1L2(l1=0.0, l2=0.0)
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
			
class L6(tf.keras.layers.Layer):
	def __init__(self):
		super(L6, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L1L2(l1=0.0, l2=0.0)
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
			
class L7(tf.keras.layers.Layer):
	def __init__(self):
		super(L7, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L1L2(l1=0.0, l2=0.0)
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
			
class L8(tf.keras.layers.Layer):
	def __init__(self):
		super(L8, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L1L2(l1=0.0, l2=0.0)
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
			
class L9(tf.keras.layers.Layer):
	def __init__(self):
		super(L9, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L1L2(l1=0.0, l2=0.0)
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
			
class L13(tf.keras.layers.Layer):
	def __init__(self):
		super(L13, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L1L2(l1=0.0, l2=0.0)
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
			
class L30(tf.keras.layers.Layer):
	def __init__(self):
		super(L30, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L1L2(l1=0.0, l2=0.0)
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
			
class L37(tf.keras.layers.Layer):
	def __init__(self):
		super(L37, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L1L2(l1=0.0, l2=0.0)
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
			
class L42(tf.keras.layers.Layer):
	def __init__(self):
		super(L42, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L1L2(l1=0.0, l2=0.0)
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
			
class L47(tf.keras.layers.Layer):
	def __init__(self):
		super(L47, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L1L2(l1=0.0, l2=0.0)
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
			
class L50(tf.keras.layers.Layer):
	def __init__(self):
		super(L50, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L1L2(l1=0.0, l2=0.0)
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
			
class L11(tf.keras.layers.Layer):
	def __init__(self):
		super(L11, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.067)
		constrain = None

		
		self.w = self.add_weight(
			shape=(1,),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		if True:
			self.w = tf.broadcast_to(self.w, shape=input_shape)
		

	def call(self, inputs):
		return (tf.add(inputs, self.w))
			
class L12(tf.keras.layers.Layer):
	def __init__(self):
		super(L12, self).__init__()
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
	
class L28(tf.keras.layers.Layer):
	def __init__(self):
		super(L28, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.067)
		constrain = None

		
		self.w = self.add_weight(
			shape=(1,),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		if True:
			self.w = tf.broadcast_to(self.w, shape=input_shape)
		

	def call(self, inputs):
		return (tf.add(inputs, self.w))
			
class L35(tf.keras.layers.Layer):
	def __init__(self):
		super(L35, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.067)
		constrain = None

		
		self.w = self.add_weight(
			shape=(1,),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		if True:
			self.w = tf.broadcast_to(self.w, shape=input_shape)
		

	def call(self, inputs):
		return (tf.add(inputs, self.w))
			
class L36(tf.keras.layers.Layer):
	def __init__(self):
		super(L36, self).__init__()
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
	
class L40(tf.keras.layers.Layer):
	def __init__(self):
		super(L40, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.067)
		constrain = None

		
		self.w = self.add_weight(
			shape=(1,),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		if True:
			self.w = tf.broadcast_to(self.w, shape=input_shape)
		

	def call(self, inputs):
		return (tf.add(inputs, self.w))
			
class L41(tf.keras.layers.Layer):
	def __init__(self):
		super(L41, self).__init__()
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
	
class L45(tf.keras.layers.Layer):
	def __init__(self):
		super(L45, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.067)
		constrain = None

		
		self.w = self.add_weight(
			shape=(1,),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		if True:
			self.w = tf.broadcast_to(self.w, shape=input_shape)
		

	def call(self, inputs):
		return (tf.add(inputs, self.w))
			
class L46(tf.keras.layers.Layer):
	def __init__(self):
		super(L46, self).__init__()
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
	
class L48(tf.keras.layers.Layer):
	def __init__(self):
		super(L48, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.067)
		constrain = None

		
		self.w = self.add_weight(
			shape=(1,),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		if True:
			self.w = tf.broadcast_to(self.w, shape=input_shape)
		

	def call(self, inputs):
		return (tf.add(inputs, self.w))
			
class L49(tf.keras.layers.Layer):
	def __init__(self):
		super(L49, self).__init__()
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
	
layer_5_out = tf.keras.Input(shape=[28, 28], name="5, Input", batch_size=32)
		

layer_37 = L37()
layer_37_out = layer_37(layer_5_out)

layer_6 = L6()
try:
	try:
		layer_6_concat = tf.keras.layers.concatenate([layer_5_out, layer_37_out])
		layer_6_out = layer_6(layer_6_concat)
	except ValueError:
		layer_6_concat = tf.keras.layers.concatenate([layer_5_out, layer_37_out], axis=-2)
		layer_6_out = layer_6(layer_6_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_6 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_5_out, layer_37_out]:
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



layer_7 = L7()
try:
	try:
		layer_7_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_37_out])
		layer_7_out = layer_7(layer_7_concat)
	except ValueError:
		layer_7_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_37_out], axis=-2)
		layer_7_out = layer_7(layer_7_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_7 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_5_out, layer_6_out, layer_37_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_7_concat = strong_concat_7(new_list_of_inputs)
	layer_7_out = layer_7(layer_7_concat)
if layer_7.count_params() != 0:
	if 7 in set_weigths:
		if len(layer_7.get_weights()) > 1:
			new_w = [get_new_weigths(layer_7, set_weigths[7])]+layer_7.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_7, set_weigths[7])]
		layer_7.set_weights(new_w)


try:
	count = 1
	for s in layer_7_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #7:',count)
except AttributeError:
	for i in layer_7_out:
		count = 1
		for s in layer_7_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #7:',count)



layer_8 = L8()
try:
	try:
		layer_8_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_37_out])
		layer_8_out = layer_8(layer_8_concat)
	except ValueError:
		layer_8_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_37_out], axis=-2)
		layer_8_out = layer_8(layer_8_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_8 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_5_out, layer_6_out, layer_7_out, layer_37_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_8_concat = strong_concat_8(new_list_of_inputs)
	layer_8_out = layer_8(layer_8_concat)
if layer_8.count_params() != 0:
	if 8 in set_weigths:
		if len(layer_8.get_weights()) > 1:
			new_w = [get_new_weigths(layer_8, set_weigths[8])]+layer_8.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_8, set_weigths[8])]
		layer_8.set_weights(new_w)


try:
	count = 1
	for s in layer_8_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #8:',count)
except AttributeError:
	for i in layer_8_out:
		count = 1
		for s in layer_8_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #8:',count)



layer_9 = L9()
try:
	try:
		layer_9_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_37_out])
		layer_9_out = layer_9(layer_9_concat)
	except ValueError:
		layer_9_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_37_out], axis=-2)
		layer_9_out = layer_9(layer_9_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_9 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_37_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_9_concat = strong_concat_9(new_list_of_inputs)
	layer_9_out = layer_9(layer_9_concat)
if layer_9.count_params() != 0:
	if 9 in set_weigths:
		if len(layer_9.get_weights()) > 1:
			new_w = [get_new_weigths(layer_9, set_weigths[9])]+layer_9.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_9, set_weigths[9])]
		layer_9.set_weights(new_w)


try:
	count = 1
	for s in layer_9_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #9:',count)
except AttributeError:
	for i in layer_9_out:
		count = 1
		for s in layer_9_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #9:',count)



layer_42 = L42()
try:
	try:
		layer_42_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_37_out])
		layer_42_out = layer_42(layer_42_concat)
	except ValueError:
		layer_42_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_37_out], axis=-2)
		layer_42_out = layer_42(layer_42_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_42 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_37_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_42_concat = strong_concat_42(new_list_of_inputs)
	layer_42_out = layer_42(layer_42_concat)
if layer_42.count_params() != 0:
	if 42 in set_weigths:
		if len(layer_42.get_weights()) > 1:
			new_w = [get_new_weigths(layer_42, set_weigths[42])]+layer_42.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_42, set_weigths[42])]
		layer_42.set_weights(new_w)


try:
	count = 1
	for s in layer_42_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #42:',count)
except AttributeError:
	for i in layer_42_out:
		count = 1
		for s in layer_42_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #42:',count)



layer_50 = L50()
try:
	try:
		layer_50_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_37_out, layer_42_out])
		layer_50_out = layer_50(layer_50_concat)
	except ValueError:
		layer_50_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_37_out, layer_42_out], axis=-2)
		layer_50_out = layer_50(layer_50_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_50 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_37_out, layer_42_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_50_concat = strong_concat_50(new_list_of_inputs)
	layer_50_out = layer_50(layer_50_concat)
if layer_50.count_params() != 0:
	if 50 in set_weigths:
		if len(layer_50.get_weights()) > 1:
			new_w = [get_new_weigths(layer_50, set_weigths[50])]+layer_50.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_50, set_weigths[50])]
		layer_50.set_weights(new_w)


try:
	count = 1
	for s in layer_50_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #50:',count)
except AttributeError:
	for i in layer_50_out:
		count = 1
		for s in layer_50_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #50:',count)



layer_7 = L7()
try:
	try:
		layer_7_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_37_out])
		layer_7_out = layer_7(layer_7_concat)
	except ValueError:
		layer_7_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_37_out], axis=-2)
		layer_7_out = layer_7(layer_7_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_7 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_5_out, layer_6_out, layer_37_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_7_concat = strong_concat_7(new_list_of_inputs)
	layer_7_out = layer_7(layer_7_concat)
if layer_7.count_params() != 0:
	if 7 in set_weigths:
		if len(layer_7.get_weights()) > 1:
			new_w = [get_new_weigths(layer_7, set_weigths[7])]+layer_7.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_7, set_weigths[7])]
		layer_7.set_weights(new_w)


try:
	count = 1
	for s in layer_7_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #7:',count)
except AttributeError:
	for i in layer_7_out:
		count = 1
		for s in layer_7_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #7:',count)



layer_8 = L8()
try:
	try:
		layer_8_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_37_out])
		layer_8_out = layer_8(layer_8_concat)
	except ValueError:
		layer_8_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_37_out], axis=-2)
		layer_8_out = layer_8(layer_8_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_8 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_5_out, layer_6_out, layer_7_out, layer_37_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_8_concat = strong_concat_8(new_list_of_inputs)
	layer_8_out = layer_8(layer_8_concat)
if layer_8.count_params() != 0:
	if 8 in set_weigths:
		if len(layer_8.get_weights()) > 1:
			new_w = [get_new_weigths(layer_8, set_weigths[8])]+layer_8.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_8, set_weigths[8])]
		layer_8.set_weights(new_w)


try:
	count = 1
	for s in layer_8_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #8:',count)
except AttributeError:
	for i in layer_8_out:
		count = 1
		for s in layer_8_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #8:',count)



layer_9 = L9()
try:
	try:
		layer_9_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_37_out])
		layer_9_out = layer_9(layer_9_concat)
	except ValueError:
		layer_9_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_37_out], axis=-2)
		layer_9_out = layer_9(layer_9_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_9 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_37_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_9_concat = strong_concat_9(new_list_of_inputs)
	layer_9_out = layer_9(layer_9_concat)
if layer_9.count_params() != 0:
	if 9 in set_weigths:
		if len(layer_9.get_weights()) > 1:
			new_w = [get_new_weigths(layer_9, set_weigths[9])]+layer_9.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_9, set_weigths[9])]
		layer_9.set_weights(new_w)


try:
	count = 1
	for s in layer_9_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #9:',count)
except AttributeError:
	for i in layer_9_out:
		count = 1
		for s in layer_9_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #9:',count)



layer_13 = L13()
try:
	try:
		layer_13_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_37_out, layer_42_out, layer_50_out])
		layer_13_out = layer_13(layer_13_concat)
	except ValueError:
		layer_13_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_37_out, layer_42_out, layer_50_out], axis=-2)
		layer_13_out = layer_13(layer_13_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_13 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_37_out, layer_42_out, layer_50_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_13_concat = strong_concat_13(new_list_of_inputs)
	layer_13_out = layer_13(layer_13_concat)
if layer_13.count_params() != 0:
	if 13 in set_weigths:
		if len(layer_13.get_weights()) > 1:
			new_w = [get_new_weigths(layer_13, set_weigths[13])]+layer_13.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_13, set_weigths[13])]
		layer_13.set_weights(new_w)


try:
	count = 1
	for s in layer_13_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #13:',count)
except AttributeError:
	for i in layer_13_out:
		count = 1
		for s in layer_13_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #13:',count)



layer_30 = L30()
try:
	try:
		layer_30_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_37_out, layer_42_out, layer_50_out])
		layer_30_out = layer_30(layer_30_concat)
	except ValueError:
		layer_30_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_37_out, layer_42_out, layer_50_out], axis=-2)
		layer_30_out = layer_30(layer_30_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_30 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_37_out, layer_42_out, layer_50_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_30_concat = strong_concat_30(new_list_of_inputs)
	layer_30_out = layer_30(layer_30_concat)
if layer_30.count_params() != 0:
	if 30 in set_weigths:
		if len(layer_30.get_weights()) > 1:
			new_w = [get_new_weigths(layer_30, set_weigths[30])]+layer_30.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_30, set_weigths[30])]
		layer_30.set_weights(new_w)


try:
	count = 1
	for s in layer_30_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #30:',count)
except AttributeError:
	for i in layer_30_out:
		count = 1
		for s in layer_30_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #30:',count)



layer_42 = L42()
try:
	try:
		layer_42_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_37_out])
		layer_42_out = layer_42(layer_42_concat)
	except ValueError:
		layer_42_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_37_out], axis=-2)
		layer_42_out = layer_42(layer_42_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_42 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_37_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_42_concat = strong_concat_42(new_list_of_inputs)
	layer_42_out = layer_42(layer_42_concat)
if layer_42.count_params() != 0:
	if 42 in set_weigths:
		if len(layer_42.get_weights()) > 1:
			new_w = [get_new_weigths(layer_42, set_weigths[42])]+layer_42.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_42, set_weigths[42])]
		layer_42.set_weights(new_w)


try:
	count = 1
	for s in layer_42_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #42:',count)
except AttributeError:
	for i in layer_42_out:
		count = 1
		for s in layer_42_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #42:',count)



layer_47 = L47()
try:
	try:
		layer_47_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_50_out])
		layer_47_out = layer_47(layer_47_concat)
	except ValueError:
		layer_47_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_50_out], axis=-2)
		layer_47_out = layer_47(layer_47_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_47 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_50_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_47_concat = strong_concat_47(new_list_of_inputs)
	layer_47_out = layer_47(layer_47_concat)
if layer_47.count_params() != 0:
	if 47 in set_weigths:
		if len(layer_47.get_weights()) > 1:
			new_w = [get_new_weigths(layer_47, set_weigths[47])]+layer_47.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_47, set_weigths[47])]
		layer_47.set_weights(new_w)


try:
	count = 1
	for s in layer_47_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #47:',count)
except AttributeError:
	for i in layer_47_out:
		count = 1
		for s in layer_47_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #47:',count)



layer_48 = L48()
try:
	try:
		layer_48_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out])
		layer_48_out = layer_48(layer_48_concat)
	except ValueError:
		layer_48_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out], axis=-2)
		layer_48_out = layer_48(layer_48_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_48 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_48_concat = strong_concat_48(new_list_of_inputs)
	layer_48_out = layer_48(layer_48_concat)
if layer_48.count_params() != 0:
	if 48 in set_weigths:
		if len(layer_48.get_weights()) > 1:
			new_w = [get_new_weigths(layer_48, set_weigths[48])]+layer_48.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_48, set_weigths[48])]
		layer_48.set_weights(new_w)


try:
	count = 1
	for s in layer_48_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #48:',count)
except AttributeError:
	for i in layer_48_out:
		count = 1
		for s in layer_48_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #48:',count)



layer_49 = L49()
try:
	try:
		layer_49_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out])
		layer_49_out = layer_49(layer_49_concat)
	except ValueError:
		layer_49_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out], axis=-2)
		layer_49_out = layer_49(layer_49_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_49 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_49_concat = strong_concat_49(new_list_of_inputs)
	layer_49_out = layer_49(layer_49_concat)
if layer_49.count_params() != 0:
	if 49 in set_weigths:
		if len(layer_49.get_weights()) > 1:
			new_w = [get_new_weigths(layer_49, set_weigths[49])]+layer_49.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_49, set_weigths[49])]
		layer_49.set_weights(new_w)


try:
	count = 1
	for s in layer_49_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #49:',count)
except AttributeError:
	for i in layer_49_out:
		count = 1
		for s in layer_49_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #49:',count)



layer_50 = L50()
try:
	try:
		layer_50_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_37_out, layer_42_out])
		layer_50_out = layer_50(layer_50_concat)
	except ValueError:
		layer_50_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_37_out, layer_42_out], axis=-2)
		layer_50_out = layer_50(layer_50_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_50 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_37_out, layer_42_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_50_concat = strong_concat_50(new_list_of_inputs)
	layer_50_out = layer_50(layer_50_concat)
if layer_50.count_params() != 0:
	if 50 in set_weigths:
		if len(layer_50.get_weights()) > 1:
			new_w = [get_new_weigths(layer_50, set_weigths[50])]+layer_50.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_50, set_weigths[50])]
		layer_50.set_weights(new_w)


try:
	count = 1
	for s in layer_50_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #50:',count)
except AttributeError:
	for i in layer_50_out:
		count = 1
		for s in layer_50_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #50:',count)



layer_35 = L35()
try:
	try:
		layer_35_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out])
		layer_35_out = layer_35(layer_35_concat)
	except ValueError:
		layer_35_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out], axis=-2)
		layer_35_out = layer_35(layer_35_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_35 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_35_concat = strong_concat_35(new_list_of_inputs)
	layer_35_out = layer_35(layer_35_concat)
if layer_35.count_params() != 0:
	if 35 in set_weigths:
		if len(layer_35.get_weights()) > 1:
			new_w = [get_new_weigths(layer_35, set_weigths[35])]+layer_35.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_35, set_weigths[35])]
		layer_35.set_weights(new_w)


try:
	count = 1
	for s in layer_35_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #35:',count)
except AttributeError:
	for i in layer_35_out:
		count = 1
		for s in layer_35_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #35:',count)



layer_36 = L36()
try:
	try:
		layer_36_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out])
		layer_36_out = layer_36(layer_36_concat)
	except ValueError:
		layer_36_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out], axis=-2)
		layer_36_out = layer_36(layer_36_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_36 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_36_concat = strong_concat_36(new_list_of_inputs)
	layer_36_out = layer_36(layer_36_concat)
if layer_36.count_params() != 0:
	if 36 in set_weigths:
		if len(layer_36.get_weights()) > 1:
			new_w = [get_new_weigths(layer_36, set_weigths[36])]+layer_36.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_36, set_weigths[36])]
		layer_36.set_weights(new_w)


try:
	count = 1
	for s in layer_36_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #36:',count)
except AttributeError:
	for i in layer_36_out:
		count = 1
		for s in layer_36_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #36:',count)



layer_40 = L40()
try:
	try:
		layer_40_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out])
		layer_40_out = layer_40(layer_40_concat)
	except ValueError:
		layer_40_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out], axis=-2)
		layer_40_out = layer_40(layer_40_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_40 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_40_concat = strong_concat_40(new_list_of_inputs)
	layer_40_out = layer_40(layer_40_concat)
if layer_40.count_params() != 0:
	if 40 in set_weigths:
		if len(layer_40.get_weights()) > 1:
			new_w = [get_new_weigths(layer_40, set_weigths[40])]+layer_40.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_40, set_weigths[40])]
		layer_40.set_weights(new_w)


try:
	count = 1
	for s in layer_40_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #40:',count)
except AttributeError:
	for i in layer_40_out:
		count = 1
		for s in layer_40_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #40:',count)



layer_41 = L41()
try:
	try:
		layer_41_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out])
		layer_41_out = layer_41(layer_41_concat)
	except ValueError:
		layer_41_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out], axis=-2)
		layer_41_out = layer_41(layer_41_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_41 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_41_concat = strong_concat_41(new_list_of_inputs)
	layer_41_out = layer_41(layer_41_concat)
if layer_41.count_params() != 0:
	if 41 in set_weigths:
		if len(layer_41.get_weights()) > 1:
			new_w = [get_new_weigths(layer_41, set_weigths[41])]+layer_41.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_41, set_weigths[41])]
		layer_41.set_weights(new_w)


try:
	count = 1
	for s in layer_41_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #41:',count)
except AttributeError:
	for i in layer_41_out:
		count = 1
		for s in layer_41_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #41:',count)



layer_11 = L11()
try:
	try:
		layer_11_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out])
		layer_11_out = layer_11(layer_11_concat)
	except ValueError:
		layer_11_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out], axis=-2)
		layer_11_out = layer_11(layer_11_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_11 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_11_concat = strong_concat_11(new_list_of_inputs)
	layer_11_out = layer_11(layer_11_concat)
if layer_11.count_params() != 0:
	if 11 in set_weigths:
		if len(layer_11.get_weights()) > 1:
			new_w = [get_new_weigths(layer_11, set_weigths[11])]+layer_11.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_11, set_weigths[11])]
		layer_11.set_weights(new_w)


try:
	count = 1
	for s in layer_11_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #11:',count)
except AttributeError:
	for i in layer_11_out:
		count = 1
		for s in layer_11_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #11:',count)



layer_12 = L12()
try:
	try:
		layer_12_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out])
		layer_12_out = layer_12(layer_12_concat)
	except ValueError:
		layer_12_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out], axis=-2)
		layer_12_out = layer_12(layer_12_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_12 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_12_concat = strong_concat_12(new_list_of_inputs)
	layer_12_out = layer_12(layer_12_concat)
if layer_12.count_params() != 0:
	if 12 in set_weigths:
		if len(layer_12.get_weights()) > 1:
			new_w = [get_new_weigths(layer_12, set_weigths[12])]+layer_12.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_12, set_weigths[12])]
		layer_12.set_weights(new_w)


try:
	count = 1
	for s in layer_12_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #12:',count)
except AttributeError:
	for i in layer_12_out:
		count = 1
		for s in layer_12_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #12:',count)



layer_45 = L45()
try:
	try:
		layer_45_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out])
		layer_45_out = layer_45(layer_45_concat)
	except ValueError:
		layer_45_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out], axis=-2)
		layer_45_out = layer_45(layer_45_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_45 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_45_concat = strong_concat_45(new_list_of_inputs)
	layer_45_out = layer_45(layer_45_concat)
if layer_45.count_params() != 0:
	if 45 in set_weigths:
		if len(layer_45.get_weights()) > 1:
			new_w = [get_new_weigths(layer_45, set_weigths[45])]+layer_45.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_45, set_weigths[45])]
		layer_45.set_weights(new_w)


try:
	count = 1
	for s in layer_45_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #45:',count)
except AttributeError:
	for i in layer_45_out:
		count = 1
		for s in layer_45_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #45:',count)



layer_46 = L46()
try:
	try:
		layer_46_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out])
		layer_46_out = layer_46(layer_46_concat)
	except ValueError:
		layer_46_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out], axis=-2)
		layer_46_out = layer_46(layer_46_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_46 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_46_concat = strong_concat_46(new_list_of_inputs)
	layer_46_out = layer_46(layer_46_concat)
if layer_46.count_params() != 0:
	if 46 in set_weigths:
		if len(layer_46.get_weights()) > 1:
			new_w = [get_new_weigths(layer_46, set_weigths[46])]+layer_46.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_46, set_weigths[46])]
		layer_46.set_weights(new_w)


try:
	count = 1
	for s in layer_46_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #46:',count)
except AttributeError:
	for i in layer_46_out:
		count = 1
		for s in layer_46_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #46:',count)



layer_13 = L13()
try:
	try:
		layer_13_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_37_out, layer_42_out, layer_50_out])
		layer_13_out = layer_13(layer_13_concat)
	except ValueError:
		layer_13_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_37_out, layer_42_out, layer_50_out], axis=-2)
		layer_13_out = layer_13(layer_13_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_13 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_37_out, layer_42_out, layer_50_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_13_concat = strong_concat_13(new_list_of_inputs)
	layer_13_out = layer_13(layer_13_concat)
if layer_13.count_params() != 0:
	if 13 in set_weigths:
		if len(layer_13.get_weights()) > 1:
			new_w = [get_new_weigths(layer_13, set_weigths[13])]+layer_13.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_13, set_weigths[13])]
		layer_13.set_weights(new_w)


try:
	count = 1
	for s in layer_13_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #13:',count)
except AttributeError:
	for i in layer_13_out:
		count = 1
		for s in layer_13_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #13:',count)



layer_47 = L47()
try:
	try:
		layer_47_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_50_out])
		layer_47_out = layer_47(layer_47_concat)
	except ValueError:
		layer_47_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_50_out], axis=-2)
		layer_47_out = layer_47(layer_47_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_47 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_50_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_47_concat = strong_concat_47(new_list_of_inputs)
	layer_47_out = layer_47(layer_47_concat)
if layer_47.count_params() != 0:
	if 47 in set_weigths:
		if len(layer_47.get_weights()) > 1:
			new_w = [get_new_weigths(layer_47, set_weigths[47])]+layer_47.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_47, set_weigths[47])]
		layer_47.set_weights(new_w)


try:
	count = 1
	for s in layer_47_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #47:',count)
except AttributeError:
	for i in layer_47_out:
		count = 1
		for s in layer_47_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #47:',count)



layer_48 = L48()
try:
	try:
		layer_48_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out])
		layer_48_out = layer_48(layer_48_concat)
	except ValueError:
		layer_48_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out], axis=-2)
		layer_48_out = layer_48(layer_48_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_48 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_48_concat = strong_concat_48(new_list_of_inputs)
	layer_48_out = layer_48(layer_48_concat)
if layer_48.count_params() != 0:
	if 48 in set_weigths:
		if len(layer_48.get_weights()) > 1:
			new_w = [get_new_weigths(layer_48, set_weigths[48])]+layer_48.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_48, set_weigths[48])]
		layer_48.set_weights(new_w)


try:
	count = 1
	for s in layer_48_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #48:',count)
except AttributeError:
	for i in layer_48_out:
		count = 1
		for s in layer_48_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #48:',count)



layer_49 = L49()
try:
	try:
		layer_49_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out])
		layer_49_out = layer_49(layer_49_concat)
	except ValueError:
		layer_49_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out], axis=-2)
		layer_49_out = layer_49(layer_49_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_49 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_49_concat = strong_concat_49(new_list_of_inputs)
	layer_49_out = layer_49(layer_49_concat)
if layer_49.count_params() != 0:
	if 49 in set_weigths:
		if len(layer_49.get_weights()) > 1:
			new_w = [get_new_weigths(layer_49, set_weigths[49])]+layer_49.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_49, set_weigths[49])]
		layer_49.set_weights(new_w)


try:
	count = 1
	for s in layer_49_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #49:',count)
except AttributeError:
	for i in layer_49_out:
		count = 1
		for s in layer_49_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #49:',count)



layer_28 = L28()
try:
	try:
		layer_28_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out])
		layer_28_out = layer_28(layer_28_concat)
	except ValueError:
		layer_28_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out], axis=-2)
		layer_28_out = layer_28(layer_28_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_28 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_28_concat = strong_concat_28(new_list_of_inputs)
	layer_28_out = layer_28(layer_28_concat)
if layer_28.count_params() != 0:
	if 28 in set_weigths:
		if len(layer_28.get_weights()) > 1:
			new_w = [get_new_weigths(layer_28, set_weigths[28])]+layer_28.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_28, set_weigths[28])]
		layer_28.set_weights(new_w)


try:
	count = 1
	for s in layer_28_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #28:',count)
except AttributeError:
	for i in layer_28_out:
		count = 1
		for s in layer_28_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #28:',count)



layer_29 = L29()
try:
	try:
		layer_29_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out])
		layer_29_out = layer_29(layer_29_concat)
	except ValueError:
		layer_29_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out], axis=-2)
		layer_29_out = layer_29(layer_29_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_29 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_29_concat = strong_concat_29(new_list_of_inputs)
	layer_29_out = layer_29(layer_29_concat)
if layer_29.count_params() != 0:
	if 29 in set_weigths:
		if len(layer_29.get_weights()) > 1:
			new_w = [get_new_weigths(layer_29, set_weigths[29])]+layer_29.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_29, set_weigths[29])]
		layer_29.set_weights(new_w)


try:
	count = 1
	for s in layer_29_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #29:',count)
except AttributeError:
	for i in layer_29_out:
		count = 1
		for s in layer_29_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #29:',count)



layer_30 = L30()
try:
	try:
		layer_30_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_37_out, layer_42_out, layer_50_out])
		layer_30_out = layer_30(layer_30_concat)
	except ValueError:
		layer_30_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_37_out, layer_42_out, layer_50_out], axis=-2)
		layer_30_out = layer_30(layer_30_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_30 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_37_out, layer_42_out, layer_50_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_30_concat = strong_concat_30(new_list_of_inputs)
	layer_30_out = layer_30(layer_30_concat)
if layer_30.count_params() != 0:
	if 30 in set_weigths:
		if len(layer_30.get_weights()) > 1:
			new_w = [get_new_weigths(layer_30, set_weigths[30])]+layer_30.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_30, set_weigths[30])]
		layer_30.set_weights(new_w)


try:
	count = 1
	for s in layer_30_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #30:',count)
except AttributeError:
	for i in layer_30_out:
		count = 1
		for s in layer_30_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #30:',count)



layer_28 = L28()
try:
	try:
		layer_28_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out])
		layer_28_out = layer_28(layer_28_concat)
	except ValueError:
		layer_28_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out], axis=-2)
		layer_28_out = layer_28(layer_28_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_28 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_28_concat = strong_concat_28(new_list_of_inputs)
	layer_28_out = layer_28(layer_28_concat)
if layer_28.count_params() != 0:
	if 28 in set_weigths:
		if len(layer_28.get_weights()) > 1:
			new_w = [get_new_weigths(layer_28, set_weigths[28])]+layer_28.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_28, set_weigths[28])]
		layer_28.set_weights(new_w)


try:
	count = 1
	for s in layer_28_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #28:',count)
except AttributeError:
	for i in layer_28_out:
		count = 1
		for s in layer_28_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #28:',count)



layer_29 = L29()
try:
	try:
		layer_29_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out])
		layer_29_out = layer_29(layer_29_concat)
	except ValueError:
		layer_29_concat = tf.keras.layers.concatenate([layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out], axis=-2)
		layer_29_out = layer_29(layer_29_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_29 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_13_out, layer_30_out, layer_37_out, layer_42_out, layer_47_out, layer_50_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_29_concat = strong_concat_29(new_list_of_inputs)
	layer_29_out = layer_29(layer_29_concat)
if layer_29.count_params() != 0:
	if 29 in set_weigths:
		if len(layer_29.get_weights()) > 1:
			new_w = [get_new_weigths(layer_29, set_weigths[29])]+layer_29.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_29, set_weigths[29])]
		layer_29.set_weights(new_w)


try:
	count = 1
	for s in layer_29_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #29:',count)
except AttributeError:
	for i in layer_29_out:
		count = 1
		for s in layer_29_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #29:',count)



layer_35_out = tf.keras.layers.Flatten()(layer_35_out)
layer_35_out = tf.keras.layers.Dense(10, activation='sigmoid')(layer_35_out)
model = tf.keras.Model(
    inputs=[layer_5_out],
    outputs=[layer_35_out],
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
	