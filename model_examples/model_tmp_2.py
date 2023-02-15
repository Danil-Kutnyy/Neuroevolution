
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

	
class L344(tf.keras.layers.Layer):
	def __init__(self):
		super(L344, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L3(tf.keras.layers.Layer):
	def __init__(self):
		super(L3, self).__init__()
	def build(self, input_shape):
		if len(input_shape.as_list()) < 3:
			def i_o(x):
				return x
			self.m_op = i_o
		elif len(input_shape.as_list()) < 4:
			if tf.keras.layers.GlobalMaxPool2D == tf.keras.layers.GlobalMaxPool2D:
				self.m_op =  tf.keras.layers.GlobalMaxPool1D()
			elif tf.keras.layers.GlobalMaxPool2D == tf.keras.layers.GlobalAveragePooling2D:
				self.m_op =  tf.keras.layers.GlobalAveragePooling1D()
			else:
				self.m_op =  tf.keras.layers.GlobalMaxPool2D()
		else:
			self.m_op =  tf.keras.layers.GlobalMaxPool2D()

	def call(self, inputs):
		return self.m_op(inputs)
			
class L4(tf.keras.layers.Layer):
	def __init__(self):
		super(L4, self).__init__()
	def build(self, input_shape):
		if len(input_shape.as_list()) < 3:
			def i_o(x):
				return x
			self.m_op = i_o
		elif len(input_shape.as_list()) < 4:
			if tf.keras.layers.GlobalMaxPool2D == tf.keras.layers.GlobalMaxPool2D:
				self.m_op =  tf.keras.layers.GlobalMaxPool1D()
			elif tf.keras.layers.GlobalMaxPool2D == tf.keras.layers.GlobalAveragePooling2D:
				self.m_op =  tf.keras.layers.GlobalAveragePooling1D()
			else:
				self.m_op =  tf.keras.layers.GlobalMaxPool2D()
		else:
			self.m_op =  tf.keras.layers.GlobalMaxPool2D()

	def call(self, inputs):
		return self.m_op(inputs)
			
class L173(tf.keras.layers.Layer):
	def __init__(self):
		super(L173, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L174(tf.keras.layers.Layer):
	def __init__(self):
		super(L174, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L175(tf.keras.layers.Layer):
	def __init__(self):
		super(L175, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L176(tf.keras.layers.Layer):
	def __init__(self):
		super(L176, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L177(tf.keras.layers.Layer):
	def __init__(self):
		super(L177, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L178(tf.keras.layers.Layer):
	def __init__(self):
		super(L178, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L179(tf.keras.layers.Layer):
	def __init__(self):
		super(L179, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L180(tf.keras.layers.Layer):
	def __init__(self):
		super(L180, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L181(tf.keras.layers.Layer):
	def __init__(self):
		super(L181, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L182(tf.keras.layers.Layer):
	def __init__(self):
		super(L182, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L183(tf.keras.layers.Layer):
	def __init__(self):
		super(L183, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L184(tf.keras.layers.Layer):
	def __init__(self):
		super(L184, self).__init__()
	def build(self, input_shape):
		if len(input_shape.as_list()) < 3:
			def i_o(x):
				return x
			self.m_op = i_o
		elif len(input_shape.as_list()) < 4:
			if tf.keras.layers.GlobalMaxPool2D == tf.keras.layers.GlobalMaxPool2D:
				self.m_op =  tf.keras.layers.GlobalMaxPool1D()
			elif tf.keras.layers.GlobalMaxPool2D == tf.keras.layers.GlobalAveragePooling2D:
				self.m_op =  tf.keras.layers.GlobalAveragePooling1D()
			else:
				self.m_op =  tf.keras.layers.GlobalMaxPool2D()
		else:
			self.m_op =  tf.keras.layers.GlobalMaxPool2D()

	def call(self, inputs):
		return self.m_op(inputs)
			
class L185(tf.keras.layers.Layer):
	def __init__(self):
		super(L185, self).__init__()
	def build(self, input_shape):
		if len(input_shape.as_list()) < 3:
			def i_o(x):
				return x
			self.m_op = i_o
		elif len(input_shape.as_list()) < 4:
			if tf.keras.layers.GlobalMaxPool2D == tf.keras.layers.GlobalMaxPool2D:
				self.m_op =  tf.keras.layers.GlobalMaxPool1D()
			elif tf.keras.layers.GlobalMaxPool2D == tf.keras.layers.GlobalAveragePooling2D:
				self.m_op =  tf.keras.layers.GlobalAveragePooling1D()
			else:
				self.m_op =  tf.keras.layers.GlobalMaxPool2D()
		else:
			self.m_op =  tf.keras.layers.GlobalMaxPool2D()

	def call(self, inputs):
		return self.m_op(inputs)
			
class L345(tf.keras.layers.Layer):
	def __init__(self):
		super(L345, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L346(tf.keras.layers.Layer):
	def __init__(self):
		super(L346, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L347(tf.keras.layers.Layer):
	def __init__(self):
		super(L347, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L348(tf.keras.layers.Layer):
	def __init__(self):
		super(L348, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L349(tf.keras.layers.Layer):
	def __init__(self):
		super(L349, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L350(tf.keras.layers.Layer):
	def __init__(self):
		super(L350, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L351(tf.keras.layers.Layer):
	def __init__(self):
		super(L351, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L352(tf.keras.layers.Layer):
	def __init__(self):
		super(L352, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L353(tf.keras.layers.Layer):
	def __init__(self):
		super(L353, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L354(tf.keras.layers.Layer):
	def __init__(self):
		super(L354, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L355(tf.keras.layers.Layer):
	def __init__(self):
		super(L355, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L356(tf.keras.layers.Layer):
	def __init__(self):
		super(L356, self).__init__()
	def build(self, input_shape):
		if len(input_shape.as_list()) < 3:
			def i_o(x):
				return x
			self.m_op = i_o
		elif len(input_shape.as_list()) < 4:
			if tf.keras.layers.GlobalMaxPool2D == tf.keras.layers.GlobalMaxPool2D:
				self.m_op =  tf.keras.layers.GlobalMaxPool1D()
			elif tf.keras.layers.GlobalMaxPool2D == tf.keras.layers.GlobalAveragePooling2D:
				self.m_op =  tf.keras.layers.GlobalAveragePooling1D()
			else:
				self.m_op =  tf.keras.layers.GlobalMaxPool2D()
		else:
			self.m_op =  tf.keras.layers.GlobalMaxPool2D()

	def call(self, inputs):
		return self.m_op(inputs)
			
class L357(tf.keras.layers.Layer):
	def __init__(self):
		super(L357, self).__init__()
	def build(self, input_shape):
		if len(input_shape.as_list()) < 3:
			def i_o(x):
				return x
			self.m_op = i_o
		elif len(input_shape.as_list()) < 4:
			if tf.keras.layers.GlobalMaxPool2D == tf.keras.layers.GlobalMaxPool2D:
				self.m_op =  tf.keras.layers.GlobalMaxPool1D()
			elif tf.keras.layers.GlobalMaxPool2D == tf.keras.layers.GlobalAveragePooling2D:
				self.m_op =  tf.keras.layers.GlobalAveragePooling1D()
			else:
				self.m_op =  tf.keras.layers.GlobalMaxPool2D()
		else:
			self.m_op =  tf.keras.layers.GlobalMaxPool2D()

	def call(self, inputs):
		return self.m_op(inputs)
			
class L75(tf.keras.layers.Layer):
	def __init__(self):
		super(L75, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L76(tf.keras.layers.Layer):
	def __init__(self):
		super(L76, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L77(tf.keras.layers.Layer):
	def __init__(self):
		super(L77, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L78(tf.keras.layers.Layer):
	def __init__(self):
		super(L78, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L79(tf.keras.layers.Layer):
	def __init__(self):
		super(L79, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L80(tf.keras.layers.Layer):
	def __init__(self):
		super(L80, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L81(tf.keras.layers.Layer):
	def __init__(self):
		super(L81, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L82(tf.keras.layers.Layer):
	def __init__(self):
		super(L82, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L83(tf.keras.layers.Layer):
	def __init__(self):
		super(L83, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L84(tf.keras.layers.Layer):
	def __init__(self):
		super(L84, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L85(tf.keras.layers.Layer):
	def __init__(self):
		super(L85, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L86(tf.keras.layers.Layer):
	def __init__(self):
		super(L86, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L87(tf.keras.layers.Layer):
	def __init__(self):
		super(L87, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L88(tf.keras.layers.Layer):
	def __init__(self):
		super(L88, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L89(tf.keras.layers.Layer):
	def __init__(self):
		super(L89, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L90(tf.keras.layers.Layer):
	def __init__(self):
		super(L90, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L91(tf.keras.layers.Layer):
	def __init__(self):
		super(L91, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L92(tf.keras.layers.Layer):
	def __init__(self):
		super(L92, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L93(tf.keras.layers.Layer):
	def __init__(self):
		super(L93, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L94(tf.keras.layers.Layer):
	def __init__(self):
		super(L94, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L95(tf.keras.layers.Layer):
	def __init__(self):
		super(L95, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L96(tf.keras.layers.Layer):
	def __init__(self):
		super(L96, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L97(tf.keras.layers.Layer):
	def __init__(self):
		super(L97, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L98(tf.keras.layers.Layer):
	def __init__(self):
		super(L98, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L99(tf.keras.layers.Layer):
	def __init__(self):
		super(L99, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L100(tf.keras.layers.Layer):
	def __init__(self):
		super(L100, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L101(tf.keras.layers.Layer):
	def __init__(self):
		super(L101, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L102(tf.keras.layers.Layer):
	def __init__(self):
		super(L102, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L103(tf.keras.layers.Layer):
	def __init__(self):
		super(L103, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L104(tf.keras.layers.Layer):
	def __init__(self):
		super(L104, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L105(tf.keras.layers.Layer):
	def __init__(self):
		super(L105, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L106(tf.keras.layers.Layer):
	def __init__(self):
		super(L106, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L107(tf.keras.layers.Layer):
	def __init__(self):
		super(L107, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L108(tf.keras.layers.Layer):
	def __init__(self):
		super(L108, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L109(tf.keras.layers.Layer):
	def __init__(self):
		super(L109, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L110(tf.keras.layers.Layer):
	def __init__(self):
		super(L110, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L111(tf.keras.layers.Layer):
	def __init__(self):
		super(L111, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L112(tf.keras.layers.Layer):
	def __init__(self):
		super(L112, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L113(tf.keras.layers.Layer):
	def __init__(self):
		super(L113, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L114(tf.keras.layers.Layer):
	def __init__(self):
		super(L114, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L115(tf.keras.layers.Layer):
	def __init__(self):
		super(L115, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L116(tf.keras.layers.Layer):
	def __init__(self):
		super(L116, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L117(tf.keras.layers.Layer):
	def __init__(self):
		super(L117, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L118(tf.keras.layers.Layer):
	def __init__(self):
		super(L118, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L119(tf.keras.layers.Layer):
	def __init__(self):
		super(L119, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L120(tf.keras.layers.Layer):
	def __init__(self):
		super(L120, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L121(tf.keras.layers.Layer):
	def __init__(self):
		super(L121, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L122(tf.keras.layers.Layer):
	def __init__(self):
		super(L122, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L123(tf.keras.layers.Layer):
	def __init__(self):
		super(L123, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L124(tf.keras.layers.Layer):
	def __init__(self):
		super(L124, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L125(tf.keras.layers.Layer):
	def __init__(self):
		super(L125, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L126(tf.keras.layers.Layer):
	def __init__(self):
		super(L126, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L127(tf.keras.layers.Layer):
	def __init__(self):
		super(L127, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L128(tf.keras.layers.Layer):
	def __init__(self):
		super(L128, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L129(tf.keras.layers.Layer):
	def __init__(self):
		super(L129, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L130(tf.keras.layers.Layer):
	def __init__(self):
		super(L130, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L131(tf.keras.layers.Layer):
	def __init__(self):
		super(L131, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L132(tf.keras.layers.Layer):
	def __init__(self):
		super(L132, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L133(tf.keras.layers.Layer):
	def __init__(self):
		super(L133, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L134(tf.keras.layers.Layer):
	def __init__(self):
		super(L134, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L135(tf.keras.layers.Layer):
	def __init__(self):
		super(L135, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L136(tf.keras.layers.Layer):
	def __init__(self):
		super(L136, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L137(tf.keras.layers.Layer):
	def __init__(self):
		super(L137, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L138(tf.keras.layers.Layer):
	def __init__(self):
		super(L138, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L139(tf.keras.layers.Layer):
	def __init__(self):
		super(L139, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L140(tf.keras.layers.Layer):
	def __init__(self):
		super(L140, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L141(tf.keras.layers.Layer):
	def __init__(self):
		super(L141, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L142(tf.keras.layers.Layer):
	def __init__(self):
		super(L142, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L143(tf.keras.layers.Layer):
	def __init__(self):
		super(L143, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L144(tf.keras.layers.Layer):
	def __init__(self):
		super(L144, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L145(tf.keras.layers.Layer):
	def __init__(self):
		super(L145, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L146(tf.keras.layers.Layer):
	def __init__(self):
		super(L146, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L147(tf.keras.layers.Layer):
	def __init__(self):
		super(L147, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L148(tf.keras.layers.Layer):
	def __init__(self):
		super(L148, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L149(tf.keras.layers.Layer):
	def __init__(self):
		super(L149, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L150(tf.keras.layers.Layer):
	def __init__(self):
		super(L150, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L151(tf.keras.layers.Layer):
	def __init__(self):
		super(L151, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L152(tf.keras.layers.Layer):
	def __init__(self):
		super(L152, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L153(tf.keras.layers.Layer):
	def __init__(self):
		super(L153, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L154(tf.keras.layers.Layer):
	def __init__(self):
		super(L154, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L155(tf.keras.layers.Layer):
	def __init__(self):
		super(L155, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L156(tf.keras.layers.Layer):
	def __init__(self):
		super(L156, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L157(tf.keras.layers.Layer):
	def __init__(self):
		super(L157, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L158(tf.keras.layers.Layer):
	def __init__(self):
		super(L158, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L159(tf.keras.layers.Layer):
	def __init__(self):
		super(L159, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L160(tf.keras.layers.Layer):
	def __init__(self):
		super(L160, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L161(tf.keras.layers.Layer):
	def __init__(self):
		super(L161, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L162(tf.keras.layers.Layer):
	def __init__(self):
		super(L162, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L163(tf.keras.layers.Layer):
	def __init__(self):
		super(L163, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L164(tf.keras.layers.Layer):
	def __init__(self):
		super(L164, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L165(tf.keras.layers.Layer):
	def __init__(self):
		super(L165, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L166(tf.keras.layers.Layer):
	def __init__(self):
		super(L166, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L167(tf.keras.layers.Layer):
	def __init__(self):
		super(L167, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L168(tf.keras.layers.Layer):
	def __init__(self):
		super(L168, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L169(tf.keras.layers.Layer):
	def __init__(self):
		super(L169, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L170(tf.keras.layers.Layer):
	def __init__(self):
		super(L170, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L171(tf.keras.layers.Layer):
	def __init__(self):
		super(L171, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L172(tf.keras.layers.Layer):
	def __init__(self):
		super(L172, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L247(tf.keras.layers.Layer):
	def __init__(self):
		super(L247, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L248(tf.keras.layers.Layer):
	def __init__(self):
		super(L248, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L249(tf.keras.layers.Layer):
	def __init__(self):
		super(L249, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L250(tf.keras.layers.Layer):
	def __init__(self):
		super(L250, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L251(tf.keras.layers.Layer):
	def __init__(self):
		super(L251, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L252(tf.keras.layers.Layer):
	def __init__(self):
		super(L252, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L253(tf.keras.layers.Layer):
	def __init__(self):
		super(L253, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L254(tf.keras.layers.Layer):
	def __init__(self):
		super(L254, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L255(tf.keras.layers.Layer):
	def __init__(self):
		super(L255, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L256(tf.keras.layers.Layer):
	def __init__(self):
		super(L256, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L257(tf.keras.layers.Layer):
	def __init__(self):
		super(L257, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L258(tf.keras.layers.Layer):
	def __init__(self):
		super(L258, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L259(tf.keras.layers.Layer):
	def __init__(self):
		super(L259, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L260(tf.keras.layers.Layer):
	def __init__(self):
		super(L260, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L261(tf.keras.layers.Layer):
	def __init__(self):
		super(L261, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L262(tf.keras.layers.Layer):
	def __init__(self):
		super(L262, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L263(tf.keras.layers.Layer):
	def __init__(self):
		super(L263, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L264(tf.keras.layers.Layer):
	def __init__(self):
		super(L264, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L265(tf.keras.layers.Layer):
	def __init__(self):
		super(L265, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L266(tf.keras.layers.Layer):
	def __init__(self):
		super(L266, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L267(tf.keras.layers.Layer):
	def __init__(self):
		super(L267, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L268(tf.keras.layers.Layer):
	def __init__(self):
		super(L268, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L269(tf.keras.layers.Layer):
	def __init__(self):
		super(L269, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L270(tf.keras.layers.Layer):
	def __init__(self):
		super(L270, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L271(tf.keras.layers.Layer):
	def __init__(self):
		super(L271, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L272(tf.keras.layers.Layer):
	def __init__(self):
		super(L272, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L273(tf.keras.layers.Layer):
	def __init__(self):
		super(L273, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L274(tf.keras.layers.Layer):
	def __init__(self):
		super(L274, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L275(tf.keras.layers.Layer):
	def __init__(self):
		super(L275, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L276(tf.keras.layers.Layer):
	def __init__(self):
		super(L276, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L277(tf.keras.layers.Layer):
	def __init__(self):
		super(L277, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L278(tf.keras.layers.Layer):
	def __init__(self):
		super(L278, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L279(tf.keras.layers.Layer):
	def __init__(self):
		super(L279, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L280(tf.keras.layers.Layer):
	def __init__(self):
		super(L280, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L281(tf.keras.layers.Layer):
	def __init__(self):
		super(L281, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L282(tf.keras.layers.Layer):
	def __init__(self):
		super(L282, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L283(tf.keras.layers.Layer):
	def __init__(self):
		super(L283, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L284(tf.keras.layers.Layer):
	def __init__(self):
		super(L284, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L285(tf.keras.layers.Layer):
	def __init__(self):
		super(L285, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L286(tf.keras.layers.Layer):
	def __init__(self):
		super(L286, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L287(tf.keras.layers.Layer):
	def __init__(self):
		super(L287, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L288(tf.keras.layers.Layer):
	def __init__(self):
		super(L288, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L289(tf.keras.layers.Layer):
	def __init__(self):
		super(L289, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L290(tf.keras.layers.Layer):
	def __init__(self):
		super(L290, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L291(tf.keras.layers.Layer):
	def __init__(self):
		super(L291, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L292(tf.keras.layers.Layer):
	def __init__(self):
		super(L292, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L293(tf.keras.layers.Layer):
	def __init__(self):
		super(L293, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L294(tf.keras.layers.Layer):
	def __init__(self):
		super(L294, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L295(tf.keras.layers.Layer):
	def __init__(self):
		super(L295, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L296(tf.keras.layers.Layer):
	def __init__(self):
		super(L296, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L297(tf.keras.layers.Layer):
	def __init__(self):
		super(L297, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L298(tf.keras.layers.Layer):
	def __init__(self):
		super(L298, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L299(tf.keras.layers.Layer):
	def __init__(self):
		super(L299, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L300(tf.keras.layers.Layer):
	def __init__(self):
		super(L300, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L301(tf.keras.layers.Layer):
	def __init__(self):
		super(L301, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L302(tf.keras.layers.Layer):
	def __init__(self):
		super(L302, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L303(tf.keras.layers.Layer):
	def __init__(self):
		super(L303, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L304(tf.keras.layers.Layer):
	def __init__(self):
		super(L304, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L305(tf.keras.layers.Layer):
	def __init__(self):
		super(L305, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L306(tf.keras.layers.Layer):
	def __init__(self):
		super(L306, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L307(tf.keras.layers.Layer):
	def __init__(self):
		super(L307, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L308(tf.keras.layers.Layer):
	def __init__(self):
		super(L308, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L309(tf.keras.layers.Layer):
	def __init__(self):
		super(L309, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L310(tf.keras.layers.Layer):
	def __init__(self):
		super(L310, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L311(tf.keras.layers.Layer):
	def __init__(self):
		super(L311, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L312(tf.keras.layers.Layer):
	def __init__(self):
		super(L312, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L313(tf.keras.layers.Layer):
	def __init__(self):
		super(L313, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L314(tf.keras.layers.Layer):
	def __init__(self):
		super(L314, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L315(tf.keras.layers.Layer):
	def __init__(self):
		super(L315, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L316(tf.keras.layers.Layer):
	def __init__(self):
		super(L316, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L317(tf.keras.layers.Layer):
	def __init__(self):
		super(L317, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L318(tf.keras.layers.Layer):
	def __init__(self):
		super(L318, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L319(tf.keras.layers.Layer):
	def __init__(self):
		super(L319, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L320(tf.keras.layers.Layer):
	def __init__(self):
		super(L320, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L321(tf.keras.layers.Layer):
	def __init__(self):
		super(L321, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L322(tf.keras.layers.Layer):
	def __init__(self):
		super(L322, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L323(tf.keras.layers.Layer):
	def __init__(self):
		super(L323, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L324(tf.keras.layers.Layer):
	def __init__(self):
		super(L324, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L325(tf.keras.layers.Layer):
	def __init__(self):
		super(L325, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L326(tf.keras.layers.Layer):
	def __init__(self):
		super(L326, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L327(tf.keras.layers.Layer):
	def __init__(self):
		super(L327, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L328(tf.keras.layers.Layer):
	def __init__(self):
		super(L328, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L329(tf.keras.layers.Layer):
	def __init__(self):
		super(L329, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L330(tf.keras.layers.Layer):
	def __init__(self):
		super(L330, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L331(tf.keras.layers.Layer):
	def __init__(self):
		super(L331, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L332(tf.keras.layers.Layer):
	def __init__(self):
		super(L332, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L333(tf.keras.layers.Layer):
	def __init__(self):
		super(L333, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L334(tf.keras.layers.Layer):
	def __init__(self):
		super(L334, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L335(tf.keras.layers.Layer):
	def __init__(self):
		super(L335, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L336(tf.keras.layers.Layer):
	def __init__(self):
		super(L336, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L337(tf.keras.layers.Layer):
	def __init__(self):
		super(L337, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L338(tf.keras.layers.Layer):
	def __init__(self):
		super(L338, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L339(tf.keras.layers.Layer):
	def __init__(self):
		super(L339, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L340(tf.keras.layers.Layer):
	def __init__(self):
		super(L340, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L341(tf.keras.layers.Layer):
	def __init__(self):
		super(L341, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L342(tf.keras.layers.Layer):
	def __init__(self):
		super(L342, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
			
class L343(tf.keras.layers.Layer):
	def __init__(self):
		super(L343, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L2(l2=0.001)
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
	
layer_3_out = tf.keras.Input(shape=[28, 28], name="3, Input", batch_size=32)
		

layer_4 = L4()
layer_4_out = layer_4(layer_3_out)

layer_173 = L173()
try:
	try:
		layer_173_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out])
		layer_173_out = layer_173(layer_173_concat)
	except ValueError:
		layer_173_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out], axis=-2)
		layer_173_out = layer_173(layer_173_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_173 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_173_concat = strong_concat_173(new_list_of_inputs)
	layer_173_out = layer_173(layer_173_concat)
if layer_173.count_params() != 0:
	if 173 in set_weigths:
		if len(layer_173.get_weights()) > 1:
			new_w = [get_new_weigths(layer_173, set_weigths[173])]+layer_173.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_173, set_weigths[173])]
		layer_173.set_weights(new_w)


try:
	count = 1
	for s in layer_173_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #173:',count)
except AttributeError:
	for i in layer_173_out:
		count = 1
		for s in layer_173_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #173:',count)



layer_174 = L174()
try:
	try:
		layer_174_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out])
		layer_174_out = layer_174(layer_174_concat)
	except ValueError:
		layer_174_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out], axis=-2)
		layer_174_out = layer_174(layer_174_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_174 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_174_concat = strong_concat_174(new_list_of_inputs)
	layer_174_out = layer_174(layer_174_concat)
if layer_174.count_params() != 0:
	if 174 in set_weigths:
		if len(layer_174.get_weights()) > 1:
			new_w = [get_new_weigths(layer_174, set_weigths[174])]+layer_174.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_174, set_weigths[174])]
		layer_174.set_weights(new_w)


try:
	count = 1
	for s in layer_174_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #174:',count)
except AttributeError:
	for i in layer_174_out:
		count = 1
		for s in layer_174_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #174:',count)



layer_175 = L175()
try:
	try:
		layer_175_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out])
		layer_175_out = layer_175(layer_175_concat)
	except ValueError:
		layer_175_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out], axis=-2)
		layer_175_out = layer_175(layer_175_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_175 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_175_concat = strong_concat_175(new_list_of_inputs)
	layer_175_out = layer_175(layer_175_concat)
if layer_175.count_params() != 0:
	if 175 in set_weigths:
		if len(layer_175.get_weights()) > 1:
			new_w = [get_new_weigths(layer_175, set_weigths[175])]+layer_175.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_175, set_weigths[175])]
		layer_175.set_weights(new_w)


try:
	count = 1
	for s in layer_175_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #175:',count)
except AttributeError:
	for i in layer_175_out:
		count = 1
		for s in layer_175_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #175:',count)



layer_176 = L176()
try:
	try:
		layer_176_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out])
		layer_176_out = layer_176(layer_176_concat)
	except ValueError:
		layer_176_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out], axis=-2)
		layer_176_out = layer_176(layer_176_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_176 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_176_concat = strong_concat_176(new_list_of_inputs)
	layer_176_out = layer_176(layer_176_concat)
if layer_176.count_params() != 0:
	if 176 in set_weigths:
		if len(layer_176.get_weights()) > 1:
			new_w = [get_new_weigths(layer_176, set_weigths[176])]+layer_176.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_176, set_weigths[176])]
		layer_176.set_weights(new_w)


try:
	count = 1
	for s in layer_176_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #176:',count)
except AttributeError:
	for i in layer_176_out:
		count = 1
		for s in layer_176_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #176:',count)



layer_177 = L177()
try:
	try:
		layer_177_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out])
		layer_177_out = layer_177(layer_177_concat)
	except ValueError:
		layer_177_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out], axis=-2)
		layer_177_out = layer_177(layer_177_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_177 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_177_concat = strong_concat_177(new_list_of_inputs)
	layer_177_out = layer_177(layer_177_concat)
if layer_177.count_params() != 0:
	if 177 in set_weigths:
		if len(layer_177.get_weights()) > 1:
			new_w = [get_new_weigths(layer_177, set_weigths[177])]+layer_177.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_177, set_weigths[177])]
		layer_177.set_weights(new_w)


try:
	count = 1
	for s in layer_177_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #177:',count)
except AttributeError:
	for i in layer_177_out:
		count = 1
		for s in layer_177_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #177:',count)



layer_178 = L178()
try:
	try:
		layer_178_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out])
		layer_178_out = layer_178(layer_178_concat)
	except ValueError:
		layer_178_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out], axis=-2)
		layer_178_out = layer_178(layer_178_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_178 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_178_concat = strong_concat_178(new_list_of_inputs)
	layer_178_out = layer_178(layer_178_concat)
if layer_178.count_params() != 0:
	if 178 in set_weigths:
		if len(layer_178.get_weights()) > 1:
			new_w = [get_new_weigths(layer_178, set_weigths[178])]+layer_178.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_178, set_weigths[178])]
		layer_178.set_weights(new_w)


try:
	count = 1
	for s in layer_178_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #178:',count)
except AttributeError:
	for i in layer_178_out:
		count = 1
		for s in layer_178_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #178:',count)



layer_179 = L179()
try:
	try:
		layer_179_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out])
		layer_179_out = layer_179(layer_179_concat)
	except ValueError:
		layer_179_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out], axis=-2)
		layer_179_out = layer_179(layer_179_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_179 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_179_concat = strong_concat_179(new_list_of_inputs)
	layer_179_out = layer_179(layer_179_concat)
if layer_179.count_params() != 0:
	if 179 in set_weigths:
		if len(layer_179.get_weights()) > 1:
			new_w = [get_new_weigths(layer_179, set_weigths[179])]+layer_179.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_179, set_weigths[179])]
		layer_179.set_weights(new_w)


try:
	count = 1
	for s in layer_179_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #179:',count)
except AttributeError:
	for i in layer_179_out:
		count = 1
		for s in layer_179_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #179:',count)



layer_180 = L180()
try:
	try:
		layer_180_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out])
		layer_180_out = layer_180(layer_180_concat)
	except ValueError:
		layer_180_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out], axis=-2)
		layer_180_out = layer_180(layer_180_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_180 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_180_concat = strong_concat_180(new_list_of_inputs)
	layer_180_out = layer_180(layer_180_concat)
if layer_180.count_params() != 0:
	if 180 in set_weigths:
		if len(layer_180.get_weights()) > 1:
			new_w = [get_new_weigths(layer_180, set_weigths[180])]+layer_180.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_180, set_weigths[180])]
		layer_180.set_weights(new_w)


try:
	count = 1
	for s in layer_180_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #180:',count)
except AttributeError:
	for i in layer_180_out:
		count = 1
		for s in layer_180_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #180:',count)



layer_352 = L352()
try:
	try:
		layer_352_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out])
		layer_352_out = layer_352(layer_352_concat)
	except ValueError:
		layer_352_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out], axis=-2)
		layer_352_out = layer_352(layer_352_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_352 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_352_concat = strong_concat_352(new_list_of_inputs)
	layer_352_out = layer_352(layer_352_concat)
if layer_352.count_params() != 0:
	if 352 in set_weigths:
		if len(layer_352.get_weights()) > 1:
			new_w = [get_new_weigths(layer_352, set_weigths[352])]+layer_352.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_352, set_weigths[352])]
		layer_352.set_weights(new_w)


try:
	count = 1
	for s in layer_352_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #352:',count)
except AttributeError:
	for i in layer_352_out:
		count = 1
		for s in layer_352_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #352:',count)



layer_353 = L353()
try:
	try:
		layer_353_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_352_out])
		layer_353_out = layer_353(layer_353_concat)
	except ValueError:
		layer_353_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_352_out], axis=-2)
		layer_353_out = layer_353(layer_353_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_353 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_352_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_353_concat = strong_concat_353(new_list_of_inputs)
	layer_353_out = layer_353(layer_353_concat)
if layer_353.count_params() != 0:
	if 353 in set_weigths:
		if len(layer_353.get_weights()) > 1:
			new_w = [get_new_weigths(layer_353, set_weigths[353])]+layer_353.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_353, set_weigths[353])]
		layer_353.set_weights(new_w)


try:
	count = 1
	for s in layer_353_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #353:',count)
except AttributeError:
	for i in layer_353_out:
		count = 1
		for s in layer_353_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #353:',count)



layer_354 = L354()
try:
	try:
		layer_354_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_352_out, layer_353_out])
		layer_354_out = layer_354(layer_354_concat)
	except ValueError:
		layer_354_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_352_out, layer_353_out], axis=-2)
		layer_354_out = layer_354(layer_354_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_354 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_352_out, layer_353_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_354_concat = strong_concat_354(new_list_of_inputs)
	layer_354_out = layer_354(layer_354_concat)
if layer_354.count_params() != 0:
	if 354 in set_weigths:
		if len(layer_354.get_weights()) > 1:
			new_w = [get_new_weigths(layer_354, set_weigths[354])]+layer_354.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_354, set_weigths[354])]
		layer_354.set_weights(new_w)


try:
	count = 1
	for s in layer_354_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #354:',count)
except AttributeError:
	for i in layer_354_out:
		count = 1
		for s in layer_354_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #354:',count)



layer_355 = L355()
try:
	try:
		layer_355_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_352_out, layer_353_out, layer_354_out])
		layer_355_out = layer_355(layer_355_concat)
	except ValueError:
		layer_355_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_352_out, layer_353_out, layer_354_out], axis=-2)
		layer_355_out = layer_355(layer_355_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_355 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_352_out, layer_353_out, layer_354_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_355_concat = strong_concat_355(new_list_of_inputs)
	layer_355_out = layer_355(layer_355_concat)
if layer_355.count_params() != 0:
	if 355 in set_weigths:
		if len(layer_355.get_weights()) > 1:
			new_w = [get_new_weigths(layer_355, set_weigths[355])]+layer_355.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_355, set_weigths[355])]
		layer_355.set_weights(new_w)


try:
	count = 1
	for s in layer_355_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #355:',count)
except AttributeError:
	for i in layer_355_out:
		count = 1
		for s in layer_355_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #355:',count)



layer_356 = L356()
try:
	try:
		layer_356_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out])
		layer_356_out = layer_356(layer_356_concat)
	except ValueError:
		layer_356_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out], axis=-2)
		layer_356_out = layer_356(layer_356_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_356 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_356_concat = strong_concat_356(new_list_of_inputs)
	layer_356_out = layer_356(layer_356_concat)
if layer_356.count_params() != 0:
	if 356 in set_weigths:
		if len(layer_356.get_weights()) > 1:
			new_w = [get_new_weigths(layer_356, set_weigths[356])]+layer_356.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_356, set_weigths[356])]
		layer_356.set_weights(new_w)


try:
	count = 1
	for s in layer_356_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #356:',count)
except AttributeError:
	for i in layer_356_out:
		count = 1
		for s in layer_356_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #356:',count)



layer_357 = L357()
try:
	try:
		layer_357_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out])
		layer_357_out = layer_357(layer_357_concat)
	except ValueError:
		layer_357_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out], axis=-2)
		layer_357_out = layer_357(layer_357_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_357 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_357_concat = strong_concat_357(new_list_of_inputs)
	layer_357_out = layer_357(layer_357_concat)
if layer_357.count_params() != 0:
	if 357 in set_weigths:
		if len(layer_357.get_weights()) > 1:
			new_w = [get_new_weigths(layer_357, set_weigths[357])]+layer_357.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_357, set_weigths[357])]
		layer_357.set_weights(new_w)


try:
	count = 1
	for s in layer_357_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #357:',count)
except AttributeError:
	for i in layer_357_out:
		count = 1
		for s in layer_357_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #357:',count)



layer_173 = L173()
try:
	try:
		layer_173_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out])
		layer_173_out = layer_173(layer_173_concat)
	except ValueError:
		layer_173_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out], axis=-2)
		layer_173_out = layer_173(layer_173_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_173 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_173_concat = strong_concat_173(new_list_of_inputs)
	layer_173_out = layer_173(layer_173_concat)
if layer_173.count_params() != 0:
	if 173 in set_weigths:
		if len(layer_173.get_weights()) > 1:
			new_w = [get_new_weigths(layer_173, set_weigths[173])]+layer_173.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_173, set_weigths[173])]
		layer_173.set_weights(new_w)


try:
	count = 1
	for s in layer_173_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #173:',count)
except AttributeError:
	for i in layer_173_out:
		count = 1
		for s in layer_173_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #173:',count)



layer_174 = L174()
try:
	try:
		layer_174_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out])
		layer_174_out = layer_174(layer_174_concat)
	except ValueError:
		layer_174_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out], axis=-2)
		layer_174_out = layer_174(layer_174_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_174 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_174_concat = strong_concat_174(new_list_of_inputs)
	layer_174_out = layer_174(layer_174_concat)
if layer_174.count_params() != 0:
	if 174 in set_weigths:
		if len(layer_174.get_weights()) > 1:
			new_w = [get_new_weigths(layer_174, set_weigths[174])]+layer_174.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_174, set_weigths[174])]
		layer_174.set_weights(new_w)


try:
	count = 1
	for s in layer_174_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #174:',count)
except AttributeError:
	for i in layer_174_out:
		count = 1
		for s in layer_174_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #174:',count)



layer_175 = L175()
try:
	try:
		layer_175_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out])
		layer_175_out = layer_175(layer_175_concat)
	except ValueError:
		layer_175_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out], axis=-2)
		layer_175_out = layer_175(layer_175_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_175 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_175_concat = strong_concat_175(new_list_of_inputs)
	layer_175_out = layer_175(layer_175_concat)
if layer_175.count_params() != 0:
	if 175 in set_weigths:
		if len(layer_175.get_weights()) > 1:
			new_w = [get_new_weigths(layer_175, set_weigths[175])]+layer_175.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_175, set_weigths[175])]
		layer_175.set_weights(new_w)


try:
	count = 1
	for s in layer_175_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #175:',count)
except AttributeError:
	for i in layer_175_out:
		count = 1
		for s in layer_175_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #175:',count)



layer_176 = L176()
try:
	try:
		layer_176_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out])
		layer_176_out = layer_176(layer_176_concat)
	except ValueError:
		layer_176_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out], axis=-2)
		layer_176_out = layer_176(layer_176_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_176 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_176_concat = strong_concat_176(new_list_of_inputs)
	layer_176_out = layer_176(layer_176_concat)
if layer_176.count_params() != 0:
	if 176 in set_weigths:
		if len(layer_176.get_weights()) > 1:
			new_w = [get_new_weigths(layer_176, set_weigths[176])]+layer_176.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_176, set_weigths[176])]
		layer_176.set_weights(new_w)


try:
	count = 1
	for s in layer_176_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #176:',count)
except AttributeError:
	for i in layer_176_out:
		count = 1
		for s in layer_176_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #176:',count)



layer_177 = L177()
try:
	try:
		layer_177_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out])
		layer_177_out = layer_177(layer_177_concat)
	except ValueError:
		layer_177_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out], axis=-2)
		layer_177_out = layer_177(layer_177_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_177 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_177_concat = strong_concat_177(new_list_of_inputs)
	layer_177_out = layer_177(layer_177_concat)
if layer_177.count_params() != 0:
	if 177 in set_weigths:
		if len(layer_177.get_weights()) > 1:
			new_w = [get_new_weigths(layer_177, set_weigths[177])]+layer_177.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_177, set_weigths[177])]
		layer_177.set_weights(new_w)


try:
	count = 1
	for s in layer_177_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #177:',count)
except AttributeError:
	for i in layer_177_out:
		count = 1
		for s in layer_177_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #177:',count)



layer_178 = L178()
try:
	try:
		layer_178_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out])
		layer_178_out = layer_178(layer_178_concat)
	except ValueError:
		layer_178_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out], axis=-2)
		layer_178_out = layer_178(layer_178_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_178 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_178_concat = strong_concat_178(new_list_of_inputs)
	layer_178_out = layer_178(layer_178_concat)
if layer_178.count_params() != 0:
	if 178 in set_weigths:
		if len(layer_178.get_weights()) > 1:
			new_w = [get_new_weigths(layer_178, set_weigths[178])]+layer_178.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_178, set_weigths[178])]
		layer_178.set_weights(new_w)


try:
	count = 1
	for s in layer_178_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #178:',count)
except AttributeError:
	for i in layer_178_out:
		count = 1
		for s in layer_178_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #178:',count)



layer_179 = L179()
try:
	try:
		layer_179_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out])
		layer_179_out = layer_179(layer_179_concat)
	except ValueError:
		layer_179_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out], axis=-2)
		layer_179_out = layer_179(layer_179_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_179 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_179_concat = strong_concat_179(new_list_of_inputs)
	layer_179_out = layer_179(layer_179_concat)
if layer_179.count_params() != 0:
	if 179 in set_weigths:
		if len(layer_179.get_weights()) > 1:
			new_w = [get_new_weigths(layer_179, set_weigths[179])]+layer_179.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_179, set_weigths[179])]
		layer_179.set_weights(new_w)


try:
	count = 1
	for s in layer_179_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #179:',count)
except AttributeError:
	for i in layer_179_out:
		count = 1
		for s in layer_179_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #179:',count)



layer_180 = L180()
try:
	try:
		layer_180_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out])
		layer_180_out = layer_180(layer_180_concat)
	except ValueError:
		layer_180_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out], axis=-2)
		layer_180_out = layer_180(layer_180_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_180 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_180_concat = strong_concat_180(new_list_of_inputs)
	layer_180_out = layer_180(layer_180_concat)
if layer_180.count_params() != 0:
	if 180 in set_weigths:
		if len(layer_180.get_weights()) > 1:
			new_w = [get_new_weigths(layer_180, set_weigths[180])]+layer_180.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_180, set_weigths[180])]
		layer_180.set_weights(new_w)


try:
	count = 1
	for s in layer_180_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #180:',count)
except AttributeError:
	for i in layer_180_out:
		count = 1
		for s in layer_180_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #180:',count)



layer_345 = L345()
try:
	try:
		layer_345_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_345_out = layer_345(layer_345_concat)
	except ValueError:
		layer_345_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_345_out = layer_345(layer_345_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_345 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_345_concat = strong_concat_345(new_list_of_inputs)
	layer_345_out = layer_345(layer_345_concat)
if layer_345.count_params() != 0:
	if 345 in set_weigths:
		if len(layer_345.get_weights()) > 1:
			new_w = [get_new_weigths(layer_345, set_weigths[345])]+layer_345.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_345, set_weigths[345])]
		layer_345.set_weights(new_w)


try:
	count = 1
	for s in layer_345_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #345:',count)
except AttributeError:
	for i in layer_345_out:
		count = 1
		for s in layer_345_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #345:',count)



layer_352 = L352()
try:
	try:
		layer_352_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out])
		layer_352_out = layer_352(layer_352_concat)
	except ValueError:
		layer_352_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out], axis=-2)
		layer_352_out = layer_352(layer_352_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_352 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_352_concat = strong_concat_352(new_list_of_inputs)
	layer_352_out = layer_352(layer_352_concat)
if layer_352.count_params() != 0:
	if 352 in set_weigths:
		if len(layer_352.get_weights()) > 1:
			new_w = [get_new_weigths(layer_352, set_weigths[352])]+layer_352.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_352, set_weigths[352])]
		layer_352.set_weights(new_w)


try:
	count = 1
	for s in layer_352_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #352:',count)
except AttributeError:
	for i in layer_352_out:
		count = 1
		for s in layer_352_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #352:',count)



layer_353 = L353()
try:
	try:
		layer_353_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_352_out])
		layer_353_out = layer_353(layer_353_concat)
	except ValueError:
		layer_353_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_352_out], axis=-2)
		layer_353_out = layer_353(layer_353_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_353 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_352_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_353_concat = strong_concat_353(new_list_of_inputs)
	layer_353_out = layer_353(layer_353_concat)
if layer_353.count_params() != 0:
	if 353 in set_weigths:
		if len(layer_353.get_weights()) > 1:
			new_w = [get_new_weigths(layer_353, set_weigths[353])]+layer_353.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_353, set_weigths[353])]
		layer_353.set_weights(new_w)


try:
	count = 1
	for s in layer_353_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #353:',count)
except AttributeError:
	for i in layer_353_out:
		count = 1
		for s in layer_353_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #353:',count)



layer_354 = L354()
try:
	try:
		layer_354_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_352_out, layer_353_out])
		layer_354_out = layer_354(layer_354_concat)
	except ValueError:
		layer_354_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_352_out, layer_353_out], axis=-2)
		layer_354_out = layer_354(layer_354_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_354 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_352_out, layer_353_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_354_concat = strong_concat_354(new_list_of_inputs)
	layer_354_out = layer_354(layer_354_concat)
if layer_354.count_params() != 0:
	if 354 in set_weigths:
		if len(layer_354.get_weights()) > 1:
			new_w = [get_new_weigths(layer_354, set_weigths[354])]+layer_354.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_354, set_weigths[354])]
		layer_354.set_weights(new_w)


try:
	count = 1
	for s in layer_354_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #354:',count)
except AttributeError:
	for i in layer_354_out:
		count = 1
		for s in layer_354_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #354:',count)



layer_355 = L355()
try:
	try:
		layer_355_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_352_out, layer_353_out, layer_354_out])
		layer_355_out = layer_355(layer_355_concat)
	except ValueError:
		layer_355_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_352_out, layer_353_out, layer_354_out], axis=-2)
		layer_355_out = layer_355(layer_355_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_355 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_352_out, layer_353_out, layer_354_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_355_concat = strong_concat_355(new_list_of_inputs)
	layer_355_out = layer_355(layer_355_concat)
if layer_355.count_params() != 0:
	if 355 in set_weigths:
		if len(layer_355.get_weights()) > 1:
			new_w = [get_new_weigths(layer_355, set_weigths[355])]+layer_355.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_355, set_weigths[355])]
		layer_355.set_weights(new_w)


try:
	count = 1
	for s in layer_355_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #355:',count)
except AttributeError:
	for i in layer_355_out:
		count = 1
		for s in layer_355_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #355:',count)



layer_356 = L356()
try:
	try:
		layer_356_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out])
		layer_356_out = layer_356(layer_356_concat)
	except ValueError:
		layer_356_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out], axis=-2)
		layer_356_out = layer_356(layer_356_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_356 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_356_concat = strong_concat_356(new_list_of_inputs)
	layer_356_out = layer_356(layer_356_concat)
if layer_356.count_params() != 0:
	if 356 in set_weigths:
		if len(layer_356.get_weights()) > 1:
			new_w = [get_new_weigths(layer_356, set_weigths[356])]+layer_356.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_356, set_weigths[356])]
		layer_356.set_weights(new_w)


try:
	count = 1
	for s in layer_356_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #356:',count)
except AttributeError:
	for i in layer_356_out:
		count = 1
		for s in layer_356_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #356:',count)



layer_357 = L357()
try:
	try:
		layer_357_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out])
		layer_357_out = layer_357(layer_357_concat)
	except ValueError:
		layer_357_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out], axis=-2)
		layer_357_out = layer_357(layer_357_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_357 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_357_concat = strong_concat_357(new_list_of_inputs)
	layer_357_out = layer_357(layer_357_concat)
if layer_357.count_params() != 0:
	if 357 in set_weigths:
		if len(layer_357.get_weights()) > 1:
			new_w = [get_new_weigths(layer_357, set_weigths[357])]+layer_357.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_357, set_weigths[357])]
		layer_357.set_weights(new_w)


try:
	count = 1
	for s in layer_357_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #357:',count)
except AttributeError:
	for i in layer_357_out:
		count = 1
		for s in layer_357_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #357:',count)



layer_181 = L181()
try:
	try:
		layer_181_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_345_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_181_out = layer_181(layer_181_concat)
	except ValueError:
		layer_181_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_345_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_181_out = layer_181(layer_181_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_181 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_345_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_181_concat = strong_concat_181(new_list_of_inputs)
	layer_181_out = layer_181(layer_181_concat)
if layer_181.count_params() != 0:
	if 181 in set_weigths:
		if len(layer_181.get_weights()) > 1:
			new_w = [get_new_weigths(layer_181, set_weigths[181])]+layer_181.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_181, set_weigths[181])]
		layer_181.set_weights(new_w)


try:
	count = 1
	for s in layer_181_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #181:',count)
except AttributeError:
	for i in layer_181_out:
		count = 1
		for s in layer_181_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #181:',count)



layer_182 = L182()
try:
	try:
		layer_182_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_345_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_182_out = layer_182(layer_182_concat)
	except ValueError:
		layer_182_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_345_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_182_out = layer_182(layer_182_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_182 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_345_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_182_concat = strong_concat_182(new_list_of_inputs)
	layer_182_out = layer_182(layer_182_concat)
if layer_182.count_params() != 0:
	if 182 in set_weigths:
		if len(layer_182.get_weights()) > 1:
			new_w = [get_new_weigths(layer_182, set_weigths[182])]+layer_182.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_182, set_weigths[182])]
		layer_182.set_weights(new_w)


try:
	count = 1
	for s in layer_182_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #182:',count)
except AttributeError:
	for i in layer_182_out:
		count = 1
		for s in layer_182_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #182:',count)



layer_183 = L183()
try:
	try:
		layer_183_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_345_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_183_out = layer_183(layer_183_concat)
	except ValueError:
		layer_183_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_345_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_183_out = layer_183(layer_183_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_183 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_345_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_183_concat = strong_concat_183(new_list_of_inputs)
	layer_183_out = layer_183(layer_183_concat)
if layer_183.count_params() != 0:
	if 183 in set_weigths:
		if len(layer_183.get_weights()) > 1:
			new_w = [get_new_weigths(layer_183, set_weigths[183])]+layer_183.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_183, set_weigths[183])]
		layer_183.set_weights(new_w)


try:
	count = 1
	for s in layer_183_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #183:',count)
except AttributeError:
	for i in layer_183_out:
		count = 1
		for s in layer_183_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #183:',count)



layer_184 = L184()
try:
	try:
		layer_184_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_345_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_184_out = layer_184(layer_184_concat)
	except ValueError:
		layer_184_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_345_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_184_out = layer_184(layer_184_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_184 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_345_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_184_concat = strong_concat_184(new_list_of_inputs)
	layer_184_out = layer_184(layer_184_concat)
if layer_184.count_params() != 0:
	if 184 in set_weigths:
		if len(layer_184.get_weights()) > 1:
			new_w = [get_new_weigths(layer_184, set_weigths[184])]+layer_184.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_184, set_weigths[184])]
		layer_184.set_weights(new_w)


try:
	count = 1
	for s in layer_184_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #184:',count)
except AttributeError:
	for i in layer_184_out:
		count = 1
		for s in layer_184_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #184:',count)



layer_185 = L185()
try:
	try:
		layer_185_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_345_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_185_out = layer_185(layer_185_concat)
	except ValueError:
		layer_185_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_345_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_185_out = layer_185(layer_185_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_185 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_345_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_185_concat = strong_concat_185(new_list_of_inputs)
	layer_185_out = layer_185(layer_185_concat)
if layer_185.count_params() != 0:
	if 185 in set_weigths:
		if len(layer_185.get_weights()) > 1:
			new_w = [get_new_weigths(layer_185, set_weigths[185])]+layer_185.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_185, set_weigths[185])]
		layer_185.set_weights(new_w)


try:
	count = 1
	for s in layer_185_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #185:',count)
except AttributeError:
	for i in layer_185_out:
		count = 1
		for s in layer_185_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #185:',count)



layer_345 = L345()
try:
	try:
		layer_345_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_345_out = layer_345(layer_345_concat)
	except ValueError:
		layer_345_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_345_out = layer_345(layer_345_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_345 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_345_concat = strong_concat_345(new_list_of_inputs)
	layer_345_out = layer_345(layer_345_concat)
if layer_345.count_params() != 0:
	if 345 in set_weigths:
		if len(layer_345.get_weights()) > 1:
			new_w = [get_new_weigths(layer_345, set_weigths[345])]+layer_345.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_345, set_weigths[345])]
		layer_345.set_weights(new_w)


try:
	count = 1
	for s in layer_345_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #345:',count)
except AttributeError:
	for i in layer_345_out:
		count = 1
		for s in layer_345_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #345:',count)



layer_346 = L346()
try:
	try:
		layer_346_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_346_out = layer_346(layer_346_concat)
	except ValueError:
		layer_346_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_346_out = layer_346(layer_346_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_346 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_346_concat = strong_concat_346(new_list_of_inputs)
	layer_346_out = layer_346(layer_346_concat)
if layer_346.count_params() != 0:
	if 346 in set_weigths:
		if len(layer_346.get_weights()) > 1:
			new_w = [get_new_weigths(layer_346, set_weigths[346])]+layer_346.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_346, set_weigths[346])]
		layer_346.set_weights(new_w)


try:
	count = 1
	for s in layer_346_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #346:',count)
except AttributeError:
	for i in layer_346_out:
		count = 1
		for s in layer_346_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #346:',count)



layer_347 = L347()
try:
	try:
		layer_347_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_347_out = layer_347(layer_347_concat)
	except ValueError:
		layer_347_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_347_out = layer_347(layer_347_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_347 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_347_concat = strong_concat_347(new_list_of_inputs)
	layer_347_out = layer_347(layer_347_concat)
if layer_347.count_params() != 0:
	if 347 in set_weigths:
		if len(layer_347.get_weights()) > 1:
			new_w = [get_new_weigths(layer_347, set_weigths[347])]+layer_347.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_347, set_weigths[347])]
		layer_347.set_weights(new_w)


try:
	count = 1
	for s in layer_347_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #347:',count)
except AttributeError:
	for i in layer_347_out:
		count = 1
		for s in layer_347_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #347:',count)



layer_348 = L348()
try:
	try:
		layer_348_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_348_out = layer_348(layer_348_concat)
	except ValueError:
		layer_348_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_348_out = layer_348(layer_348_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_348 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_348_concat = strong_concat_348(new_list_of_inputs)
	layer_348_out = layer_348(layer_348_concat)
if layer_348.count_params() != 0:
	if 348 in set_weigths:
		if len(layer_348.get_weights()) > 1:
			new_w = [get_new_weigths(layer_348, set_weigths[348])]+layer_348.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_348, set_weigths[348])]
		layer_348.set_weights(new_w)


try:
	count = 1
	for s in layer_348_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #348:',count)
except AttributeError:
	for i in layer_348_out:
		count = 1
		for s in layer_348_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #348:',count)



layer_349 = L349()
try:
	try:
		layer_349_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_349_out = layer_349(layer_349_concat)
	except ValueError:
		layer_349_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_349_out = layer_349(layer_349_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_349 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_349_concat = strong_concat_349(new_list_of_inputs)
	layer_349_out = layer_349(layer_349_concat)
if layer_349.count_params() != 0:
	if 349 in set_weigths:
		if len(layer_349.get_weights()) > 1:
			new_w = [get_new_weigths(layer_349, set_weigths[349])]+layer_349.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_349, set_weigths[349])]
		layer_349.set_weights(new_w)


try:
	count = 1
	for s in layer_349_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #349:',count)
except AttributeError:
	for i in layer_349_out:
		count = 1
		for s in layer_349_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #349:',count)



layer_350 = L350()
try:
	try:
		layer_350_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_350_out = layer_350(layer_350_concat)
	except ValueError:
		layer_350_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_350_out = layer_350(layer_350_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_350 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_350_concat = strong_concat_350(new_list_of_inputs)
	layer_350_out = layer_350(layer_350_concat)
if layer_350.count_params() != 0:
	if 350 in set_weigths:
		if len(layer_350.get_weights()) > 1:
			new_w = [get_new_weigths(layer_350, set_weigths[350])]+layer_350.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_350, set_weigths[350])]
		layer_350.set_weights(new_w)


try:
	count = 1
	for s in layer_350_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #350:',count)
except AttributeError:
	for i in layer_350_out:
		count = 1
		for s in layer_350_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #350:',count)



layer_351 = L351()
try:
	try:
		layer_351_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_351_out = layer_351(layer_351_concat)
	except ValueError:
		layer_351_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_351_out = layer_351(layer_351_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_351 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_351_concat = strong_concat_351(new_list_of_inputs)
	layer_351_out = layer_351(layer_351_concat)
if layer_351.count_params() != 0:
	if 351 in set_weigths:
		if len(layer_351.get_weights()) > 1:
			new_w = [get_new_weigths(layer_351, set_weigths[351])]+layer_351.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_351, set_weigths[351])]
		layer_351.set_weights(new_w)


try:
	count = 1
	for s in layer_351_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #351:',count)
except AttributeError:
	for i in layer_351_out:
		count = 1
		for s in layer_351_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #351:',count)



layer_75 = L75()
try:
	try:
		layer_75_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_75_out = layer_75(layer_75_concat)
	except ValueError:
		layer_75_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_75_out = layer_75(layer_75_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_75 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_75_concat = strong_concat_75(new_list_of_inputs)
	layer_75_out = layer_75(layer_75_concat)
if layer_75.count_params() != 0:
	if 75 in set_weigths:
		if len(layer_75.get_weights()) > 1:
			new_w = [get_new_weigths(layer_75, set_weigths[75])]+layer_75.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_75, set_weigths[75])]
		layer_75.set_weights(new_w)


try:
	count = 1
	for s in layer_75_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #75:',count)
except AttributeError:
	for i in layer_75_out:
		count = 1
		for s in layer_75_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #75:',count)



layer_76 = L76()
try:
	try:
		layer_76_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_76_out = layer_76(layer_76_concat)
	except ValueError:
		layer_76_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_76_out = layer_76(layer_76_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_76 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_76_concat = strong_concat_76(new_list_of_inputs)
	layer_76_out = layer_76(layer_76_concat)
if layer_76.count_params() != 0:
	if 76 in set_weigths:
		if len(layer_76.get_weights()) > 1:
			new_w = [get_new_weigths(layer_76, set_weigths[76])]+layer_76.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_76, set_weigths[76])]
		layer_76.set_weights(new_w)


try:
	count = 1
	for s in layer_76_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #76:',count)
except AttributeError:
	for i in layer_76_out:
		count = 1
		for s in layer_76_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #76:',count)



layer_77 = L77()
try:
	try:
		layer_77_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_77_out = layer_77(layer_77_concat)
	except ValueError:
		layer_77_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_77_out = layer_77(layer_77_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_77 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_77_concat = strong_concat_77(new_list_of_inputs)
	layer_77_out = layer_77(layer_77_concat)
if layer_77.count_params() != 0:
	if 77 in set_weigths:
		if len(layer_77.get_weights()) > 1:
			new_w = [get_new_weigths(layer_77, set_weigths[77])]+layer_77.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_77, set_weigths[77])]
		layer_77.set_weights(new_w)


try:
	count = 1
	for s in layer_77_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #77:',count)
except AttributeError:
	for i in layer_77_out:
		count = 1
		for s in layer_77_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #77:',count)



layer_78 = L78()
try:
	try:
		layer_78_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_78_out = layer_78(layer_78_concat)
	except ValueError:
		layer_78_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_78_out = layer_78(layer_78_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_78 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_78_concat = strong_concat_78(new_list_of_inputs)
	layer_78_out = layer_78(layer_78_concat)
if layer_78.count_params() != 0:
	if 78 in set_weigths:
		if len(layer_78.get_weights()) > 1:
			new_w = [get_new_weigths(layer_78, set_weigths[78])]+layer_78.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_78, set_weigths[78])]
		layer_78.set_weights(new_w)


try:
	count = 1
	for s in layer_78_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #78:',count)
except AttributeError:
	for i in layer_78_out:
		count = 1
		for s in layer_78_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #78:',count)



layer_79 = L79()
try:
	try:
		layer_79_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_79_out = layer_79(layer_79_concat)
	except ValueError:
		layer_79_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_79_out = layer_79(layer_79_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_79 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_79_concat = strong_concat_79(new_list_of_inputs)
	layer_79_out = layer_79(layer_79_concat)
if layer_79.count_params() != 0:
	if 79 in set_weigths:
		if len(layer_79.get_weights()) > 1:
			new_w = [get_new_weigths(layer_79, set_weigths[79])]+layer_79.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_79, set_weigths[79])]
		layer_79.set_weights(new_w)


try:
	count = 1
	for s in layer_79_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #79:',count)
except AttributeError:
	for i in layer_79_out:
		count = 1
		for s in layer_79_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #79:',count)



layer_80 = L80()
try:
	try:
		layer_80_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_80_out = layer_80(layer_80_concat)
	except ValueError:
		layer_80_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_80_out = layer_80(layer_80_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_80 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_80_concat = strong_concat_80(new_list_of_inputs)
	layer_80_out = layer_80(layer_80_concat)
if layer_80.count_params() != 0:
	if 80 in set_weigths:
		if len(layer_80.get_weights()) > 1:
			new_w = [get_new_weigths(layer_80, set_weigths[80])]+layer_80.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_80, set_weigths[80])]
		layer_80.set_weights(new_w)


try:
	count = 1
	for s in layer_80_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #80:',count)
except AttributeError:
	for i in layer_80_out:
		count = 1
		for s in layer_80_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #80:',count)



layer_81 = L81()
try:
	try:
		layer_81_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_81_out = layer_81(layer_81_concat)
	except ValueError:
		layer_81_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_81_out = layer_81(layer_81_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_81 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_81_concat = strong_concat_81(new_list_of_inputs)
	layer_81_out = layer_81(layer_81_concat)
if layer_81.count_params() != 0:
	if 81 in set_weigths:
		if len(layer_81.get_weights()) > 1:
			new_w = [get_new_weigths(layer_81, set_weigths[81])]+layer_81.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_81, set_weigths[81])]
		layer_81.set_weights(new_w)


try:
	count = 1
	for s in layer_81_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #81:',count)
except AttributeError:
	for i in layer_81_out:
		count = 1
		for s in layer_81_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #81:',count)



layer_82 = L82()
try:
	try:
		layer_82_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_82_out = layer_82(layer_82_concat)
	except ValueError:
		layer_82_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_82_out = layer_82(layer_82_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_82 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_82_concat = strong_concat_82(new_list_of_inputs)
	layer_82_out = layer_82(layer_82_concat)
if layer_82.count_params() != 0:
	if 82 in set_weigths:
		if len(layer_82.get_weights()) > 1:
			new_w = [get_new_weigths(layer_82, set_weigths[82])]+layer_82.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_82, set_weigths[82])]
		layer_82.set_weights(new_w)


try:
	count = 1
	for s in layer_82_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #82:',count)
except AttributeError:
	for i in layer_82_out:
		count = 1
		for s in layer_82_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #82:',count)



layer_83 = L83()
try:
	try:
		layer_83_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_83_out = layer_83(layer_83_concat)
	except ValueError:
		layer_83_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_83_out = layer_83(layer_83_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_83 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_83_concat = strong_concat_83(new_list_of_inputs)
	layer_83_out = layer_83(layer_83_concat)
if layer_83.count_params() != 0:
	if 83 in set_weigths:
		if len(layer_83.get_weights()) > 1:
			new_w = [get_new_weigths(layer_83, set_weigths[83])]+layer_83.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_83, set_weigths[83])]
		layer_83.set_weights(new_w)


try:
	count = 1
	for s in layer_83_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #83:',count)
except AttributeError:
	for i in layer_83_out:
		count = 1
		for s in layer_83_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #83:',count)



layer_84 = L84()
try:
	try:
		layer_84_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_84_out = layer_84(layer_84_concat)
	except ValueError:
		layer_84_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_84_out = layer_84(layer_84_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_84 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_84_concat = strong_concat_84(new_list_of_inputs)
	layer_84_out = layer_84(layer_84_concat)
if layer_84.count_params() != 0:
	if 84 in set_weigths:
		if len(layer_84.get_weights()) > 1:
			new_w = [get_new_weigths(layer_84, set_weigths[84])]+layer_84.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_84, set_weigths[84])]
		layer_84.set_weights(new_w)


try:
	count = 1
	for s in layer_84_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #84:',count)
except AttributeError:
	for i in layer_84_out:
		count = 1
		for s in layer_84_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #84:',count)



layer_85 = L85()
try:
	try:
		layer_85_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_85_out = layer_85(layer_85_concat)
	except ValueError:
		layer_85_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_85_out = layer_85(layer_85_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_85 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_85_concat = strong_concat_85(new_list_of_inputs)
	layer_85_out = layer_85(layer_85_concat)
if layer_85.count_params() != 0:
	if 85 in set_weigths:
		if len(layer_85.get_weights()) > 1:
			new_w = [get_new_weigths(layer_85, set_weigths[85])]+layer_85.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_85, set_weigths[85])]
		layer_85.set_weights(new_w)


try:
	count = 1
	for s in layer_85_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #85:',count)
except AttributeError:
	for i in layer_85_out:
		count = 1
		for s in layer_85_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #85:',count)



layer_86 = L86()
try:
	try:
		layer_86_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_86_out = layer_86(layer_86_concat)
	except ValueError:
		layer_86_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_86_out = layer_86(layer_86_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_86 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_86_concat = strong_concat_86(new_list_of_inputs)
	layer_86_out = layer_86(layer_86_concat)
if layer_86.count_params() != 0:
	if 86 in set_weigths:
		if len(layer_86.get_weights()) > 1:
			new_w = [get_new_weigths(layer_86, set_weigths[86])]+layer_86.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_86, set_weigths[86])]
		layer_86.set_weights(new_w)


try:
	count = 1
	for s in layer_86_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #86:',count)
except AttributeError:
	for i in layer_86_out:
		count = 1
		for s in layer_86_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #86:',count)



layer_87 = L87()
try:
	try:
		layer_87_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_87_out = layer_87(layer_87_concat)
	except ValueError:
		layer_87_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_87_out = layer_87(layer_87_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_87 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_87_concat = strong_concat_87(new_list_of_inputs)
	layer_87_out = layer_87(layer_87_concat)
if layer_87.count_params() != 0:
	if 87 in set_weigths:
		if len(layer_87.get_weights()) > 1:
			new_w = [get_new_weigths(layer_87, set_weigths[87])]+layer_87.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_87, set_weigths[87])]
		layer_87.set_weights(new_w)


try:
	count = 1
	for s in layer_87_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #87:',count)
except AttributeError:
	for i in layer_87_out:
		count = 1
		for s in layer_87_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #87:',count)



layer_88 = L88()
try:
	try:
		layer_88_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_88_out = layer_88(layer_88_concat)
	except ValueError:
		layer_88_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_88_out = layer_88(layer_88_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_88 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_88_concat = strong_concat_88(new_list_of_inputs)
	layer_88_out = layer_88(layer_88_concat)
if layer_88.count_params() != 0:
	if 88 in set_weigths:
		if len(layer_88.get_weights()) > 1:
			new_w = [get_new_weigths(layer_88, set_weigths[88])]+layer_88.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_88, set_weigths[88])]
		layer_88.set_weights(new_w)


try:
	count = 1
	for s in layer_88_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #88:',count)
except AttributeError:
	for i in layer_88_out:
		count = 1
		for s in layer_88_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #88:',count)



layer_89 = L89()
try:
	try:
		layer_89_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_89_out = layer_89(layer_89_concat)
	except ValueError:
		layer_89_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_89_out = layer_89(layer_89_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_89 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_89_concat = strong_concat_89(new_list_of_inputs)
	layer_89_out = layer_89(layer_89_concat)
if layer_89.count_params() != 0:
	if 89 in set_weigths:
		if len(layer_89.get_weights()) > 1:
			new_w = [get_new_weigths(layer_89, set_weigths[89])]+layer_89.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_89, set_weigths[89])]
		layer_89.set_weights(new_w)


try:
	count = 1
	for s in layer_89_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #89:',count)
except AttributeError:
	for i in layer_89_out:
		count = 1
		for s in layer_89_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #89:',count)



layer_90 = L90()
try:
	try:
		layer_90_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_90_out = layer_90(layer_90_concat)
	except ValueError:
		layer_90_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_90_out = layer_90(layer_90_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_90 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_90_concat = strong_concat_90(new_list_of_inputs)
	layer_90_out = layer_90(layer_90_concat)
if layer_90.count_params() != 0:
	if 90 in set_weigths:
		if len(layer_90.get_weights()) > 1:
			new_w = [get_new_weigths(layer_90, set_weigths[90])]+layer_90.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_90, set_weigths[90])]
		layer_90.set_weights(new_w)


try:
	count = 1
	for s in layer_90_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #90:',count)
except AttributeError:
	for i in layer_90_out:
		count = 1
		for s in layer_90_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #90:',count)



layer_91 = L91()
try:
	try:
		layer_91_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_91_out = layer_91(layer_91_concat)
	except ValueError:
		layer_91_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_91_out = layer_91(layer_91_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_91 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_91_concat = strong_concat_91(new_list_of_inputs)
	layer_91_out = layer_91(layer_91_concat)
if layer_91.count_params() != 0:
	if 91 in set_weigths:
		if len(layer_91.get_weights()) > 1:
			new_w = [get_new_weigths(layer_91, set_weigths[91])]+layer_91.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_91, set_weigths[91])]
		layer_91.set_weights(new_w)


try:
	count = 1
	for s in layer_91_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #91:',count)
except AttributeError:
	for i in layer_91_out:
		count = 1
		for s in layer_91_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #91:',count)



layer_92 = L92()
try:
	try:
		layer_92_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_92_out = layer_92(layer_92_concat)
	except ValueError:
		layer_92_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_92_out = layer_92(layer_92_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_92 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_92_concat = strong_concat_92(new_list_of_inputs)
	layer_92_out = layer_92(layer_92_concat)
if layer_92.count_params() != 0:
	if 92 in set_weigths:
		if len(layer_92.get_weights()) > 1:
			new_w = [get_new_weigths(layer_92, set_weigths[92])]+layer_92.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_92, set_weigths[92])]
		layer_92.set_weights(new_w)


try:
	count = 1
	for s in layer_92_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #92:',count)
except AttributeError:
	for i in layer_92_out:
		count = 1
		for s in layer_92_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #92:',count)



layer_93 = L93()
try:
	try:
		layer_93_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_93_out = layer_93(layer_93_concat)
	except ValueError:
		layer_93_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_93_out = layer_93(layer_93_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_93 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_93_concat = strong_concat_93(new_list_of_inputs)
	layer_93_out = layer_93(layer_93_concat)
if layer_93.count_params() != 0:
	if 93 in set_weigths:
		if len(layer_93.get_weights()) > 1:
			new_w = [get_new_weigths(layer_93, set_weigths[93])]+layer_93.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_93, set_weigths[93])]
		layer_93.set_weights(new_w)


try:
	count = 1
	for s in layer_93_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #93:',count)
except AttributeError:
	for i in layer_93_out:
		count = 1
		for s in layer_93_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #93:',count)



layer_94 = L94()
try:
	try:
		layer_94_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_94_out = layer_94(layer_94_concat)
	except ValueError:
		layer_94_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_94_out = layer_94(layer_94_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_94 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_94_concat = strong_concat_94(new_list_of_inputs)
	layer_94_out = layer_94(layer_94_concat)
if layer_94.count_params() != 0:
	if 94 in set_weigths:
		if len(layer_94.get_weights()) > 1:
			new_w = [get_new_weigths(layer_94, set_weigths[94])]+layer_94.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_94, set_weigths[94])]
		layer_94.set_weights(new_w)


try:
	count = 1
	for s in layer_94_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #94:',count)
except AttributeError:
	for i in layer_94_out:
		count = 1
		for s in layer_94_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #94:',count)



layer_95 = L95()
try:
	try:
		layer_95_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_95_out = layer_95(layer_95_concat)
	except ValueError:
		layer_95_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_95_out = layer_95(layer_95_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_95 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_95_concat = strong_concat_95(new_list_of_inputs)
	layer_95_out = layer_95(layer_95_concat)
if layer_95.count_params() != 0:
	if 95 in set_weigths:
		if len(layer_95.get_weights()) > 1:
			new_w = [get_new_weigths(layer_95, set_weigths[95])]+layer_95.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_95, set_weigths[95])]
		layer_95.set_weights(new_w)


try:
	count = 1
	for s in layer_95_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #95:',count)
except AttributeError:
	for i in layer_95_out:
		count = 1
		for s in layer_95_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #95:',count)



layer_96 = L96()
try:
	try:
		layer_96_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_96_out = layer_96(layer_96_concat)
	except ValueError:
		layer_96_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_96_out = layer_96(layer_96_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_96 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_96_concat = strong_concat_96(new_list_of_inputs)
	layer_96_out = layer_96(layer_96_concat)
if layer_96.count_params() != 0:
	if 96 in set_weigths:
		if len(layer_96.get_weights()) > 1:
			new_w = [get_new_weigths(layer_96, set_weigths[96])]+layer_96.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_96, set_weigths[96])]
		layer_96.set_weights(new_w)


try:
	count = 1
	for s in layer_96_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #96:',count)
except AttributeError:
	for i in layer_96_out:
		count = 1
		for s in layer_96_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #96:',count)



layer_97 = L97()
try:
	try:
		layer_97_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_97_out = layer_97(layer_97_concat)
	except ValueError:
		layer_97_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_97_out = layer_97(layer_97_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_97 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_97_concat = strong_concat_97(new_list_of_inputs)
	layer_97_out = layer_97(layer_97_concat)
if layer_97.count_params() != 0:
	if 97 in set_weigths:
		if len(layer_97.get_weights()) > 1:
			new_w = [get_new_weigths(layer_97, set_weigths[97])]+layer_97.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_97, set_weigths[97])]
		layer_97.set_weights(new_w)


try:
	count = 1
	for s in layer_97_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #97:',count)
except AttributeError:
	for i in layer_97_out:
		count = 1
		for s in layer_97_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #97:',count)



layer_98 = L98()
try:
	try:
		layer_98_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_98_out = layer_98(layer_98_concat)
	except ValueError:
		layer_98_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_98_out = layer_98(layer_98_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_98 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_98_concat = strong_concat_98(new_list_of_inputs)
	layer_98_out = layer_98(layer_98_concat)
if layer_98.count_params() != 0:
	if 98 in set_weigths:
		if len(layer_98.get_weights()) > 1:
			new_w = [get_new_weigths(layer_98, set_weigths[98])]+layer_98.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_98, set_weigths[98])]
		layer_98.set_weights(new_w)


try:
	count = 1
	for s in layer_98_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #98:',count)
except AttributeError:
	for i in layer_98_out:
		count = 1
		for s in layer_98_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #98:',count)



layer_99 = L99()
try:
	try:
		layer_99_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_99_out = layer_99(layer_99_concat)
	except ValueError:
		layer_99_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_99_out = layer_99(layer_99_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_99 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_99_concat = strong_concat_99(new_list_of_inputs)
	layer_99_out = layer_99(layer_99_concat)
if layer_99.count_params() != 0:
	if 99 in set_weigths:
		if len(layer_99.get_weights()) > 1:
			new_w = [get_new_weigths(layer_99, set_weigths[99])]+layer_99.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_99, set_weigths[99])]
		layer_99.set_weights(new_w)


try:
	count = 1
	for s in layer_99_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #99:',count)
except AttributeError:
	for i in layer_99_out:
		count = 1
		for s in layer_99_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #99:',count)



layer_100 = L100()
try:
	try:
		layer_100_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_100_out = layer_100(layer_100_concat)
	except ValueError:
		layer_100_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_100_out = layer_100(layer_100_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_100 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_100_concat = strong_concat_100(new_list_of_inputs)
	layer_100_out = layer_100(layer_100_concat)
if layer_100.count_params() != 0:
	if 100 in set_weigths:
		if len(layer_100.get_weights()) > 1:
			new_w = [get_new_weigths(layer_100, set_weigths[100])]+layer_100.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_100, set_weigths[100])]
		layer_100.set_weights(new_w)


try:
	count = 1
	for s in layer_100_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #100:',count)
except AttributeError:
	for i in layer_100_out:
		count = 1
		for s in layer_100_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #100:',count)



layer_101 = L101()
try:
	try:
		layer_101_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_101_out = layer_101(layer_101_concat)
	except ValueError:
		layer_101_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_101_out = layer_101(layer_101_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_101 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_101_concat = strong_concat_101(new_list_of_inputs)
	layer_101_out = layer_101(layer_101_concat)
if layer_101.count_params() != 0:
	if 101 in set_weigths:
		if len(layer_101.get_weights()) > 1:
			new_w = [get_new_weigths(layer_101, set_weigths[101])]+layer_101.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_101, set_weigths[101])]
		layer_101.set_weights(new_w)


try:
	count = 1
	for s in layer_101_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #101:',count)
except AttributeError:
	for i in layer_101_out:
		count = 1
		for s in layer_101_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #101:',count)



layer_102 = L102()
try:
	try:
		layer_102_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_102_out = layer_102(layer_102_concat)
	except ValueError:
		layer_102_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_102_out = layer_102(layer_102_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_102 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_102_concat = strong_concat_102(new_list_of_inputs)
	layer_102_out = layer_102(layer_102_concat)
if layer_102.count_params() != 0:
	if 102 in set_weigths:
		if len(layer_102.get_weights()) > 1:
			new_w = [get_new_weigths(layer_102, set_weigths[102])]+layer_102.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_102, set_weigths[102])]
		layer_102.set_weights(new_w)


try:
	count = 1
	for s in layer_102_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #102:',count)
except AttributeError:
	for i in layer_102_out:
		count = 1
		for s in layer_102_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #102:',count)



layer_103 = L103()
try:
	try:
		layer_103_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_103_out = layer_103(layer_103_concat)
	except ValueError:
		layer_103_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_103_out = layer_103(layer_103_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_103 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_103_concat = strong_concat_103(new_list_of_inputs)
	layer_103_out = layer_103(layer_103_concat)
if layer_103.count_params() != 0:
	if 103 in set_weigths:
		if len(layer_103.get_weights()) > 1:
			new_w = [get_new_weigths(layer_103, set_weigths[103])]+layer_103.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_103, set_weigths[103])]
		layer_103.set_weights(new_w)


try:
	count = 1
	for s in layer_103_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #103:',count)
except AttributeError:
	for i in layer_103_out:
		count = 1
		for s in layer_103_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #103:',count)



layer_104 = L104()
try:
	try:
		layer_104_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_104_out = layer_104(layer_104_concat)
	except ValueError:
		layer_104_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_104_out = layer_104(layer_104_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_104 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_104_concat = strong_concat_104(new_list_of_inputs)
	layer_104_out = layer_104(layer_104_concat)
if layer_104.count_params() != 0:
	if 104 in set_weigths:
		if len(layer_104.get_weights()) > 1:
			new_w = [get_new_weigths(layer_104, set_weigths[104])]+layer_104.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_104, set_weigths[104])]
		layer_104.set_weights(new_w)


try:
	count = 1
	for s in layer_104_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #104:',count)
except AttributeError:
	for i in layer_104_out:
		count = 1
		for s in layer_104_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #104:',count)



layer_105 = L105()
try:
	try:
		layer_105_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_105_out = layer_105(layer_105_concat)
	except ValueError:
		layer_105_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_105_out = layer_105(layer_105_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_105 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_105_concat = strong_concat_105(new_list_of_inputs)
	layer_105_out = layer_105(layer_105_concat)
if layer_105.count_params() != 0:
	if 105 in set_weigths:
		if len(layer_105.get_weights()) > 1:
			new_w = [get_new_weigths(layer_105, set_weigths[105])]+layer_105.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_105, set_weigths[105])]
		layer_105.set_weights(new_w)


try:
	count = 1
	for s in layer_105_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #105:',count)
except AttributeError:
	for i in layer_105_out:
		count = 1
		for s in layer_105_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #105:',count)



layer_106 = L106()
try:
	try:
		layer_106_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_106_out = layer_106(layer_106_concat)
	except ValueError:
		layer_106_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_106_out = layer_106(layer_106_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_106 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_106_concat = strong_concat_106(new_list_of_inputs)
	layer_106_out = layer_106(layer_106_concat)
if layer_106.count_params() != 0:
	if 106 in set_weigths:
		if len(layer_106.get_weights()) > 1:
			new_w = [get_new_weigths(layer_106, set_weigths[106])]+layer_106.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_106, set_weigths[106])]
		layer_106.set_weights(new_w)


try:
	count = 1
	for s in layer_106_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #106:',count)
except AttributeError:
	for i in layer_106_out:
		count = 1
		for s in layer_106_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #106:',count)



layer_107 = L107()
try:
	try:
		layer_107_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_107_out = layer_107(layer_107_concat)
	except ValueError:
		layer_107_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_107_out = layer_107(layer_107_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_107 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_107_concat = strong_concat_107(new_list_of_inputs)
	layer_107_out = layer_107(layer_107_concat)
if layer_107.count_params() != 0:
	if 107 in set_weigths:
		if len(layer_107.get_weights()) > 1:
			new_w = [get_new_weigths(layer_107, set_weigths[107])]+layer_107.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_107, set_weigths[107])]
		layer_107.set_weights(new_w)


try:
	count = 1
	for s in layer_107_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #107:',count)
except AttributeError:
	for i in layer_107_out:
		count = 1
		for s in layer_107_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #107:',count)



layer_108 = L108()
try:
	try:
		layer_108_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_108_out = layer_108(layer_108_concat)
	except ValueError:
		layer_108_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_108_out = layer_108(layer_108_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_108 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_108_concat = strong_concat_108(new_list_of_inputs)
	layer_108_out = layer_108(layer_108_concat)
if layer_108.count_params() != 0:
	if 108 in set_weigths:
		if len(layer_108.get_weights()) > 1:
			new_w = [get_new_weigths(layer_108, set_weigths[108])]+layer_108.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_108, set_weigths[108])]
		layer_108.set_weights(new_w)


try:
	count = 1
	for s in layer_108_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #108:',count)
except AttributeError:
	for i in layer_108_out:
		count = 1
		for s in layer_108_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #108:',count)



layer_109 = L109()
try:
	try:
		layer_109_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_109_out = layer_109(layer_109_concat)
	except ValueError:
		layer_109_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_109_out = layer_109(layer_109_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_109 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_109_concat = strong_concat_109(new_list_of_inputs)
	layer_109_out = layer_109(layer_109_concat)
if layer_109.count_params() != 0:
	if 109 in set_weigths:
		if len(layer_109.get_weights()) > 1:
			new_w = [get_new_weigths(layer_109, set_weigths[109])]+layer_109.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_109, set_weigths[109])]
		layer_109.set_weights(new_w)


try:
	count = 1
	for s in layer_109_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #109:',count)
except AttributeError:
	for i in layer_109_out:
		count = 1
		for s in layer_109_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #109:',count)



layer_110 = L110()
try:
	try:
		layer_110_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_110_out = layer_110(layer_110_concat)
	except ValueError:
		layer_110_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_110_out = layer_110(layer_110_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_110 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_110_concat = strong_concat_110(new_list_of_inputs)
	layer_110_out = layer_110(layer_110_concat)
if layer_110.count_params() != 0:
	if 110 in set_weigths:
		if len(layer_110.get_weights()) > 1:
			new_w = [get_new_weigths(layer_110, set_weigths[110])]+layer_110.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_110, set_weigths[110])]
		layer_110.set_weights(new_w)


try:
	count = 1
	for s in layer_110_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #110:',count)
except AttributeError:
	for i in layer_110_out:
		count = 1
		for s in layer_110_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #110:',count)



layer_111 = L111()
try:
	try:
		layer_111_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_111_out = layer_111(layer_111_concat)
	except ValueError:
		layer_111_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_111_out = layer_111(layer_111_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_111 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_111_concat = strong_concat_111(new_list_of_inputs)
	layer_111_out = layer_111(layer_111_concat)
if layer_111.count_params() != 0:
	if 111 in set_weigths:
		if len(layer_111.get_weights()) > 1:
			new_w = [get_new_weigths(layer_111, set_weigths[111])]+layer_111.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_111, set_weigths[111])]
		layer_111.set_weights(new_w)


try:
	count = 1
	for s in layer_111_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #111:',count)
except AttributeError:
	for i in layer_111_out:
		count = 1
		for s in layer_111_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #111:',count)



layer_112 = L112()
try:
	try:
		layer_112_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_112_out = layer_112(layer_112_concat)
	except ValueError:
		layer_112_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_112_out = layer_112(layer_112_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_112 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_112_concat = strong_concat_112(new_list_of_inputs)
	layer_112_out = layer_112(layer_112_concat)
if layer_112.count_params() != 0:
	if 112 in set_weigths:
		if len(layer_112.get_weights()) > 1:
			new_w = [get_new_weigths(layer_112, set_weigths[112])]+layer_112.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_112, set_weigths[112])]
		layer_112.set_weights(new_w)


try:
	count = 1
	for s in layer_112_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #112:',count)
except AttributeError:
	for i in layer_112_out:
		count = 1
		for s in layer_112_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #112:',count)



layer_113 = L113()
try:
	try:
		layer_113_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_113_out = layer_113(layer_113_concat)
	except ValueError:
		layer_113_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_113_out = layer_113(layer_113_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_113 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_113_concat = strong_concat_113(new_list_of_inputs)
	layer_113_out = layer_113(layer_113_concat)
if layer_113.count_params() != 0:
	if 113 in set_weigths:
		if len(layer_113.get_weights()) > 1:
			new_w = [get_new_weigths(layer_113, set_weigths[113])]+layer_113.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_113, set_weigths[113])]
		layer_113.set_weights(new_w)


try:
	count = 1
	for s in layer_113_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #113:',count)
except AttributeError:
	for i in layer_113_out:
		count = 1
		for s in layer_113_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #113:',count)



layer_114 = L114()
try:
	try:
		layer_114_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_114_out = layer_114(layer_114_concat)
	except ValueError:
		layer_114_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_114_out = layer_114(layer_114_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_114 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_114_concat = strong_concat_114(new_list_of_inputs)
	layer_114_out = layer_114(layer_114_concat)
if layer_114.count_params() != 0:
	if 114 in set_weigths:
		if len(layer_114.get_weights()) > 1:
			new_w = [get_new_weigths(layer_114, set_weigths[114])]+layer_114.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_114, set_weigths[114])]
		layer_114.set_weights(new_w)


try:
	count = 1
	for s in layer_114_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #114:',count)
except AttributeError:
	for i in layer_114_out:
		count = 1
		for s in layer_114_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #114:',count)



layer_115 = L115()
try:
	try:
		layer_115_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_115_out = layer_115(layer_115_concat)
	except ValueError:
		layer_115_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_115_out = layer_115(layer_115_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_115 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_115_concat = strong_concat_115(new_list_of_inputs)
	layer_115_out = layer_115(layer_115_concat)
if layer_115.count_params() != 0:
	if 115 in set_weigths:
		if len(layer_115.get_weights()) > 1:
			new_w = [get_new_weigths(layer_115, set_weigths[115])]+layer_115.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_115, set_weigths[115])]
		layer_115.set_weights(new_w)


try:
	count = 1
	for s in layer_115_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #115:',count)
except AttributeError:
	for i in layer_115_out:
		count = 1
		for s in layer_115_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #115:',count)



layer_116 = L116()
try:
	try:
		layer_116_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_116_out = layer_116(layer_116_concat)
	except ValueError:
		layer_116_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_116_out = layer_116(layer_116_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_116 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_116_concat = strong_concat_116(new_list_of_inputs)
	layer_116_out = layer_116(layer_116_concat)
if layer_116.count_params() != 0:
	if 116 in set_weigths:
		if len(layer_116.get_weights()) > 1:
			new_w = [get_new_weigths(layer_116, set_weigths[116])]+layer_116.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_116, set_weigths[116])]
		layer_116.set_weights(new_w)


try:
	count = 1
	for s in layer_116_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #116:',count)
except AttributeError:
	for i in layer_116_out:
		count = 1
		for s in layer_116_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #116:',count)



layer_117 = L117()
try:
	try:
		layer_117_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_117_out = layer_117(layer_117_concat)
	except ValueError:
		layer_117_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_117_out = layer_117(layer_117_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_117 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_117_concat = strong_concat_117(new_list_of_inputs)
	layer_117_out = layer_117(layer_117_concat)
if layer_117.count_params() != 0:
	if 117 in set_weigths:
		if len(layer_117.get_weights()) > 1:
			new_w = [get_new_weigths(layer_117, set_weigths[117])]+layer_117.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_117, set_weigths[117])]
		layer_117.set_weights(new_w)


try:
	count = 1
	for s in layer_117_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #117:',count)
except AttributeError:
	for i in layer_117_out:
		count = 1
		for s in layer_117_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #117:',count)



layer_118 = L118()
try:
	try:
		layer_118_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_118_out = layer_118(layer_118_concat)
	except ValueError:
		layer_118_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_118_out = layer_118(layer_118_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_118 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_118_concat = strong_concat_118(new_list_of_inputs)
	layer_118_out = layer_118(layer_118_concat)
if layer_118.count_params() != 0:
	if 118 in set_weigths:
		if len(layer_118.get_weights()) > 1:
			new_w = [get_new_weigths(layer_118, set_weigths[118])]+layer_118.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_118, set_weigths[118])]
		layer_118.set_weights(new_w)


try:
	count = 1
	for s in layer_118_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #118:',count)
except AttributeError:
	for i in layer_118_out:
		count = 1
		for s in layer_118_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #118:',count)



layer_119 = L119()
try:
	try:
		layer_119_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_119_out = layer_119(layer_119_concat)
	except ValueError:
		layer_119_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_119_out = layer_119(layer_119_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_119 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_119_concat = strong_concat_119(new_list_of_inputs)
	layer_119_out = layer_119(layer_119_concat)
if layer_119.count_params() != 0:
	if 119 in set_weigths:
		if len(layer_119.get_weights()) > 1:
			new_w = [get_new_weigths(layer_119, set_weigths[119])]+layer_119.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_119, set_weigths[119])]
		layer_119.set_weights(new_w)


try:
	count = 1
	for s in layer_119_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #119:',count)
except AttributeError:
	for i in layer_119_out:
		count = 1
		for s in layer_119_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #119:',count)



layer_120 = L120()
try:
	try:
		layer_120_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_120_out = layer_120(layer_120_concat)
	except ValueError:
		layer_120_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_120_out = layer_120(layer_120_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_120 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_120_concat = strong_concat_120(new_list_of_inputs)
	layer_120_out = layer_120(layer_120_concat)
if layer_120.count_params() != 0:
	if 120 in set_weigths:
		if len(layer_120.get_weights()) > 1:
			new_w = [get_new_weigths(layer_120, set_weigths[120])]+layer_120.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_120, set_weigths[120])]
		layer_120.set_weights(new_w)


try:
	count = 1
	for s in layer_120_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #120:',count)
except AttributeError:
	for i in layer_120_out:
		count = 1
		for s in layer_120_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #120:',count)



layer_121 = L121()
try:
	try:
		layer_121_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_121_out = layer_121(layer_121_concat)
	except ValueError:
		layer_121_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_121_out = layer_121(layer_121_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_121 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_121_concat = strong_concat_121(new_list_of_inputs)
	layer_121_out = layer_121(layer_121_concat)
if layer_121.count_params() != 0:
	if 121 in set_weigths:
		if len(layer_121.get_weights()) > 1:
			new_w = [get_new_weigths(layer_121, set_weigths[121])]+layer_121.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_121, set_weigths[121])]
		layer_121.set_weights(new_w)


try:
	count = 1
	for s in layer_121_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #121:',count)
except AttributeError:
	for i in layer_121_out:
		count = 1
		for s in layer_121_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #121:',count)



layer_122 = L122()
try:
	try:
		layer_122_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_122_out = layer_122(layer_122_concat)
	except ValueError:
		layer_122_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_122_out = layer_122(layer_122_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_122 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_122_concat = strong_concat_122(new_list_of_inputs)
	layer_122_out = layer_122(layer_122_concat)
if layer_122.count_params() != 0:
	if 122 in set_weigths:
		if len(layer_122.get_weights()) > 1:
			new_w = [get_new_weigths(layer_122, set_weigths[122])]+layer_122.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_122, set_weigths[122])]
		layer_122.set_weights(new_w)


try:
	count = 1
	for s in layer_122_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #122:',count)
except AttributeError:
	for i in layer_122_out:
		count = 1
		for s in layer_122_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #122:',count)



layer_123 = L123()
try:
	try:
		layer_123_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_123_out = layer_123(layer_123_concat)
	except ValueError:
		layer_123_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_123_out = layer_123(layer_123_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_123 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_123_concat = strong_concat_123(new_list_of_inputs)
	layer_123_out = layer_123(layer_123_concat)
if layer_123.count_params() != 0:
	if 123 in set_weigths:
		if len(layer_123.get_weights()) > 1:
			new_w = [get_new_weigths(layer_123, set_weigths[123])]+layer_123.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_123, set_weigths[123])]
		layer_123.set_weights(new_w)


try:
	count = 1
	for s in layer_123_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #123:',count)
except AttributeError:
	for i in layer_123_out:
		count = 1
		for s in layer_123_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #123:',count)



layer_124 = L124()
try:
	try:
		layer_124_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_124_out = layer_124(layer_124_concat)
	except ValueError:
		layer_124_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_124_out = layer_124(layer_124_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_124 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_124_concat = strong_concat_124(new_list_of_inputs)
	layer_124_out = layer_124(layer_124_concat)
if layer_124.count_params() != 0:
	if 124 in set_weigths:
		if len(layer_124.get_weights()) > 1:
			new_w = [get_new_weigths(layer_124, set_weigths[124])]+layer_124.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_124, set_weigths[124])]
		layer_124.set_weights(new_w)


try:
	count = 1
	for s in layer_124_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #124:',count)
except AttributeError:
	for i in layer_124_out:
		count = 1
		for s in layer_124_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #124:',count)



layer_125 = L125()
try:
	try:
		layer_125_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_125_out = layer_125(layer_125_concat)
	except ValueError:
		layer_125_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_125_out = layer_125(layer_125_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_125 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_125_concat = strong_concat_125(new_list_of_inputs)
	layer_125_out = layer_125(layer_125_concat)
if layer_125.count_params() != 0:
	if 125 in set_weigths:
		if len(layer_125.get_weights()) > 1:
			new_w = [get_new_weigths(layer_125, set_weigths[125])]+layer_125.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_125, set_weigths[125])]
		layer_125.set_weights(new_w)


try:
	count = 1
	for s in layer_125_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #125:',count)
except AttributeError:
	for i in layer_125_out:
		count = 1
		for s in layer_125_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #125:',count)



layer_126 = L126()
try:
	try:
		layer_126_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_126_out = layer_126(layer_126_concat)
	except ValueError:
		layer_126_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_126_out = layer_126(layer_126_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_126 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_126_concat = strong_concat_126(new_list_of_inputs)
	layer_126_out = layer_126(layer_126_concat)
if layer_126.count_params() != 0:
	if 126 in set_weigths:
		if len(layer_126.get_weights()) > 1:
			new_w = [get_new_weigths(layer_126, set_weigths[126])]+layer_126.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_126, set_weigths[126])]
		layer_126.set_weights(new_w)


try:
	count = 1
	for s in layer_126_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #126:',count)
except AttributeError:
	for i in layer_126_out:
		count = 1
		for s in layer_126_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #126:',count)



layer_127 = L127()
try:
	try:
		layer_127_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_127_out = layer_127(layer_127_concat)
	except ValueError:
		layer_127_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_127_out = layer_127(layer_127_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_127 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_127_concat = strong_concat_127(new_list_of_inputs)
	layer_127_out = layer_127(layer_127_concat)
if layer_127.count_params() != 0:
	if 127 in set_weigths:
		if len(layer_127.get_weights()) > 1:
			new_w = [get_new_weigths(layer_127, set_weigths[127])]+layer_127.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_127, set_weigths[127])]
		layer_127.set_weights(new_w)


try:
	count = 1
	for s in layer_127_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #127:',count)
except AttributeError:
	for i in layer_127_out:
		count = 1
		for s in layer_127_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #127:',count)



layer_128 = L128()
try:
	try:
		layer_128_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_128_out = layer_128(layer_128_concat)
	except ValueError:
		layer_128_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_128_out = layer_128(layer_128_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_128 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_128_concat = strong_concat_128(new_list_of_inputs)
	layer_128_out = layer_128(layer_128_concat)
if layer_128.count_params() != 0:
	if 128 in set_weigths:
		if len(layer_128.get_weights()) > 1:
			new_w = [get_new_weigths(layer_128, set_weigths[128])]+layer_128.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_128, set_weigths[128])]
		layer_128.set_weights(new_w)


try:
	count = 1
	for s in layer_128_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #128:',count)
except AttributeError:
	for i in layer_128_out:
		count = 1
		for s in layer_128_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #128:',count)



layer_129 = L129()
try:
	try:
		layer_129_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_129_out = layer_129(layer_129_concat)
	except ValueError:
		layer_129_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_129_out = layer_129(layer_129_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_129 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_129_concat = strong_concat_129(new_list_of_inputs)
	layer_129_out = layer_129(layer_129_concat)
if layer_129.count_params() != 0:
	if 129 in set_weigths:
		if len(layer_129.get_weights()) > 1:
			new_w = [get_new_weigths(layer_129, set_weigths[129])]+layer_129.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_129, set_weigths[129])]
		layer_129.set_weights(new_w)


try:
	count = 1
	for s in layer_129_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #129:',count)
except AttributeError:
	for i in layer_129_out:
		count = 1
		for s in layer_129_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #129:',count)



layer_130 = L130()
try:
	try:
		layer_130_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_130_out = layer_130(layer_130_concat)
	except ValueError:
		layer_130_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_130_out = layer_130(layer_130_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_130 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_130_concat = strong_concat_130(new_list_of_inputs)
	layer_130_out = layer_130(layer_130_concat)
if layer_130.count_params() != 0:
	if 130 in set_weigths:
		if len(layer_130.get_weights()) > 1:
			new_w = [get_new_weigths(layer_130, set_weigths[130])]+layer_130.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_130, set_weigths[130])]
		layer_130.set_weights(new_w)


try:
	count = 1
	for s in layer_130_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #130:',count)
except AttributeError:
	for i in layer_130_out:
		count = 1
		for s in layer_130_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #130:',count)



layer_131 = L131()
try:
	try:
		layer_131_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_131_out = layer_131(layer_131_concat)
	except ValueError:
		layer_131_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_131_out = layer_131(layer_131_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_131 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_131_concat = strong_concat_131(new_list_of_inputs)
	layer_131_out = layer_131(layer_131_concat)
if layer_131.count_params() != 0:
	if 131 in set_weigths:
		if len(layer_131.get_weights()) > 1:
			new_w = [get_new_weigths(layer_131, set_weigths[131])]+layer_131.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_131, set_weigths[131])]
		layer_131.set_weights(new_w)


try:
	count = 1
	for s in layer_131_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #131:',count)
except AttributeError:
	for i in layer_131_out:
		count = 1
		for s in layer_131_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #131:',count)



layer_132 = L132()
try:
	try:
		layer_132_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_132_out = layer_132(layer_132_concat)
	except ValueError:
		layer_132_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_132_out = layer_132(layer_132_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_132 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_132_concat = strong_concat_132(new_list_of_inputs)
	layer_132_out = layer_132(layer_132_concat)
if layer_132.count_params() != 0:
	if 132 in set_weigths:
		if len(layer_132.get_weights()) > 1:
			new_w = [get_new_weigths(layer_132, set_weigths[132])]+layer_132.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_132, set_weigths[132])]
		layer_132.set_weights(new_w)


try:
	count = 1
	for s in layer_132_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #132:',count)
except AttributeError:
	for i in layer_132_out:
		count = 1
		for s in layer_132_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #132:',count)



layer_133 = L133()
try:
	try:
		layer_133_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_133_out = layer_133(layer_133_concat)
	except ValueError:
		layer_133_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_133_out = layer_133(layer_133_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_133 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_133_concat = strong_concat_133(new_list_of_inputs)
	layer_133_out = layer_133(layer_133_concat)
if layer_133.count_params() != 0:
	if 133 in set_weigths:
		if len(layer_133.get_weights()) > 1:
			new_w = [get_new_weigths(layer_133, set_weigths[133])]+layer_133.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_133, set_weigths[133])]
		layer_133.set_weights(new_w)


try:
	count = 1
	for s in layer_133_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #133:',count)
except AttributeError:
	for i in layer_133_out:
		count = 1
		for s in layer_133_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #133:',count)



layer_134 = L134()
try:
	try:
		layer_134_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_134_out = layer_134(layer_134_concat)
	except ValueError:
		layer_134_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_134_out = layer_134(layer_134_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_134 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_134_concat = strong_concat_134(new_list_of_inputs)
	layer_134_out = layer_134(layer_134_concat)
if layer_134.count_params() != 0:
	if 134 in set_weigths:
		if len(layer_134.get_weights()) > 1:
			new_w = [get_new_weigths(layer_134, set_weigths[134])]+layer_134.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_134, set_weigths[134])]
		layer_134.set_weights(new_w)


try:
	count = 1
	for s in layer_134_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #134:',count)
except AttributeError:
	for i in layer_134_out:
		count = 1
		for s in layer_134_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #134:',count)



layer_135 = L135()
try:
	try:
		layer_135_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_135_out = layer_135(layer_135_concat)
	except ValueError:
		layer_135_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_135_out = layer_135(layer_135_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_135 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_135_concat = strong_concat_135(new_list_of_inputs)
	layer_135_out = layer_135(layer_135_concat)
if layer_135.count_params() != 0:
	if 135 in set_weigths:
		if len(layer_135.get_weights()) > 1:
			new_w = [get_new_weigths(layer_135, set_weigths[135])]+layer_135.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_135, set_weigths[135])]
		layer_135.set_weights(new_w)


try:
	count = 1
	for s in layer_135_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #135:',count)
except AttributeError:
	for i in layer_135_out:
		count = 1
		for s in layer_135_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #135:',count)



layer_136 = L136()
try:
	try:
		layer_136_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_136_out = layer_136(layer_136_concat)
	except ValueError:
		layer_136_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_136_out = layer_136(layer_136_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_136 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_136_concat = strong_concat_136(new_list_of_inputs)
	layer_136_out = layer_136(layer_136_concat)
if layer_136.count_params() != 0:
	if 136 in set_weigths:
		if len(layer_136.get_weights()) > 1:
			new_w = [get_new_weigths(layer_136, set_weigths[136])]+layer_136.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_136, set_weigths[136])]
		layer_136.set_weights(new_w)


try:
	count = 1
	for s in layer_136_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #136:',count)
except AttributeError:
	for i in layer_136_out:
		count = 1
		for s in layer_136_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #136:',count)



layer_137 = L137()
try:
	try:
		layer_137_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_137_out = layer_137(layer_137_concat)
	except ValueError:
		layer_137_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_137_out = layer_137(layer_137_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_137 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_137_concat = strong_concat_137(new_list_of_inputs)
	layer_137_out = layer_137(layer_137_concat)
if layer_137.count_params() != 0:
	if 137 in set_weigths:
		if len(layer_137.get_weights()) > 1:
			new_w = [get_new_weigths(layer_137, set_weigths[137])]+layer_137.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_137, set_weigths[137])]
		layer_137.set_weights(new_w)


try:
	count = 1
	for s in layer_137_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #137:',count)
except AttributeError:
	for i in layer_137_out:
		count = 1
		for s in layer_137_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #137:',count)



layer_138 = L138()
try:
	try:
		layer_138_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_138_out = layer_138(layer_138_concat)
	except ValueError:
		layer_138_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_138_out = layer_138(layer_138_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_138 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_138_concat = strong_concat_138(new_list_of_inputs)
	layer_138_out = layer_138(layer_138_concat)
if layer_138.count_params() != 0:
	if 138 in set_weigths:
		if len(layer_138.get_weights()) > 1:
			new_w = [get_new_weigths(layer_138, set_weigths[138])]+layer_138.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_138, set_weigths[138])]
		layer_138.set_weights(new_w)


try:
	count = 1
	for s in layer_138_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #138:',count)
except AttributeError:
	for i in layer_138_out:
		count = 1
		for s in layer_138_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #138:',count)



layer_139 = L139()
try:
	try:
		layer_139_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_139_out = layer_139(layer_139_concat)
	except ValueError:
		layer_139_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_139_out = layer_139(layer_139_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_139 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_139_concat = strong_concat_139(new_list_of_inputs)
	layer_139_out = layer_139(layer_139_concat)
if layer_139.count_params() != 0:
	if 139 in set_weigths:
		if len(layer_139.get_weights()) > 1:
			new_w = [get_new_weigths(layer_139, set_weigths[139])]+layer_139.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_139, set_weigths[139])]
		layer_139.set_weights(new_w)


try:
	count = 1
	for s in layer_139_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #139:',count)
except AttributeError:
	for i in layer_139_out:
		count = 1
		for s in layer_139_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #139:',count)



layer_140 = L140()
try:
	try:
		layer_140_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_140_out = layer_140(layer_140_concat)
	except ValueError:
		layer_140_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_140_out = layer_140(layer_140_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_140 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_140_concat = strong_concat_140(new_list_of_inputs)
	layer_140_out = layer_140(layer_140_concat)
if layer_140.count_params() != 0:
	if 140 in set_weigths:
		if len(layer_140.get_weights()) > 1:
			new_w = [get_new_weigths(layer_140, set_weigths[140])]+layer_140.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_140, set_weigths[140])]
		layer_140.set_weights(new_w)


try:
	count = 1
	for s in layer_140_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #140:',count)
except AttributeError:
	for i in layer_140_out:
		count = 1
		for s in layer_140_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #140:',count)



layer_141 = L141()
try:
	try:
		layer_141_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_141_out = layer_141(layer_141_concat)
	except ValueError:
		layer_141_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_141_out = layer_141(layer_141_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_141 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_141_concat = strong_concat_141(new_list_of_inputs)
	layer_141_out = layer_141(layer_141_concat)
if layer_141.count_params() != 0:
	if 141 in set_weigths:
		if len(layer_141.get_weights()) > 1:
			new_w = [get_new_weigths(layer_141, set_weigths[141])]+layer_141.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_141, set_weigths[141])]
		layer_141.set_weights(new_w)


try:
	count = 1
	for s in layer_141_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #141:',count)
except AttributeError:
	for i in layer_141_out:
		count = 1
		for s in layer_141_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #141:',count)



layer_142 = L142()
try:
	try:
		layer_142_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_142_out = layer_142(layer_142_concat)
	except ValueError:
		layer_142_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_142_out = layer_142(layer_142_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_142 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_142_concat = strong_concat_142(new_list_of_inputs)
	layer_142_out = layer_142(layer_142_concat)
if layer_142.count_params() != 0:
	if 142 in set_weigths:
		if len(layer_142.get_weights()) > 1:
			new_w = [get_new_weigths(layer_142, set_weigths[142])]+layer_142.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_142, set_weigths[142])]
		layer_142.set_weights(new_w)


try:
	count = 1
	for s in layer_142_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #142:',count)
except AttributeError:
	for i in layer_142_out:
		count = 1
		for s in layer_142_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #142:',count)



layer_143 = L143()
try:
	try:
		layer_143_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_143_out = layer_143(layer_143_concat)
	except ValueError:
		layer_143_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_143_out = layer_143(layer_143_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_143 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_143_concat = strong_concat_143(new_list_of_inputs)
	layer_143_out = layer_143(layer_143_concat)
if layer_143.count_params() != 0:
	if 143 in set_weigths:
		if len(layer_143.get_weights()) > 1:
			new_w = [get_new_weigths(layer_143, set_weigths[143])]+layer_143.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_143, set_weigths[143])]
		layer_143.set_weights(new_w)


try:
	count = 1
	for s in layer_143_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #143:',count)
except AttributeError:
	for i in layer_143_out:
		count = 1
		for s in layer_143_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #143:',count)



layer_144 = L144()
try:
	try:
		layer_144_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_144_out = layer_144(layer_144_concat)
	except ValueError:
		layer_144_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_144_out = layer_144(layer_144_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_144 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_144_concat = strong_concat_144(new_list_of_inputs)
	layer_144_out = layer_144(layer_144_concat)
if layer_144.count_params() != 0:
	if 144 in set_weigths:
		if len(layer_144.get_weights()) > 1:
			new_w = [get_new_weigths(layer_144, set_weigths[144])]+layer_144.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_144, set_weigths[144])]
		layer_144.set_weights(new_w)


try:
	count = 1
	for s in layer_144_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #144:',count)
except AttributeError:
	for i in layer_144_out:
		count = 1
		for s in layer_144_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #144:',count)



layer_145 = L145()
try:
	try:
		layer_145_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_145_out = layer_145(layer_145_concat)
	except ValueError:
		layer_145_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_145_out = layer_145(layer_145_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_145 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_145_concat = strong_concat_145(new_list_of_inputs)
	layer_145_out = layer_145(layer_145_concat)
if layer_145.count_params() != 0:
	if 145 in set_weigths:
		if len(layer_145.get_weights()) > 1:
			new_w = [get_new_weigths(layer_145, set_weigths[145])]+layer_145.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_145, set_weigths[145])]
		layer_145.set_weights(new_w)


try:
	count = 1
	for s in layer_145_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #145:',count)
except AttributeError:
	for i in layer_145_out:
		count = 1
		for s in layer_145_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #145:',count)



layer_146 = L146()
try:
	try:
		layer_146_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_146_out = layer_146(layer_146_concat)
	except ValueError:
		layer_146_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_146_out = layer_146(layer_146_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_146 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_146_concat = strong_concat_146(new_list_of_inputs)
	layer_146_out = layer_146(layer_146_concat)
if layer_146.count_params() != 0:
	if 146 in set_weigths:
		if len(layer_146.get_weights()) > 1:
			new_w = [get_new_weigths(layer_146, set_weigths[146])]+layer_146.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_146, set_weigths[146])]
		layer_146.set_weights(new_w)


try:
	count = 1
	for s in layer_146_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #146:',count)
except AttributeError:
	for i in layer_146_out:
		count = 1
		for s in layer_146_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #146:',count)



layer_147 = L147()
try:
	try:
		layer_147_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_147_out = layer_147(layer_147_concat)
	except ValueError:
		layer_147_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_147_out = layer_147(layer_147_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_147 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_147_concat = strong_concat_147(new_list_of_inputs)
	layer_147_out = layer_147(layer_147_concat)
if layer_147.count_params() != 0:
	if 147 in set_weigths:
		if len(layer_147.get_weights()) > 1:
			new_w = [get_new_weigths(layer_147, set_weigths[147])]+layer_147.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_147, set_weigths[147])]
		layer_147.set_weights(new_w)


try:
	count = 1
	for s in layer_147_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #147:',count)
except AttributeError:
	for i in layer_147_out:
		count = 1
		for s in layer_147_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #147:',count)



layer_148 = L148()
try:
	try:
		layer_148_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_148_out = layer_148(layer_148_concat)
	except ValueError:
		layer_148_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_148_out = layer_148(layer_148_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_148 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_148_concat = strong_concat_148(new_list_of_inputs)
	layer_148_out = layer_148(layer_148_concat)
if layer_148.count_params() != 0:
	if 148 in set_weigths:
		if len(layer_148.get_weights()) > 1:
			new_w = [get_new_weigths(layer_148, set_weigths[148])]+layer_148.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_148, set_weigths[148])]
		layer_148.set_weights(new_w)


try:
	count = 1
	for s in layer_148_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #148:',count)
except AttributeError:
	for i in layer_148_out:
		count = 1
		for s in layer_148_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #148:',count)



layer_149 = L149()
try:
	try:
		layer_149_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_149_out = layer_149(layer_149_concat)
	except ValueError:
		layer_149_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_149_out = layer_149(layer_149_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_149 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_149_concat = strong_concat_149(new_list_of_inputs)
	layer_149_out = layer_149(layer_149_concat)
if layer_149.count_params() != 0:
	if 149 in set_weigths:
		if len(layer_149.get_weights()) > 1:
			new_w = [get_new_weigths(layer_149, set_weigths[149])]+layer_149.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_149, set_weigths[149])]
		layer_149.set_weights(new_w)


try:
	count = 1
	for s in layer_149_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #149:',count)
except AttributeError:
	for i in layer_149_out:
		count = 1
		for s in layer_149_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #149:',count)



layer_150 = L150()
try:
	try:
		layer_150_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_150_out = layer_150(layer_150_concat)
	except ValueError:
		layer_150_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_150_out = layer_150(layer_150_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_150 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_150_concat = strong_concat_150(new_list_of_inputs)
	layer_150_out = layer_150(layer_150_concat)
if layer_150.count_params() != 0:
	if 150 in set_weigths:
		if len(layer_150.get_weights()) > 1:
			new_w = [get_new_weigths(layer_150, set_weigths[150])]+layer_150.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_150, set_weigths[150])]
		layer_150.set_weights(new_w)


try:
	count = 1
	for s in layer_150_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #150:',count)
except AttributeError:
	for i in layer_150_out:
		count = 1
		for s in layer_150_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #150:',count)



layer_151 = L151()
try:
	try:
		layer_151_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_151_out = layer_151(layer_151_concat)
	except ValueError:
		layer_151_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_151_out = layer_151(layer_151_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_151 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_151_concat = strong_concat_151(new_list_of_inputs)
	layer_151_out = layer_151(layer_151_concat)
if layer_151.count_params() != 0:
	if 151 in set_weigths:
		if len(layer_151.get_weights()) > 1:
			new_w = [get_new_weigths(layer_151, set_weigths[151])]+layer_151.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_151, set_weigths[151])]
		layer_151.set_weights(new_w)


try:
	count = 1
	for s in layer_151_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #151:',count)
except AttributeError:
	for i in layer_151_out:
		count = 1
		for s in layer_151_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #151:',count)



layer_152 = L152()
try:
	try:
		layer_152_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_152_out = layer_152(layer_152_concat)
	except ValueError:
		layer_152_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_152_out = layer_152(layer_152_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_152 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_152_concat = strong_concat_152(new_list_of_inputs)
	layer_152_out = layer_152(layer_152_concat)
if layer_152.count_params() != 0:
	if 152 in set_weigths:
		if len(layer_152.get_weights()) > 1:
			new_w = [get_new_weigths(layer_152, set_weigths[152])]+layer_152.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_152, set_weigths[152])]
		layer_152.set_weights(new_w)


try:
	count = 1
	for s in layer_152_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #152:',count)
except AttributeError:
	for i in layer_152_out:
		count = 1
		for s in layer_152_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #152:',count)



layer_153 = L153()
try:
	try:
		layer_153_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_153_out = layer_153(layer_153_concat)
	except ValueError:
		layer_153_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_153_out = layer_153(layer_153_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_153 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_153_concat = strong_concat_153(new_list_of_inputs)
	layer_153_out = layer_153(layer_153_concat)
if layer_153.count_params() != 0:
	if 153 in set_weigths:
		if len(layer_153.get_weights()) > 1:
			new_w = [get_new_weigths(layer_153, set_weigths[153])]+layer_153.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_153, set_weigths[153])]
		layer_153.set_weights(new_w)


try:
	count = 1
	for s in layer_153_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #153:',count)
except AttributeError:
	for i in layer_153_out:
		count = 1
		for s in layer_153_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #153:',count)



layer_154 = L154()
try:
	try:
		layer_154_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_154_out = layer_154(layer_154_concat)
	except ValueError:
		layer_154_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_154_out = layer_154(layer_154_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_154 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_154_concat = strong_concat_154(new_list_of_inputs)
	layer_154_out = layer_154(layer_154_concat)
if layer_154.count_params() != 0:
	if 154 in set_weigths:
		if len(layer_154.get_weights()) > 1:
			new_w = [get_new_weigths(layer_154, set_weigths[154])]+layer_154.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_154, set_weigths[154])]
		layer_154.set_weights(new_w)


try:
	count = 1
	for s in layer_154_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #154:',count)
except AttributeError:
	for i in layer_154_out:
		count = 1
		for s in layer_154_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #154:',count)



layer_155 = L155()
try:
	try:
		layer_155_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_155_out = layer_155(layer_155_concat)
	except ValueError:
		layer_155_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_155_out = layer_155(layer_155_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_155 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_155_concat = strong_concat_155(new_list_of_inputs)
	layer_155_out = layer_155(layer_155_concat)
if layer_155.count_params() != 0:
	if 155 in set_weigths:
		if len(layer_155.get_weights()) > 1:
			new_w = [get_new_weigths(layer_155, set_weigths[155])]+layer_155.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_155, set_weigths[155])]
		layer_155.set_weights(new_w)


try:
	count = 1
	for s in layer_155_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #155:',count)
except AttributeError:
	for i in layer_155_out:
		count = 1
		for s in layer_155_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #155:',count)



layer_156 = L156()
try:
	try:
		layer_156_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_156_out = layer_156(layer_156_concat)
	except ValueError:
		layer_156_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_156_out = layer_156(layer_156_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_156 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_156_concat = strong_concat_156(new_list_of_inputs)
	layer_156_out = layer_156(layer_156_concat)
if layer_156.count_params() != 0:
	if 156 in set_weigths:
		if len(layer_156.get_weights()) > 1:
			new_w = [get_new_weigths(layer_156, set_weigths[156])]+layer_156.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_156, set_weigths[156])]
		layer_156.set_weights(new_w)


try:
	count = 1
	for s in layer_156_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #156:',count)
except AttributeError:
	for i in layer_156_out:
		count = 1
		for s in layer_156_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #156:',count)



layer_157 = L157()
try:
	try:
		layer_157_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_157_out = layer_157(layer_157_concat)
	except ValueError:
		layer_157_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_157_out = layer_157(layer_157_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_157 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_157_concat = strong_concat_157(new_list_of_inputs)
	layer_157_out = layer_157(layer_157_concat)
if layer_157.count_params() != 0:
	if 157 in set_weigths:
		if len(layer_157.get_weights()) > 1:
			new_w = [get_new_weigths(layer_157, set_weigths[157])]+layer_157.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_157, set_weigths[157])]
		layer_157.set_weights(new_w)


try:
	count = 1
	for s in layer_157_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #157:',count)
except AttributeError:
	for i in layer_157_out:
		count = 1
		for s in layer_157_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #157:',count)



layer_158 = L158()
try:
	try:
		layer_158_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_158_out = layer_158(layer_158_concat)
	except ValueError:
		layer_158_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_158_out = layer_158(layer_158_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_158 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_158_concat = strong_concat_158(new_list_of_inputs)
	layer_158_out = layer_158(layer_158_concat)
if layer_158.count_params() != 0:
	if 158 in set_weigths:
		if len(layer_158.get_weights()) > 1:
			new_w = [get_new_weigths(layer_158, set_weigths[158])]+layer_158.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_158, set_weigths[158])]
		layer_158.set_weights(new_w)


try:
	count = 1
	for s in layer_158_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #158:',count)
except AttributeError:
	for i in layer_158_out:
		count = 1
		for s in layer_158_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #158:',count)



layer_159 = L159()
try:
	try:
		layer_159_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_159_out = layer_159(layer_159_concat)
	except ValueError:
		layer_159_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_159_out = layer_159(layer_159_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_159 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_159_concat = strong_concat_159(new_list_of_inputs)
	layer_159_out = layer_159(layer_159_concat)
if layer_159.count_params() != 0:
	if 159 in set_weigths:
		if len(layer_159.get_weights()) > 1:
			new_w = [get_new_weigths(layer_159, set_weigths[159])]+layer_159.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_159, set_weigths[159])]
		layer_159.set_weights(new_w)


try:
	count = 1
	for s in layer_159_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #159:',count)
except AttributeError:
	for i in layer_159_out:
		count = 1
		for s in layer_159_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #159:',count)



layer_160 = L160()
try:
	try:
		layer_160_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_160_out = layer_160(layer_160_concat)
	except ValueError:
		layer_160_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_160_out = layer_160(layer_160_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_160 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_160_concat = strong_concat_160(new_list_of_inputs)
	layer_160_out = layer_160(layer_160_concat)
if layer_160.count_params() != 0:
	if 160 in set_weigths:
		if len(layer_160.get_weights()) > 1:
			new_w = [get_new_weigths(layer_160, set_weigths[160])]+layer_160.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_160, set_weigths[160])]
		layer_160.set_weights(new_w)


try:
	count = 1
	for s in layer_160_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #160:',count)
except AttributeError:
	for i in layer_160_out:
		count = 1
		for s in layer_160_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #160:',count)



layer_161 = L161()
try:
	try:
		layer_161_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_161_out = layer_161(layer_161_concat)
	except ValueError:
		layer_161_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_161_out = layer_161(layer_161_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_161 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_161_concat = strong_concat_161(new_list_of_inputs)
	layer_161_out = layer_161(layer_161_concat)
if layer_161.count_params() != 0:
	if 161 in set_weigths:
		if len(layer_161.get_weights()) > 1:
			new_w = [get_new_weigths(layer_161, set_weigths[161])]+layer_161.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_161, set_weigths[161])]
		layer_161.set_weights(new_w)


try:
	count = 1
	for s in layer_161_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #161:',count)
except AttributeError:
	for i in layer_161_out:
		count = 1
		for s in layer_161_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #161:',count)



layer_162 = L162()
try:
	try:
		layer_162_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_162_out = layer_162(layer_162_concat)
	except ValueError:
		layer_162_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_162_out = layer_162(layer_162_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_162 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_162_concat = strong_concat_162(new_list_of_inputs)
	layer_162_out = layer_162(layer_162_concat)
if layer_162.count_params() != 0:
	if 162 in set_weigths:
		if len(layer_162.get_weights()) > 1:
			new_w = [get_new_weigths(layer_162, set_weigths[162])]+layer_162.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_162, set_weigths[162])]
		layer_162.set_weights(new_w)


try:
	count = 1
	for s in layer_162_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #162:',count)
except AttributeError:
	for i in layer_162_out:
		count = 1
		for s in layer_162_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #162:',count)



layer_163 = L163()
try:
	try:
		layer_163_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_163_out = layer_163(layer_163_concat)
	except ValueError:
		layer_163_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_163_out = layer_163(layer_163_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_163 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_163_concat = strong_concat_163(new_list_of_inputs)
	layer_163_out = layer_163(layer_163_concat)
if layer_163.count_params() != 0:
	if 163 in set_weigths:
		if len(layer_163.get_weights()) > 1:
			new_w = [get_new_weigths(layer_163, set_weigths[163])]+layer_163.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_163, set_weigths[163])]
		layer_163.set_weights(new_w)


try:
	count = 1
	for s in layer_163_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #163:',count)
except AttributeError:
	for i in layer_163_out:
		count = 1
		for s in layer_163_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #163:',count)



layer_164 = L164()
try:
	try:
		layer_164_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_164_out = layer_164(layer_164_concat)
	except ValueError:
		layer_164_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_164_out = layer_164(layer_164_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_164 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_164_concat = strong_concat_164(new_list_of_inputs)
	layer_164_out = layer_164(layer_164_concat)
if layer_164.count_params() != 0:
	if 164 in set_weigths:
		if len(layer_164.get_weights()) > 1:
			new_w = [get_new_weigths(layer_164, set_weigths[164])]+layer_164.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_164, set_weigths[164])]
		layer_164.set_weights(new_w)


try:
	count = 1
	for s in layer_164_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #164:',count)
except AttributeError:
	for i in layer_164_out:
		count = 1
		for s in layer_164_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #164:',count)



layer_165 = L165()
try:
	try:
		layer_165_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_165_out = layer_165(layer_165_concat)
	except ValueError:
		layer_165_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_165_out = layer_165(layer_165_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_165 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_165_concat = strong_concat_165(new_list_of_inputs)
	layer_165_out = layer_165(layer_165_concat)
if layer_165.count_params() != 0:
	if 165 in set_weigths:
		if len(layer_165.get_weights()) > 1:
			new_w = [get_new_weigths(layer_165, set_weigths[165])]+layer_165.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_165, set_weigths[165])]
		layer_165.set_weights(new_w)


try:
	count = 1
	for s in layer_165_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #165:',count)
except AttributeError:
	for i in layer_165_out:
		count = 1
		for s in layer_165_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #165:',count)



layer_166 = L166()
try:
	try:
		layer_166_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_166_out = layer_166(layer_166_concat)
	except ValueError:
		layer_166_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_166_out = layer_166(layer_166_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_166 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_166_concat = strong_concat_166(new_list_of_inputs)
	layer_166_out = layer_166(layer_166_concat)
if layer_166.count_params() != 0:
	if 166 in set_weigths:
		if len(layer_166.get_weights()) > 1:
			new_w = [get_new_weigths(layer_166, set_weigths[166])]+layer_166.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_166, set_weigths[166])]
		layer_166.set_weights(new_w)


try:
	count = 1
	for s in layer_166_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #166:',count)
except AttributeError:
	for i in layer_166_out:
		count = 1
		for s in layer_166_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #166:',count)



layer_167 = L167()
try:
	try:
		layer_167_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_167_out = layer_167(layer_167_concat)
	except ValueError:
		layer_167_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_167_out = layer_167(layer_167_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_167 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_167_concat = strong_concat_167(new_list_of_inputs)
	layer_167_out = layer_167(layer_167_concat)
if layer_167.count_params() != 0:
	if 167 in set_weigths:
		if len(layer_167.get_weights()) > 1:
			new_w = [get_new_weigths(layer_167, set_weigths[167])]+layer_167.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_167, set_weigths[167])]
		layer_167.set_weights(new_w)


try:
	count = 1
	for s in layer_167_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #167:',count)
except AttributeError:
	for i in layer_167_out:
		count = 1
		for s in layer_167_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #167:',count)



layer_168 = L168()
try:
	try:
		layer_168_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_168_out = layer_168(layer_168_concat)
	except ValueError:
		layer_168_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_168_out = layer_168(layer_168_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_168 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_168_concat = strong_concat_168(new_list_of_inputs)
	layer_168_out = layer_168(layer_168_concat)
if layer_168.count_params() != 0:
	if 168 in set_weigths:
		if len(layer_168.get_weights()) > 1:
			new_w = [get_new_weigths(layer_168, set_weigths[168])]+layer_168.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_168, set_weigths[168])]
		layer_168.set_weights(new_w)


try:
	count = 1
	for s in layer_168_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #168:',count)
except AttributeError:
	for i in layer_168_out:
		count = 1
		for s in layer_168_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #168:',count)



layer_169 = L169()
try:
	try:
		layer_169_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_169_out = layer_169(layer_169_concat)
	except ValueError:
		layer_169_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_169_out = layer_169(layer_169_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_169 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_169_concat = strong_concat_169(new_list_of_inputs)
	layer_169_out = layer_169(layer_169_concat)
if layer_169.count_params() != 0:
	if 169 in set_weigths:
		if len(layer_169.get_weights()) > 1:
			new_w = [get_new_weigths(layer_169, set_weigths[169])]+layer_169.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_169, set_weigths[169])]
		layer_169.set_weights(new_w)


try:
	count = 1
	for s in layer_169_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #169:',count)
except AttributeError:
	for i in layer_169_out:
		count = 1
		for s in layer_169_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #169:',count)



layer_170 = L170()
try:
	try:
		layer_170_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_170_out = layer_170(layer_170_concat)
	except ValueError:
		layer_170_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_170_out = layer_170(layer_170_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_170 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_170_concat = strong_concat_170(new_list_of_inputs)
	layer_170_out = layer_170(layer_170_concat)
if layer_170.count_params() != 0:
	if 170 in set_weigths:
		if len(layer_170.get_weights()) > 1:
			new_w = [get_new_weigths(layer_170, set_weigths[170])]+layer_170.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_170, set_weigths[170])]
		layer_170.set_weights(new_w)


try:
	count = 1
	for s in layer_170_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #170:',count)
except AttributeError:
	for i in layer_170_out:
		count = 1
		for s in layer_170_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #170:',count)



layer_171 = L171()
try:
	try:
		layer_171_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_171_out = layer_171(layer_171_concat)
	except ValueError:
		layer_171_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_171_out = layer_171(layer_171_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_171 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_171_concat = strong_concat_171(new_list_of_inputs)
	layer_171_out = layer_171(layer_171_concat)
if layer_171.count_params() != 0:
	if 171 in set_weigths:
		if len(layer_171.get_weights()) > 1:
			new_w = [get_new_weigths(layer_171, set_weigths[171])]+layer_171.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_171, set_weigths[171])]
		layer_171.set_weights(new_w)


try:
	count = 1
	for s in layer_171_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #171:',count)
except AttributeError:
	for i in layer_171_out:
		count = 1
		for s in layer_171_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #171:',count)



layer_172 = L172()
try:
	try:
		layer_172_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_172_out = layer_172(layer_172_concat)
	except ValueError:
		layer_172_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_172_out = layer_172(layer_172_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_172 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_172_concat = strong_concat_172(new_list_of_inputs)
	layer_172_out = layer_172(layer_172_concat)
if layer_172.count_params() != 0:
	if 172 in set_weigths:
		if len(layer_172.get_weights()) > 1:
			new_w = [get_new_weigths(layer_172, set_weigths[172])]+layer_172.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_172, set_weigths[172])]
		layer_172.set_weights(new_w)


try:
	count = 1
	for s in layer_172_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #172:',count)
except AttributeError:
	for i in layer_172_out:
		count = 1
		for s in layer_172_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #172:',count)



layer_182 = L182()
try:
	try:
		layer_182_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_345_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_182_out = layer_182(layer_182_concat)
	except ValueError:
		layer_182_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_345_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_182_out = layer_182(layer_182_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_182 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_345_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_182_concat = strong_concat_182(new_list_of_inputs)
	layer_182_out = layer_182(layer_182_concat)
if layer_182.count_params() != 0:
	if 182 in set_weigths:
		if len(layer_182.get_weights()) > 1:
			new_w = [get_new_weigths(layer_182, set_weigths[182])]+layer_182.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_182, set_weigths[182])]
		layer_182.set_weights(new_w)


try:
	count = 1
	for s in layer_182_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #182:',count)
except AttributeError:
	for i in layer_182_out:
		count = 1
		for s in layer_182_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #182:',count)



layer_183 = L183()
try:
	try:
		layer_183_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_345_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_183_out = layer_183(layer_183_concat)
	except ValueError:
		layer_183_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_345_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_183_out = layer_183(layer_183_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_183 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_345_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_183_concat = strong_concat_183(new_list_of_inputs)
	layer_183_out = layer_183(layer_183_concat)
if layer_183.count_params() != 0:
	if 183 in set_weigths:
		if len(layer_183.get_weights()) > 1:
			new_w = [get_new_weigths(layer_183, set_weigths[183])]+layer_183.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_183, set_weigths[183])]
		layer_183.set_weights(new_w)


try:
	count = 1
	for s in layer_183_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #183:',count)
except AttributeError:
	for i in layer_183_out:
		count = 1
		for s in layer_183_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #183:',count)



layer_184 = L184()
try:
	try:
		layer_184_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_345_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_184_out = layer_184(layer_184_concat)
	except ValueError:
		layer_184_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_345_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_184_out = layer_184(layer_184_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_184 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_345_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_184_concat = strong_concat_184(new_list_of_inputs)
	layer_184_out = layer_184(layer_184_concat)
if layer_184.count_params() != 0:
	if 184 in set_weigths:
		if len(layer_184.get_weights()) > 1:
			new_w = [get_new_weigths(layer_184, set_weigths[184])]+layer_184.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_184, set_weigths[184])]
		layer_184.set_weights(new_w)


try:
	count = 1
	for s in layer_184_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #184:',count)
except AttributeError:
	for i in layer_184_out:
		count = 1
		for s in layer_184_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #184:',count)



layer_185 = L185()
try:
	try:
		layer_185_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_345_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_185_out = layer_185(layer_185_concat)
	except ValueError:
		layer_185_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_345_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_185_out = layer_185(layer_185_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_185 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_345_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_185_concat = strong_concat_185(new_list_of_inputs)
	layer_185_out = layer_185(layer_185_concat)
if layer_185.count_params() != 0:
	if 185 in set_weigths:
		if len(layer_185.get_weights()) > 1:
			new_w = [get_new_weigths(layer_185, set_weigths[185])]+layer_185.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_185, set_weigths[185])]
		layer_185.set_weights(new_w)


try:
	count = 1
	for s in layer_185_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #185:',count)
except AttributeError:
	for i in layer_185_out:
		count = 1
		for s in layer_185_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #185:',count)



layer_247 = L247()
try:
	try:
		layer_247_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_247_out = layer_247(layer_247_concat)
	except ValueError:
		layer_247_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_247_out = layer_247(layer_247_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_247 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_247_concat = strong_concat_247(new_list_of_inputs)
	layer_247_out = layer_247(layer_247_concat)
if layer_247.count_params() != 0:
	if 247 in set_weigths:
		if len(layer_247.get_weights()) > 1:
			new_w = [get_new_weigths(layer_247, set_weigths[247])]+layer_247.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_247, set_weigths[247])]
		layer_247.set_weights(new_w)


try:
	count = 1
	for s in layer_247_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #247:',count)
except AttributeError:
	for i in layer_247_out:
		count = 1
		for s in layer_247_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #247:',count)



layer_248 = L248()
try:
	try:
		layer_248_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_248_out = layer_248(layer_248_concat)
	except ValueError:
		layer_248_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_248_out = layer_248(layer_248_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_248 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_248_concat = strong_concat_248(new_list_of_inputs)
	layer_248_out = layer_248(layer_248_concat)
if layer_248.count_params() != 0:
	if 248 in set_weigths:
		if len(layer_248.get_weights()) > 1:
			new_w = [get_new_weigths(layer_248, set_weigths[248])]+layer_248.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_248, set_weigths[248])]
		layer_248.set_weights(new_w)


try:
	count = 1
	for s in layer_248_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #248:',count)
except AttributeError:
	for i in layer_248_out:
		count = 1
		for s in layer_248_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #248:',count)



layer_249 = L249()
try:
	try:
		layer_249_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_249_out = layer_249(layer_249_concat)
	except ValueError:
		layer_249_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_249_out = layer_249(layer_249_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_249 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_249_concat = strong_concat_249(new_list_of_inputs)
	layer_249_out = layer_249(layer_249_concat)
if layer_249.count_params() != 0:
	if 249 in set_weigths:
		if len(layer_249.get_weights()) > 1:
			new_w = [get_new_weigths(layer_249, set_weigths[249])]+layer_249.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_249, set_weigths[249])]
		layer_249.set_weights(new_w)


try:
	count = 1
	for s in layer_249_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #249:',count)
except AttributeError:
	for i in layer_249_out:
		count = 1
		for s in layer_249_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #249:',count)



layer_250 = L250()
try:
	try:
		layer_250_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_250_out = layer_250(layer_250_concat)
	except ValueError:
		layer_250_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_250_out = layer_250(layer_250_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_250 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_250_concat = strong_concat_250(new_list_of_inputs)
	layer_250_out = layer_250(layer_250_concat)
if layer_250.count_params() != 0:
	if 250 in set_weigths:
		if len(layer_250.get_weights()) > 1:
			new_w = [get_new_weigths(layer_250, set_weigths[250])]+layer_250.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_250, set_weigths[250])]
		layer_250.set_weights(new_w)


try:
	count = 1
	for s in layer_250_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #250:',count)
except AttributeError:
	for i in layer_250_out:
		count = 1
		for s in layer_250_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #250:',count)



layer_251 = L251()
try:
	try:
		layer_251_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_251_out = layer_251(layer_251_concat)
	except ValueError:
		layer_251_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_251_out = layer_251(layer_251_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_251 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_251_concat = strong_concat_251(new_list_of_inputs)
	layer_251_out = layer_251(layer_251_concat)
if layer_251.count_params() != 0:
	if 251 in set_weigths:
		if len(layer_251.get_weights()) > 1:
			new_w = [get_new_weigths(layer_251, set_weigths[251])]+layer_251.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_251, set_weigths[251])]
		layer_251.set_weights(new_w)


try:
	count = 1
	for s in layer_251_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #251:',count)
except AttributeError:
	for i in layer_251_out:
		count = 1
		for s in layer_251_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #251:',count)



layer_252 = L252()
try:
	try:
		layer_252_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_252_out = layer_252(layer_252_concat)
	except ValueError:
		layer_252_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_252_out = layer_252(layer_252_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_252 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_252_concat = strong_concat_252(new_list_of_inputs)
	layer_252_out = layer_252(layer_252_concat)
if layer_252.count_params() != 0:
	if 252 in set_weigths:
		if len(layer_252.get_weights()) > 1:
			new_w = [get_new_weigths(layer_252, set_weigths[252])]+layer_252.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_252, set_weigths[252])]
		layer_252.set_weights(new_w)


try:
	count = 1
	for s in layer_252_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #252:',count)
except AttributeError:
	for i in layer_252_out:
		count = 1
		for s in layer_252_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #252:',count)



layer_253 = L253()
try:
	try:
		layer_253_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_253_out = layer_253(layer_253_concat)
	except ValueError:
		layer_253_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_253_out = layer_253(layer_253_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_253 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_253_concat = strong_concat_253(new_list_of_inputs)
	layer_253_out = layer_253(layer_253_concat)
if layer_253.count_params() != 0:
	if 253 in set_weigths:
		if len(layer_253.get_weights()) > 1:
			new_w = [get_new_weigths(layer_253, set_weigths[253])]+layer_253.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_253, set_weigths[253])]
		layer_253.set_weights(new_w)


try:
	count = 1
	for s in layer_253_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #253:',count)
except AttributeError:
	for i in layer_253_out:
		count = 1
		for s in layer_253_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #253:',count)



layer_254 = L254()
try:
	try:
		layer_254_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_254_out = layer_254(layer_254_concat)
	except ValueError:
		layer_254_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_254_out = layer_254(layer_254_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_254 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_254_concat = strong_concat_254(new_list_of_inputs)
	layer_254_out = layer_254(layer_254_concat)
if layer_254.count_params() != 0:
	if 254 in set_weigths:
		if len(layer_254.get_weights()) > 1:
			new_w = [get_new_weigths(layer_254, set_weigths[254])]+layer_254.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_254, set_weigths[254])]
		layer_254.set_weights(new_w)


try:
	count = 1
	for s in layer_254_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #254:',count)
except AttributeError:
	for i in layer_254_out:
		count = 1
		for s in layer_254_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #254:',count)



layer_255 = L255()
try:
	try:
		layer_255_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_255_out = layer_255(layer_255_concat)
	except ValueError:
		layer_255_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_255_out = layer_255(layer_255_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_255 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_255_concat = strong_concat_255(new_list_of_inputs)
	layer_255_out = layer_255(layer_255_concat)
if layer_255.count_params() != 0:
	if 255 in set_weigths:
		if len(layer_255.get_weights()) > 1:
			new_w = [get_new_weigths(layer_255, set_weigths[255])]+layer_255.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_255, set_weigths[255])]
		layer_255.set_weights(new_w)


try:
	count = 1
	for s in layer_255_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #255:',count)
except AttributeError:
	for i in layer_255_out:
		count = 1
		for s in layer_255_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #255:',count)



layer_256 = L256()
try:
	try:
		layer_256_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_256_out = layer_256(layer_256_concat)
	except ValueError:
		layer_256_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_256_out = layer_256(layer_256_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_256 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_256_concat = strong_concat_256(new_list_of_inputs)
	layer_256_out = layer_256(layer_256_concat)
if layer_256.count_params() != 0:
	if 256 in set_weigths:
		if len(layer_256.get_weights()) > 1:
			new_w = [get_new_weigths(layer_256, set_weigths[256])]+layer_256.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_256, set_weigths[256])]
		layer_256.set_weights(new_w)


try:
	count = 1
	for s in layer_256_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #256:',count)
except AttributeError:
	for i in layer_256_out:
		count = 1
		for s in layer_256_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #256:',count)



layer_257 = L257()
try:
	try:
		layer_257_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_257_out = layer_257(layer_257_concat)
	except ValueError:
		layer_257_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_257_out = layer_257(layer_257_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_257 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_257_concat = strong_concat_257(new_list_of_inputs)
	layer_257_out = layer_257(layer_257_concat)
if layer_257.count_params() != 0:
	if 257 in set_weigths:
		if len(layer_257.get_weights()) > 1:
			new_w = [get_new_weigths(layer_257, set_weigths[257])]+layer_257.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_257, set_weigths[257])]
		layer_257.set_weights(new_w)


try:
	count = 1
	for s in layer_257_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #257:',count)
except AttributeError:
	for i in layer_257_out:
		count = 1
		for s in layer_257_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #257:',count)



layer_258 = L258()
try:
	try:
		layer_258_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_258_out = layer_258(layer_258_concat)
	except ValueError:
		layer_258_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_258_out = layer_258(layer_258_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_258 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_258_concat = strong_concat_258(new_list_of_inputs)
	layer_258_out = layer_258(layer_258_concat)
if layer_258.count_params() != 0:
	if 258 in set_weigths:
		if len(layer_258.get_weights()) > 1:
			new_w = [get_new_weigths(layer_258, set_weigths[258])]+layer_258.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_258, set_weigths[258])]
		layer_258.set_weights(new_w)


try:
	count = 1
	for s in layer_258_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #258:',count)
except AttributeError:
	for i in layer_258_out:
		count = 1
		for s in layer_258_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #258:',count)



layer_259 = L259()
try:
	try:
		layer_259_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_259_out = layer_259(layer_259_concat)
	except ValueError:
		layer_259_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_259_out = layer_259(layer_259_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_259 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_259_concat = strong_concat_259(new_list_of_inputs)
	layer_259_out = layer_259(layer_259_concat)
if layer_259.count_params() != 0:
	if 259 in set_weigths:
		if len(layer_259.get_weights()) > 1:
			new_w = [get_new_weigths(layer_259, set_weigths[259])]+layer_259.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_259, set_weigths[259])]
		layer_259.set_weights(new_w)


try:
	count = 1
	for s in layer_259_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #259:',count)
except AttributeError:
	for i in layer_259_out:
		count = 1
		for s in layer_259_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #259:',count)



layer_260 = L260()
try:
	try:
		layer_260_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_260_out = layer_260(layer_260_concat)
	except ValueError:
		layer_260_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_260_out = layer_260(layer_260_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_260 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_260_concat = strong_concat_260(new_list_of_inputs)
	layer_260_out = layer_260(layer_260_concat)
if layer_260.count_params() != 0:
	if 260 in set_weigths:
		if len(layer_260.get_weights()) > 1:
			new_w = [get_new_weigths(layer_260, set_weigths[260])]+layer_260.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_260, set_weigths[260])]
		layer_260.set_weights(new_w)


try:
	count = 1
	for s in layer_260_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #260:',count)
except AttributeError:
	for i in layer_260_out:
		count = 1
		for s in layer_260_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #260:',count)



layer_261 = L261()
try:
	try:
		layer_261_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_261_out = layer_261(layer_261_concat)
	except ValueError:
		layer_261_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_261_out = layer_261(layer_261_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_261 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_261_concat = strong_concat_261(new_list_of_inputs)
	layer_261_out = layer_261(layer_261_concat)
if layer_261.count_params() != 0:
	if 261 in set_weigths:
		if len(layer_261.get_weights()) > 1:
			new_w = [get_new_weigths(layer_261, set_weigths[261])]+layer_261.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_261, set_weigths[261])]
		layer_261.set_weights(new_w)


try:
	count = 1
	for s in layer_261_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #261:',count)
except AttributeError:
	for i in layer_261_out:
		count = 1
		for s in layer_261_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #261:',count)



layer_262 = L262()
try:
	try:
		layer_262_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_262_out = layer_262(layer_262_concat)
	except ValueError:
		layer_262_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_262_out = layer_262(layer_262_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_262 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_262_concat = strong_concat_262(new_list_of_inputs)
	layer_262_out = layer_262(layer_262_concat)
if layer_262.count_params() != 0:
	if 262 in set_weigths:
		if len(layer_262.get_weights()) > 1:
			new_w = [get_new_weigths(layer_262, set_weigths[262])]+layer_262.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_262, set_weigths[262])]
		layer_262.set_weights(new_w)


try:
	count = 1
	for s in layer_262_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #262:',count)
except AttributeError:
	for i in layer_262_out:
		count = 1
		for s in layer_262_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #262:',count)



layer_263 = L263()
try:
	try:
		layer_263_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_263_out = layer_263(layer_263_concat)
	except ValueError:
		layer_263_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_263_out = layer_263(layer_263_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_263 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_263_concat = strong_concat_263(new_list_of_inputs)
	layer_263_out = layer_263(layer_263_concat)
if layer_263.count_params() != 0:
	if 263 in set_weigths:
		if len(layer_263.get_weights()) > 1:
			new_w = [get_new_weigths(layer_263, set_weigths[263])]+layer_263.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_263, set_weigths[263])]
		layer_263.set_weights(new_w)


try:
	count = 1
	for s in layer_263_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #263:',count)
except AttributeError:
	for i in layer_263_out:
		count = 1
		for s in layer_263_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #263:',count)



layer_264 = L264()
try:
	try:
		layer_264_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_264_out = layer_264(layer_264_concat)
	except ValueError:
		layer_264_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_264_out = layer_264(layer_264_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_264 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_264_concat = strong_concat_264(new_list_of_inputs)
	layer_264_out = layer_264(layer_264_concat)
if layer_264.count_params() != 0:
	if 264 in set_weigths:
		if len(layer_264.get_weights()) > 1:
			new_w = [get_new_weigths(layer_264, set_weigths[264])]+layer_264.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_264, set_weigths[264])]
		layer_264.set_weights(new_w)


try:
	count = 1
	for s in layer_264_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #264:',count)
except AttributeError:
	for i in layer_264_out:
		count = 1
		for s in layer_264_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #264:',count)



layer_265 = L265()
try:
	try:
		layer_265_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_265_out = layer_265(layer_265_concat)
	except ValueError:
		layer_265_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_265_out = layer_265(layer_265_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_265 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_265_concat = strong_concat_265(new_list_of_inputs)
	layer_265_out = layer_265(layer_265_concat)
if layer_265.count_params() != 0:
	if 265 in set_weigths:
		if len(layer_265.get_weights()) > 1:
			new_w = [get_new_weigths(layer_265, set_weigths[265])]+layer_265.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_265, set_weigths[265])]
		layer_265.set_weights(new_w)


try:
	count = 1
	for s in layer_265_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #265:',count)
except AttributeError:
	for i in layer_265_out:
		count = 1
		for s in layer_265_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #265:',count)



layer_266 = L266()
try:
	try:
		layer_266_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_266_out = layer_266(layer_266_concat)
	except ValueError:
		layer_266_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_266_out = layer_266(layer_266_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_266 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_266_concat = strong_concat_266(new_list_of_inputs)
	layer_266_out = layer_266(layer_266_concat)
if layer_266.count_params() != 0:
	if 266 in set_weigths:
		if len(layer_266.get_weights()) > 1:
			new_w = [get_new_weigths(layer_266, set_weigths[266])]+layer_266.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_266, set_weigths[266])]
		layer_266.set_weights(new_w)


try:
	count = 1
	for s in layer_266_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #266:',count)
except AttributeError:
	for i in layer_266_out:
		count = 1
		for s in layer_266_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #266:',count)



layer_267 = L267()
try:
	try:
		layer_267_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_267_out = layer_267(layer_267_concat)
	except ValueError:
		layer_267_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_267_out = layer_267(layer_267_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_267 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_267_concat = strong_concat_267(new_list_of_inputs)
	layer_267_out = layer_267(layer_267_concat)
if layer_267.count_params() != 0:
	if 267 in set_weigths:
		if len(layer_267.get_weights()) > 1:
			new_w = [get_new_weigths(layer_267, set_weigths[267])]+layer_267.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_267, set_weigths[267])]
		layer_267.set_weights(new_w)


try:
	count = 1
	for s in layer_267_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #267:',count)
except AttributeError:
	for i in layer_267_out:
		count = 1
		for s in layer_267_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #267:',count)



layer_268 = L268()
try:
	try:
		layer_268_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_268_out = layer_268(layer_268_concat)
	except ValueError:
		layer_268_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_268_out = layer_268(layer_268_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_268 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_268_concat = strong_concat_268(new_list_of_inputs)
	layer_268_out = layer_268(layer_268_concat)
if layer_268.count_params() != 0:
	if 268 in set_weigths:
		if len(layer_268.get_weights()) > 1:
			new_w = [get_new_weigths(layer_268, set_weigths[268])]+layer_268.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_268, set_weigths[268])]
		layer_268.set_weights(new_w)


try:
	count = 1
	for s in layer_268_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #268:',count)
except AttributeError:
	for i in layer_268_out:
		count = 1
		for s in layer_268_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #268:',count)



layer_269 = L269()
try:
	try:
		layer_269_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_269_out = layer_269(layer_269_concat)
	except ValueError:
		layer_269_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_269_out = layer_269(layer_269_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_269 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_269_concat = strong_concat_269(new_list_of_inputs)
	layer_269_out = layer_269(layer_269_concat)
if layer_269.count_params() != 0:
	if 269 in set_weigths:
		if len(layer_269.get_weights()) > 1:
			new_w = [get_new_weigths(layer_269, set_weigths[269])]+layer_269.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_269, set_weigths[269])]
		layer_269.set_weights(new_w)


try:
	count = 1
	for s in layer_269_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #269:',count)
except AttributeError:
	for i in layer_269_out:
		count = 1
		for s in layer_269_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #269:',count)



layer_270 = L270()
try:
	try:
		layer_270_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_270_out = layer_270(layer_270_concat)
	except ValueError:
		layer_270_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_270_out = layer_270(layer_270_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_270 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_270_concat = strong_concat_270(new_list_of_inputs)
	layer_270_out = layer_270(layer_270_concat)
if layer_270.count_params() != 0:
	if 270 in set_weigths:
		if len(layer_270.get_weights()) > 1:
			new_w = [get_new_weigths(layer_270, set_weigths[270])]+layer_270.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_270, set_weigths[270])]
		layer_270.set_weights(new_w)


try:
	count = 1
	for s in layer_270_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #270:',count)
except AttributeError:
	for i in layer_270_out:
		count = 1
		for s in layer_270_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #270:',count)



layer_271 = L271()
try:
	try:
		layer_271_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_271_out = layer_271(layer_271_concat)
	except ValueError:
		layer_271_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_271_out = layer_271(layer_271_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_271 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_271_concat = strong_concat_271(new_list_of_inputs)
	layer_271_out = layer_271(layer_271_concat)
if layer_271.count_params() != 0:
	if 271 in set_weigths:
		if len(layer_271.get_weights()) > 1:
			new_w = [get_new_weigths(layer_271, set_weigths[271])]+layer_271.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_271, set_weigths[271])]
		layer_271.set_weights(new_w)


try:
	count = 1
	for s in layer_271_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #271:',count)
except AttributeError:
	for i in layer_271_out:
		count = 1
		for s in layer_271_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #271:',count)



layer_272 = L272()
try:
	try:
		layer_272_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_272_out = layer_272(layer_272_concat)
	except ValueError:
		layer_272_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_272_out = layer_272(layer_272_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_272 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_272_concat = strong_concat_272(new_list_of_inputs)
	layer_272_out = layer_272(layer_272_concat)
if layer_272.count_params() != 0:
	if 272 in set_weigths:
		if len(layer_272.get_weights()) > 1:
			new_w = [get_new_weigths(layer_272, set_weigths[272])]+layer_272.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_272, set_weigths[272])]
		layer_272.set_weights(new_w)


try:
	count = 1
	for s in layer_272_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #272:',count)
except AttributeError:
	for i in layer_272_out:
		count = 1
		for s in layer_272_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #272:',count)



layer_273 = L273()
try:
	try:
		layer_273_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_273_out = layer_273(layer_273_concat)
	except ValueError:
		layer_273_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_273_out = layer_273(layer_273_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_273 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_273_concat = strong_concat_273(new_list_of_inputs)
	layer_273_out = layer_273(layer_273_concat)
if layer_273.count_params() != 0:
	if 273 in set_weigths:
		if len(layer_273.get_weights()) > 1:
			new_w = [get_new_weigths(layer_273, set_weigths[273])]+layer_273.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_273, set_weigths[273])]
		layer_273.set_weights(new_w)


try:
	count = 1
	for s in layer_273_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #273:',count)
except AttributeError:
	for i in layer_273_out:
		count = 1
		for s in layer_273_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #273:',count)



layer_274 = L274()
try:
	try:
		layer_274_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_274_out = layer_274(layer_274_concat)
	except ValueError:
		layer_274_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_274_out = layer_274(layer_274_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_274 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_274_concat = strong_concat_274(new_list_of_inputs)
	layer_274_out = layer_274(layer_274_concat)
if layer_274.count_params() != 0:
	if 274 in set_weigths:
		if len(layer_274.get_weights()) > 1:
			new_w = [get_new_weigths(layer_274, set_weigths[274])]+layer_274.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_274, set_weigths[274])]
		layer_274.set_weights(new_w)


try:
	count = 1
	for s in layer_274_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #274:',count)
except AttributeError:
	for i in layer_274_out:
		count = 1
		for s in layer_274_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #274:',count)



layer_275 = L275()
try:
	try:
		layer_275_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_275_out = layer_275(layer_275_concat)
	except ValueError:
		layer_275_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_275_out = layer_275(layer_275_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_275 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_275_concat = strong_concat_275(new_list_of_inputs)
	layer_275_out = layer_275(layer_275_concat)
if layer_275.count_params() != 0:
	if 275 in set_weigths:
		if len(layer_275.get_weights()) > 1:
			new_w = [get_new_weigths(layer_275, set_weigths[275])]+layer_275.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_275, set_weigths[275])]
		layer_275.set_weights(new_w)


try:
	count = 1
	for s in layer_275_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #275:',count)
except AttributeError:
	for i in layer_275_out:
		count = 1
		for s in layer_275_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #275:',count)



layer_276 = L276()
try:
	try:
		layer_276_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_276_out = layer_276(layer_276_concat)
	except ValueError:
		layer_276_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_276_out = layer_276(layer_276_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_276 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_276_concat = strong_concat_276(new_list_of_inputs)
	layer_276_out = layer_276(layer_276_concat)
if layer_276.count_params() != 0:
	if 276 in set_weigths:
		if len(layer_276.get_weights()) > 1:
			new_w = [get_new_weigths(layer_276, set_weigths[276])]+layer_276.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_276, set_weigths[276])]
		layer_276.set_weights(new_w)


try:
	count = 1
	for s in layer_276_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #276:',count)
except AttributeError:
	for i in layer_276_out:
		count = 1
		for s in layer_276_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #276:',count)



layer_277 = L277()
try:
	try:
		layer_277_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_277_out = layer_277(layer_277_concat)
	except ValueError:
		layer_277_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_277_out = layer_277(layer_277_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_277 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_277_concat = strong_concat_277(new_list_of_inputs)
	layer_277_out = layer_277(layer_277_concat)
if layer_277.count_params() != 0:
	if 277 in set_weigths:
		if len(layer_277.get_weights()) > 1:
			new_w = [get_new_weigths(layer_277, set_weigths[277])]+layer_277.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_277, set_weigths[277])]
		layer_277.set_weights(new_w)


try:
	count = 1
	for s in layer_277_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #277:',count)
except AttributeError:
	for i in layer_277_out:
		count = 1
		for s in layer_277_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #277:',count)



layer_278 = L278()
try:
	try:
		layer_278_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_278_out = layer_278(layer_278_concat)
	except ValueError:
		layer_278_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_278_out = layer_278(layer_278_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_278 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_278_concat = strong_concat_278(new_list_of_inputs)
	layer_278_out = layer_278(layer_278_concat)
if layer_278.count_params() != 0:
	if 278 in set_weigths:
		if len(layer_278.get_weights()) > 1:
			new_w = [get_new_weigths(layer_278, set_weigths[278])]+layer_278.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_278, set_weigths[278])]
		layer_278.set_weights(new_w)


try:
	count = 1
	for s in layer_278_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #278:',count)
except AttributeError:
	for i in layer_278_out:
		count = 1
		for s in layer_278_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #278:',count)



layer_279 = L279()
try:
	try:
		layer_279_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_279_out = layer_279(layer_279_concat)
	except ValueError:
		layer_279_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_279_out = layer_279(layer_279_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_279 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_279_concat = strong_concat_279(new_list_of_inputs)
	layer_279_out = layer_279(layer_279_concat)
if layer_279.count_params() != 0:
	if 279 in set_weigths:
		if len(layer_279.get_weights()) > 1:
			new_w = [get_new_weigths(layer_279, set_weigths[279])]+layer_279.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_279, set_weigths[279])]
		layer_279.set_weights(new_w)


try:
	count = 1
	for s in layer_279_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #279:',count)
except AttributeError:
	for i in layer_279_out:
		count = 1
		for s in layer_279_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #279:',count)



layer_280 = L280()
try:
	try:
		layer_280_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_280_out = layer_280(layer_280_concat)
	except ValueError:
		layer_280_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_280_out = layer_280(layer_280_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_280 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_280_concat = strong_concat_280(new_list_of_inputs)
	layer_280_out = layer_280(layer_280_concat)
if layer_280.count_params() != 0:
	if 280 in set_weigths:
		if len(layer_280.get_weights()) > 1:
			new_w = [get_new_weigths(layer_280, set_weigths[280])]+layer_280.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_280, set_weigths[280])]
		layer_280.set_weights(new_w)


try:
	count = 1
	for s in layer_280_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #280:',count)
except AttributeError:
	for i in layer_280_out:
		count = 1
		for s in layer_280_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #280:',count)



layer_281 = L281()
try:
	try:
		layer_281_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_281_out = layer_281(layer_281_concat)
	except ValueError:
		layer_281_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_281_out = layer_281(layer_281_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_281 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_281_concat = strong_concat_281(new_list_of_inputs)
	layer_281_out = layer_281(layer_281_concat)
if layer_281.count_params() != 0:
	if 281 in set_weigths:
		if len(layer_281.get_weights()) > 1:
			new_w = [get_new_weigths(layer_281, set_weigths[281])]+layer_281.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_281, set_weigths[281])]
		layer_281.set_weights(new_w)


try:
	count = 1
	for s in layer_281_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #281:',count)
except AttributeError:
	for i in layer_281_out:
		count = 1
		for s in layer_281_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #281:',count)



layer_282 = L282()
try:
	try:
		layer_282_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_282_out = layer_282(layer_282_concat)
	except ValueError:
		layer_282_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_282_out = layer_282(layer_282_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_282 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_282_concat = strong_concat_282(new_list_of_inputs)
	layer_282_out = layer_282(layer_282_concat)
if layer_282.count_params() != 0:
	if 282 in set_weigths:
		if len(layer_282.get_weights()) > 1:
			new_w = [get_new_weigths(layer_282, set_weigths[282])]+layer_282.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_282, set_weigths[282])]
		layer_282.set_weights(new_w)


try:
	count = 1
	for s in layer_282_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #282:',count)
except AttributeError:
	for i in layer_282_out:
		count = 1
		for s in layer_282_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #282:',count)



layer_283 = L283()
try:
	try:
		layer_283_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_283_out = layer_283(layer_283_concat)
	except ValueError:
		layer_283_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_283_out = layer_283(layer_283_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_283 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_283_concat = strong_concat_283(new_list_of_inputs)
	layer_283_out = layer_283(layer_283_concat)
if layer_283.count_params() != 0:
	if 283 in set_weigths:
		if len(layer_283.get_weights()) > 1:
			new_w = [get_new_weigths(layer_283, set_weigths[283])]+layer_283.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_283, set_weigths[283])]
		layer_283.set_weights(new_w)


try:
	count = 1
	for s in layer_283_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #283:',count)
except AttributeError:
	for i in layer_283_out:
		count = 1
		for s in layer_283_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #283:',count)



layer_284 = L284()
try:
	try:
		layer_284_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_284_out = layer_284(layer_284_concat)
	except ValueError:
		layer_284_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_284_out = layer_284(layer_284_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_284 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_284_concat = strong_concat_284(new_list_of_inputs)
	layer_284_out = layer_284(layer_284_concat)
if layer_284.count_params() != 0:
	if 284 in set_weigths:
		if len(layer_284.get_weights()) > 1:
			new_w = [get_new_weigths(layer_284, set_weigths[284])]+layer_284.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_284, set_weigths[284])]
		layer_284.set_weights(new_w)


try:
	count = 1
	for s in layer_284_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #284:',count)
except AttributeError:
	for i in layer_284_out:
		count = 1
		for s in layer_284_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #284:',count)



layer_285 = L285()
try:
	try:
		layer_285_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_285_out = layer_285(layer_285_concat)
	except ValueError:
		layer_285_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_285_out = layer_285(layer_285_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_285 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_285_concat = strong_concat_285(new_list_of_inputs)
	layer_285_out = layer_285(layer_285_concat)
if layer_285.count_params() != 0:
	if 285 in set_weigths:
		if len(layer_285.get_weights()) > 1:
			new_w = [get_new_weigths(layer_285, set_weigths[285])]+layer_285.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_285, set_weigths[285])]
		layer_285.set_weights(new_w)


try:
	count = 1
	for s in layer_285_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #285:',count)
except AttributeError:
	for i in layer_285_out:
		count = 1
		for s in layer_285_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #285:',count)



layer_286 = L286()
try:
	try:
		layer_286_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_286_out = layer_286(layer_286_concat)
	except ValueError:
		layer_286_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_286_out = layer_286(layer_286_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_286 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_286_concat = strong_concat_286(new_list_of_inputs)
	layer_286_out = layer_286(layer_286_concat)
if layer_286.count_params() != 0:
	if 286 in set_weigths:
		if len(layer_286.get_weights()) > 1:
			new_w = [get_new_weigths(layer_286, set_weigths[286])]+layer_286.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_286, set_weigths[286])]
		layer_286.set_weights(new_w)


try:
	count = 1
	for s in layer_286_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #286:',count)
except AttributeError:
	for i in layer_286_out:
		count = 1
		for s in layer_286_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #286:',count)



layer_287 = L287()
try:
	try:
		layer_287_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_287_out = layer_287(layer_287_concat)
	except ValueError:
		layer_287_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_287_out = layer_287(layer_287_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_287 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_287_concat = strong_concat_287(new_list_of_inputs)
	layer_287_out = layer_287(layer_287_concat)
if layer_287.count_params() != 0:
	if 287 in set_weigths:
		if len(layer_287.get_weights()) > 1:
			new_w = [get_new_weigths(layer_287, set_weigths[287])]+layer_287.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_287, set_weigths[287])]
		layer_287.set_weights(new_w)


try:
	count = 1
	for s in layer_287_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #287:',count)
except AttributeError:
	for i in layer_287_out:
		count = 1
		for s in layer_287_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #287:',count)



layer_288 = L288()
try:
	try:
		layer_288_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_288_out = layer_288(layer_288_concat)
	except ValueError:
		layer_288_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_288_out = layer_288(layer_288_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_288 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_288_concat = strong_concat_288(new_list_of_inputs)
	layer_288_out = layer_288(layer_288_concat)
if layer_288.count_params() != 0:
	if 288 in set_weigths:
		if len(layer_288.get_weights()) > 1:
			new_w = [get_new_weigths(layer_288, set_weigths[288])]+layer_288.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_288, set_weigths[288])]
		layer_288.set_weights(new_w)


try:
	count = 1
	for s in layer_288_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #288:',count)
except AttributeError:
	for i in layer_288_out:
		count = 1
		for s in layer_288_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #288:',count)



layer_289 = L289()
try:
	try:
		layer_289_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_289_out = layer_289(layer_289_concat)
	except ValueError:
		layer_289_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_289_out = layer_289(layer_289_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_289 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_289_concat = strong_concat_289(new_list_of_inputs)
	layer_289_out = layer_289(layer_289_concat)
if layer_289.count_params() != 0:
	if 289 in set_weigths:
		if len(layer_289.get_weights()) > 1:
			new_w = [get_new_weigths(layer_289, set_weigths[289])]+layer_289.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_289, set_weigths[289])]
		layer_289.set_weights(new_w)


try:
	count = 1
	for s in layer_289_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #289:',count)
except AttributeError:
	for i in layer_289_out:
		count = 1
		for s in layer_289_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #289:',count)



layer_290 = L290()
try:
	try:
		layer_290_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_290_out = layer_290(layer_290_concat)
	except ValueError:
		layer_290_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_290_out = layer_290(layer_290_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_290 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_290_concat = strong_concat_290(new_list_of_inputs)
	layer_290_out = layer_290(layer_290_concat)
if layer_290.count_params() != 0:
	if 290 in set_weigths:
		if len(layer_290.get_weights()) > 1:
			new_w = [get_new_weigths(layer_290, set_weigths[290])]+layer_290.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_290, set_weigths[290])]
		layer_290.set_weights(new_w)


try:
	count = 1
	for s in layer_290_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #290:',count)
except AttributeError:
	for i in layer_290_out:
		count = 1
		for s in layer_290_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #290:',count)



layer_291 = L291()
try:
	try:
		layer_291_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_291_out = layer_291(layer_291_concat)
	except ValueError:
		layer_291_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_291_out = layer_291(layer_291_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_291 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_291_concat = strong_concat_291(new_list_of_inputs)
	layer_291_out = layer_291(layer_291_concat)
if layer_291.count_params() != 0:
	if 291 in set_weigths:
		if len(layer_291.get_weights()) > 1:
			new_w = [get_new_weigths(layer_291, set_weigths[291])]+layer_291.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_291, set_weigths[291])]
		layer_291.set_weights(new_w)


try:
	count = 1
	for s in layer_291_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #291:',count)
except AttributeError:
	for i in layer_291_out:
		count = 1
		for s in layer_291_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #291:',count)



layer_292 = L292()
try:
	try:
		layer_292_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_292_out = layer_292(layer_292_concat)
	except ValueError:
		layer_292_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_292_out = layer_292(layer_292_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_292 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_292_concat = strong_concat_292(new_list_of_inputs)
	layer_292_out = layer_292(layer_292_concat)
if layer_292.count_params() != 0:
	if 292 in set_weigths:
		if len(layer_292.get_weights()) > 1:
			new_w = [get_new_weigths(layer_292, set_weigths[292])]+layer_292.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_292, set_weigths[292])]
		layer_292.set_weights(new_w)


try:
	count = 1
	for s in layer_292_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #292:',count)
except AttributeError:
	for i in layer_292_out:
		count = 1
		for s in layer_292_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #292:',count)



layer_293 = L293()
try:
	try:
		layer_293_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_293_out = layer_293(layer_293_concat)
	except ValueError:
		layer_293_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_293_out = layer_293(layer_293_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_293 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_293_concat = strong_concat_293(new_list_of_inputs)
	layer_293_out = layer_293(layer_293_concat)
if layer_293.count_params() != 0:
	if 293 in set_weigths:
		if len(layer_293.get_weights()) > 1:
			new_w = [get_new_weigths(layer_293, set_weigths[293])]+layer_293.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_293, set_weigths[293])]
		layer_293.set_weights(new_w)


try:
	count = 1
	for s in layer_293_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #293:',count)
except AttributeError:
	for i in layer_293_out:
		count = 1
		for s in layer_293_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #293:',count)



layer_294 = L294()
try:
	try:
		layer_294_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_294_out = layer_294(layer_294_concat)
	except ValueError:
		layer_294_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_294_out = layer_294(layer_294_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_294 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_294_concat = strong_concat_294(new_list_of_inputs)
	layer_294_out = layer_294(layer_294_concat)
if layer_294.count_params() != 0:
	if 294 in set_weigths:
		if len(layer_294.get_weights()) > 1:
			new_w = [get_new_weigths(layer_294, set_weigths[294])]+layer_294.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_294, set_weigths[294])]
		layer_294.set_weights(new_w)


try:
	count = 1
	for s in layer_294_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #294:',count)
except AttributeError:
	for i in layer_294_out:
		count = 1
		for s in layer_294_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #294:',count)



layer_295 = L295()
try:
	try:
		layer_295_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_295_out = layer_295(layer_295_concat)
	except ValueError:
		layer_295_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_295_out = layer_295(layer_295_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_295 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_295_concat = strong_concat_295(new_list_of_inputs)
	layer_295_out = layer_295(layer_295_concat)
if layer_295.count_params() != 0:
	if 295 in set_weigths:
		if len(layer_295.get_weights()) > 1:
			new_w = [get_new_weigths(layer_295, set_weigths[295])]+layer_295.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_295, set_weigths[295])]
		layer_295.set_weights(new_w)


try:
	count = 1
	for s in layer_295_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #295:',count)
except AttributeError:
	for i in layer_295_out:
		count = 1
		for s in layer_295_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #295:',count)



layer_296 = L296()
try:
	try:
		layer_296_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_296_out = layer_296(layer_296_concat)
	except ValueError:
		layer_296_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_296_out = layer_296(layer_296_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_296 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_296_concat = strong_concat_296(new_list_of_inputs)
	layer_296_out = layer_296(layer_296_concat)
if layer_296.count_params() != 0:
	if 296 in set_weigths:
		if len(layer_296.get_weights()) > 1:
			new_w = [get_new_weigths(layer_296, set_weigths[296])]+layer_296.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_296, set_weigths[296])]
		layer_296.set_weights(new_w)


try:
	count = 1
	for s in layer_296_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #296:',count)
except AttributeError:
	for i in layer_296_out:
		count = 1
		for s in layer_296_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #296:',count)



layer_297 = L297()
try:
	try:
		layer_297_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_297_out = layer_297(layer_297_concat)
	except ValueError:
		layer_297_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_297_out = layer_297(layer_297_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_297 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_297_concat = strong_concat_297(new_list_of_inputs)
	layer_297_out = layer_297(layer_297_concat)
if layer_297.count_params() != 0:
	if 297 in set_weigths:
		if len(layer_297.get_weights()) > 1:
			new_w = [get_new_weigths(layer_297, set_weigths[297])]+layer_297.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_297, set_weigths[297])]
		layer_297.set_weights(new_w)


try:
	count = 1
	for s in layer_297_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #297:',count)
except AttributeError:
	for i in layer_297_out:
		count = 1
		for s in layer_297_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #297:',count)



layer_298 = L298()
try:
	try:
		layer_298_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_298_out = layer_298(layer_298_concat)
	except ValueError:
		layer_298_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_298_out = layer_298(layer_298_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_298 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_298_concat = strong_concat_298(new_list_of_inputs)
	layer_298_out = layer_298(layer_298_concat)
if layer_298.count_params() != 0:
	if 298 in set_weigths:
		if len(layer_298.get_weights()) > 1:
			new_w = [get_new_weigths(layer_298, set_weigths[298])]+layer_298.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_298, set_weigths[298])]
		layer_298.set_weights(new_w)


try:
	count = 1
	for s in layer_298_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #298:',count)
except AttributeError:
	for i in layer_298_out:
		count = 1
		for s in layer_298_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #298:',count)



layer_299 = L299()
try:
	try:
		layer_299_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_299_out = layer_299(layer_299_concat)
	except ValueError:
		layer_299_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_299_out = layer_299(layer_299_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_299 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_299_concat = strong_concat_299(new_list_of_inputs)
	layer_299_out = layer_299(layer_299_concat)
if layer_299.count_params() != 0:
	if 299 in set_weigths:
		if len(layer_299.get_weights()) > 1:
			new_w = [get_new_weigths(layer_299, set_weigths[299])]+layer_299.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_299, set_weigths[299])]
		layer_299.set_weights(new_w)


try:
	count = 1
	for s in layer_299_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #299:',count)
except AttributeError:
	for i in layer_299_out:
		count = 1
		for s in layer_299_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #299:',count)



layer_300 = L300()
try:
	try:
		layer_300_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_300_out = layer_300(layer_300_concat)
	except ValueError:
		layer_300_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_300_out = layer_300(layer_300_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_300 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_300_concat = strong_concat_300(new_list_of_inputs)
	layer_300_out = layer_300(layer_300_concat)
if layer_300.count_params() != 0:
	if 300 in set_weigths:
		if len(layer_300.get_weights()) > 1:
			new_w = [get_new_weigths(layer_300, set_weigths[300])]+layer_300.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_300, set_weigths[300])]
		layer_300.set_weights(new_w)


try:
	count = 1
	for s in layer_300_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #300:',count)
except AttributeError:
	for i in layer_300_out:
		count = 1
		for s in layer_300_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #300:',count)



layer_301 = L301()
try:
	try:
		layer_301_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_301_out = layer_301(layer_301_concat)
	except ValueError:
		layer_301_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_301_out = layer_301(layer_301_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_301 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_301_concat = strong_concat_301(new_list_of_inputs)
	layer_301_out = layer_301(layer_301_concat)
if layer_301.count_params() != 0:
	if 301 in set_weigths:
		if len(layer_301.get_weights()) > 1:
			new_w = [get_new_weigths(layer_301, set_weigths[301])]+layer_301.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_301, set_weigths[301])]
		layer_301.set_weights(new_w)


try:
	count = 1
	for s in layer_301_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #301:',count)
except AttributeError:
	for i in layer_301_out:
		count = 1
		for s in layer_301_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #301:',count)



layer_302 = L302()
try:
	try:
		layer_302_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_302_out = layer_302(layer_302_concat)
	except ValueError:
		layer_302_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_302_out = layer_302(layer_302_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_302 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_302_concat = strong_concat_302(new_list_of_inputs)
	layer_302_out = layer_302(layer_302_concat)
if layer_302.count_params() != 0:
	if 302 in set_weigths:
		if len(layer_302.get_weights()) > 1:
			new_w = [get_new_weigths(layer_302, set_weigths[302])]+layer_302.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_302, set_weigths[302])]
		layer_302.set_weights(new_w)


try:
	count = 1
	for s in layer_302_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #302:',count)
except AttributeError:
	for i in layer_302_out:
		count = 1
		for s in layer_302_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #302:',count)



layer_303 = L303()
try:
	try:
		layer_303_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_303_out = layer_303(layer_303_concat)
	except ValueError:
		layer_303_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_303_out = layer_303(layer_303_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_303 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_303_concat = strong_concat_303(new_list_of_inputs)
	layer_303_out = layer_303(layer_303_concat)
if layer_303.count_params() != 0:
	if 303 in set_weigths:
		if len(layer_303.get_weights()) > 1:
			new_w = [get_new_weigths(layer_303, set_weigths[303])]+layer_303.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_303, set_weigths[303])]
		layer_303.set_weights(new_w)


try:
	count = 1
	for s in layer_303_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #303:',count)
except AttributeError:
	for i in layer_303_out:
		count = 1
		for s in layer_303_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #303:',count)



layer_304 = L304()
try:
	try:
		layer_304_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_304_out = layer_304(layer_304_concat)
	except ValueError:
		layer_304_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_304_out = layer_304(layer_304_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_304 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_304_concat = strong_concat_304(new_list_of_inputs)
	layer_304_out = layer_304(layer_304_concat)
if layer_304.count_params() != 0:
	if 304 in set_weigths:
		if len(layer_304.get_weights()) > 1:
			new_w = [get_new_weigths(layer_304, set_weigths[304])]+layer_304.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_304, set_weigths[304])]
		layer_304.set_weights(new_w)


try:
	count = 1
	for s in layer_304_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #304:',count)
except AttributeError:
	for i in layer_304_out:
		count = 1
		for s in layer_304_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #304:',count)



layer_305 = L305()
try:
	try:
		layer_305_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_305_out = layer_305(layer_305_concat)
	except ValueError:
		layer_305_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_305_out = layer_305(layer_305_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_305 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_305_concat = strong_concat_305(new_list_of_inputs)
	layer_305_out = layer_305(layer_305_concat)
if layer_305.count_params() != 0:
	if 305 in set_weigths:
		if len(layer_305.get_weights()) > 1:
			new_w = [get_new_weigths(layer_305, set_weigths[305])]+layer_305.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_305, set_weigths[305])]
		layer_305.set_weights(new_w)


try:
	count = 1
	for s in layer_305_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #305:',count)
except AttributeError:
	for i in layer_305_out:
		count = 1
		for s in layer_305_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #305:',count)



layer_306 = L306()
try:
	try:
		layer_306_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_306_out = layer_306(layer_306_concat)
	except ValueError:
		layer_306_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_306_out = layer_306(layer_306_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_306 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_306_concat = strong_concat_306(new_list_of_inputs)
	layer_306_out = layer_306(layer_306_concat)
if layer_306.count_params() != 0:
	if 306 in set_weigths:
		if len(layer_306.get_weights()) > 1:
			new_w = [get_new_weigths(layer_306, set_weigths[306])]+layer_306.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_306, set_weigths[306])]
		layer_306.set_weights(new_w)


try:
	count = 1
	for s in layer_306_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #306:',count)
except AttributeError:
	for i in layer_306_out:
		count = 1
		for s in layer_306_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #306:',count)



layer_307 = L307()
try:
	try:
		layer_307_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_307_out = layer_307(layer_307_concat)
	except ValueError:
		layer_307_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_307_out = layer_307(layer_307_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_307 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_307_concat = strong_concat_307(new_list_of_inputs)
	layer_307_out = layer_307(layer_307_concat)
if layer_307.count_params() != 0:
	if 307 in set_weigths:
		if len(layer_307.get_weights()) > 1:
			new_w = [get_new_weigths(layer_307, set_weigths[307])]+layer_307.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_307, set_weigths[307])]
		layer_307.set_weights(new_w)


try:
	count = 1
	for s in layer_307_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #307:',count)
except AttributeError:
	for i in layer_307_out:
		count = 1
		for s in layer_307_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #307:',count)



layer_308 = L308()
try:
	try:
		layer_308_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_308_out = layer_308(layer_308_concat)
	except ValueError:
		layer_308_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_308_out = layer_308(layer_308_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_308 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_308_concat = strong_concat_308(new_list_of_inputs)
	layer_308_out = layer_308(layer_308_concat)
if layer_308.count_params() != 0:
	if 308 in set_weigths:
		if len(layer_308.get_weights()) > 1:
			new_w = [get_new_weigths(layer_308, set_weigths[308])]+layer_308.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_308, set_weigths[308])]
		layer_308.set_weights(new_w)


try:
	count = 1
	for s in layer_308_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #308:',count)
except AttributeError:
	for i in layer_308_out:
		count = 1
		for s in layer_308_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #308:',count)



layer_309 = L309()
try:
	try:
		layer_309_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_309_out = layer_309(layer_309_concat)
	except ValueError:
		layer_309_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_309_out = layer_309(layer_309_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_309 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_309_concat = strong_concat_309(new_list_of_inputs)
	layer_309_out = layer_309(layer_309_concat)
if layer_309.count_params() != 0:
	if 309 in set_weigths:
		if len(layer_309.get_weights()) > 1:
			new_w = [get_new_weigths(layer_309, set_weigths[309])]+layer_309.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_309, set_weigths[309])]
		layer_309.set_weights(new_w)


try:
	count = 1
	for s in layer_309_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #309:',count)
except AttributeError:
	for i in layer_309_out:
		count = 1
		for s in layer_309_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #309:',count)



layer_310 = L310()
try:
	try:
		layer_310_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_310_out = layer_310(layer_310_concat)
	except ValueError:
		layer_310_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_310_out = layer_310(layer_310_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_310 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_310_concat = strong_concat_310(new_list_of_inputs)
	layer_310_out = layer_310(layer_310_concat)
if layer_310.count_params() != 0:
	if 310 in set_weigths:
		if len(layer_310.get_weights()) > 1:
			new_w = [get_new_weigths(layer_310, set_weigths[310])]+layer_310.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_310, set_weigths[310])]
		layer_310.set_weights(new_w)


try:
	count = 1
	for s in layer_310_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #310:',count)
except AttributeError:
	for i in layer_310_out:
		count = 1
		for s in layer_310_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #310:',count)



layer_311 = L311()
try:
	try:
		layer_311_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_311_out = layer_311(layer_311_concat)
	except ValueError:
		layer_311_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_311_out = layer_311(layer_311_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_311 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_311_concat = strong_concat_311(new_list_of_inputs)
	layer_311_out = layer_311(layer_311_concat)
if layer_311.count_params() != 0:
	if 311 in set_weigths:
		if len(layer_311.get_weights()) > 1:
			new_w = [get_new_weigths(layer_311, set_weigths[311])]+layer_311.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_311, set_weigths[311])]
		layer_311.set_weights(new_w)


try:
	count = 1
	for s in layer_311_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #311:',count)
except AttributeError:
	for i in layer_311_out:
		count = 1
		for s in layer_311_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #311:',count)



layer_312 = L312()
try:
	try:
		layer_312_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_312_out = layer_312(layer_312_concat)
	except ValueError:
		layer_312_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_312_out = layer_312(layer_312_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_312 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_312_concat = strong_concat_312(new_list_of_inputs)
	layer_312_out = layer_312(layer_312_concat)
if layer_312.count_params() != 0:
	if 312 in set_weigths:
		if len(layer_312.get_weights()) > 1:
			new_w = [get_new_weigths(layer_312, set_weigths[312])]+layer_312.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_312, set_weigths[312])]
		layer_312.set_weights(new_w)


try:
	count = 1
	for s in layer_312_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #312:',count)
except AttributeError:
	for i in layer_312_out:
		count = 1
		for s in layer_312_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #312:',count)



layer_313 = L313()
try:
	try:
		layer_313_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_313_out = layer_313(layer_313_concat)
	except ValueError:
		layer_313_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_313_out = layer_313(layer_313_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_313 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_313_concat = strong_concat_313(new_list_of_inputs)
	layer_313_out = layer_313(layer_313_concat)
if layer_313.count_params() != 0:
	if 313 in set_weigths:
		if len(layer_313.get_weights()) > 1:
			new_w = [get_new_weigths(layer_313, set_weigths[313])]+layer_313.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_313, set_weigths[313])]
		layer_313.set_weights(new_w)


try:
	count = 1
	for s in layer_313_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #313:',count)
except AttributeError:
	for i in layer_313_out:
		count = 1
		for s in layer_313_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #313:',count)



layer_314 = L314()
try:
	try:
		layer_314_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_314_out = layer_314(layer_314_concat)
	except ValueError:
		layer_314_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_314_out = layer_314(layer_314_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_314 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_314_concat = strong_concat_314(new_list_of_inputs)
	layer_314_out = layer_314(layer_314_concat)
if layer_314.count_params() != 0:
	if 314 in set_weigths:
		if len(layer_314.get_weights()) > 1:
			new_w = [get_new_weigths(layer_314, set_weigths[314])]+layer_314.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_314, set_weigths[314])]
		layer_314.set_weights(new_w)


try:
	count = 1
	for s in layer_314_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #314:',count)
except AttributeError:
	for i in layer_314_out:
		count = 1
		for s in layer_314_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #314:',count)



layer_315 = L315()
try:
	try:
		layer_315_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_315_out = layer_315(layer_315_concat)
	except ValueError:
		layer_315_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_315_out = layer_315(layer_315_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_315 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_315_concat = strong_concat_315(new_list_of_inputs)
	layer_315_out = layer_315(layer_315_concat)
if layer_315.count_params() != 0:
	if 315 in set_weigths:
		if len(layer_315.get_weights()) > 1:
			new_w = [get_new_weigths(layer_315, set_weigths[315])]+layer_315.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_315, set_weigths[315])]
		layer_315.set_weights(new_w)


try:
	count = 1
	for s in layer_315_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #315:',count)
except AttributeError:
	for i in layer_315_out:
		count = 1
		for s in layer_315_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #315:',count)



layer_316 = L316()
try:
	try:
		layer_316_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_316_out = layer_316(layer_316_concat)
	except ValueError:
		layer_316_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_316_out = layer_316(layer_316_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_316 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_316_concat = strong_concat_316(new_list_of_inputs)
	layer_316_out = layer_316(layer_316_concat)
if layer_316.count_params() != 0:
	if 316 in set_weigths:
		if len(layer_316.get_weights()) > 1:
			new_w = [get_new_weigths(layer_316, set_weigths[316])]+layer_316.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_316, set_weigths[316])]
		layer_316.set_weights(new_w)


try:
	count = 1
	for s in layer_316_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #316:',count)
except AttributeError:
	for i in layer_316_out:
		count = 1
		for s in layer_316_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #316:',count)



layer_317 = L317()
try:
	try:
		layer_317_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_317_out = layer_317(layer_317_concat)
	except ValueError:
		layer_317_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_317_out = layer_317(layer_317_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_317 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_317_concat = strong_concat_317(new_list_of_inputs)
	layer_317_out = layer_317(layer_317_concat)
if layer_317.count_params() != 0:
	if 317 in set_weigths:
		if len(layer_317.get_weights()) > 1:
			new_w = [get_new_weigths(layer_317, set_weigths[317])]+layer_317.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_317, set_weigths[317])]
		layer_317.set_weights(new_w)


try:
	count = 1
	for s in layer_317_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #317:',count)
except AttributeError:
	for i in layer_317_out:
		count = 1
		for s in layer_317_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #317:',count)



layer_318 = L318()
try:
	try:
		layer_318_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_318_out = layer_318(layer_318_concat)
	except ValueError:
		layer_318_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_318_out = layer_318(layer_318_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_318 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_318_concat = strong_concat_318(new_list_of_inputs)
	layer_318_out = layer_318(layer_318_concat)
if layer_318.count_params() != 0:
	if 318 in set_weigths:
		if len(layer_318.get_weights()) > 1:
			new_w = [get_new_weigths(layer_318, set_weigths[318])]+layer_318.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_318, set_weigths[318])]
		layer_318.set_weights(new_w)


try:
	count = 1
	for s in layer_318_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #318:',count)
except AttributeError:
	for i in layer_318_out:
		count = 1
		for s in layer_318_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #318:',count)



layer_319 = L319()
try:
	try:
		layer_319_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_319_out = layer_319(layer_319_concat)
	except ValueError:
		layer_319_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_319_out = layer_319(layer_319_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_319 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_319_concat = strong_concat_319(new_list_of_inputs)
	layer_319_out = layer_319(layer_319_concat)
if layer_319.count_params() != 0:
	if 319 in set_weigths:
		if len(layer_319.get_weights()) > 1:
			new_w = [get_new_weigths(layer_319, set_weigths[319])]+layer_319.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_319, set_weigths[319])]
		layer_319.set_weights(new_w)


try:
	count = 1
	for s in layer_319_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #319:',count)
except AttributeError:
	for i in layer_319_out:
		count = 1
		for s in layer_319_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #319:',count)



layer_320 = L320()
try:
	try:
		layer_320_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_320_out = layer_320(layer_320_concat)
	except ValueError:
		layer_320_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_320_out = layer_320(layer_320_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_320 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_320_concat = strong_concat_320(new_list_of_inputs)
	layer_320_out = layer_320(layer_320_concat)
if layer_320.count_params() != 0:
	if 320 in set_weigths:
		if len(layer_320.get_weights()) > 1:
			new_w = [get_new_weigths(layer_320, set_weigths[320])]+layer_320.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_320, set_weigths[320])]
		layer_320.set_weights(new_w)


try:
	count = 1
	for s in layer_320_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #320:',count)
except AttributeError:
	for i in layer_320_out:
		count = 1
		for s in layer_320_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #320:',count)



layer_321 = L321()
try:
	try:
		layer_321_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_321_out = layer_321(layer_321_concat)
	except ValueError:
		layer_321_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_321_out = layer_321(layer_321_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_321 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_321_concat = strong_concat_321(new_list_of_inputs)
	layer_321_out = layer_321(layer_321_concat)
if layer_321.count_params() != 0:
	if 321 in set_weigths:
		if len(layer_321.get_weights()) > 1:
			new_w = [get_new_weigths(layer_321, set_weigths[321])]+layer_321.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_321, set_weigths[321])]
		layer_321.set_weights(new_w)


try:
	count = 1
	for s in layer_321_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #321:',count)
except AttributeError:
	for i in layer_321_out:
		count = 1
		for s in layer_321_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #321:',count)



layer_322 = L322()
try:
	try:
		layer_322_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_322_out = layer_322(layer_322_concat)
	except ValueError:
		layer_322_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_322_out = layer_322(layer_322_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_322 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_322_concat = strong_concat_322(new_list_of_inputs)
	layer_322_out = layer_322(layer_322_concat)
if layer_322.count_params() != 0:
	if 322 in set_weigths:
		if len(layer_322.get_weights()) > 1:
			new_w = [get_new_weigths(layer_322, set_weigths[322])]+layer_322.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_322, set_weigths[322])]
		layer_322.set_weights(new_w)


try:
	count = 1
	for s in layer_322_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #322:',count)
except AttributeError:
	for i in layer_322_out:
		count = 1
		for s in layer_322_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #322:',count)



layer_323 = L323()
try:
	try:
		layer_323_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_323_out = layer_323(layer_323_concat)
	except ValueError:
		layer_323_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_323_out = layer_323(layer_323_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_323 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_323_concat = strong_concat_323(new_list_of_inputs)
	layer_323_out = layer_323(layer_323_concat)
if layer_323.count_params() != 0:
	if 323 in set_weigths:
		if len(layer_323.get_weights()) > 1:
			new_w = [get_new_weigths(layer_323, set_weigths[323])]+layer_323.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_323, set_weigths[323])]
		layer_323.set_weights(new_w)


try:
	count = 1
	for s in layer_323_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #323:',count)
except AttributeError:
	for i in layer_323_out:
		count = 1
		for s in layer_323_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #323:',count)



layer_324 = L324()
try:
	try:
		layer_324_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_324_out = layer_324(layer_324_concat)
	except ValueError:
		layer_324_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_324_out = layer_324(layer_324_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_324 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_324_concat = strong_concat_324(new_list_of_inputs)
	layer_324_out = layer_324(layer_324_concat)
if layer_324.count_params() != 0:
	if 324 in set_weigths:
		if len(layer_324.get_weights()) > 1:
			new_w = [get_new_weigths(layer_324, set_weigths[324])]+layer_324.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_324, set_weigths[324])]
		layer_324.set_weights(new_w)


try:
	count = 1
	for s in layer_324_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #324:',count)
except AttributeError:
	for i in layer_324_out:
		count = 1
		for s in layer_324_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #324:',count)



layer_325 = L325()
try:
	try:
		layer_325_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_325_out = layer_325(layer_325_concat)
	except ValueError:
		layer_325_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_325_out = layer_325(layer_325_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_325 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_325_concat = strong_concat_325(new_list_of_inputs)
	layer_325_out = layer_325(layer_325_concat)
if layer_325.count_params() != 0:
	if 325 in set_weigths:
		if len(layer_325.get_weights()) > 1:
			new_w = [get_new_weigths(layer_325, set_weigths[325])]+layer_325.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_325, set_weigths[325])]
		layer_325.set_weights(new_w)


try:
	count = 1
	for s in layer_325_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #325:',count)
except AttributeError:
	for i in layer_325_out:
		count = 1
		for s in layer_325_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #325:',count)



layer_326 = L326()
try:
	try:
		layer_326_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_326_out = layer_326(layer_326_concat)
	except ValueError:
		layer_326_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_326_out = layer_326(layer_326_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_326 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_326_concat = strong_concat_326(new_list_of_inputs)
	layer_326_out = layer_326(layer_326_concat)
if layer_326.count_params() != 0:
	if 326 in set_weigths:
		if len(layer_326.get_weights()) > 1:
			new_w = [get_new_weigths(layer_326, set_weigths[326])]+layer_326.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_326, set_weigths[326])]
		layer_326.set_weights(new_w)


try:
	count = 1
	for s in layer_326_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #326:',count)
except AttributeError:
	for i in layer_326_out:
		count = 1
		for s in layer_326_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #326:',count)



layer_327 = L327()
try:
	try:
		layer_327_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_327_out = layer_327(layer_327_concat)
	except ValueError:
		layer_327_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_327_out = layer_327(layer_327_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_327 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_327_concat = strong_concat_327(new_list_of_inputs)
	layer_327_out = layer_327(layer_327_concat)
if layer_327.count_params() != 0:
	if 327 in set_weigths:
		if len(layer_327.get_weights()) > 1:
			new_w = [get_new_weigths(layer_327, set_weigths[327])]+layer_327.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_327, set_weigths[327])]
		layer_327.set_weights(new_w)


try:
	count = 1
	for s in layer_327_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #327:',count)
except AttributeError:
	for i in layer_327_out:
		count = 1
		for s in layer_327_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #327:',count)



layer_328 = L328()
try:
	try:
		layer_328_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_328_out = layer_328(layer_328_concat)
	except ValueError:
		layer_328_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_328_out = layer_328(layer_328_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_328 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_328_concat = strong_concat_328(new_list_of_inputs)
	layer_328_out = layer_328(layer_328_concat)
if layer_328.count_params() != 0:
	if 328 in set_weigths:
		if len(layer_328.get_weights()) > 1:
			new_w = [get_new_weigths(layer_328, set_weigths[328])]+layer_328.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_328, set_weigths[328])]
		layer_328.set_weights(new_w)


try:
	count = 1
	for s in layer_328_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #328:',count)
except AttributeError:
	for i in layer_328_out:
		count = 1
		for s in layer_328_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #328:',count)



layer_329 = L329()
try:
	try:
		layer_329_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_329_out = layer_329(layer_329_concat)
	except ValueError:
		layer_329_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_329_out = layer_329(layer_329_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_329 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_329_concat = strong_concat_329(new_list_of_inputs)
	layer_329_out = layer_329(layer_329_concat)
if layer_329.count_params() != 0:
	if 329 in set_weigths:
		if len(layer_329.get_weights()) > 1:
			new_w = [get_new_weigths(layer_329, set_weigths[329])]+layer_329.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_329, set_weigths[329])]
		layer_329.set_weights(new_w)


try:
	count = 1
	for s in layer_329_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #329:',count)
except AttributeError:
	for i in layer_329_out:
		count = 1
		for s in layer_329_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #329:',count)



layer_330 = L330()
try:
	try:
		layer_330_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_330_out = layer_330(layer_330_concat)
	except ValueError:
		layer_330_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_330_out = layer_330(layer_330_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_330 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_330_concat = strong_concat_330(new_list_of_inputs)
	layer_330_out = layer_330(layer_330_concat)
if layer_330.count_params() != 0:
	if 330 in set_weigths:
		if len(layer_330.get_weights()) > 1:
			new_w = [get_new_weigths(layer_330, set_weigths[330])]+layer_330.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_330, set_weigths[330])]
		layer_330.set_weights(new_w)


try:
	count = 1
	for s in layer_330_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #330:',count)
except AttributeError:
	for i in layer_330_out:
		count = 1
		for s in layer_330_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #330:',count)



layer_331 = L331()
try:
	try:
		layer_331_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_331_out = layer_331(layer_331_concat)
	except ValueError:
		layer_331_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_331_out = layer_331(layer_331_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_331 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_331_concat = strong_concat_331(new_list_of_inputs)
	layer_331_out = layer_331(layer_331_concat)
if layer_331.count_params() != 0:
	if 331 in set_weigths:
		if len(layer_331.get_weights()) > 1:
			new_w = [get_new_weigths(layer_331, set_weigths[331])]+layer_331.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_331, set_weigths[331])]
		layer_331.set_weights(new_w)


try:
	count = 1
	for s in layer_331_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #331:',count)
except AttributeError:
	for i in layer_331_out:
		count = 1
		for s in layer_331_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #331:',count)



layer_332 = L332()
try:
	try:
		layer_332_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_332_out = layer_332(layer_332_concat)
	except ValueError:
		layer_332_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_332_out = layer_332(layer_332_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_332 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_332_concat = strong_concat_332(new_list_of_inputs)
	layer_332_out = layer_332(layer_332_concat)
if layer_332.count_params() != 0:
	if 332 in set_weigths:
		if len(layer_332.get_weights()) > 1:
			new_w = [get_new_weigths(layer_332, set_weigths[332])]+layer_332.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_332, set_weigths[332])]
		layer_332.set_weights(new_w)


try:
	count = 1
	for s in layer_332_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #332:',count)
except AttributeError:
	for i in layer_332_out:
		count = 1
		for s in layer_332_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #332:',count)



layer_333 = L333()
try:
	try:
		layer_333_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_333_out = layer_333(layer_333_concat)
	except ValueError:
		layer_333_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_333_out = layer_333(layer_333_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_333 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_333_concat = strong_concat_333(new_list_of_inputs)
	layer_333_out = layer_333(layer_333_concat)
if layer_333.count_params() != 0:
	if 333 in set_weigths:
		if len(layer_333.get_weights()) > 1:
			new_w = [get_new_weigths(layer_333, set_weigths[333])]+layer_333.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_333, set_weigths[333])]
		layer_333.set_weights(new_w)


try:
	count = 1
	for s in layer_333_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #333:',count)
except AttributeError:
	for i in layer_333_out:
		count = 1
		for s in layer_333_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #333:',count)



layer_334 = L334()
try:
	try:
		layer_334_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_334_out = layer_334(layer_334_concat)
	except ValueError:
		layer_334_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_334_out = layer_334(layer_334_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_334 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_334_concat = strong_concat_334(new_list_of_inputs)
	layer_334_out = layer_334(layer_334_concat)
if layer_334.count_params() != 0:
	if 334 in set_weigths:
		if len(layer_334.get_weights()) > 1:
			new_w = [get_new_weigths(layer_334, set_weigths[334])]+layer_334.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_334, set_weigths[334])]
		layer_334.set_weights(new_w)


try:
	count = 1
	for s in layer_334_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #334:',count)
except AttributeError:
	for i in layer_334_out:
		count = 1
		for s in layer_334_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #334:',count)



layer_335 = L335()
try:
	try:
		layer_335_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_335_out = layer_335(layer_335_concat)
	except ValueError:
		layer_335_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_335_out = layer_335(layer_335_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_335 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_335_concat = strong_concat_335(new_list_of_inputs)
	layer_335_out = layer_335(layer_335_concat)
if layer_335.count_params() != 0:
	if 335 in set_weigths:
		if len(layer_335.get_weights()) > 1:
			new_w = [get_new_weigths(layer_335, set_weigths[335])]+layer_335.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_335, set_weigths[335])]
		layer_335.set_weights(new_w)


try:
	count = 1
	for s in layer_335_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #335:',count)
except AttributeError:
	for i in layer_335_out:
		count = 1
		for s in layer_335_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #335:',count)



layer_336 = L336()
try:
	try:
		layer_336_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_336_out = layer_336(layer_336_concat)
	except ValueError:
		layer_336_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_336_out = layer_336(layer_336_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_336 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_336_concat = strong_concat_336(new_list_of_inputs)
	layer_336_out = layer_336(layer_336_concat)
if layer_336.count_params() != 0:
	if 336 in set_weigths:
		if len(layer_336.get_weights()) > 1:
			new_w = [get_new_weigths(layer_336, set_weigths[336])]+layer_336.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_336, set_weigths[336])]
		layer_336.set_weights(new_w)


try:
	count = 1
	for s in layer_336_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #336:',count)
except AttributeError:
	for i in layer_336_out:
		count = 1
		for s in layer_336_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #336:',count)



layer_337 = L337()
try:
	try:
		layer_337_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_337_out = layer_337(layer_337_concat)
	except ValueError:
		layer_337_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_337_out = layer_337(layer_337_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_337 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_337_concat = strong_concat_337(new_list_of_inputs)
	layer_337_out = layer_337(layer_337_concat)
if layer_337.count_params() != 0:
	if 337 in set_weigths:
		if len(layer_337.get_weights()) > 1:
			new_w = [get_new_weigths(layer_337, set_weigths[337])]+layer_337.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_337, set_weigths[337])]
		layer_337.set_weights(new_w)


try:
	count = 1
	for s in layer_337_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #337:',count)
except AttributeError:
	for i in layer_337_out:
		count = 1
		for s in layer_337_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #337:',count)



layer_338 = L338()
try:
	try:
		layer_338_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_338_out = layer_338(layer_338_concat)
	except ValueError:
		layer_338_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_338_out = layer_338(layer_338_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_338 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_338_concat = strong_concat_338(new_list_of_inputs)
	layer_338_out = layer_338(layer_338_concat)
if layer_338.count_params() != 0:
	if 338 in set_weigths:
		if len(layer_338.get_weights()) > 1:
			new_w = [get_new_weigths(layer_338, set_weigths[338])]+layer_338.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_338, set_weigths[338])]
		layer_338.set_weights(new_w)


try:
	count = 1
	for s in layer_338_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #338:',count)
except AttributeError:
	for i in layer_338_out:
		count = 1
		for s in layer_338_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #338:',count)



layer_339 = L339()
try:
	try:
		layer_339_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_339_out = layer_339(layer_339_concat)
	except ValueError:
		layer_339_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_339_out = layer_339(layer_339_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_339 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_339_concat = strong_concat_339(new_list_of_inputs)
	layer_339_out = layer_339(layer_339_concat)
if layer_339.count_params() != 0:
	if 339 in set_weigths:
		if len(layer_339.get_weights()) > 1:
			new_w = [get_new_weigths(layer_339, set_weigths[339])]+layer_339.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_339, set_weigths[339])]
		layer_339.set_weights(new_w)


try:
	count = 1
	for s in layer_339_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #339:',count)
except AttributeError:
	for i in layer_339_out:
		count = 1
		for s in layer_339_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #339:',count)



layer_340 = L340()
try:
	try:
		layer_340_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_340_out = layer_340(layer_340_concat)
	except ValueError:
		layer_340_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_340_out = layer_340(layer_340_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_340 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_340_concat = strong_concat_340(new_list_of_inputs)
	layer_340_out = layer_340(layer_340_concat)
if layer_340.count_params() != 0:
	if 340 in set_weigths:
		if len(layer_340.get_weights()) > 1:
			new_w = [get_new_weigths(layer_340, set_weigths[340])]+layer_340.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_340, set_weigths[340])]
		layer_340.set_weights(new_w)


try:
	count = 1
	for s in layer_340_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #340:',count)
except AttributeError:
	for i in layer_340_out:
		count = 1
		for s in layer_340_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #340:',count)



layer_341 = L341()
try:
	try:
		layer_341_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_341_out = layer_341(layer_341_concat)
	except ValueError:
		layer_341_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_341_out = layer_341(layer_341_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_341 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_341_concat = strong_concat_341(new_list_of_inputs)
	layer_341_out = layer_341(layer_341_concat)
if layer_341.count_params() != 0:
	if 341 in set_weigths:
		if len(layer_341.get_weights()) > 1:
			new_w = [get_new_weigths(layer_341, set_weigths[341])]+layer_341.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_341, set_weigths[341])]
		layer_341.set_weights(new_w)


try:
	count = 1
	for s in layer_341_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #341:',count)
except AttributeError:
	for i in layer_341_out:
		count = 1
		for s in layer_341_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #341:',count)



layer_342 = L342()
try:
	try:
		layer_342_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_342_out = layer_342(layer_342_concat)
	except ValueError:
		layer_342_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_342_out = layer_342(layer_342_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_342 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_342_concat = strong_concat_342(new_list_of_inputs)
	layer_342_out = layer_342(layer_342_concat)
if layer_342.count_params() != 0:
	if 342 in set_weigths:
		if len(layer_342.get_weights()) > 1:
			new_w = [get_new_weigths(layer_342, set_weigths[342])]+layer_342.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_342, set_weigths[342])]
		layer_342.set_weights(new_w)


try:
	count = 1
	for s in layer_342_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #342:',count)
except AttributeError:
	for i in layer_342_out:
		count = 1
		for s in layer_342_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #342:',count)



layer_343 = L343()
try:
	try:
		layer_343_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_343_out = layer_343(layer_343_concat)
	except ValueError:
		layer_343_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_343_out = layer_343(layer_343_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_343 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_343_concat = strong_concat_343(new_list_of_inputs)
	layer_343_out = layer_343(layer_343_concat)
if layer_343.count_params() != 0:
	if 343 in set_weigths:
		if len(layer_343.get_weights()) > 1:
			new_w = [get_new_weigths(layer_343, set_weigths[343])]+layer_343.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_343, set_weigths[343])]
		layer_343.set_weights(new_w)


try:
	count = 1
	for s in layer_343_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #343:',count)
except AttributeError:
	for i in layer_343_out:
		count = 1
		for s in layer_343_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #343:',count)



layer_344 = L344()
try:
	try:
		layer_344_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_344_out = layer_344(layer_344_concat)
	except ValueError:
		layer_344_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_344_out = layer_344(layer_344_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_344 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_344_concat = strong_concat_344(new_list_of_inputs)
	layer_344_out = layer_344(layer_344_concat)
if layer_344.count_params() != 0:
	if 344 in set_weigths:
		if len(layer_344.get_weights()) > 1:
			new_w = [get_new_weigths(layer_344, set_weigths[344])]+layer_344.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_344, set_weigths[344])]
		layer_344.set_weights(new_w)


try:
	count = 1
	for s in layer_344_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #344:',count)
except AttributeError:
	for i in layer_344_out:
		count = 1
		for s in layer_344_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #344:',count)



layer_346 = L346()
try:
	try:
		layer_346_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_346_out = layer_346(layer_346_concat)
	except ValueError:
		layer_346_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_346_out = layer_346(layer_346_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_346 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_346_concat = strong_concat_346(new_list_of_inputs)
	layer_346_out = layer_346(layer_346_concat)
if layer_346.count_params() != 0:
	if 346 in set_weigths:
		if len(layer_346.get_weights()) > 1:
			new_w = [get_new_weigths(layer_346, set_weigths[346])]+layer_346.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_346, set_weigths[346])]
		layer_346.set_weights(new_w)


try:
	count = 1
	for s in layer_346_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #346:',count)
except AttributeError:
	for i in layer_346_out:
		count = 1
		for s in layer_346_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #346:',count)



layer_347 = L347()
try:
	try:
		layer_347_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_347_out = layer_347(layer_347_concat)
	except ValueError:
		layer_347_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_347_out = layer_347(layer_347_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_347 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_347_concat = strong_concat_347(new_list_of_inputs)
	layer_347_out = layer_347(layer_347_concat)
if layer_347.count_params() != 0:
	if 347 in set_weigths:
		if len(layer_347.get_weights()) > 1:
			new_w = [get_new_weigths(layer_347, set_weigths[347])]+layer_347.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_347, set_weigths[347])]
		layer_347.set_weights(new_w)


try:
	count = 1
	for s in layer_347_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #347:',count)
except AttributeError:
	for i in layer_347_out:
		count = 1
		for s in layer_347_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #347:',count)



layer_348 = L348()
try:
	try:
		layer_348_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_348_out = layer_348(layer_348_concat)
	except ValueError:
		layer_348_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_348_out = layer_348(layer_348_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_348 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_348_concat = strong_concat_348(new_list_of_inputs)
	layer_348_out = layer_348(layer_348_concat)
if layer_348.count_params() != 0:
	if 348 in set_weigths:
		if len(layer_348.get_weights()) > 1:
			new_w = [get_new_weigths(layer_348, set_weigths[348])]+layer_348.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_348, set_weigths[348])]
		layer_348.set_weights(new_w)


try:
	count = 1
	for s in layer_348_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #348:',count)
except AttributeError:
	for i in layer_348_out:
		count = 1
		for s in layer_348_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #348:',count)



layer_349 = L349()
try:
	try:
		layer_349_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_349_out = layer_349(layer_349_concat)
	except ValueError:
		layer_349_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_349_out = layer_349(layer_349_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_349 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_349_concat = strong_concat_349(new_list_of_inputs)
	layer_349_out = layer_349(layer_349_concat)
if layer_349.count_params() != 0:
	if 349 in set_weigths:
		if len(layer_349.get_weights()) > 1:
			new_w = [get_new_weigths(layer_349, set_weigths[349])]+layer_349.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_349, set_weigths[349])]
		layer_349.set_weights(new_w)


try:
	count = 1
	for s in layer_349_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #349:',count)
except AttributeError:
	for i in layer_349_out:
		count = 1
		for s in layer_349_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #349:',count)



layer_350 = L350()
try:
	try:
		layer_350_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_350_out = layer_350(layer_350_concat)
	except ValueError:
		layer_350_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_350_out = layer_350(layer_350_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_350 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_350_concat = strong_concat_350(new_list_of_inputs)
	layer_350_out = layer_350(layer_350_concat)
if layer_350.count_params() != 0:
	if 350 in set_weigths:
		if len(layer_350.get_weights()) > 1:
			new_w = [get_new_weigths(layer_350, set_weigths[350])]+layer_350.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_350, set_weigths[350])]
		layer_350.set_weights(new_w)


try:
	count = 1
	for s in layer_350_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #350:',count)
except AttributeError:
	for i in layer_350_out:
		count = 1
		for s in layer_350_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #350:',count)



layer_351 = L351()
try:
	try:
		layer_351_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_351_out = layer_351(layer_351_concat)
	except ValueError:
		layer_351_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_351_out = layer_351(layer_351_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_351 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_351_concat = strong_concat_351(new_list_of_inputs)
	layer_351_out = layer_351(layer_351_concat)
if layer_351.count_params() != 0:
	if 351 in set_weigths:
		if len(layer_351.get_weights()) > 1:
			new_w = [get_new_weigths(layer_351, set_weigths[351])]+layer_351.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_351, set_weigths[351])]
		layer_351.set_weights(new_w)


try:
	count = 1
	for s in layer_351_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #351:',count)
except AttributeError:
	for i in layer_351_out:
		count = 1
		for s in layer_351_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #351:',count)



layer_247 = L247()
try:
	try:
		layer_247_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_247_out = layer_247(layer_247_concat)
	except ValueError:
		layer_247_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_247_out = layer_247(layer_247_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_247 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_247_concat = strong_concat_247(new_list_of_inputs)
	layer_247_out = layer_247(layer_247_concat)
if layer_247.count_params() != 0:
	if 247 in set_weigths:
		if len(layer_247.get_weights()) > 1:
			new_w = [get_new_weigths(layer_247, set_weigths[247])]+layer_247.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_247, set_weigths[247])]
		layer_247.set_weights(new_w)


try:
	count = 1
	for s in layer_247_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #247:',count)
except AttributeError:
	for i in layer_247_out:
		count = 1
		for s in layer_247_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #247:',count)



layer_248 = L248()
try:
	try:
		layer_248_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_248_out = layer_248(layer_248_concat)
	except ValueError:
		layer_248_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_248_out = layer_248(layer_248_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_248 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_248_concat = strong_concat_248(new_list_of_inputs)
	layer_248_out = layer_248(layer_248_concat)
if layer_248.count_params() != 0:
	if 248 in set_weigths:
		if len(layer_248.get_weights()) > 1:
			new_w = [get_new_weigths(layer_248, set_weigths[248])]+layer_248.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_248, set_weigths[248])]
		layer_248.set_weights(new_w)


try:
	count = 1
	for s in layer_248_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #248:',count)
except AttributeError:
	for i in layer_248_out:
		count = 1
		for s in layer_248_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #248:',count)



layer_249 = L249()
try:
	try:
		layer_249_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_249_out = layer_249(layer_249_concat)
	except ValueError:
		layer_249_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_249_out = layer_249(layer_249_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_249 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_249_concat = strong_concat_249(new_list_of_inputs)
	layer_249_out = layer_249(layer_249_concat)
if layer_249.count_params() != 0:
	if 249 in set_weigths:
		if len(layer_249.get_weights()) > 1:
			new_w = [get_new_weigths(layer_249, set_weigths[249])]+layer_249.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_249, set_weigths[249])]
		layer_249.set_weights(new_w)


try:
	count = 1
	for s in layer_249_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #249:',count)
except AttributeError:
	for i in layer_249_out:
		count = 1
		for s in layer_249_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #249:',count)



layer_250 = L250()
try:
	try:
		layer_250_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_250_out = layer_250(layer_250_concat)
	except ValueError:
		layer_250_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_250_out = layer_250(layer_250_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_250 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_250_concat = strong_concat_250(new_list_of_inputs)
	layer_250_out = layer_250(layer_250_concat)
if layer_250.count_params() != 0:
	if 250 in set_weigths:
		if len(layer_250.get_weights()) > 1:
			new_w = [get_new_weigths(layer_250, set_weigths[250])]+layer_250.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_250, set_weigths[250])]
		layer_250.set_weights(new_w)


try:
	count = 1
	for s in layer_250_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #250:',count)
except AttributeError:
	for i in layer_250_out:
		count = 1
		for s in layer_250_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #250:',count)



layer_251 = L251()
try:
	try:
		layer_251_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_251_out = layer_251(layer_251_concat)
	except ValueError:
		layer_251_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_251_out = layer_251(layer_251_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_251 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_251_concat = strong_concat_251(new_list_of_inputs)
	layer_251_out = layer_251(layer_251_concat)
if layer_251.count_params() != 0:
	if 251 in set_weigths:
		if len(layer_251.get_weights()) > 1:
			new_w = [get_new_weigths(layer_251, set_weigths[251])]+layer_251.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_251, set_weigths[251])]
		layer_251.set_weights(new_w)


try:
	count = 1
	for s in layer_251_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #251:',count)
except AttributeError:
	for i in layer_251_out:
		count = 1
		for s in layer_251_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #251:',count)



layer_252 = L252()
try:
	try:
		layer_252_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_252_out = layer_252(layer_252_concat)
	except ValueError:
		layer_252_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_252_out = layer_252(layer_252_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_252 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_252_concat = strong_concat_252(new_list_of_inputs)
	layer_252_out = layer_252(layer_252_concat)
if layer_252.count_params() != 0:
	if 252 in set_weigths:
		if len(layer_252.get_weights()) > 1:
			new_w = [get_new_weigths(layer_252, set_weigths[252])]+layer_252.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_252, set_weigths[252])]
		layer_252.set_weights(new_w)


try:
	count = 1
	for s in layer_252_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #252:',count)
except AttributeError:
	for i in layer_252_out:
		count = 1
		for s in layer_252_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #252:',count)



layer_253 = L253()
try:
	try:
		layer_253_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_253_out = layer_253(layer_253_concat)
	except ValueError:
		layer_253_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_253_out = layer_253(layer_253_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_253 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_253_concat = strong_concat_253(new_list_of_inputs)
	layer_253_out = layer_253(layer_253_concat)
if layer_253.count_params() != 0:
	if 253 in set_weigths:
		if len(layer_253.get_weights()) > 1:
			new_w = [get_new_weigths(layer_253, set_weigths[253])]+layer_253.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_253, set_weigths[253])]
		layer_253.set_weights(new_w)


try:
	count = 1
	for s in layer_253_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #253:',count)
except AttributeError:
	for i in layer_253_out:
		count = 1
		for s in layer_253_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #253:',count)



layer_254 = L254()
try:
	try:
		layer_254_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_254_out = layer_254(layer_254_concat)
	except ValueError:
		layer_254_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_254_out = layer_254(layer_254_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_254 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_254_concat = strong_concat_254(new_list_of_inputs)
	layer_254_out = layer_254(layer_254_concat)
if layer_254.count_params() != 0:
	if 254 in set_weigths:
		if len(layer_254.get_weights()) > 1:
			new_w = [get_new_weigths(layer_254, set_weigths[254])]+layer_254.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_254, set_weigths[254])]
		layer_254.set_weights(new_w)


try:
	count = 1
	for s in layer_254_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #254:',count)
except AttributeError:
	for i in layer_254_out:
		count = 1
		for s in layer_254_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #254:',count)



layer_255 = L255()
try:
	try:
		layer_255_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_255_out = layer_255(layer_255_concat)
	except ValueError:
		layer_255_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_255_out = layer_255(layer_255_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_255 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_255_concat = strong_concat_255(new_list_of_inputs)
	layer_255_out = layer_255(layer_255_concat)
if layer_255.count_params() != 0:
	if 255 in set_weigths:
		if len(layer_255.get_weights()) > 1:
			new_w = [get_new_weigths(layer_255, set_weigths[255])]+layer_255.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_255, set_weigths[255])]
		layer_255.set_weights(new_w)


try:
	count = 1
	for s in layer_255_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #255:',count)
except AttributeError:
	for i in layer_255_out:
		count = 1
		for s in layer_255_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #255:',count)



layer_256 = L256()
try:
	try:
		layer_256_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_256_out = layer_256(layer_256_concat)
	except ValueError:
		layer_256_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_256_out = layer_256(layer_256_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_256 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_256_concat = strong_concat_256(new_list_of_inputs)
	layer_256_out = layer_256(layer_256_concat)
if layer_256.count_params() != 0:
	if 256 in set_weigths:
		if len(layer_256.get_weights()) > 1:
			new_w = [get_new_weigths(layer_256, set_weigths[256])]+layer_256.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_256, set_weigths[256])]
		layer_256.set_weights(new_w)


try:
	count = 1
	for s in layer_256_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #256:',count)
except AttributeError:
	for i in layer_256_out:
		count = 1
		for s in layer_256_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #256:',count)



layer_257 = L257()
try:
	try:
		layer_257_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_257_out = layer_257(layer_257_concat)
	except ValueError:
		layer_257_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_257_out = layer_257(layer_257_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_257 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_257_concat = strong_concat_257(new_list_of_inputs)
	layer_257_out = layer_257(layer_257_concat)
if layer_257.count_params() != 0:
	if 257 in set_weigths:
		if len(layer_257.get_weights()) > 1:
			new_w = [get_new_weigths(layer_257, set_weigths[257])]+layer_257.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_257, set_weigths[257])]
		layer_257.set_weights(new_w)


try:
	count = 1
	for s in layer_257_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #257:',count)
except AttributeError:
	for i in layer_257_out:
		count = 1
		for s in layer_257_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #257:',count)



layer_258 = L258()
try:
	try:
		layer_258_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_258_out = layer_258(layer_258_concat)
	except ValueError:
		layer_258_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_258_out = layer_258(layer_258_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_258 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_258_concat = strong_concat_258(new_list_of_inputs)
	layer_258_out = layer_258(layer_258_concat)
if layer_258.count_params() != 0:
	if 258 in set_weigths:
		if len(layer_258.get_weights()) > 1:
			new_w = [get_new_weigths(layer_258, set_weigths[258])]+layer_258.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_258, set_weigths[258])]
		layer_258.set_weights(new_w)


try:
	count = 1
	for s in layer_258_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #258:',count)
except AttributeError:
	for i in layer_258_out:
		count = 1
		for s in layer_258_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #258:',count)



layer_259 = L259()
try:
	try:
		layer_259_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_259_out = layer_259(layer_259_concat)
	except ValueError:
		layer_259_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_259_out = layer_259(layer_259_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_259 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_259_concat = strong_concat_259(new_list_of_inputs)
	layer_259_out = layer_259(layer_259_concat)
if layer_259.count_params() != 0:
	if 259 in set_weigths:
		if len(layer_259.get_weights()) > 1:
			new_w = [get_new_weigths(layer_259, set_weigths[259])]+layer_259.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_259, set_weigths[259])]
		layer_259.set_weights(new_w)


try:
	count = 1
	for s in layer_259_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #259:',count)
except AttributeError:
	for i in layer_259_out:
		count = 1
		for s in layer_259_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #259:',count)



layer_260 = L260()
try:
	try:
		layer_260_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_260_out = layer_260(layer_260_concat)
	except ValueError:
		layer_260_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_260_out = layer_260(layer_260_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_260 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_260_concat = strong_concat_260(new_list_of_inputs)
	layer_260_out = layer_260(layer_260_concat)
if layer_260.count_params() != 0:
	if 260 in set_weigths:
		if len(layer_260.get_weights()) > 1:
			new_w = [get_new_weigths(layer_260, set_weigths[260])]+layer_260.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_260, set_weigths[260])]
		layer_260.set_weights(new_w)


try:
	count = 1
	for s in layer_260_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #260:',count)
except AttributeError:
	for i in layer_260_out:
		count = 1
		for s in layer_260_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #260:',count)



layer_261 = L261()
try:
	try:
		layer_261_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_261_out = layer_261(layer_261_concat)
	except ValueError:
		layer_261_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_261_out = layer_261(layer_261_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_261 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_261_concat = strong_concat_261(new_list_of_inputs)
	layer_261_out = layer_261(layer_261_concat)
if layer_261.count_params() != 0:
	if 261 in set_weigths:
		if len(layer_261.get_weights()) > 1:
			new_w = [get_new_weigths(layer_261, set_weigths[261])]+layer_261.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_261, set_weigths[261])]
		layer_261.set_weights(new_w)


try:
	count = 1
	for s in layer_261_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #261:',count)
except AttributeError:
	for i in layer_261_out:
		count = 1
		for s in layer_261_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #261:',count)



layer_262 = L262()
try:
	try:
		layer_262_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_262_out = layer_262(layer_262_concat)
	except ValueError:
		layer_262_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_262_out = layer_262(layer_262_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_262 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_262_concat = strong_concat_262(new_list_of_inputs)
	layer_262_out = layer_262(layer_262_concat)
if layer_262.count_params() != 0:
	if 262 in set_weigths:
		if len(layer_262.get_weights()) > 1:
			new_w = [get_new_weigths(layer_262, set_weigths[262])]+layer_262.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_262, set_weigths[262])]
		layer_262.set_weights(new_w)


try:
	count = 1
	for s in layer_262_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #262:',count)
except AttributeError:
	for i in layer_262_out:
		count = 1
		for s in layer_262_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #262:',count)



layer_263 = L263()
try:
	try:
		layer_263_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_263_out = layer_263(layer_263_concat)
	except ValueError:
		layer_263_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_263_out = layer_263(layer_263_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_263 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_263_concat = strong_concat_263(new_list_of_inputs)
	layer_263_out = layer_263(layer_263_concat)
if layer_263.count_params() != 0:
	if 263 in set_weigths:
		if len(layer_263.get_weights()) > 1:
			new_w = [get_new_weigths(layer_263, set_weigths[263])]+layer_263.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_263, set_weigths[263])]
		layer_263.set_weights(new_w)


try:
	count = 1
	for s in layer_263_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #263:',count)
except AttributeError:
	for i in layer_263_out:
		count = 1
		for s in layer_263_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #263:',count)



layer_264 = L264()
try:
	try:
		layer_264_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_264_out = layer_264(layer_264_concat)
	except ValueError:
		layer_264_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_264_out = layer_264(layer_264_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_264 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_264_concat = strong_concat_264(new_list_of_inputs)
	layer_264_out = layer_264(layer_264_concat)
if layer_264.count_params() != 0:
	if 264 in set_weigths:
		if len(layer_264.get_weights()) > 1:
			new_w = [get_new_weigths(layer_264, set_weigths[264])]+layer_264.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_264, set_weigths[264])]
		layer_264.set_weights(new_w)


try:
	count = 1
	for s in layer_264_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #264:',count)
except AttributeError:
	for i in layer_264_out:
		count = 1
		for s in layer_264_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #264:',count)



layer_265 = L265()
try:
	try:
		layer_265_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_265_out = layer_265(layer_265_concat)
	except ValueError:
		layer_265_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_265_out = layer_265(layer_265_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_265 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_265_concat = strong_concat_265(new_list_of_inputs)
	layer_265_out = layer_265(layer_265_concat)
if layer_265.count_params() != 0:
	if 265 in set_weigths:
		if len(layer_265.get_weights()) > 1:
			new_w = [get_new_weigths(layer_265, set_weigths[265])]+layer_265.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_265, set_weigths[265])]
		layer_265.set_weights(new_w)


try:
	count = 1
	for s in layer_265_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #265:',count)
except AttributeError:
	for i in layer_265_out:
		count = 1
		for s in layer_265_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #265:',count)



layer_266 = L266()
try:
	try:
		layer_266_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_266_out = layer_266(layer_266_concat)
	except ValueError:
		layer_266_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_266_out = layer_266(layer_266_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_266 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_266_concat = strong_concat_266(new_list_of_inputs)
	layer_266_out = layer_266(layer_266_concat)
if layer_266.count_params() != 0:
	if 266 in set_weigths:
		if len(layer_266.get_weights()) > 1:
			new_w = [get_new_weigths(layer_266, set_weigths[266])]+layer_266.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_266, set_weigths[266])]
		layer_266.set_weights(new_w)


try:
	count = 1
	for s in layer_266_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #266:',count)
except AttributeError:
	for i in layer_266_out:
		count = 1
		for s in layer_266_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #266:',count)



layer_267 = L267()
try:
	try:
		layer_267_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_267_out = layer_267(layer_267_concat)
	except ValueError:
		layer_267_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_267_out = layer_267(layer_267_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_267 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_267_concat = strong_concat_267(new_list_of_inputs)
	layer_267_out = layer_267(layer_267_concat)
if layer_267.count_params() != 0:
	if 267 in set_weigths:
		if len(layer_267.get_weights()) > 1:
			new_w = [get_new_weigths(layer_267, set_weigths[267])]+layer_267.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_267, set_weigths[267])]
		layer_267.set_weights(new_w)


try:
	count = 1
	for s in layer_267_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #267:',count)
except AttributeError:
	for i in layer_267_out:
		count = 1
		for s in layer_267_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #267:',count)



layer_268 = L268()
try:
	try:
		layer_268_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_268_out = layer_268(layer_268_concat)
	except ValueError:
		layer_268_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_268_out = layer_268(layer_268_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_268 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_268_concat = strong_concat_268(new_list_of_inputs)
	layer_268_out = layer_268(layer_268_concat)
if layer_268.count_params() != 0:
	if 268 in set_weigths:
		if len(layer_268.get_weights()) > 1:
			new_w = [get_new_weigths(layer_268, set_weigths[268])]+layer_268.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_268, set_weigths[268])]
		layer_268.set_weights(new_w)


try:
	count = 1
	for s in layer_268_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #268:',count)
except AttributeError:
	for i in layer_268_out:
		count = 1
		for s in layer_268_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #268:',count)



layer_269 = L269()
try:
	try:
		layer_269_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_269_out = layer_269(layer_269_concat)
	except ValueError:
		layer_269_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_269_out = layer_269(layer_269_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_269 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_269_concat = strong_concat_269(new_list_of_inputs)
	layer_269_out = layer_269(layer_269_concat)
if layer_269.count_params() != 0:
	if 269 in set_weigths:
		if len(layer_269.get_weights()) > 1:
			new_w = [get_new_weigths(layer_269, set_weigths[269])]+layer_269.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_269, set_weigths[269])]
		layer_269.set_weights(new_w)


try:
	count = 1
	for s in layer_269_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #269:',count)
except AttributeError:
	for i in layer_269_out:
		count = 1
		for s in layer_269_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #269:',count)



layer_270 = L270()
try:
	try:
		layer_270_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_270_out = layer_270(layer_270_concat)
	except ValueError:
		layer_270_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_270_out = layer_270(layer_270_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_270 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_270_concat = strong_concat_270(new_list_of_inputs)
	layer_270_out = layer_270(layer_270_concat)
if layer_270.count_params() != 0:
	if 270 in set_weigths:
		if len(layer_270.get_weights()) > 1:
			new_w = [get_new_weigths(layer_270, set_weigths[270])]+layer_270.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_270, set_weigths[270])]
		layer_270.set_weights(new_w)


try:
	count = 1
	for s in layer_270_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #270:',count)
except AttributeError:
	for i in layer_270_out:
		count = 1
		for s in layer_270_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #270:',count)



layer_271 = L271()
try:
	try:
		layer_271_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_271_out = layer_271(layer_271_concat)
	except ValueError:
		layer_271_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_271_out = layer_271(layer_271_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_271 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_271_concat = strong_concat_271(new_list_of_inputs)
	layer_271_out = layer_271(layer_271_concat)
if layer_271.count_params() != 0:
	if 271 in set_weigths:
		if len(layer_271.get_weights()) > 1:
			new_w = [get_new_weigths(layer_271, set_weigths[271])]+layer_271.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_271, set_weigths[271])]
		layer_271.set_weights(new_w)


try:
	count = 1
	for s in layer_271_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #271:',count)
except AttributeError:
	for i in layer_271_out:
		count = 1
		for s in layer_271_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #271:',count)



layer_272 = L272()
try:
	try:
		layer_272_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_272_out = layer_272(layer_272_concat)
	except ValueError:
		layer_272_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_272_out = layer_272(layer_272_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_272 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_272_concat = strong_concat_272(new_list_of_inputs)
	layer_272_out = layer_272(layer_272_concat)
if layer_272.count_params() != 0:
	if 272 in set_weigths:
		if len(layer_272.get_weights()) > 1:
			new_w = [get_new_weigths(layer_272, set_weigths[272])]+layer_272.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_272, set_weigths[272])]
		layer_272.set_weights(new_w)


try:
	count = 1
	for s in layer_272_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #272:',count)
except AttributeError:
	for i in layer_272_out:
		count = 1
		for s in layer_272_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #272:',count)



layer_273 = L273()
try:
	try:
		layer_273_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_273_out = layer_273(layer_273_concat)
	except ValueError:
		layer_273_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_273_out = layer_273(layer_273_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_273 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_273_concat = strong_concat_273(new_list_of_inputs)
	layer_273_out = layer_273(layer_273_concat)
if layer_273.count_params() != 0:
	if 273 in set_weigths:
		if len(layer_273.get_weights()) > 1:
			new_w = [get_new_weigths(layer_273, set_weigths[273])]+layer_273.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_273, set_weigths[273])]
		layer_273.set_weights(new_w)


try:
	count = 1
	for s in layer_273_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #273:',count)
except AttributeError:
	for i in layer_273_out:
		count = 1
		for s in layer_273_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #273:',count)



layer_274 = L274()
try:
	try:
		layer_274_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_274_out = layer_274(layer_274_concat)
	except ValueError:
		layer_274_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_274_out = layer_274(layer_274_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_274 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_274_concat = strong_concat_274(new_list_of_inputs)
	layer_274_out = layer_274(layer_274_concat)
if layer_274.count_params() != 0:
	if 274 in set_weigths:
		if len(layer_274.get_weights()) > 1:
			new_w = [get_new_weigths(layer_274, set_weigths[274])]+layer_274.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_274, set_weigths[274])]
		layer_274.set_weights(new_w)


try:
	count = 1
	for s in layer_274_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #274:',count)
except AttributeError:
	for i in layer_274_out:
		count = 1
		for s in layer_274_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #274:',count)



layer_275 = L275()
try:
	try:
		layer_275_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_275_out = layer_275(layer_275_concat)
	except ValueError:
		layer_275_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_275_out = layer_275(layer_275_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_275 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_275_concat = strong_concat_275(new_list_of_inputs)
	layer_275_out = layer_275(layer_275_concat)
if layer_275.count_params() != 0:
	if 275 in set_weigths:
		if len(layer_275.get_weights()) > 1:
			new_w = [get_new_weigths(layer_275, set_weigths[275])]+layer_275.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_275, set_weigths[275])]
		layer_275.set_weights(new_w)


try:
	count = 1
	for s in layer_275_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #275:',count)
except AttributeError:
	for i in layer_275_out:
		count = 1
		for s in layer_275_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #275:',count)



layer_276 = L276()
try:
	try:
		layer_276_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_276_out = layer_276(layer_276_concat)
	except ValueError:
		layer_276_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_276_out = layer_276(layer_276_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_276 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_276_concat = strong_concat_276(new_list_of_inputs)
	layer_276_out = layer_276(layer_276_concat)
if layer_276.count_params() != 0:
	if 276 in set_weigths:
		if len(layer_276.get_weights()) > 1:
			new_w = [get_new_weigths(layer_276, set_weigths[276])]+layer_276.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_276, set_weigths[276])]
		layer_276.set_weights(new_w)


try:
	count = 1
	for s in layer_276_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #276:',count)
except AttributeError:
	for i in layer_276_out:
		count = 1
		for s in layer_276_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #276:',count)



layer_277 = L277()
try:
	try:
		layer_277_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_277_out = layer_277(layer_277_concat)
	except ValueError:
		layer_277_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_277_out = layer_277(layer_277_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_277 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_277_concat = strong_concat_277(new_list_of_inputs)
	layer_277_out = layer_277(layer_277_concat)
if layer_277.count_params() != 0:
	if 277 in set_weigths:
		if len(layer_277.get_weights()) > 1:
			new_w = [get_new_weigths(layer_277, set_weigths[277])]+layer_277.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_277, set_weigths[277])]
		layer_277.set_weights(new_w)


try:
	count = 1
	for s in layer_277_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #277:',count)
except AttributeError:
	for i in layer_277_out:
		count = 1
		for s in layer_277_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #277:',count)



layer_278 = L278()
try:
	try:
		layer_278_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_278_out = layer_278(layer_278_concat)
	except ValueError:
		layer_278_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_278_out = layer_278(layer_278_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_278 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_278_concat = strong_concat_278(new_list_of_inputs)
	layer_278_out = layer_278(layer_278_concat)
if layer_278.count_params() != 0:
	if 278 in set_weigths:
		if len(layer_278.get_weights()) > 1:
			new_w = [get_new_weigths(layer_278, set_weigths[278])]+layer_278.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_278, set_weigths[278])]
		layer_278.set_weights(new_w)


try:
	count = 1
	for s in layer_278_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #278:',count)
except AttributeError:
	for i in layer_278_out:
		count = 1
		for s in layer_278_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #278:',count)



layer_279 = L279()
try:
	try:
		layer_279_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_279_out = layer_279(layer_279_concat)
	except ValueError:
		layer_279_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_279_out = layer_279(layer_279_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_279 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_279_concat = strong_concat_279(new_list_of_inputs)
	layer_279_out = layer_279(layer_279_concat)
if layer_279.count_params() != 0:
	if 279 in set_weigths:
		if len(layer_279.get_weights()) > 1:
			new_w = [get_new_weigths(layer_279, set_weigths[279])]+layer_279.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_279, set_weigths[279])]
		layer_279.set_weights(new_w)


try:
	count = 1
	for s in layer_279_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #279:',count)
except AttributeError:
	for i in layer_279_out:
		count = 1
		for s in layer_279_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #279:',count)



layer_280 = L280()
try:
	try:
		layer_280_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_280_out = layer_280(layer_280_concat)
	except ValueError:
		layer_280_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_280_out = layer_280(layer_280_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_280 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_280_concat = strong_concat_280(new_list_of_inputs)
	layer_280_out = layer_280(layer_280_concat)
if layer_280.count_params() != 0:
	if 280 in set_weigths:
		if len(layer_280.get_weights()) > 1:
			new_w = [get_new_weigths(layer_280, set_weigths[280])]+layer_280.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_280, set_weigths[280])]
		layer_280.set_weights(new_w)


try:
	count = 1
	for s in layer_280_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #280:',count)
except AttributeError:
	for i in layer_280_out:
		count = 1
		for s in layer_280_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #280:',count)



layer_281 = L281()
try:
	try:
		layer_281_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_281_out = layer_281(layer_281_concat)
	except ValueError:
		layer_281_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_281_out = layer_281(layer_281_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_281 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_281_concat = strong_concat_281(new_list_of_inputs)
	layer_281_out = layer_281(layer_281_concat)
if layer_281.count_params() != 0:
	if 281 in set_weigths:
		if len(layer_281.get_weights()) > 1:
			new_w = [get_new_weigths(layer_281, set_weigths[281])]+layer_281.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_281, set_weigths[281])]
		layer_281.set_weights(new_w)


try:
	count = 1
	for s in layer_281_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #281:',count)
except AttributeError:
	for i in layer_281_out:
		count = 1
		for s in layer_281_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #281:',count)



layer_282 = L282()
try:
	try:
		layer_282_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_282_out = layer_282(layer_282_concat)
	except ValueError:
		layer_282_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_282_out = layer_282(layer_282_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_282 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_282_concat = strong_concat_282(new_list_of_inputs)
	layer_282_out = layer_282(layer_282_concat)
if layer_282.count_params() != 0:
	if 282 in set_weigths:
		if len(layer_282.get_weights()) > 1:
			new_w = [get_new_weigths(layer_282, set_weigths[282])]+layer_282.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_282, set_weigths[282])]
		layer_282.set_weights(new_w)


try:
	count = 1
	for s in layer_282_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #282:',count)
except AttributeError:
	for i in layer_282_out:
		count = 1
		for s in layer_282_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #282:',count)



layer_283 = L283()
try:
	try:
		layer_283_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_283_out = layer_283(layer_283_concat)
	except ValueError:
		layer_283_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_283_out = layer_283(layer_283_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_283 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_283_concat = strong_concat_283(new_list_of_inputs)
	layer_283_out = layer_283(layer_283_concat)
if layer_283.count_params() != 0:
	if 283 in set_weigths:
		if len(layer_283.get_weights()) > 1:
			new_w = [get_new_weigths(layer_283, set_weigths[283])]+layer_283.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_283, set_weigths[283])]
		layer_283.set_weights(new_w)


try:
	count = 1
	for s in layer_283_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #283:',count)
except AttributeError:
	for i in layer_283_out:
		count = 1
		for s in layer_283_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #283:',count)



layer_284 = L284()
try:
	try:
		layer_284_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_284_out = layer_284(layer_284_concat)
	except ValueError:
		layer_284_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_284_out = layer_284(layer_284_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_284 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_284_concat = strong_concat_284(new_list_of_inputs)
	layer_284_out = layer_284(layer_284_concat)
if layer_284.count_params() != 0:
	if 284 in set_weigths:
		if len(layer_284.get_weights()) > 1:
			new_w = [get_new_weigths(layer_284, set_weigths[284])]+layer_284.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_284, set_weigths[284])]
		layer_284.set_weights(new_w)


try:
	count = 1
	for s in layer_284_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #284:',count)
except AttributeError:
	for i in layer_284_out:
		count = 1
		for s in layer_284_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #284:',count)



layer_285 = L285()
try:
	try:
		layer_285_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_285_out = layer_285(layer_285_concat)
	except ValueError:
		layer_285_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_285_out = layer_285(layer_285_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_285 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_285_concat = strong_concat_285(new_list_of_inputs)
	layer_285_out = layer_285(layer_285_concat)
if layer_285.count_params() != 0:
	if 285 in set_weigths:
		if len(layer_285.get_weights()) > 1:
			new_w = [get_new_weigths(layer_285, set_weigths[285])]+layer_285.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_285, set_weigths[285])]
		layer_285.set_weights(new_w)


try:
	count = 1
	for s in layer_285_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #285:',count)
except AttributeError:
	for i in layer_285_out:
		count = 1
		for s in layer_285_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #285:',count)



layer_286 = L286()
try:
	try:
		layer_286_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_286_out = layer_286(layer_286_concat)
	except ValueError:
		layer_286_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_286_out = layer_286(layer_286_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_286 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_286_concat = strong_concat_286(new_list_of_inputs)
	layer_286_out = layer_286(layer_286_concat)
if layer_286.count_params() != 0:
	if 286 in set_weigths:
		if len(layer_286.get_weights()) > 1:
			new_w = [get_new_weigths(layer_286, set_weigths[286])]+layer_286.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_286, set_weigths[286])]
		layer_286.set_weights(new_w)


try:
	count = 1
	for s in layer_286_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #286:',count)
except AttributeError:
	for i in layer_286_out:
		count = 1
		for s in layer_286_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #286:',count)



layer_287 = L287()
try:
	try:
		layer_287_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_287_out = layer_287(layer_287_concat)
	except ValueError:
		layer_287_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_287_out = layer_287(layer_287_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_287 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_287_concat = strong_concat_287(new_list_of_inputs)
	layer_287_out = layer_287(layer_287_concat)
if layer_287.count_params() != 0:
	if 287 in set_weigths:
		if len(layer_287.get_weights()) > 1:
			new_w = [get_new_weigths(layer_287, set_weigths[287])]+layer_287.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_287, set_weigths[287])]
		layer_287.set_weights(new_w)


try:
	count = 1
	for s in layer_287_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #287:',count)
except AttributeError:
	for i in layer_287_out:
		count = 1
		for s in layer_287_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #287:',count)



layer_288 = L288()
try:
	try:
		layer_288_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_288_out = layer_288(layer_288_concat)
	except ValueError:
		layer_288_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_288_out = layer_288(layer_288_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_288 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_288_concat = strong_concat_288(new_list_of_inputs)
	layer_288_out = layer_288(layer_288_concat)
if layer_288.count_params() != 0:
	if 288 in set_weigths:
		if len(layer_288.get_weights()) > 1:
			new_w = [get_new_weigths(layer_288, set_weigths[288])]+layer_288.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_288, set_weigths[288])]
		layer_288.set_weights(new_w)


try:
	count = 1
	for s in layer_288_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #288:',count)
except AttributeError:
	for i in layer_288_out:
		count = 1
		for s in layer_288_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #288:',count)



layer_289 = L289()
try:
	try:
		layer_289_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_289_out = layer_289(layer_289_concat)
	except ValueError:
		layer_289_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_289_out = layer_289(layer_289_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_289 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_289_concat = strong_concat_289(new_list_of_inputs)
	layer_289_out = layer_289(layer_289_concat)
if layer_289.count_params() != 0:
	if 289 in set_weigths:
		if len(layer_289.get_weights()) > 1:
			new_w = [get_new_weigths(layer_289, set_weigths[289])]+layer_289.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_289, set_weigths[289])]
		layer_289.set_weights(new_w)


try:
	count = 1
	for s in layer_289_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #289:',count)
except AttributeError:
	for i in layer_289_out:
		count = 1
		for s in layer_289_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #289:',count)



layer_290 = L290()
try:
	try:
		layer_290_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_290_out = layer_290(layer_290_concat)
	except ValueError:
		layer_290_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_290_out = layer_290(layer_290_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_290 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_290_concat = strong_concat_290(new_list_of_inputs)
	layer_290_out = layer_290(layer_290_concat)
if layer_290.count_params() != 0:
	if 290 in set_weigths:
		if len(layer_290.get_weights()) > 1:
			new_w = [get_new_weigths(layer_290, set_weigths[290])]+layer_290.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_290, set_weigths[290])]
		layer_290.set_weights(new_w)


try:
	count = 1
	for s in layer_290_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #290:',count)
except AttributeError:
	for i in layer_290_out:
		count = 1
		for s in layer_290_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #290:',count)



layer_291 = L291()
try:
	try:
		layer_291_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_291_out = layer_291(layer_291_concat)
	except ValueError:
		layer_291_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_291_out = layer_291(layer_291_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_291 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_291_concat = strong_concat_291(new_list_of_inputs)
	layer_291_out = layer_291(layer_291_concat)
if layer_291.count_params() != 0:
	if 291 in set_weigths:
		if len(layer_291.get_weights()) > 1:
			new_w = [get_new_weigths(layer_291, set_weigths[291])]+layer_291.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_291, set_weigths[291])]
		layer_291.set_weights(new_w)


try:
	count = 1
	for s in layer_291_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #291:',count)
except AttributeError:
	for i in layer_291_out:
		count = 1
		for s in layer_291_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #291:',count)



layer_292 = L292()
try:
	try:
		layer_292_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_292_out = layer_292(layer_292_concat)
	except ValueError:
		layer_292_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_292_out = layer_292(layer_292_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_292 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_292_concat = strong_concat_292(new_list_of_inputs)
	layer_292_out = layer_292(layer_292_concat)
if layer_292.count_params() != 0:
	if 292 in set_weigths:
		if len(layer_292.get_weights()) > 1:
			new_w = [get_new_weigths(layer_292, set_weigths[292])]+layer_292.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_292, set_weigths[292])]
		layer_292.set_weights(new_w)


try:
	count = 1
	for s in layer_292_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #292:',count)
except AttributeError:
	for i in layer_292_out:
		count = 1
		for s in layer_292_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #292:',count)



layer_293 = L293()
try:
	try:
		layer_293_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_293_out = layer_293(layer_293_concat)
	except ValueError:
		layer_293_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_293_out = layer_293(layer_293_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_293 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_293_concat = strong_concat_293(new_list_of_inputs)
	layer_293_out = layer_293(layer_293_concat)
if layer_293.count_params() != 0:
	if 293 in set_weigths:
		if len(layer_293.get_weights()) > 1:
			new_w = [get_new_weigths(layer_293, set_weigths[293])]+layer_293.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_293, set_weigths[293])]
		layer_293.set_weights(new_w)


try:
	count = 1
	for s in layer_293_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #293:',count)
except AttributeError:
	for i in layer_293_out:
		count = 1
		for s in layer_293_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #293:',count)



layer_294 = L294()
try:
	try:
		layer_294_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_294_out = layer_294(layer_294_concat)
	except ValueError:
		layer_294_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_294_out = layer_294(layer_294_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_294 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_294_concat = strong_concat_294(new_list_of_inputs)
	layer_294_out = layer_294(layer_294_concat)
if layer_294.count_params() != 0:
	if 294 in set_weigths:
		if len(layer_294.get_weights()) > 1:
			new_w = [get_new_weigths(layer_294, set_weigths[294])]+layer_294.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_294, set_weigths[294])]
		layer_294.set_weights(new_w)


try:
	count = 1
	for s in layer_294_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #294:',count)
except AttributeError:
	for i in layer_294_out:
		count = 1
		for s in layer_294_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #294:',count)



layer_295 = L295()
try:
	try:
		layer_295_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_295_out = layer_295(layer_295_concat)
	except ValueError:
		layer_295_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_295_out = layer_295(layer_295_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_295 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_295_concat = strong_concat_295(new_list_of_inputs)
	layer_295_out = layer_295(layer_295_concat)
if layer_295.count_params() != 0:
	if 295 in set_weigths:
		if len(layer_295.get_weights()) > 1:
			new_w = [get_new_weigths(layer_295, set_weigths[295])]+layer_295.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_295, set_weigths[295])]
		layer_295.set_weights(new_w)


try:
	count = 1
	for s in layer_295_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #295:',count)
except AttributeError:
	for i in layer_295_out:
		count = 1
		for s in layer_295_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #295:',count)



layer_296 = L296()
try:
	try:
		layer_296_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_296_out = layer_296(layer_296_concat)
	except ValueError:
		layer_296_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_296_out = layer_296(layer_296_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_296 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_296_concat = strong_concat_296(new_list_of_inputs)
	layer_296_out = layer_296(layer_296_concat)
if layer_296.count_params() != 0:
	if 296 in set_weigths:
		if len(layer_296.get_weights()) > 1:
			new_w = [get_new_weigths(layer_296, set_weigths[296])]+layer_296.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_296, set_weigths[296])]
		layer_296.set_weights(new_w)


try:
	count = 1
	for s in layer_296_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #296:',count)
except AttributeError:
	for i in layer_296_out:
		count = 1
		for s in layer_296_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #296:',count)



layer_297 = L297()
try:
	try:
		layer_297_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_297_out = layer_297(layer_297_concat)
	except ValueError:
		layer_297_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_297_out = layer_297(layer_297_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_297 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_297_concat = strong_concat_297(new_list_of_inputs)
	layer_297_out = layer_297(layer_297_concat)
if layer_297.count_params() != 0:
	if 297 in set_weigths:
		if len(layer_297.get_weights()) > 1:
			new_w = [get_new_weigths(layer_297, set_weigths[297])]+layer_297.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_297, set_weigths[297])]
		layer_297.set_weights(new_w)


try:
	count = 1
	for s in layer_297_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #297:',count)
except AttributeError:
	for i in layer_297_out:
		count = 1
		for s in layer_297_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #297:',count)



layer_298 = L298()
try:
	try:
		layer_298_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_298_out = layer_298(layer_298_concat)
	except ValueError:
		layer_298_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_298_out = layer_298(layer_298_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_298 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_298_concat = strong_concat_298(new_list_of_inputs)
	layer_298_out = layer_298(layer_298_concat)
if layer_298.count_params() != 0:
	if 298 in set_weigths:
		if len(layer_298.get_weights()) > 1:
			new_w = [get_new_weigths(layer_298, set_weigths[298])]+layer_298.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_298, set_weigths[298])]
		layer_298.set_weights(new_w)


try:
	count = 1
	for s in layer_298_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #298:',count)
except AttributeError:
	for i in layer_298_out:
		count = 1
		for s in layer_298_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #298:',count)



layer_299 = L299()
try:
	try:
		layer_299_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_299_out = layer_299(layer_299_concat)
	except ValueError:
		layer_299_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_299_out = layer_299(layer_299_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_299 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_299_concat = strong_concat_299(new_list_of_inputs)
	layer_299_out = layer_299(layer_299_concat)
if layer_299.count_params() != 0:
	if 299 in set_weigths:
		if len(layer_299.get_weights()) > 1:
			new_w = [get_new_weigths(layer_299, set_weigths[299])]+layer_299.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_299, set_weigths[299])]
		layer_299.set_weights(new_w)


try:
	count = 1
	for s in layer_299_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #299:',count)
except AttributeError:
	for i in layer_299_out:
		count = 1
		for s in layer_299_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #299:',count)



layer_300 = L300()
try:
	try:
		layer_300_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_300_out = layer_300(layer_300_concat)
	except ValueError:
		layer_300_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_300_out = layer_300(layer_300_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_300 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_300_concat = strong_concat_300(new_list_of_inputs)
	layer_300_out = layer_300(layer_300_concat)
if layer_300.count_params() != 0:
	if 300 in set_weigths:
		if len(layer_300.get_weights()) > 1:
			new_w = [get_new_weigths(layer_300, set_weigths[300])]+layer_300.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_300, set_weigths[300])]
		layer_300.set_weights(new_w)


try:
	count = 1
	for s in layer_300_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #300:',count)
except AttributeError:
	for i in layer_300_out:
		count = 1
		for s in layer_300_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #300:',count)



layer_301 = L301()
try:
	try:
		layer_301_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_301_out = layer_301(layer_301_concat)
	except ValueError:
		layer_301_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_301_out = layer_301(layer_301_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_301 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_301_concat = strong_concat_301(new_list_of_inputs)
	layer_301_out = layer_301(layer_301_concat)
if layer_301.count_params() != 0:
	if 301 in set_weigths:
		if len(layer_301.get_weights()) > 1:
			new_w = [get_new_weigths(layer_301, set_weigths[301])]+layer_301.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_301, set_weigths[301])]
		layer_301.set_weights(new_w)


try:
	count = 1
	for s in layer_301_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #301:',count)
except AttributeError:
	for i in layer_301_out:
		count = 1
		for s in layer_301_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #301:',count)



layer_302 = L302()
try:
	try:
		layer_302_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_302_out = layer_302(layer_302_concat)
	except ValueError:
		layer_302_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_302_out = layer_302(layer_302_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_302 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_302_concat = strong_concat_302(new_list_of_inputs)
	layer_302_out = layer_302(layer_302_concat)
if layer_302.count_params() != 0:
	if 302 in set_weigths:
		if len(layer_302.get_weights()) > 1:
			new_w = [get_new_weigths(layer_302, set_weigths[302])]+layer_302.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_302, set_weigths[302])]
		layer_302.set_weights(new_w)


try:
	count = 1
	for s in layer_302_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #302:',count)
except AttributeError:
	for i in layer_302_out:
		count = 1
		for s in layer_302_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #302:',count)



layer_303 = L303()
try:
	try:
		layer_303_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_303_out = layer_303(layer_303_concat)
	except ValueError:
		layer_303_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_303_out = layer_303(layer_303_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_303 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_303_concat = strong_concat_303(new_list_of_inputs)
	layer_303_out = layer_303(layer_303_concat)
if layer_303.count_params() != 0:
	if 303 in set_weigths:
		if len(layer_303.get_weights()) > 1:
			new_w = [get_new_weigths(layer_303, set_weigths[303])]+layer_303.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_303, set_weigths[303])]
		layer_303.set_weights(new_w)


try:
	count = 1
	for s in layer_303_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #303:',count)
except AttributeError:
	for i in layer_303_out:
		count = 1
		for s in layer_303_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #303:',count)



layer_304 = L304()
try:
	try:
		layer_304_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_304_out = layer_304(layer_304_concat)
	except ValueError:
		layer_304_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_304_out = layer_304(layer_304_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_304 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_304_concat = strong_concat_304(new_list_of_inputs)
	layer_304_out = layer_304(layer_304_concat)
if layer_304.count_params() != 0:
	if 304 in set_weigths:
		if len(layer_304.get_weights()) > 1:
			new_w = [get_new_weigths(layer_304, set_weigths[304])]+layer_304.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_304, set_weigths[304])]
		layer_304.set_weights(new_w)


try:
	count = 1
	for s in layer_304_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #304:',count)
except AttributeError:
	for i in layer_304_out:
		count = 1
		for s in layer_304_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #304:',count)



layer_305 = L305()
try:
	try:
		layer_305_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_305_out = layer_305(layer_305_concat)
	except ValueError:
		layer_305_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_305_out = layer_305(layer_305_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_305 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_305_concat = strong_concat_305(new_list_of_inputs)
	layer_305_out = layer_305(layer_305_concat)
if layer_305.count_params() != 0:
	if 305 in set_weigths:
		if len(layer_305.get_weights()) > 1:
			new_w = [get_new_weigths(layer_305, set_weigths[305])]+layer_305.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_305, set_weigths[305])]
		layer_305.set_weights(new_w)


try:
	count = 1
	for s in layer_305_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #305:',count)
except AttributeError:
	for i in layer_305_out:
		count = 1
		for s in layer_305_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #305:',count)



layer_306 = L306()
try:
	try:
		layer_306_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_306_out = layer_306(layer_306_concat)
	except ValueError:
		layer_306_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_306_out = layer_306(layer_306_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_306 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_306_concat = strong_concat_306(new_list_of_inputs)
	layer_306_out = layer_306(layer_306_concat)
if layer_306.count_params() != 0:
	if 306 in set_weigths:
		if len(layer_306.get_weights()) > 1:
			new_w = [get_new_weigths(layer_306, set_weigths[306])]+layer_306.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_306, set_weigths[306])]
		layer_306.set_weights(new_w)


try:
	count = 1
	for s in layer_306_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #306:',count)
except AttributeError:
	for i in layer_306_out:
		count = 1
		for s in layer_306_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #306:',count)



layer_307 = L307()
try:
	try:
		layer_307_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_307_out = layer_307(layer_307_concat)
	except ValueError:
		layer_307_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_307_out = layer_307(layer_307_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_307 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_307_concat = strong_concat_307(new_list_of_inputs)
	layer_307_out = layer_307(layer_307_concat)
if layer_307.count_params() != 0:
	if 307 in set_weigths:
		if len(layer_307.get_weights()) > 1:
			new_w = [get_new_weigths(layer_307, set_weigths[307])]+layer_307.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_307, set_weigths[307])]
		layer_307.set_weights(new_w)


try:
	count = 1
	for s in layer_307_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #307:',count)
except AttributeError:
	for i in layer_307_out:
		count = 1
		for s in layer_307_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #307:',count)



layer_308 = L308()
try:
	try:
		layer_308_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_308_out = layer_308(layer_308_concat)
	except ValueError:
		layer_308_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_308_out = layer_308(layer_308_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_308 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_308_concat = strong_concat_308(new_list_of_inputs)
	layer_308_out = layer_308(layer_308_concat)
if layer_308.count_params() != 0:
	if 308 in set_weigths:
		if len(layer_308.get_weights()) > 1:
			new_w = [get_new_weigths(layer_308, set_weigths[308])]+layer_308.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_308, set_weigths[308])]
		layer_308.set_weights(new_w)


try:
	count = 1
	for s in layer_308_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #308:',count)
except AttributeError:
	for i in layer_308_out:
		count = 1
		for s in layer_308_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #308:',count)



layer_309 = L309()
try:
	try:
		layer_309_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_309_out = layer_309(layer_309_concat)
	except ValueError:
		layer_309_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_309_out = layer_309(layer_309_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_309 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_309_concat = strong_concat_309(new_list_of_inputs)
	layer_309_out = layer_309(layer_309_concat)
if layer_309.count_params() != 0:
	if 309 in set_weigths:
		if len(layer_309.get_weights()) > 1:
			new_w = [get_new_weigths(layer_309, set_weigths[309])]+layer_309.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_309, set_weigths[309])]
		layer_309.set_weights(new_w)


try:
	count = 1
	for s in layer_309_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #309:',count)
except AttributeError:
	for i in layer_309_out:
		count = 1
		for s in layer_309_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #309:',count)



layer_310 = L310()
try:
	try:
		layer_310_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_310_out = layer_310(layer_310_concat)
	except ValueError:
		layer_310_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_310_out = layer_310(layer_310_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_310 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_310_concat = strong_concat_310(new_list_of_inputs)
	layer_310_out = layer_310(layer_310_concat)
if layer_310.count_params() != 0:
	if 310 in set_weigths:
		if len(layer_310.get_weights()) > 1:
			new_w = [get_new_weigths(layer_310, set_weigths[310])]+layer_310.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_310, set_weigths[310])]
		layer_310.set_weights(new_w)


try:
	count = 1
	for s in layer_310_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #310:',count)
except AttributeError:
	for i in layer_310_out:
		count = 1
		for s in layer_310_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #310:',count)



layer_311 = L311()
try:
	try:
		layer_311_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_311_out = layer_311(layer_311_concat)
	except ValueError:
		layer_311_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_311_out = layer_311(layer_311_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_311 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_311_concat = strong_concat_311(new_list_of_inputs)
	layer_311_out = layer_311(layer_311_concat)
if layer_311.count_params() != 0:
	if 311 in set_weigths:
		if len(layer_311.get_weights()) > 1:
			new_w = [get_new_weigths(layer_311, set_weigths[311])]+layer_311.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_311, set_weigths[311])]
		layer_311.set_weights(new_w)


try:
	count = 1
	for s in layer_311_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #311:',count)
except AttributeError:
	for i in layer_311_out:
		count = 1
		for s in layer_311_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #311:',count)



layer_312 = L312()
try:
	try:
		layer_312_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_312_out = layer_312(layer_312_concat)
	except ValueError:
		layer_312_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_312_out = layer_312(layer_312_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_312 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_312_concat = strong_concat_312(new_list_of_inputs)
	layer_312_out = layer_312(layer_312_concat)
if layer_312.count_params() != 0:
	if 312 in set_weigths:
		if len(layer_312.get_weights()) > 1:
			new_w = [get_new_weigths(layer_312, set_weigths[312])]+layer_312.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_312, set_weigths[312])]
		layer_312.set_weights(new_w)


try:
	count = 1
	for s in layer_312_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #312:',count)
except AttributeError:
	for i in layer_312_out:
		count = 1
		for s in layer_312_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #312:',count)



layer_313 = L313()
try:
	try:
		layer_313_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_313_out = layer_313(layer_313_concat)
	except ValueError:
		layer_313_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_313_out = layer_313(layer_313_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_313 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_313_concat = strong_concat_313(new_list_of_inputs)
	layer_313_out = layer_313(layer_313_concat)
if layer_313.count_params() != 0:
	if 313 in set_weigths:
		if len(layer_313.get_weights()) > 1:
			new_w = [get_new_weigths(layer_313, set_weigths[313])]+layer_313.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_313, set_weigths[313])]
		layer_313.set_weights(new_w)


try:
	count = 1
	for s in layer_313_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #313:',count)
except AttributeError:
	for i in layer_313_out:
		count = 1
		for s in layer_313_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #313:',count)



layer_314 = L314()
try:
	try:
		layer_314_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_314_out = layer_314(layer_314_concat)
	except ValueError:
		layer_314_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_314_out = layer_314(layer_314_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_314 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_314_concat = strong_concat_314(new_list_of_inputs)
	layer_314_out = layer_314(layer_314_concat)
if layer_314.count_params() != 0:
	if 314 in set_weigths:
		if len(layer_314.get_weights()) > 1:
			new_w = [get_new_weigths(layer_314, set_weigths[314])]+layer_314.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_314, set_weigths[314])]
		layer_314.set_weights(new_w)


try:
	count = 1
	for s in layer_314_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #314:',count)
except AttributeError:
	for i in layer_314_out:
		count = 1
		for s in layer_314_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #314:',count)



layer_315 = L315()
try:
	try:
		layer_315_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_315_out = layer_315(layer_315_concat)
	except ValueError:
		layer_315_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_315_out = layer_315(layer_315_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_315 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_315_concat = strong_concat_315(new_list_of_inputs)
	layer_315_out = layer_315(layer_315_concat)
if layer_315.count_params() != 0:
	if 315 in set_weigths:
		if len(layer_315.get_weights()) > 1:
			new_w = [get_new_weigths(layer_315, set_weigths[315])]+layer_315.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_315, set_weigths[315])]
		layer_315.set_weights(new_w)


try:
	count = 1
	for s in layer_315_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #315:',count)
except AttributeError:
	for i in layer_315_out:
		count = 1
		for s in layer_315_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #315:',count)



layer_316 = L316()
try:
	try:
		layer_316_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_316_out = layer_316(layer_316_concat)
	except ValueError:
		layer_316_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_316_out = layer_316(layer_316_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_316 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_316_concat = strong_concat_316(new_list_of_inputs)
	layer_316_out = layer_316(layer_316_concat)
if layer_316.count_params() != 0:
	if 316 in set_weigths:
		if len(layer_316.get_weights()) > 1:
			new_w = [get_new_weigths(layer_316, set_weigths[316])]+layer_316.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_316, set_weigths[316])]
		layer_316.set_weights(new_w)


try:
	count = 1
	for s in layer_316_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #316:',count)
except AttributeError:
	for i in layer_316_out:
		count = 1
		for s in layer_316_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #316:',count)



layer_317 = L317()
try:
	try:
		layer_317_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_317_out = layer_317(layer_317_concat)
	except ValueError:
		layer_317_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_317_out = layer_317(layer_317_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_317 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_317_concat = strong_concat_317(new_list_of_inputs)
	layer_317_out = layer_317(layer_317_concat)
if layer_317.count_params() != 0:
	if 317 in set_weigths:
		if len(layer_317.get_weights()) > 1:
			new_w = [get_new_weigths(layer_317, set_weigths[317])]+layer_317.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_317, set_weigths[317])]
		layer_317.set_weights(new_w)


try:
	count = 1
	for s in layer_317_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #317:',count)
except AttributeError:
	for i in layer_317_out:
		count = 1
		for s in layer_317_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #317:',count)



layer_318 = L318()
try:
	try:
		layer_318_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_318_out = layer_318(layer_318_concat)
	except ValueError:
		layer_318_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_318_out = layer_318(layer_318_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_318 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_318_concat = strong_concat_318(new_list_of_inputs)
	layer_318_out = layer_318(layer_318_concat)
if layer_318.count_params() != 0:
	if 318 in set_weigths:
		if len(layer_318.get_weights()) > 1:
			new_w = [get_new_weigths(layer_318, set_weigths[318])]+layer_318.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_318, set_weigths[318])]
		layer_318.set_weights(new_w)


try:
	count = 1
	for s in layer_318_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #318:',count)
except AttributeError:
	for i in layer_318_out:
		count = 1
		for s in layer_318_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #318:',count)



layer_319 = L319()
try:
	try:
		layer_319_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_319_out = layer_319(layer_319_concat)
	except ValueError:
		layer_319_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_319_out = layer_319(layer_319_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_319 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_319_concat = strong_concat_319(new_list_of_inputs)
	layer_319_out = layer_319(layer_319_concat)
if layer_319.count_params() != 0:
	if 319 in set_weigths:
		if len(layer_319.get_weights()) > 1:
			new_w = [get_new_weigths(layer_319, set_weigths[319])]+layer_319.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_319, set_weigths[319])]
		layer_319.set_weights(new_w)


try:
	count = 1
	for s in layer_319_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #319:',count)
except AttributeError:
	for i in layer_319_out:
		count = 1
		for s in layer_319_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #319:',count)



layer_320 = L320()
try:
	try:
		layer_320_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_320_out = layer_320(layer_320_concat)
	except ValueError:
		layer_320_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_320_out = layer_320(layer_320_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_320 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_320_concat = strong_concat_320(new_list_of_inputs)
	layer_320_out = layer_320(layer_320_concat)
if layer_320.count_params() != 0:
	if 320 in set_weigths:
		if len(layer_320.get_weights()) > 1:
			new_w = [get_new_weigths(layer_320, set_weigths[320])]+layer_320.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_320, set_weigths[320])]
		layer_320.set_weights(new_w)


try:
	count = 1
	for s in layer_320_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #320:',count)
except AttributeError:
	for i in layer_320_out:
		count = 1
		for s in layer_320_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #320:',count)



layer_321 = L321()
try:
	try:
		layer_321_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_321_out = layer_321(layer_321_concat)
	except ValueError:
		layer_321_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_321_out = layer_321(layer_321_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_321 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_321_concat = strong_concat_321(new_list_of_inputs)
	layer_321_out = layer_321(layer_321_concat)
if layer_321.count_params() != 0:
	if 321 in set_weigths:
		if len(layer_321.get_weights()) > 1:
			new_w = [get_new_weigths(layer_321, set_weigths[321])]+layer_321.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_321, set_weigths[321])]
		layer_321.set_weights(new_w)


try:
	count = 1
	for s in layer_321_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #321:',count)
except AttributeError:
	for i in layer_321_out:
		count = 1
		for s in layer_321_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #321:',count)



layer_322 = L322()
try:
	try:
		layer_322_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_322_out = layer_322(layer_322_concat)
	except ValueError:
		layer_322_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_322_out = layer_322(layer_322_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_322 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_322_concat = strong_concat_322(new_list_of_inputs)
	layer_322_out = layer_322(layer_322_concat)
if layer_322.count_params() != 0:
	if 322 in set_weigths:
		if len(layer_322.get_weights()) > 1:
			new_w = [get_new_weigths(layer_322, set_weigths[322])]+layer_322.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_322, set_weigths[322])]
		layer_322.set_weights(new_w)


try:
	count = 1
	for s in layer_322_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #322:',count)
except AttributeError:
	for i in layer_322_out:
		count = 1
		for s in layer_322_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #322:',count)



layer_323 = L323()
try:
	try:
		layer_323_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_323_out = layer_323(layer_323_concat)
	except ValueError:
		layer_323_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_323_out = layer_323(layer_323_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_323 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_323_concat = strong_concat_323(new_list_of_inputs)
	layer_323_out = layer_323(layer_323_concat)
if layer_323.count_params() != 0:
	if 323 in set_weigths:
		if len(layer_323.get_weights()) > 1:
			new_w = [get_new_weigths(layer_323, set_weigths[323])]+layer_323.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_323, set_weigths[323])]
		layer_323.set_weights(new_w)


try:
	count = 1
	for s in layer_323_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #323:',count)
except AttributeError:
	for i in layer_323_out:
		count = 1
		for s in layer_323_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #323:',count)



layer_324 = L324()
try:
	try:
		layer_324_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_324_out = layer_324(layer_324_concat)
	except ValueError:
		layer_324_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_324_out = layer_324(layer_324_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_324 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_324_concat = strong_concat_324(new_list_of_inputs)
	layer_324_out = layer_324(layer_324_concat)
if layer_324.count_params() != 0:
	if 324 in set_weigths:
		if len(layer_324.get_weights()) > 1:
			new_w = [get_new_weigths(layer_324, set_weigths[324])]+layer_324.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_324, set_weigths[324])]
		layer_324.set_weights(new_w)


try:
	count = 1
	for s in layer_324_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #324:',count)
except AttributeError:
	for i in layer_324_out:
		count = 1
		for s in layer_324_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #324:',count)



layer_325 = L325()
try:
	try:
		layer_325_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_325_out = layer_325(layer_325_concat)
	except ValueError:
		layer_325_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_325_out = layer_325(layer_325_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_325 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_325_concat = strong_concat_325(new_list_of_inputs)
	layer_325_out = layer_325(layer_325_concat)
if layer_325.count_params() != 0:
	if 325 in set_weigths:
		if len(layer_325.get_weights()) > 1:
			new_w = [get_new_weigths(layer_325, set_weigths[325])]+layer_325.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_325, set_weigths[325])]
		layer_325.set_weights(new_w)


try:
	count = 1
	for s in layer_325_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #325:',count)
except AttributeError:
	for i in layer_325_out:
		count = 1
		for s in layer_325_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #325:',count)



layer_326 = L326()
try:
	try:
		layer_326_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_326_out = layer_326(layer_326_concat)
	except ValueError:
		layer_326_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_326_out = layer_326(layer_326_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_326 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_326_concat = strong_concat_326(new_list_of_inputs)
	layer_326_out = layer_326(layer_326_concat)
if layer_326.count_params() != 0:
	if 326 in set_weigths:
		if len(layer_326.get_weights()) > 1:
			new_w = [get_new_weigths(layer_326, set_weigths[326])]+layer_326.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_326, set_weigths[326])]
		layer_326.set_weights(new_w)


try:
	count = 1
	for s in layer_326_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #326:',count)
except AttributeError:
	for i in layer_326_out:
		count = 1
		for s in layer_326_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #326:',count)



layer_327 = L327()
try:
	try:
		layer_327_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_327_out = layer_327(layer_327_concat)
	except ValueError:
		layer_327_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_327_out = layer_327(layer_327_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_327 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_327_concat = strong_concat_327(new_list_of_inputs)
	layer_327_out = layer_327(layer_327_concat)
if layer_327.count_params() != 0:
	if 327 in set_weigths:
		if len(layer_327.get_weights()) > 1:
			new_w = [get_new_weigths(layer_327, set_weigths[327])]+layer_327.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_327, set_weigths[327])]
		layer_327.set_weights(new_w)


try:
	count = 1
	for s in layer_327_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #327:',count)
except AttributeError:
	for i in layer_327_out:
		count = 1
		for s in layer_327_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #327:',count)



layer_328 = L328()
try:
	try:
		layer_328_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_328_out = layer_328(layer_328_concat)
	except ValueError:
		layer_328_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_328_out = layer_328(layer_328_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_328 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_328_concat = strong_concat_328(new_list_of_inputs)
	layer_328_out = layer_328(layer_328_concat)
if layer_328.count_params() != 0:
	if 328 in set_weigths:
		if len(layer_328.get_weights()) > 1:
			new_w = [get_new_weigths(layer_328, set_weigths[328])]+layer_328.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_328, set_weigths[328])]
		layer_328.set_weights(new_w)


try:
	count = 1
	for s in layer_328_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #328:',count)
except AttributeError:
	for i in layer_328_out:
		count = 1
		for s in layer_328_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #328:',count)



layer_329 = L329()
try:
	try:
		layer_329_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_329_out = layer_329(layer_329_concat)
	except ValueError:
		layer_329_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_329_out = layer_329(layer_329_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_329 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_329_concat = strong_concat_329(new_list_of_inputs)
	layer_329_out = layer_329(layer_329_concat)
if layer_329.count_params() != 0:
	if 329 in set_weigths:
		if len(layer_329.get_weights()) > 1:
			new_w = [get_new_weigths(layer_329, set_weigths[329])]+layer_329.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_329, set_weigths[329])]
		layer_329.set_weights(new_w)


try:
	count = 1
	for s in layer_329_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #329:',count)
except AttributeError:
	for i in layer_329_out:
		count = 1
		for s in layer_329_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #329:',count)



layer_330 = L330()
try:
	try:
		layer_330_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_330_out = layer_330(layer_330_concat)
	except ValueError:
		layer_330_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_330_out = layer_330(layer_330_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_330 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_330_concat = strong_concat_330(new_list_of_inputs)
	layer_330_out = layer_330(layer_330_concat)
if layer_330.count_params() != 0:
	if 330 in set_weigths:
		if len(layer_330.get_weights()) > 1:
			new_w = [get_new_weigths(layer_330, set_weigths[330])]+layer_330.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_330, set_weigths[330])]
		layer_330.set_weights(new_w)


try:
	count = 1
	for s in layer_330_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #330:',count)
except AttributeError:
	for i in layer_330_out:
		count = 1
		for s in layer_330_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #330:',count)



layer_331 = L331()
try:
	try:
		layer_331_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_331_out = layer_331(layer_331_concat)
	except ValueError:
		layer_331_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_331_out = layer_331(layer_331_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_331 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_331_concat = strong_concat_331(new_list_of_inputs)
	layer_331_out = layer_331(layer_331_concat)
if layer_331.count_params() != 0:
	if 331 in set_weigths:
		if len(layer_331.get_weights()) > 1:
			new_w = [get_new_weigths(layer_331, set_weigths[331])]+layer_331.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_331, set_weigths[331])]
		layer_331.set_weights(new_w)


try:
	count = 1
	for s in layer_331_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #331:',count)
except AttributeError:
	for i in layer_331_out:
		count = 1
		for s in layer_331_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #331:',count)



layer_332 = L332()
try:
	try:
		layer_332_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_332_out = layer_332(layer_332_concat)
	except ValueError:
		layer_332_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_332_out = layer_332(layer_332_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_332 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_332_concat = strong_concat_332(new_list_of_inputs)
	layer_332_out = layer_332(layer_332_concat)
if layer_332.count_params() != 0:
	if 332 in set_weigths:
		if len(layer_332.get_weights()) > 1:
			new_w = [get_new_weigths(layer_332, set_weigths[332])]+layer_332.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_332, set_weigths[332])]
		layer_332.set_weights(new_w)


try:
	count = 1
	for s in layer_332_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #332:',count)
except AttributeError:
	for i in layer_332_out:
		count = 1
		for s in layer_332_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #332:',count)



layer_333 = L333()
try:
	try:
		layer_333_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_333_out = layer_333(layer_333_concat)
	except ValueError:
		layer_333_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_333_out = layer_333(layer_333_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_333 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_333_concat = strong_concat_333(new_list_of_inputs)
	layer_333_out = layer_333(layer_333_concat)
if layer_333.count_params() != 0:
	if 333 in set_weigths:
		if len(layer_333.get_weights()) > 1:
			new_w = [get_new_weigths(layer_333, set_weigths[333])]+layer_333.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_333, set_weigths[333])]
		layer_333.set_weights(new_w)


try:
	count = 1
	for s in layer_333_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #333:',count)
except AttributeError:
	for i in layer_333_out:
		count = 1
		for s in layer_333_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #333:',count)



layer_334 = L334()
try:
	try:
		layer_334_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_334_out = layer_334(layer_334_concat)
	except ValueError:
		layer_334_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_334_out = layer_334(layer_334_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_334 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_334_concat = strong_concat_334(new_list_of_inputs)
	layer_334_out = layer_334(layer_334_concat)
if layer_334.count_params() != 0:
	if 334 in set_weigths:
		if len(layer_334.get_weights()) > 1:
			new_w = [get_new_weigths(layer_334, set_weigths[334])]+layer_334.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_334, set_weigths[334])]
		layer_334.set_weights(new_w)


try:
	count = 1
	for s in layer_334_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #334:',count)
except AttributeError:
	for i in layer_334_out:
		count = 1
		for s in layer_334_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #334:',count)



layer_335 = L335()
try:
	try:
		layer_335_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_335_out = layer_335(layer_335_concat)
	except ValueError:
		layer_335_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_335_out = layer_335(layer_335_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_335 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_335_concat = strong_concat_335(new_list_of_inputs)
	layer_335_out = layer_335(layer_335_concat)
if layer_335.count_params() != 0:
	if 335 in set_weigths:
		if len(layer_335.get_weights()) > 1:
			new_w = [get_new_weigths(layer_335, set_weigths[335])]+layer_335.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_335, set_weigths[335])]
		layer_335.set_weights(new_w)


try:
	count = 1
	for s in layer_335_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #335:',count)
except AttributeError:
	for i in layer_335_out:
		count = 1
		for s in layer_335_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #335:',count)



layer_336 = L336()
try:
	try:
		layer_336_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_336_out = layer_336(layer_336_concat)
	except ValueError:
		layer_336_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_336_out = layer_336(layer_336_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_336 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_336_concat = strong_concat_336(new_list_of_inputs)
	layer_336_out = layer_336(layer_336_concat)
if layer_336.count_params() != 0:
	if 336 in set_weigths:
		if len(layer_336.get_weights()) > 1:
			new_w = [get_new_weigths(layer_336, set_weigths[336])]+layer_336.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_336, set_weigths[336])]
		layer_336.set_weights(new_w)


try:
	count = 1
	for s in layer_336_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #336:',count)
except AttributeError:
	for i in layer_336_out:
		count = 1
		for s in layer_336_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #336:',count)



layer_337 = L337()
try:
	try:
		layer_337_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_337_out = layer_337(layer_337_concat)
	except ValueError:
		layer_337_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_337_out = layer_337(layer_337_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_337 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_337_concat = strong_concat_337(new_list_of_inputs)
	layer_337_out = layer_337(layer_337_concat)
if layer_337.count_params() != 0:
	if 337 in set_weigths:
		if len(layer_337.get_weights()) > 1:
			new_w = [get_new_weigths(layer_337, set_weigths[337])]+layer_337.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_337, set_weigths[337])]
		layer_337.set_weights(new_w)


try:
	count = 1
	for s in layer_337_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #337:',count)
except AttributeError:
	for i in layer_337_out:
		count = 1
		for s in layer_337_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #337:',count)



layer_338 = L338()
try:
	try:
		layer_338_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_338_out = layer_338(layer_338_concat)
	except ValueError:
		layer_338_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_338_out = layer_338(layer_338_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_338 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_338_concat = strong_concat_338(new_list_of_inputs)
	layer_338_out = layer_338(layer_338_concat)
if layer_338.count_params() != 0:
	if 338 in set_weigths:
		if len(layer_338.get_weights()) > 1:
			new_w = [get_new_weigths(layer_338, set_weigths[338])]+layer_338.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_338, set_weigths[338])]
		layer_338.set_weights(new_w)


try:
	count = 1
	for s in layer_338_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #338:',count)
except AttributeError:
	for i in layer_338_out:
		count = 1
		for s in layer_338_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #338:',count)



layer_339 = L339()
try:
	try:
		layer_339_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_339_out = layer_339(layer_339_concat)
	except ValueError:
		layer_339_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_339_out = layer_339(layer_339_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_339 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_339_concat = strong_concat_339(new_list_of_inputs)
	layer_339_out = layer_339(layer_339_concat)
if layer_339.count_params() != 0:
	if 339 in set_weigths:
		if len(layer_339.get_weights()) > 1:
			new_w = [get_new_weigths(layer_339, set_weigths[339])]+layer_339.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_339, set_weigths[339])]
		layer_339.set_weights(new_w)


try:
	count = 1
	for s in layer_339_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #339:',count)
except AttributeError:
	for i in layer_339_out:
		count = 1
		for s in layer_339_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #339:',count)



layer_340 = L340()
try:
	try:
		layer_340_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_340_out = layer_340(layer_340_concat)
	except ValueError:
		layer_340_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_340_out = layer_340(layer_340_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_340 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_340_concat = strong_concat_340(new_list_of_inputs)
	layer_340_out = layer_340(layer_340_concat)
if layer_340.count_params() != 0:
	if 340 in set_weigths:
		if len(layer_340.get_weights()) > 1:
			new_w = [get_new_weigths(layer_340, set_weigths[340])]+layer_340.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_340, set_weigths[340])]
		layer_340.set_weights(new_w)


try:
	count = 1
	for s in layer_340_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #340:',count)
except AttributeError:
	for i in layer_340_out:
		count = 1
		for s in layer_340_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #340:',count)



layer_341 = L341()
try:
	try:
		layer_341_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_341_out = layer_341(layer_341_concat)
	except ValueError:
		layer_341_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_341_out = layer_341(layer_341_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_341 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_341_concat = strong_concat_341(new_list_of_inputs)
	layer_341_out = layer_341(layer_341_concat)
if layer_341.count_params() != 0:
	if 341 in set_weigths:
		if len(layer_341.get_weights()) > 1:
			new_w = [get_new_weigths(layer_341, set_weigths[341])]+layer_341.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_341, set_weigths[341])]
		layer_341.set_weights(new_w)


try:
	count = 1
	for s in layer_341_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #341:',count)
except AttributeError:
	for i in layer_341_out:
		count = 1
		for s in layer_341_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #341:',count)



layer_342 = L342()
try:
	try:
		layer_342_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_342_out = layer_342(layer_342_concat)
	except ValueError:
		layer_342_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_342_out = layer_342(layer_342_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_342 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_342_concat = strong_concat_342(new_list_of_inputs)
	layer_342_out = layer_342(layer_342_concat)
if layer_342.count_params() != 0:
	if 342 in set_weigths:
		if len(layer_342.get_weights()) > 1:
			new_w = [get_new_weigths(layer_342, set_weigths[342])]+layer_342.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_342, set_weigths[342])]
		layer_342.set_weights(new_w)


try:
	count = 1
	for s in layer_342_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #342:',count)
except AttributeError:
	for i in layer_342_out:
		count = 1
		for s in layer_342_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #342:',count)



layer_343 = L343()
try:
	try:
		layer_343_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_343_out = layer_343(layer_343_concat)
	except ValueError:
		layer_343_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_343_out = layer_343(layer_343_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_343 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_343_concat = strong_concat_343(new_list_of_inputs)
	layer_343_out = layer_343(layer_343_concat)
if layer_343.count_params() != 0:
	if 343 in set_weigths:
		if len(layer_343.get_weights()) > 1:
			new_w = [get_new_weigths(layer_343, set_weigths[343])]+layer_343.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_343, set_weigths[343])]
		layer_343.set_weights(new_w)


try:
	count = 1
	for s in layer_343_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #343:',count)
except AttributeError:
	for i in layer_343_out:
		count = 1
		for s in layer_343_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #343:',count)



layer_344 = L344()
try:
	try:
		layer_344_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out])
		layer_344_out = layer_344(layer_344_concat)
	except ValueError:
		layer_344_concat = tf.keras.layers.concatenate([layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out], axis=-2)
		layer_344_out = layer_344(layer_344_concat)
except ValueError:
	print('do strong concat!')
	strong_concat_344 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_3_out, layer_4_out, layer_173_out, layer_174_out, layer_175_out, layer_176_out, layer_177_out, layer_178_out, layer_179_out, layer_180_out, layer_181_out, layer_182_out, layer_183_out, layer_184_out, layer_185_out, layer_345_out, layer_346_out, layer_347_out, layer_348_out, layer_349_out, layer_350_out, layer_351_out, layer_352_out, layer_353_out, layer_354_out, layer_355_out, layer_356_out, layer_357_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_344_concat = strong_concat_344(new_list_of_inputs)
	layer_344_out = layer_344(layer_344_concat)
if layer_344.count_params() != 0:
	if 344 in set_weigths:
		if len(layer_344.get_weights()) > 1:
			new_w = [get_new_weigths(layer_344, set_weigths[344])]+layer_344.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_344, set_weigths[344])]
		layer_344.set_weights(new_w)


try:
	count = 1
	for s in layer_344_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #344:',count)
except AttributeError:
	for i in layer_344_out:
		count = 1
		for s in layer_344_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #344:',count)



layer_75_out = tf.keras.layers.Flatten()(layer_75_out)
layer_75_out = tf.keras.layers.Dense(10, activation='sigmoid')(layer_75_out)
model = tf.keras.Model(
    inputs=[layer_3_out],
    outputs=[layer_75_out],
)


model.summary()
tf.keras.utils.plot_model(model, "random_model_1034.png",show_shapes=True)
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

	