
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

	
class L80(tf.keras.layers.Layer):
	def __init__(self):
		super(L80, self).__init__()
		self.units = 1

	def build(self, input_shape):
		initializer = None
		if initializer == None:
			initializer = "random_normal"
		regularizer = None
		constrain = tf.keras.constraints.UnitNorm(axis=0)
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		

	def call(self, inputs):
		return tf.keras.activations.hard_sigmoid( tf.matmul(inputs, self.w) ) 
			
class L1(tf.keras.layers.Layer):
	def __init__(self):
		super(L1, self).__init__()
	def build(self, input_shape):
		self.check = False
		inp_lsit = input_shape.as_list()
		n = len(inp_lsit)
		if n == 3:
			self.check = True
		if self.check == True:
			self.i_layer = tf.keras.layers.Permute((2,1))
	def call(self, inputs):
		if self.check == True:
			return self.i_layer(inputs)
		return inputs
	
class L81(tf.keras.layers.Layer):
	def __init__(self):
		super(L81, self).__init__()
		self.units = 206

	def build(self, input_shape):
		layer_i  = tf.

	def call(self, inputs):
		return tf.keras.activations.swish( tf.matmul(inputs, self.w) ) 
			
class L82(tf.keras.layers.Layer):
	def __init__(self):
		super(L82, self).__init__()

	def call(self, inputs):
		return tf.keras.activations.swish(tf.math.negative(inputs))
			
class L83(tf.keras.layers.Layer):
	def __init__(self):
		super(L83, self).__init__()

	def call(self, inputs):
		return tf.keras.activations.selu(tf.math.negative(inputs))
			
class L84(tf.keras.layers.Layer):
	def __init__(self):
		super(L84, self).__init__()
	def build(self, input_shape):
		self.check = False
		inp_lsit = input_shape.as_list()
		n = len(inp_lsit)
		if n == 3:
			self.check = True
		if self.check == True:
			self.i_layer = tf.keras.layers.Permute((2,1))
	def call(self, inputs):
		if self.check == True:
			return self.i_layer(inputs)
		return inputs
	
class L85(tf.keras.layers.Layer):
	def __init__(self):
		super(L85, self).__init__()
	def build(self, input_shape):
		self.check = False
		inp_lsit = input_shape.as_list()
		n = len(inp_lsit)
		if n == 3:
			self.check = True
		if self.check == True:
			self.i_layer = tf.keras.layers.Permute((2,1))
	def call(self, inputs):
		if self.check == True:
			return self.i_layer(inputs)
		return inputs
	
class L0(tf.keras.layers.Layer):
	def __init__(self):
		super(L0, self).__init__()
	def build(self, input_shape):
		self.check = False
		inp_lsit = input_shape.as_list()
		n = len(inp_lsit)
		if n == 3:
			self.check = True
		if self.check == True:
			self.i_layer = tf.keras.layers.Permute((2,1))
	def call(self, inputs):
		if self.check == True:
			return self.i_layer(inputs)
		return inputs
	
class L7(tf.keras.layers.Layer):
	def __init__(self):
		super(L7, self).__init__()
	def call(self, inputs):
		return inputs
			
class L8(tf.keras.layers.Layer):
	def __init__(self):
		super(L8, self).__init__()
	def call(self, inputs):
		return inputs
			
class L9(tf.keras.layers.Layer):
	def __init__(self):
		super(L9, self).__init__()
	def call(self, inputs):
		return inputs
			
class L10(tf.keras.layers.Layer):
	def __init__(self):
		super(L10, self).__init__()
	def call(self, inputs):
		return inputs
			
class L11(tf.keras.layers.Layer):
	def __init__(self):
		super(L11, self).__init__()
	def call(self, inputs):
		return inputs
			
class L12(tf.keras.layers.Layer):
	def __init__(self):
		super(L12, self).__init__()
	def call(self, inputs):
		return inputs
			
class L13(tf.keras.layers.Layer):
	def __init__(self):
		super(L13, self).__init__()
	def call(self, inputs):
		return inputs
			
class L14(tf.keras.layers.Layer):
	def __init__(self):
		super(L14, self).__init__()
	def call(self, inputs):
		return inputs
			
class L15(tf.keras.layers.Layer):
	def __init__(self):
		super(L15, self).__init__()
	def call(self, inputs):
		return inputs
			
class L16(tf.keras.layers.Layer):
	def __init__(self):
		super(L16, self).__init__()
	def call(self, inputs):
		return inputs
			
class L17(tf.keras.layers.Layer):
	def __init__(self):
		super(L17, self).__init__()
	def call(self, inputs):
		return inputs
			
class L18(tf.keras.layers.Layer):
	def __init__(self):
		super(L18, self).__init__()
	def call(self, inputs):
		return inputs
			
class L19(tf.keras.layers.Layer):
	def __init__(self):
		super(L19, self).__init__()
	def call(self, inputs):
		return inputs
			
class L20(tf.keras.layers.Layer):
	def __init__(self):
		super(L20, self).__init__()

	def call(self, inputs):
		return tf.keras.activations.softmax((inputs))
			
class L21(tf.keras.layers.Layer):
	def __init__(self):
		super(L21, self).__init__()

	def call(self, inputs):
		return ((inputs))
			
class L22(tf.keras.layers.Layer):
	def __init__(self):
		super(L22, self).__init__()
	def build(self, input_shape):
		self.check = False
		inp_lsit = input_shape.as_list()
		n = len(inp_lsit)
		if n == 3:
			self.check = True
		if self.check == True:
			self.i_layer = tf.keras.layers.Permute((2,1))
	def call(self, inputs):
		if self.check == True:
			return self.i_layer(inputs)
		return inputs
	
class L23(tf.keras.layers.Layer):
	def __init__(self):
		super(L23, self).__init__()
	def call(self, inputs):
		return inputs
			
class L24(tf.keras.layers.Layer):
	def __init__(self):
		super(L24, self).__init__()
	def call(self, inputs):
		return inputs
			
class L25(tf.keras.layers.Layer):
	def __init__(self):
		super(L25, self).__init__()
	def call(self, inputs):
		return inputs
			
class L26(tf.keras.layers.Layer):
	def __init__(self):
		super(L26, self).__init__()
	def call(self, inputs):
		return inputs
			
class L27(tf.keras.layers.Layer):
	def __init__(self):
		super(L27, self).__init__()
	def call(self, inputs):
		return inputs
			
class L28(tf.keras.layers.Layer):
	def __init__(self):
		super(L28, self).__init__()
	def call(self, inputs):
		return inputs
			
class L29(tf.keras.layers.Layer):
	def __init__(self):
		super(L29, self).__init__()
	def call(self, inputs):
		return inputs
			
class L30(tf.keras.layers.Layer):
	def __init__(self):
		super(L30, self).__init__()
	def call(self, inputs):
		return inputs
			
class L31(tf.keras.layers.Layer):
	def __init__(self):
		super(L31, self).__init__()
	def call(self, inputs):
		return inputs
			
class L32(tf.keras.layers.Layer):
	def __init__(self):
		super(L32, self).__init__()
	def call(self, inputs):
		return inputs
			
class L33(tf.keras.layers.Layer):
	def __init__(self):
		super(L33, self).__init__()
	def call(self, inputs):
		return inputs
			
class L34(tf.keras.layers.Layer):
	def __init__(self):
		super(L34, self).__init__()
	def call(self, inputs):
		return inputs
			
class L35(tf.keras.layers.Layer):
	def __init__(self):
		super(L35, self).__init__()
	def call(self, inputs):
		return inputs
			
class L36(tf.keras.layers.Layer):
	def __init__(self):
		super(L36, self).__init__()
	def call(self, inputs):
		return inputs
			
class L37(tf.keras.layers.Layer):
	def __init__(self):
		super(L37, self).__init__()
	def call(self, inputs):
		return inputs
			
class L38(tf.keras.layers.Layer):
	def __init__(self):
		super(L38, self).__init__()
	def call(self, inputs):
		return inputs
			
class L39(tf.keras.layers.Layer):
	def __init__(self):
		super(L39, self).__init__()
		self.units = 1

	def build(self, input_shape):
		initializer = None
		if initializer == None:
			initializer = "random_normal"
		regularizer = None
		constrain = tf.keras.constraints.UnitNorm(axis=0)
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		

	def call(self, inputs):
		return tf.keras.activations.hard_sigmoid( tf.matmul(inputs, self.w) ) 
			
class L40(tf.keras.layers.Layer):
	def __init__(self):
		super(L40, self).__init__()
	def build(self, input_shape):
		self.check = False
		inp_lsit = input_shape.as_list()
		n = len(inp_lsit)
		if n == 3:
			self.check = True
		if self.check == True:
			self.i_layer = tf.keras.layers.Permute((2,1))
	def call(self, inputs):
		if self.check == True:
			return self.i_layer(inputs)
		return inputs
	
class L41(tf.keras.layers.Layer):
	def __init__(self):
		super(L41, self).__init__()
	def build(self, input_shape):
		self.check = False
		inp_lsit = input_shape.as_list()
		n = len(inp_lsit)
		if n == 3:
			self.check = True
		if self.check == True:
			self.i_layer = tf.keras.layers.Permute((2,1))
	def call(self, inputs):
		if self.check == True:
			return self.i_layer(inputs)
		return inputs
	
class L42(tf.keras.layers.Layer):
	def __init__(self):
		super(L42, self).__init__()
		self.units = 1

	def build(self, input_shape):
		initializer = None
		if initializer == None:
			initializer = "random_normal"
		regularizer = None
		constrain = tf.keras.constraints.UnitNorm(axis=0)
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		

	def call(self, inputs):
		return tf.keras.activations.hard_sigmoid( tf.matmul(inputs, self.w) ) 
			
class L43(tf.keras.layers.Layer):
	def __init__(self):
		super(L43, self).__init__()
		self.units = 1

	def build(self, input_shape):
		initializer = None
		if initializer == None:
			initializer = "random_normal"
		regularizer = None
		constrain = tf.keras.constraints.UnitNorm(axis=0)
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		

	def call(self, inputs):
		return tf.keras.activations.hard_sigmoid( tf.matmul(inputs, self.w) ) 
			
class L44(tf.keras.layers.Layer):
	def __init__(self):
		super(L44, self).__init__()
		self.units = 1

	def build(self, input_shape):
		initializer = None
		if initializer == None:
			initializer = "random_normal"
		regularizer = None
		constrain = tf.keras.constraints.UnitNorm(axis=0)
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		

	def call(self, inputs):
		return tf.keras.activations.hard_sigmoid( tf.matmul(inputs, self.w) ) 
			
class L45(tf.keras.layers.Layer):
	def __init__(self):
		super(L45, self).__init__()
		self.units = 1

	def build(self, input_shape):
		initializer = None
		if initializer == None:
			initializer = "random_normal"
		regularizer = None
		constrain = tf.keras.constraints.UnitNorm(axis=0)
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		

	def call(self, inputs):
		return tf.keras.activations.hard_sigmoid( tf.matmul(inputs, self.w) ) 
			
class L46(tf.keras.layers.Layer):
	def __init__(self):
		super(L46, self).__init__()
		self.units = 1

	def build(self, input_shape):
		initializer = None
		if initializer == None:
			initializer = "random_normal"
		regularizer = None
		constrain = tf.keras.constraints.UnitNorm(axis=0)
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		

	def call(self, inputs):
		return tf.keras.activations.hard_sigmoid( tf.matmul(inputs, self.w) ) 
			
class L47(tf.keras.layers.Layer):
	def __init__(self):
		super(L47, self).__init__()
		self.units = 1

	def build(self, input_shape):
		initializer = None
		if initializer == None:
			initializer = "random_normal"
		regularizer = None
		constrain = tf.keras.constraints.UnitNorm(axis=0)
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		

	def call(self, inputs):
		return tf.keras.activations.hard_sigmoid( tf.matmul(inputs, self.w) ) 
			
class L48(tf.keras.layers.Layer):
	def __init__(self):
		super(L48, self).__init__()
		self.units = 1

	def build(self, input_shape):
		initializer = None
		if initializer == None:
			initializer = "random_normal"
		regularizer = None
		constrain = tf.keras.constraints.UnitNorm(axis=0)
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		

	def call(self, inputs):
		return tf.keras.activations.hard_sigmoid( tf.matmul(inputs, self.w) ) 
			
class L49(tf.keras.layers.Layer):
	def __init__(self):
		super(L49, self).__init__()
		self.units = 1

	def build(self, input_shape):
		initializer = None
		if initializer == None:
			initializer = "random_normal"
		regularizer = None
		constrain = tf.keras.constraints.UnitNorm(axis=0)
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		

	def call(self, inputs):
		return tf.keras.activations.hard_sigmoid( tf.matmul(inputs, self.w) ) 
			
class L50(tf.keras.layers.Layer):
	def __init__(self):
		super(L50, self).__init__()
		self.units = 1

	def build(self, input_shape):
		initializer = None
		if initializer == None:
			initializer = "random_normal"
		regularizer = None
		constrain = tf.keras.constraints.UnitNorm(axis=0)
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		

	def call(self, inputs):
		return tf.keras.activations.hard_sigmoid( tf.matmul(inputs, self.w) ) 
			
class L51(tf.keras.layers.Layer):
	def __init__(self):
		super(L51, self).__init__()
		self.units = 1

	def build(self, input_shape):
		initializer = None
		if initializer == None:
			initializer = "random_normal"
		regularizer = None
		constrain = tf.keras.constraints.UnitNorm(axis=0)
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		

	def call(self, inputs):
		return tf.keras.activations.hard_sigmoid( tf.matmul(inputs, self.w) ) 
			
class L52(tf.keras.layers.Layer):
	def __init__(self):
		super(L52, self).__init__()
		self.units = 1

	def build(self, input_shape):
		initializer = None
		if initializer == None:
			initializer = "random_normal"
		regularizer = None
		constrain = tf.keras.constraints.UnitNorm(axis=0)
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		

	def call(self, inputs):
		return tf.keras.activations.hard_sigmoid( tf.matmul(inputs, self.w) ) 
			
class L53(tf.keras.layers.Layer):
	def __init__(self):
		super(L53, self).__init__()
		self.units = 1

	def build(self, input_shape):
		initializer = None
		if initializer == None:
			initializer = "random_normal"
		regularizer = None
		constrain = tf.keras.constraints.UnitNorm(axis=0)
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		

	def call(self, inputs):
		return tf.keras.activations.hard_sigmoid( tf.matmul(inputs, self.w) ) 
			
class L54(tf.keras.layers.Layer):
	def __init__(self):
		super(L54, self).__init__()
		self.units = 1

	def build(self, input_shape):
		initializer = None
		if initializer == None:
			initializer = "random_normal"
		regularizer = None
		constrain = tf.keras.constraints.UnitNorm(axis=0)
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		

	def call(self, inputs):
		return tf.keras.activations.hard_sigmoid( tf.matmul(inputs, self.w) ) 
			
class L55(tf.keras.layers.Layer):
	def __init__(self):
		super(L55, self).__init__()
		self.units = 1

	def build(self, input_shape):
		initializer = None
		if initializer == None:
			initializer = "random_normal"
		regularizer = None
		constrain = tf.keras.constraints.UnitNorm(axis=0)
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		

	def call(self, inputs):
		return tf.keras.activations.hard_sigmoid( tf.matmul(inputs, self.w) ) 
			
class L56(tf.keras.layers.Layer):
	def __init__(self):
		super(L56, self).__init__()
		self.units = 1

	def build(self, input_shape):
		initializer = None
		if initializer == None:
			initializer = "random_normal"
		regularizer = None
		constrain = tf.keras.constraints.UnitNorm(axis=0)
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		

	def call(self, inputs):
		return tf.keras.activations.hard_sigmoid( tf.matmul(inputs, self.w) ) 
			
class L57(tf.keras.layers.Layer):
	def __init__(self):
		super(L57, self).__init__()
		self.units = 1

	def build(self, input_shape):
		initializer = None
		if initializer == None:
			initializer = "random_normal"
		regularizer = None
		constrain = tf.keras.constraints.UnitNorm(axis=0)
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		

	def call(self, inputs):
		return tf.keras.activations.hard_sigmoid( tf.matmul(inputs, self.w) ) 
			
class L58(tf.keras.layers.Layer):
	def __init__(self):
		super(L58, self).__init__()
		self.units = 1

	def build(self, input_shape):
		initializer = None
		if initializer == None:
			initializer = "random_normal"
		regularizer = None
		constrain = tf.keras.constraints.UnitNorm(axis=0)
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		

	def call(self, inputs):
		return tf.keras.activations.hard_sigmoid( tf.matmul(inputs, self.w) ) 
			
class L59(tf.keras.layers.Layer):
	def __init__(self):
		super(L59, self).__init__()
		self.units = 1

	def build(self, input_shape):
		initializer = None
		if initializer == None:
			initializer = "random_normal"
		regularizer = None
		constrain = tf.keras.constraints.UnitNorm(axis=0)
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		

	def call(self, inputs):
		return tf.keras.activations.hard_sigmoid( tf.matmul(inputs, self.w) ) 
			
class L60(tf.keras.layers.Layer):
	def __init__(self):
		super(L60, self).__init__()
		self.units = 1

	def build(self, input_shape):
		initializer = None
		if initializer == None:
			initializer = "random_normal"
		regularizer = None
		constrain = tf.keras.constraints.UnitNorm(axis=0)
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		

	def call(self, inputs):
		return tf.keras.activations.hard_sigmoid( tf.matmul(inputs, self.w) ) 
			
class L61(tf.keras.layers.Layer):
	def __init__(self):
		super(L61, self).__init__()
		self.units = 1

	def build(self, input_shape):
		initializer = None
		if initializer == None:
			initializer = "random_normal"
		regularizer = None
		constrain = tf.keras.constraints.UnitNorm(axis=0)
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		

	def call(self, inputs):
		return tf.keras.activations.hard_sigmoid( tf.matmul(inputs, self.w) ) 
			
class L62(tf.keras.layers.Layer):
	def __init__(self):
		super(L62, self).__init__()
		self.units = 1

	def build(self, input_shape):
		initializer = None
		if initializer == None:
			initializer = "random_normal"
		regularizer = None
		constrain = tf.keras.constraints.UnitNorm(axis=0)
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		

	def call(self, inputs):
		return tf.keras.activations.hard_sigmoid( tf.matmul(inputs, self.w) ) 
			
class L63(tf.keras.layers.Layer):
	def __init__(self):
		super(L63, self).__init__()
		self.units = 1

	def build(self, input_shape):
		initializer = None
		if initializer == None:
			initializer = "random_normal"
		regularizer = None
		constrain = tf.keras.constraints.UnitNorm(axis=0)
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		

	def call(self, inputs):
		return tf.keras.activations.hard_sigmoid( tf.matmul(inputs, self.w) ) 
			
class L64(tf.keras.layers.Layer):
	def __init__(self):
		super(L64, self).__init__()
		self.units = 1

	def build(self, input_shape):
		initializer = None
		if initializer == None:
			initializer = "random_normal"
		regularizer = None
		constrain = tf.keras.constraints.UnitNorm(axis=0)
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		

	def call(self, inputs):
		return tf.keras.activations.hard_sigmoid( tf.matmul(inputs, self.w) ) 
			
class L65(tf.keras.layers.Layer):
	def __init__(self):
		super(L65, self).__init__()
		self.units = 1

	def build(self, input_shape):
		initializer = None
		if initializer == None:
			initializer = "random_normal"
		regularizer = None
		constrain = tf.keras.constraints.UnitNorm(axis=0)
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		

	def call(self, inputs):
		return tf.keras.activations.hard_sigmoid( tf.matmul(inputs, self.w) ) 
			
class L66(tf.keras.layers.Layer):
	def __init__(self):
		super(L66, self).__init__()
		self.units = 1

	def build(self, input_shape):
		initializer = None
		if initializer == None:
			initializer = "random_normal"
		regularizer = None
		constrain = tf.keras.constraints.UnitNorm(axis=0)
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		

	def call(self, inputs):
		return tf.keras.activations.hard_sigmoid( tf.matmul(inputs, self.w) ) 
			
class L67(tf.keras.layers.Layer):
	def __init__(self):
		super(L67, self).__init__()
		self.units = 1

	def build(self, input_shape):
		initializer = None
		if initializer == None:
			initializer = "random_normal"
		regularizer = None
		constrain = tf.keras.constraints.UnitNorm(axis=0)
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		

	def call(self, inputs):
		return tf.keras.activations.hard_sigmoid( tf.matmul(inputs, self.w) ) 
			
class L68(tf.keras.layers.Layer):
	def __init__(self):
		super(L68, self).__init__()
		self.units = 1

	def build(self, input_shape):
		initializer = None
		if initializer == None:
			initializer = "random_normal"
		regularizer = None
		constrain = tf.keras.constraints.UnitNorm(axis=0)
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		

	def call(self, inputs):
		return tf.keras.activations.hard_sigmoid( tf.matmul(inputs, self.w) ) 
			
class L69(tf.keras.layers.Layer):
	def __init__(self):
		super(L69, self).__init__()
		self.units = 1

	def build(self, input_shape):
		initializer = None
		if initializer == None:
			initializer = "random_normal"
		regularizer = None
		constrain = tf.keras.constraints.UnitNorm(axis=0)
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		

	def call(self, inputs):
		return tf.keras.activations.hard_sigmoid( tf.matmul(inputs, self.w) ) 
			
class L70(tf.keras.layers.Layer):
	def __init__(self):
		super(L70, self).__init__()
		self.units = 1

	def build(self, input_shape):
		initializer = None
		if initializer == None:
			initializer = "random_normal"
		regularizer = None
		constrain = tf.keras.constraints.UnitNorm(axis=0)
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		

	def call(self, inputs):
		return tf.keras.activations.hard_sigmoid( tf.matmul(inputs, self.w) ) 
			
class L71(tf.keras.layers.Layer):
	def __init__(self):
		super(L71, self).__init__()
		self.units = 1

	def build(self, input_shape):
		initializer = None
		if initializer == None:
			initializer = "random_normal"
		regularizer = None
		constrain = tf.keras.constraints.UnitNorm(axis=0)
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		

	def call(self, inputs):
		return tf.keras.activations.hard_sigmoid( tf.matmul(inputs, self.w) ) 
			
class L72(tf.keras.layers.Layer):
	def __init__(self):
		super(L72, self).__init__()
		self.units = 1

	def build(self, input_shape):
		initializer = None
		if initializer == None:
			initializer = "random_normal"
		regularizer = None
		constrain = tf.keras.constraints.UnitNorm(axis=0)
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		

	def call(self, inputs):
		return tf.keras.activations.hard_sigmoid( tf.matmul(inputs, self.w) ) 
			
class L73(tf.keras.layers.Layer):
	def __init__(self):
		super(L73, self).__init__()
		self.units = 1

	def build(self, input_shape):
		initializer = None
		if initializer == None:
			initializer = "random_normal"
		regularizer = None
		constrain = tf.keras.constraints.UnitNorm(axis=0)
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		

	def call(self, inputs):
		return tf.keras.activations.hard_sigmoid( tf.matmul(inputs, self.w) ) 
			
class L74(tf.keras.layers.Layer):
	def __init__(self):
		super(L74, self).__init__()
		self.units = 1

	def build(self, input_shape):
		initializer = None
		if initializer == None:
			initializer = "random_normal"
		regularizer = None
		constrain = tf.keras.constraints.UnitNorm(axis=0)
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		

	def call(self, inputs):
		return tf.keras.activations.hard_sigmoid( tf.matmul(inputs, self.w) ) 
			
class L75(tf.keras.layers.Layer):
	def __init__(self):
		super(L75, self).__init__()
		self.units = 1

	def build(self, input_shape):
		initializer = None
		if initializer == None:
			initializer = "random_normal"
		regularizer = None
		constrain = tf.keras.constraints.UnitNorm(axis=0)
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		

	def call(self, inputs):
		return tf.keras.activations.hard_sigmoid( tf.matmul(inputs, self.w) ) 
			
class L76(tf.keras.layers.Layer):
	def __init__(self):
		super(L76, self).__init__()
		self.units = 1

	def build(self, input_shape):
		initializer = None
		if initializer == None:
			initializer = "random_normal"
		regularizer = None
		constrain = tf.keras.constraints.UnitNorm(axis=0)
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		

	def call(self, inputs):
		return tf.keras.activations.hard_sigmoid( tf.matmul(inputs, self.w) ) 
			
class L77(tf.keras.layers.Layer):
	def __init__(self):
		super(L77, self).__init__()
		self.units = 1

	def build(self, input_shape):
		initializer = None
		if initializer == None:
			initializer = "random_normal"
		regularizer = None
		constrain = tf.keras.constraints.UnitNorm(axis=0)
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		

	def call(self, inputs):
		return tf.keras.activations.hard_sigmoid( tf.matmul(inputs, self.w) ) 
			
class L78(tf.keras.layers.Layer):
	def __init__(self):
		super(L78, self).__init__()
		self.units = 1

	def build(self, input_shape):
		initializer = None
		if initializer == None:
			initializer = "random_normal"
		regularizer = None
		constrain = tf.keras.constraints.UnitNorm(axis=0)
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		

	def call(self, inputs):
		return tf.keras.activations.hard_sigmoid( tf.matmul(inputs, self.w) ) 
			
class L79(tf.keras.layers.Layer):
	def __init__(self):
		super(L79, self).__init__()
		self.units = 1

	def build(self, input_shape):
		initializer = None
		if initializer == None:
			initializer = "random_normal"
		regularizer = None
		constrain = tf.keras.constraints.UnitNorm(axis=0)
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable=True,
		)
		

	def call(self, inputs):
		return tf.keras.activations.hard_sigmoid( tf.matmul(inputs, self.w) ) 
			
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
		

layer_84 = L84()
layer_84_out = layer_84(layer_1_out)

layer_85 = L85()
try:
	try:
		layer_85_concat = tf.keras.layers.concatenate([layer_1_out, layer_84_out])
		layer_85_out = layer_85(layer_85_concat)
	except ValueError:
		layer_85_concat = tf.keras.layers.concatenate([layer_1_out, layer_84_out], axis=-2)
		layer_85_out = layer_85(layer_85_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_85 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_84_out]:
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



layer_83 = L83()
try:
	try:
		layer_83_concat = tf.keras.layers.concatenate([layer_1_out, layer_84_out, layer_85_out])
		layer_83_out = layer_83(layer_83_concat)
	except ValueError:
		layer_83_concat = tf.keras.layers.concatenate([layer_1_out, layer_84_out, layer_85_out], axis=-2)
		layer_83_out = layer_83(layer_83_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_83 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_84_out, layer_85_out]:
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



layer_85 = L85()
try:
	try:
		layer_85_concat = tf.keras.layers.concatenate([layer_1_out, layer_84_out])
		layer_85_out = layer_85(layer_85_concat)
	except ValueError:
		layer_85_concat = tf.keras.layers.concatenate([layer_1_out, layer_84_out], axis=-2)
		layer_85_out = layer_85(layer_85_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_85 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_84_out]:
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



layer_81 = L81()
try:
	try:
		layer_81_concat = tf.keras.layers.concatenate([layer_1_out, layer_83_out, layer_84_out, layer_85_out])
		layer_81_out = layer_81(layer_81_concat)
	except ValueError:
		layer_81_concat = tf.keras.layers.concatenate([layer_1_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_81_out = layer_81(layer_81_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_81 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_83_out, layer_84_out, layer_85_out]:
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
		layer_82_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_83_out, layer_84_out, layer_85_out])
		layer_82_out = layer_82(layer_82_concat)
	except ValueError:
		layer_82_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_82_out = layer_82(layer_82_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_82 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_83_out, layer_84_out, layer_85_out]:
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



layer_0 = L0()
try:
	try:
		layer_0_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_0_out = layer_0(layer_0_concat)
	except ValueError:
		layer_0_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_0_out = layer_0(layer_0_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_0 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_0_concat = strong_concat_0(new_list_of_inputs)
	layer_0_out = layer_0(layer_0_concat)
if layer_0.count_params() != 0:
	if 0 in set_weigths:
		if len(layer_0.get_weights()) > 1:
			new_w = [get_new_weigths(layer_0, set_weigths[0])]+layer_0.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_0, set_weigths[0])]
		layer_0.set_weights(new_w)


try:
	count = 1
	for s in layer_0_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #0:',count)
except AttributeError:
	for i in layer_0_out:
		count = 1
		for s in layer_0_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #0:',count)



layer_7 = L7()
try:
	try:
		layer_7_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_7_out = layer_7(layer_7_concat)
	except ValueError:
		layer_7_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_7_out = layer_7(layer_7_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_7 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
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
		layer_8_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_8_out = layer_8(layer_8_concat)
	except ValueError:
		layer_8_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_8_out = layer_8(layer_8_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_8 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
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
		layer_9_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_9_out = layer_9(layer_9_concat)
	except ValueError:
		layer_9_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_9_out = layer_9(layer_9_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_9 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
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



layer_10 = L10()
try:
	try:
		layer_10_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_10_out = layer_10(layer_10_concat)
	except ValueError:
		layer_10_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_10_out = layer_10(layer_10_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_10 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_10_concat = strong_concat_10(new_list_of_inputs)
	layer_10_out = layer_10(layer_10_concat)
if layer_10.count_params() != 0:
	if 10 in set_weigths:
		if len(layer_10.get_weights()) > 1:
			new_w = [get_new_weigths(layer_10, set_weigths[10])]+layer_10.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_10, set_weigths[10])]
		layer_10.set_weights(new_w)


try:
	count = 1
	for s in layer_10_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #10:',count)
except AttributeError:
	for i in layer_10_out:
		count = 1
		for s in layer_10_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #10:',count)



layer_11 = L11()
try:
	try:
		layer_11_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_11_out = layer_11(layer_11_concat)
	except ValueError:
		layer_11_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_11_out = layer_11(layer_11_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_11 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
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
		layer_12_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_12_out = layer_12(layer_12_concat)
	except ValueError:
		layer_12_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_12_out = layer_12(layer_12_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_12 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
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



layer_13 = L13()
try:
	try:
		layer_13_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_13_out = layer_13(layer_13_concat)
	except ValueError:
		layer_13_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_13_out = layer_13(layer_13_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_13 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
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



layer_14 = L14()
try:
	try:
		layer_14_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_14_out = layer_14(layer_14_concat)
	except ValueError:
		layer_14_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_14_out = layer_14(layer_14_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_14 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_14_concat = strong_concat_14(new_list_of_inputs)
	layer_14_out = layer_14(layer_14_concat)
if layer_14.count_params() != 0:
	if 14 in set_weigths:
		if len(layer_14.get_weights()) > 1:
			new_w = [get_new_weigths(layer_14, set_weigths[14])]+layer_14.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_14, set_weigths[14])]
		layer_14.set_weights(new_w)


try:
	count = 1
	for s in layer_14_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #14:',count)
except AttributeError:
	for i in layer_14_out:
		count = 1
		for s in layer_14_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #14:',count)



layer_15 = L15()
try:
	try:
		layer_15_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_15_out = layer_15(layer_15_concat)
	except ValueError:
		layer_15_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_15_out = layer_15(layer_15_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_15 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_15_concat = strong_concat_15(new_list_of_inputs)
	layer_15_out = layer_15(layer_15_concat)
if layer_15.count_params() != 0:
	if 15 in set_weigths:
		if len(layer_15.get_weights()) > 1:
			new_w = [get_new_weigths(layer_15, set_weigths[15])]+layer_15.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_15, set_weigths[15])]
		layer_15.set_weights(new_w)


try:
	count = 1
	for s in layer_15_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #15:',count)
except AttributeError:
	for i in layer_15_out:
		count = 1
		for s in layer_15_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #15:',count)



layer_16 = L16()
try:
	try:
		layer_16_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_16_out = layer_16(layer_16_concat)
	except ValueError:
		layer_16_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_16_out = layer_16(layer_16_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_16 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_16_concat = strong_concat_16(new_list_of_inputs)
	layer_16_out = layer_16(layer_16_concat)
if layer_16.count_params() != 0:
	if 16 in set_weigths:
		if len(layer_16.get_weights()) > 1:
			new_w = [get_new_weigths(layer_16, set_weigths[16])]+layer_16.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_16, set_weigths[16])]
		layer_16.set_weights(new_w)


try:
	count = 1
	for s in layer_16_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #16:',count)
except AttributeError:
	for i in layer_16_out:
		count = 1
		for s in layer_16_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #16:',count)



layer_17 = L17()
try:
	try:
		layer_17_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_17_out = layer_17(layer_17_concat)
	except ValueError:
		layer_17_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_17_out = layer_17(layer_17_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_17 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_17_concat = strong_concat_17(new_list_of_inputs)
	layer_17_out = layer_17(layer_17_concat)
if layer_17.count_params() != 0:
	if 17 in set_weigths:
		if len(layer_17.get_weights()) > 1:
			new_w = [get_new_weigths(layer_17, set_weigths[17])]+layer_17.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_17, set_weigths[17])]
		layer_17.set_weights(new_w)


try:
	count = 1
	for s in layer_17_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #17:',count)
except AttributeError:
	for i in layer_17_out:
		count = 1
		for s in layer_17_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #17:',count)



layer_18 = L18()
try:
	try:
		layer_18_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_18_out = layer_18(layer_18_concat)
	except ValueError:
		layer_18_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_18_out = layer_18(layer_18_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_18 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_18_concat = strong_concat_18(new_list_of_inputs)
	layer_18_out = layer_18(layer_18_concat)
if layer_18.count_params() != 0:
	if 18 in set_weigths:
		if len(layer_18.get_weights()) > 1:
			new_w = [get_new_weigths(layer_18, set_weigths[18])]+layer_18.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_18, set_weigths[18])]
		layer_18.set_weights(new_w)


try:
	count = 1
	for s in layer_18_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #18:',count)
except AttributeError:
	for i in layer_18_out:
		count = 1
		for s in layer_18_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #18:',count)



layer_19 = L19()
try:
	try:
		layer_19_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_19_out = layer_19(layer_19_concat)
	except ValueError:
		layer_19_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_19_out = layer_19(layer_19_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_19 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_19_concat = strong_concat_19(new_list_of_inputs)
	layer_19_out = layer_19(layer_19_concat)
if layer_19.count_params() != 0:
	if 19 in set_weigths:
		if len(layer_19.get_weights()) > 1:
			new_w = [get_new_weigths(layer_19, set_weigths[19])]+layer_19.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_19, set_weigths[19])]
		layer_19.set_weights(new_w)


try:
	count = 1
	for s in layer_19_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #19:',count)
except AttributeError:
	for i in layer_19_out:
		count = 1
		for s in layer_19_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #19:',count)



layer_20 = L20()
try:
	try:
		layer_20_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_20_out = layer_20(layer_20_concat)
	except ValueError:
		layer_20_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_20_out = layer_20(layer_20_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_20 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_20_concat = strong_concat_20(new_list_of_inputs)
	layer_20_out = layer_20(layer_20_concat)
if layer_20.count_params() != 0:
	if 20 in set_weigths:
		if len(layer_20.get_weights()) > 1:
			new_w = [get_new_weigths(layer_20, set_weigths[20])]+layer_20.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_20, set_weigths[20])]
		layer_20.set_weights(new_w)


try:
	count = 1
	for s in layer_20_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #20:',count)
except AttributeError:
	for i in layer_20_out:
		count = 1
		for s in layer_20_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #20:',count)



layer_21 = L21()
try:
	try:
		layer_21_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_21_out = layer_21(layer_21_concat)
	except ValueError:
		layer_21_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_21_out = layer_21(layer_21_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_21 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_21_concat = strong_concat_21(new_list_of_inputs)
	layer_21_out = layer_21(layer_21_concat)
if layer_21.count_params() != 0:
	if 21 in set_weigths:
		if len(layer_21.get_weights()) > 1:
			new_w = [get_new_weigths(layer_21, set_weigths[21])]+layer_21.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_21, set_weigths[21])]
		layer_21.set_weights(new_w)


try:
	count = 1
	for s in layer_21_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #21:',count)
except AttributeError:
	for i in layer_21_out:
		count = 1
		for s in layer_21_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #21:',count)



layer_22 = L22()
try:
	try:
		layer_22_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_22_out = layer_22(layer_22_concat)
	except ValueError:
		layer_22_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_22_out = layer_22(layer_22_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_22 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_22_concat = strong_concat_22(new_list_of_inputs)
	layer_22_out = layer_22(layer_22_concat)
if layer_22.count_params() != 0:
	if 22 in set_weigths:
		if len(layer_22.get_weights()) > 1:
			new_w = [get_new_weigths(layer_22, set_weigths[22])]+layer_22.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_22, set_weigths[22])]
		layer_22.set_weights(new_w)


try:
	count = 1
	for s in layer_22_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #22:',count)
except AttributeError:
	for i in layer_22_out:
		count = 1
		for s in layer_22_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #22:',count)



layer_23 = L23()
try:
	try:
		layer_23_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_23_out = layer_23(layer_23_concat)
	except ValueError:
		layer_23_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_23_out = layer_23(layer_23_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_23 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_23_concat = strong_concat_23(new_list_of_inputs)
	layer_23_out = layer_23(layer_23_concat)
if layer_23.count_params() != 0:
	if 23 in set_weigths:
		if len(layer_23.get_weights()) > 1:
			new_w = [get_new_weigths(layer_23, set_weigths[23])]+layer_23.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_23, set_weigths[23])]
		layer_23.set_weights(new_w)


try:
	count = 1
	for s in layer_23_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #23:',count)
except AttributeError:
	for i in layer_23_out:
		count = 1
		for s in layer_23_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #23:',count)



layer_24 = L24()
try:
	try:
		layer_24_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_24_out = layer_24(layer_24_concat)
	except ValueError:
		layer_24_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_24_out = layer_24(layer_24_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_24 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_24_concat = strong_concat_24(new_list_of_inputs)
	layer_24_out = layer_24(layer_24_concat)
if layer_24.count_params() != 0:
	if 24 in set_weigths:
		if len(layer_24.get_weights()) > 1:
			new_w = [get_new_weigths(layer_24, set_weigths[24])]+layer_24.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_24, set_weigths[24])]
		layer_24.set_weights(new_w)


try:
	count = 1
	for s in layer_24_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #24:',count)
except AttributeError:
	for i in layer_24_out:
		count = 1
		for s in layer_24_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #24:',count)



layer_25 = L25()
try:
	try:
		layer_25_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_25_out = layer_25(layer_25_concat)
	except ValueError:
		layer_25_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_25_out = layer_25(layer_25_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_25 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_25_concat = strong_concat_25(new_list_of_inputs)
	layer_25_out = layer_25(layer_25_concat)
if layer_25.count_params() != 0:
	if 25 in set_weigths:
		if len(layer_25.get_weights()) > 1:
			new_w = [get_new_weigths(layer_25, set_weigths[25])]+layer_25.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_25, set_weigths[25])]
		layer_25.set_weights(new_w)


try:
	count = 1
	for s in layer_25_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #25:',count)
except AttributeError:
	for i in layer_25_out:
		count = 1
		for s in layer_25_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #25:',count)



layer_26 = L26()
try:
	try:
		layer_26_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_26_out = layer_26(layer_26_concat)
	except ValueError:
		layer_26_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_26_out = layer_26(layer_26_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_26 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_26_concat = strong_concat_26(new_list_of_inputs)
	layer_26_out = layer_26(layer_26_concat)
if layer_26.count_params() != 0:
	if 26 in set_weigths:
		if len(layer_26.get_weights()) > 1:
			new_w = [get_new_weigths(layer_26, set_weigths[26])]+layer_26.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_26, set_weigths[26])]
		layer_26.set_weights(new_w)


try:
	count = 1
	for s in layer_26_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #26:',count)
except AttributeError:
	for i in layer_26_out:
		count = 1
		for s in layer_26_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #26:',count)



layer_27 = L27()
try:
	try:
		layer_27_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_27_out = layer_27(layer_27_concat)
	except ValueError:
		layer_27_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_27_out = layer_27(layer_27_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_27 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_27_concat = strong_concat_27(new_list_of_inputs)
	layer_27_out = layer_27(layer_27_concat)
if layer_27.count_params() != 0:
	if 27 in set_weigths:
		if len(layer_27.get_weights()) > 1:
			new_w = [get_new_weigths(layer_27, set_weigths[27])]+layer_27.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_27, set_weigths[27])]
		layer_27.set_weights(new_w)


try:
	count = 1
	for s in layer_27_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #27:',count)
except AttributeError:
	for i in layer_27_out:
		count = 1
		for s in layer_27_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #27:',count)



layer_28 = L28()
try:
	try:
		layer_28_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_28_out = layer_28(layer_28_concat)
	except ValueError:
		layer_28_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_28_out = layer_28(layer_28_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_28 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
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
		layer_29_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_29_out = layer_29(layer_29_concat)
	except ValueError:
		layer_29_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_29_out = layer_29(layer_29_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_29 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
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
		layer_30_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_30_out = layer_30(layer_30_concat)
	except ValueError:
		layer_30_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_30_out = layer_30(layer_30_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_30 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
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



layer_31 = L31()
try:
	try:
		layer_31_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_31_out = layer_31(layer_31_concat)
	except ValueError:
		layer_31_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_31_out = layer_31(layer_31_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_31 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_31_concat = strong_concat_31(new_list_of_inputs)
	layer_31_out = layer_31(layer_31_concat)
if layer_31.count_params() != 0:
	if 31 in set_weigths:
		if len(layer_31.get_weights()) > 1:
			new_w = [get_new_weigths(layer_31, set_weigths[31])]+layer_31.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_31, set_weigths[31])]
		layer_31.set_weights(new_w)


try:
	count = 1
	for s in layer_31_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #31:',count)
except AttributeError:
	for i in layer_31_out:
		count = 1
		for s in layer_31_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #31:',count)



layer_32 = L32()
try:
	try:
		layer_32_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_32_out = layer_32(layer_32_concat)
	except ValueError:
		layer_32_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_32_out = layer_32(layer_32_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_32 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_32_concat = strong_concat_32(new_list_of_inputs)
	layer_32_out = layer_32(layer_32_concat)
if layer_32.count_params() != 0:
	if 32 in set_weigths:
		if len(layer_32.get_weights()) > 1:
			new_w = [get_new_weigths(layer_32, set_weigths[32])]+layer_32.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_32, set_weigths[32])]
		layer_32.set_weights(new_w)


try:
	count = 1
	for s in layer_32_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #32:',count)
except AttributeError:
	for i in layer_32_out:
		count = 1
		for s in layer_32_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #32:',count)



layer_33 = L33()
try:
	try:
		layer_33_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_33_out = layer_33(layer_33_concat)
	except ValueError:
		layer_33_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_33_out = layer_33(layer_33_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_33 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_33_concat = strong_concat_33(new_list_of_inputs)
	layer_33_out = layer_33(layer_33_concat)
if layer_33.count_params() != 0:
	if 33 in set_weigths:
		if len(layer_33.get_weights()) > 1:
			new_w = [get_new_weigths(layer_33, set_weigths[33])]+layer_33.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_33, set_weigths[33])]
		layer_33.set_weights(new_w)


try:
	count = 1
	for s in layer_33_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #33:',count)
except AttributeError:
	for i in layer_33_out:
		count = 1
		for s in layer_33_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #33:',count)



layer_34 = L34()
try:
	try:
		layer_34_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_34_out = layer_34(layer_34_concat)
	except ValueError:
		layer_34_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_34_out = layer_34(layer_34_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_34 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_34_concat = strong_concat_34(new_list_of_inputs)
	layer_34_out = layer_34(layer_34_concat)
if layer_34.count_params() != 0:
	if 34 in set_weigths:
		if len(layer_34.get_weights()) > 1:
			new_w = [get_new_weigths(layer_34, set_weigths[34])]+layer_34.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_34, set_weigths[34])]
		layer_34.set_weights(new_w)


try:
	count = 1
	for s in layer_34_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #34:',count)
except AttributeError:
	for i in layer_34_out:
		count = 1
		for s in layer_34_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #34:',count)



layer_35 = L35()
try:
	try:
		layer_35_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_35_out = layer_35(layer_35_concat)
	except ValueError:
		layer_35_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_35_out = layer_35(layer_35_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_35 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
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
		layer_36_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_36_out = layer_36(layer_36_concat)
	except ValueError:
		layer_36_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_36_out = layer_36(layer_36_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_36 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
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



layer_37 = L37()
try:
	try:
		layer_37_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_37_out = layer_37(layer_37_concat)
	except ValueError:
		layer_37_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_37_out = layer_37(layer_37_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_37 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_37_concat = strong_concat_37(new_list_of_inputs)
	layer_37_out = layer_37(layer_37_concat)
if layer_37.count_params() != 0:
	if 37 in set_weigths:
		if len(layer_37.get_weights()) > 1:
			new_w = [get_new_weigths(layer_37, set_weigths[37])]+layer_37.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_37, set_weigths[37])]
		layer_37.set_weights(new_w)


try:
	count = 1
	for s in layer_37_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #37:',count)
except AttributeError:
	for i in layer_37_out:
		count = 1
		for s in layer_37_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #37:',count)



layer_38 = L38()
try:
	try:
		layer_38_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_38_out = layer_38(layer_38_concat)
	except ValueError:
		layer_38_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_38_out = layer_38(layer_38_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_38 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_38_concat = strong_concat_38(new_list_of_inputs)
	layer_38_out = layer_38(layer_38_concat)
if layer_38.count_params() != 0:
	if 38 in set_weigths:
		if len(layer_38.get_weights()) > 1:
			new_w = [get_new_weigths(layer_38, set_weigths[38])]+layer_38.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_38, set_weigths[38])]
		layer_38.set_weights(new_w)


try:
	count = 1
	for s in layer_38_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #38:',count)
except AttributeError:
	for i in layer_38_out:
		count = 1
		for s in layer_38_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #38:',count)



layer_39 = L39()
try:
	try:
		layer_39_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_39_out = layer_39(layer_39_concat)
	except ValueError:
		layer_39_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_39_out = layer_39(layer_39_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_39 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_39_concat = strong_concat_39(new_list_of_inputs)
	layer_39_out = layer_39(layer_39_concat)
if layer_39.count_params() != 0:
	if 39 in set_weigths:
		if len(layer_39.get_weights()) > 1:
			new_w = [get_new_weigths(layer_39, set_weigths[39])]+layer_39.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_39, set_weigths[39])]
		layer_39.set_weights(new_w)


try:
	count = 1
	for s in layer_39_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #39:',count)
except AttributeError:
	for i in layer_39_out:
		count = 1
		for s in layer_39_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #39:',count)



layer_40 = L40()
try:
	try:
		layer_40_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_40_out = layer_40(layer_40_concat)
	except ValueError:
		layer_40_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_40_out = layer_40(layer_40_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_40 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
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
		layer_41_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_41_out = layer_41(layer_41_concat)
	except ValueError:
		layer_41_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_41_out = layer_41(layer_41_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_41 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
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



layer_42 = L42()
try:
	try:
		layer_42_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_42_out = layer_42(layer_42_concat)
	except ValueError:
		layer_42_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_42_out = layer_42(layer_42_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_42 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
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



layer_43 = L43()
try:
	try:
		layer_43_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_43_out = layer_43(layer_43_concat)
	except ValueError:
		layer_43_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_43_out = layer_43(layer_43_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_43 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_43_concat = strong_concat_43(new_list_of_inputs)
	layer_43_out = layer_43(layer_43_concat)
if layer_43.count_params() != 0:
	if 43 in set_weigths:
		if len(layer_43.get_weights()) > 1:
			new_w = [get_new_weigths(layer_43, set_weigths[43])]+layer_43.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_43, set_weigths[43])]
		layer_43.set_weights(new_w)


try:
	count = 1
	for s in layer_43_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #43:',count)
except AttributeError:
	for i in layer_43_out:
		count = 1
		for s in layer_43_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #43:',count)



layer_44 = L44()
try:
	try:
		layer_44_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_44_out = layer_44(layer_44_concat)
	except ValueError:
		layer_44_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_44_out = layer_44(layer_44_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_44 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_44_concat = strong_concat_44(new_list_of_inputs)
	layer_44_out = layer_44(layer_44_concat)
if layer_44.count_params() != 0:
	if 44 in set_weigths:
		if len(layer_44.get_weights()) > 1:
			new_w = [get_new_weigths(layer_44, set_weigths[44])]+layer_44.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_44, set_weigths[44])]
		layer_44.set_weights(new_w)


try:
	count = 1
	for s in layer_44_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #44:',count)
except AttributeError:
	for i in layer_44_out:
		count = 1
		for s in layer_44_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #44:',count)



layer_45 = L45()
try:
	try:
		layer_45_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_45_out = layer_45(layer_45_concat)
	except ValueError:
		layer_45_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_45_out = layer_45(layer_45_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_45 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
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
		layer_46_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_46_out = layer_46(layer_46_concat)
	except ValueError:
		layer_46_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_46_out = layer_46(layer_46_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_46 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
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



layer_47 = L47()
try:
	try:
		layer_47_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_47_out = layer_47(layer_47_concat)
	except ValueError:
		layer_47_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_47_out = layer_47(layer_47_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_47 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
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
		layer_48_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_48_out = layer_48(layer_48_concat)
	except ValueError:
		layer_48_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_48_out = layer_48(layer_48_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_48 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
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
		layer_49_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_49_out = layer_49(layer_49_concat)
	except ValueError:
		layer_49_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_49_out = layer_49(layer_49_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_49 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
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
		layer_50_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_50_out = layer_50(layer_50_concat)
	except ValueError:
		layer_50_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_50_out = layer_50(layer_50_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_50 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
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



layer_51 = L51()
try:
	try:
		layer_51_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_51_out = layer_51(layer_51_concat)
	except ValueError:
		layer_51_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_51_out = layer_51(layer_51_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_51 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_51_concat = strong_concat_51(new_list_of_inputs)
	layer_51_out = layer_51(layer_51_concat)
if layer_51.count_params() != 0:
	if 51 in set_weigths:
		if len(layer_51.get_weights()) > 1:
			new_w = [get_new_weigths(layer_51, set_weigths[51])]+layer_51.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_51, set_weigths[51])]
		layer_51.set_weights(new_w)


try:
	count = 1
	for s in layer_51_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #51:',count)
except AttributeError:
	for i in layer_51_out:
		count = 1
		for s in layer_51_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #51:',count)



layer_52 = L52()
try:
	try:
		layer_52_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_52_out = layer_52(layer_52_concat)
	except ValueError:
		layer_52_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_52_out = layer_52(layer_52_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_52 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_52_concat = strong_concat_52(new_list_of_inputs)
	layer_52_out = layer_52(layer_52_concat)
if layer_52.count_params() != 0:
	if 52 in set_weigths:
		if len(layer_52.get_weights()) > 1:
			new_w = [get_new_weigths(layer_52, set_weigths[52])]+layer_52.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_52, set_weigths[52])]
		layer_52.set_weights(new_w)


try:
	count = 1
	for s in layer_52_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #52:',count)
except AttributeError:
	for i in layer_52_out:
		count = 1
		for s in layer_52_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #52:',count)



layer_53 = L53()
try:
	try:
		layer_53_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_53_out = layer_53(layer_53_concat)
	except ValueError:
		layer_53_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_53_out = layer_53(layer_53_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_53 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_53_concat = strong_concat_53(new_list_of_inputs)
	layer_53_out = layer_53(layer_53_concat)
if layer_53.count_params() != 0:
	if 53 in set_weigths:
		if len(layer_53.get_weights()) > 1:
			new_w = [get_new_weigths(layer_53, set_weigths[53])]+layer_53.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_53, set_weigths[53])]
		layer_53.set_weights(new_w)


try:
	count = 1
	for s in layer_53_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #53:',count)
except AttributeError:
	for i in layer_53_out:
		count = 1
		for s in layer_53_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #53:',count)



layer_54 = L54()
try:
	try:
		layer_54_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_54_out = layer_54(layer_54_concat)
	except ValueError:
		layer_54_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_54_out = layer_54(layer_54_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_54 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_54_concat = strong_concat_54(new_list_of_inputs)
	layer_54_out = layer_54(layer_54_concat)
if layer_54.count_params() != 0:
	if 54 in set_weigths:
		if len(layer_54.get_weights()) > 1:
			new_w = [get_new_weigths(layer_54, set_weigths[54])]+layer_54.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_54, set_weigths[54])]
		layer_54.set_weights(new_w)


try:
	count = 1
	for s in layer_54_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #54:',count)
except AttributeError:
	for i in layer_54_out:
		count = 1
		for s in layer_54_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #54:',count)



layer_55 = L55()
try:
	try:
		layer_55_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_55_out = layer_55(layer_55_concat)
	except ValueError:
		layer_55_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_55_out = layer_55(layer_55_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_55 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_55_concat = strong_concat_55(new_list_of_inputs)
	layer_55_out = layer_55(layer_55_concat)
if layer_55.count_params() != 0:
	if 55 in set_weigths:
		if len(layer_55.get_weights()) > 1:
			new_w = [get_new_weigths(layer_55, set_weigths[55])]+layer_55.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_55, set_weigths[55])]
		layer_55.set_weights(new_w)


try:
	count = 1
	for s in layer_55_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #55:',count)
except AttributeError:
	for i in layer_55_out:
		count = 1
		for s in layer_55_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #55:',count)



layer_56 = L56()
try:
	try:
		layer_56_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_56_out = layer_56(layer_56_concat)
	except ValueError:
		layer_56_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_56_out = layer_56(layer_56_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_56 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_56_concat = strong_concat_56(new_list_of_inputs)
	layer_56_out = layer_56(layer_56_concat)
if layer_56.count_params() != 0:
	if 56 in set_weigths:
		if len(layer_56.get_weights()) > 1:
			new_w = [get_new_weigths(layer_56, set_weigths[56])]+layer_56.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_56, set_weigths[56])]
		layer_56.set_weights(new_w)


try:
	count = 1
	for s in layer_56_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #56:',count)
except AttributeError:
	for i in layer_56_out:
		count = 1
		for s in layer_56_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #56:',count)



layer_57 = L57()
try:
	try:
		layer_57_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_57_out = layer_57(layer_57_concat)
	except ValueError:
		layer_57_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_57_out = layer_57(layer_57_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_57 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_57_concat = strong_concat_57(new_list_of_inputs)
	layer_57_out = layer_57(layer_57_concat)
if layer_57.count_params() != 0:
	if 57 in set_weigths:
		if len(layer_57.get_weights()) > 1:
			new_w = [get_new_weigths(layer_57, set_weigths[57])]+layer_57.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_57, set_weigths[57])]
		layer_57.set_weights(new_w)


try:
	count = 1
	for s in layer_57_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #57:',count)
except AttributeError:
	for i in layer_57_out:
		count = 1
		for s in layer_57_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #57:',count)



layer_58 = L58()
try:
	try:
		layer_58_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_58_out = layer_58(layer_58_concat)
	except ValueError:
		layer_58_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_58_out = layer_58(layer_58_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_58 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_58_concat = strong_concat_58(new_list_of_inputs)
	layer_58_out = layer_58(layer_58_concat)
if layer_58.count_params() != 0:
	if 58 in set_weigths:
		if len(layer_58.get_weights()) > 1:
			new_w = [get_new_weigths(layer_58, set_weigths[58])]+layer_58.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_58, set_weigths[58])]
		layer_58.set_weights(new_w)


try:
	count = 1
	for s in layer_58_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #58:',count)
except AttributeError:
	for i in layer_58_out:
		count = 1
		for s in layer_58_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #58:',count)



layer_59 = L59()
try:
	try:
		layer_59_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_59_out = layer_59(layer_59_concat)
	except ValueError:
		layer_59_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_59_out = layer_59(layer_59_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_59 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_59_concat = strong_concat_59(new_list_of_inputs)
	layer_59_out = layer_59(layer_59_concat)
if layer_59.count_params() != 0:
	if 59 in set_weigths:
		if len(layer_59.get_weights()) > 1:
			new_w = [get_new_weigths(layer_59, set_weigths[59])]+layer_59.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_59, set_weigths[59])]
		layer_59.set_weights(new_w)


try:
	count = 1
	for s in layer_59_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #59:',count)
except AttributeError:
	for i in layer_59_out:
		count = 1
		for s in layer_59_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #59:',count)



layer_60 = L60()
try:
	try:
		layer_60_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_60_out = layer_60(layer_60_concat)
	except ValueError:
		layer_60_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_60_out = layer_60(layer_60_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_60 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_60_concat = strong_concat_60(new_list_of_inputs)
	layer_60_out = layer_60(layer_60_concat)
if layer_60.count_params() != 0:
	if 60 in set_weigths:
		if len(layer_60.get_weights()) > 1:
			new_w = [get_new_weigths(layer_60, set_weigths[60])]+layer_60.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_60, set_weigths[60])]
		layer_60.set_weights(new_w)


try:
	count = 1
	for s in layer_60_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #60:',count)
except AttributeError:
	for i in layer_60_out:
		count = 1
		for s in layer_60_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #60:',count)



layer_61 = L61()
try:
	try:
		layer_61_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_61_out = layer_61(layer_61_concat)
	except ValueError:
		layer_61_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_61_out = layer_61(layer_61_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_61 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_61_concat = strong_concat_61(new_list_of_inputs)
	layer_61_out = layer_61(layer_61_concat)
if layer_61.count_params() != 0:
	if 61 in set_weigths:
		if len(layer_61.get_weights()) > 1:
			new_w = [get_new_weigths(layer_61, set_weigths[61])]+layer_61.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_61, set_weigths[61])]
		layer_61.set_weights(new_w)


try:
	count = 1
	for s in layer_61_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #61:',count)
except AttributeError:
	for i in layer_61_out:
		count = 1
		for s in layer_61_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #61:',count)



layer_62 = L62()
try:
	try:
		layer_62_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_62_out = layer_62(layer_62_concat)
	except ValueError:
		layer_62_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_62_out = layer_62(layer_62_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_62 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_62_concat = strong_concat_62(new_list_of_inputs)
	layer_62_out = layer_62(layer_62_concat)
if layer_62.count_params() != 0:
	if 62 in set_weigths:
		if len(layer_62.get_weights()) > 1:
			new_w = [get_new_weigths(layer_62, set_weigths[62])]+layer_62.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_62, set_weigths[62])]
		layer_62.set_weights(new_w)


try:
	count = 1
	for s in layer_62_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #62:',count)
except AttributeError:
	for i in layer_62_out:
		count = 1
		for s in layer_62_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #62:',count)



layer_63 = L63()
try:
	try:
		layer_63_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_63_out = layer_63(layer_63_concat)
	except ValueError:
		layer_63_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_63_out = layer_63(layer_63_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_63 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_63_concat = strong_concat_63(new_list_of_inputs)
	layer_63_out = layer_63(layer_63_concat)
if layer_63.count_params() != 0:
	if 63 in set_weigths:
		if len(layer_63.get_weights()) > 1:
			new_w = [get_new_weigths(layer_63, set_weigths[63])]+layer_63.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_63, set_weigths[63])]
		layer_63.set_weights(new_w)


try:
	count = 1
	for s in layer_63_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #63:',count)
except AttributeError:
	for i in layer_63_out:
		count = 1
		for s in layer_63_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #63:',count)



layer_64 = L64()
try:
	try:
		layer_64_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_64_out = layer_64(layer_64_concat)
	except ValueError:
		layer_64_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_64_out = layer_64(layer_64_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_64 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_64_concat = strong_concat_64(new_list_of_inputs)
	layer_64_out = layer_64(layer_64_concat)
if layer_64.count_params() != 0:
	if 64 in set_weigths:
		if len(layer_64.get_weights()) > 1:
			new_w = [get_new_weigths(layer_64, set_weigths[64])]+layer_64.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_64, set_weigths[64])]
		layer_64.set_weights(new_w)


try:
	count = 1
	for s in layer_64_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #64:',count)
except AttributeError:
	for i in layer_64_out:
		count = 1
		for s in layer_64_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #64:',count)



layer_65 = L65()
try:
	try:
		layer_65_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_65_out = layer_65(layer_65_concat)
	except ValueError:
		layer_65_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_65_out = layer_65(layer_65_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_65 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_65_concat = strong_concat_65(new_list_of_inputs)
	layer_65_out = layer_65(layer_65_concat)
if layer_65.count_params() != 0:
	if 65 in set_weigths:
		if len(layer_65.get_weights()) > 1:
			new_w = [get_new_weigths(layer_65, set_weigths[65])]+layer_65.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_65, set_weigths[65])]
		layer_65.set_weights(new_w)


try:
	count = 1
	for s in layer_65_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #65:',count)
except AttributeError:
	for i in layer_65_out:
		count = 1
		for s in layer_65_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #65:',count)



layer_66 = L66()
try:
	try:
		layer_66_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_66_out = layer_66(layer_66_concat)
	except ValueError:
		layer_66_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_66_out = layer_66(layer_66_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_66 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_66_concat = strong_concat_66(new_list_of_inputs)
	layer_66_out = layer_66(layer_66_concat)
if layer_66.count_params() != 0:
	if 66 in set_weigths:
		if len(layer_66.get_weights()) > 1:
			new_w = [get_new_weigths(layer_66, set_weigths[66])]+layer_66.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_66, set_weigths[66])]
		layer_66.set_weights(new_w)


try:
	count = 1
	for s in layer_66_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #66:',count)
except AttributeError:
	for i in layer_66_out:
		count = 1
		for s in layer_66_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #66:',count)



layer_67 = L67()
try:
	try:
		layer_67_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_67_out = layer_67(layer_67_concat)
	except ValueError:
		layer_67_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_67_out = layer_67(layer_67_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_67 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_67_concat = strong_concat_67(new_list_of_inputs)
	layer_67_out = layer_67(layer_67_concat)
if layer_67.count_params() != 0:
	if 67 in set_weigths:
		if len(layer_67.get_weights()) > 1:
			new_w = [get_new_weigths(layer_67, set_weigths[67])]+layer_67.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_67, set_weigths[67])]
		layer_67.set_weights(new_w)


try:
	count = 1
	for s in layer_67_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #67:',count)
except AttributeError:
	for i in layer_67_out:
		count = 1
		for s in layer_67_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #67:',count)



layer_68 = L68()
try:
	try:
		layer_68_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_68_out = layer_68(layer_68_concat)
	except ValueError:
		layer_68_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_68_out = layer_68(layer_68_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_68 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_68_concat = strong_concat_68(new_list_of_inputs)
	layer_68_out = layer_68(layer_68_concat)
if layer_68.count_params() != 0:
	if 68 in set_weigths:
		if len(layer_68.get_weights()) > 1:
			new_w = [get_new_weigths(layer_68, set_weigths[68])]+layer_68.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_68, set_weigths[68])]
		layer_68.set_weights(new_w)


try:
	count = 1
	for s in layer_68_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #68:',count)
except AttributeError:
	for i in layer_68_out:
		count = 1
		for s in layer_68_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #68:',count)



layer_69 = L69()
try:
	try:
		layer_69_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_69_out = layer_69(layer_69_concat)
	except ValueError:
		layer_69_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_69_out = layer_69(layer_69_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_69 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_69_concat = strong_concat_69(new_list_of_inputs)
	layer_69_out = layer_69(layer_69_concat)
if layer_69.count_params() != 0:
	if 69 in set_weigths:
		if len(layer_69.get_weights()) > 1:
			new_w = [get_new_weigths(layer_69, set_weigths[69])]+layer_69.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_69, set_weigths[69])]
		layer_69.set_weights(new_w)


try:
	count = 1
	for s in layer_69_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #69:',count)
except AttributeError:
	for i in layer_69_out:
		count = 1
		for s in layer_69_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #69:',count)



layer_70 = L70()
try:
	try:
		layer_70_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_70_out = layer_70(layer_70_concat)
	except ValueError:
		layer_70_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_70_out = layer_70(layer_70_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_70 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_70_concat = strong_concat_70(new_list_of_inputs)
	layer_70_out = layer_70(layer_70_concat)
if layer_70.count_params() != 0:
	if 70 in set_weigths:
		if len(layer_70.get_weights()) > 1:
			new_w = [get_new_weigths(layer_70, set_weigths[70])]+layer_70.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_70, set_weigths[70])]
		layer_70.set_weights(new_w)


try:
	count = 1
	for s in layer_70_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #70:',count)
except AttributeError:
	for i in layer_70_out:
		count = 1
		for s in layer_70_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #70:',count)



layer_71 = L71()
try:
	try:
		layer_71_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_71_out = layer_71(layer_71_concat)
	except ValueError:
		layer_71_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_71_out = layer_71(layer_71_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_71 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_71_concat = strong_concat_71(new_list_of_inputs)
	layer_71_out = layer_71(layer_71_concat)
if layer_71.count_params() != 0:
	if 71 in set_weigths:
		if len(layer_71.get_weights()) > 1:
			new_w = [get_new_weigths(layer_71, set_weigths[71])]+layer_71.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_71, set_weigths[71])]
		layer_71.set_weights(new_w)


try:
	count = 1
	for s in layer_71_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #71:',count)
except AttributeError:
	for i in layer_71_out:
		count = 1
		for s in layer_71_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #71:',count)



layer_72 = L72()
try:
	try:
		layer_72_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_72_out = layer_72(layer_72_concat)
	except ValueError:
		layer_72_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_72_out = layer_72(layer_72_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_72 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_72_concat = strong_concat_72(new_list_of_inputs)
	layer_72_out = layer_72(layer_72_concat)
if layer_72.count_params() != 0:
	if 72 in set_weigths:
		if len(layer_72.get_weights()) > 1:
			new_w = [get_new_weigths(layer_72, set_weigths[72])]+layer_72.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_72, set_weigths[72])]
		layer_72.set_weights(new_w)


try:
	count = 1
	for s in layer_72_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #72:',count)
except AttributeError:
	for i in layer_72_out:
		count = 1
		for s in layer_72_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #72:',count)



layer_73 = L73()
try:
	try:
		layer_73_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_73_out = layer_73(layer_73_concat)
	except ValueError:
		layer_73_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_73_out = layer_73(layer_73_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_73 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_73_concat = strong_concat_73(new_list_of_inputs)
	layer_73_out = layer_73(layer_73_concat)
if layer_73.count_params() != 0:
	if 73 in set_weigths:
		if len(layer_73.get_weights()) > 1:
			new_w = [get_new_weigths(layer_73, set_weigths[73])]+layer_73.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_73, set_weigths[73])]
		layer_73.set_weights(new_w)


try:
	count = 1
	for s in layer_73_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #73:',count)
except AttributeError:
	for i in layer_73_out:
		count = 1
		for s in layer_73_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #73:',count)



layer_74 = L74()
try:
	try:
		layer_74_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_74_out = layer_74(layer_74_concat)
	except ValueError:
		layer_74_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_74_out = layer_74(layer_74_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_74 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_74_concat = strong_concat_74(new_list_of_inputs)
	layer_74_out = layer_74(layer_74_concat)
if layer_74.count_params() != 0:
	if 74 in set_weigths:
		if len(layer_74.get_weights()) > 1:
			new_w = [get_new_weigths(layer_74, set_weigths[74])]+layer_74.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_74, set_weigths[74])]
		layer_74.set_weights(new_w)


try:
	count = 1
	for s in layer_74_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #74:',count)
except AttributeError:
	for i in layer_74_out:
		count = 1
		for s in layer_74_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #74:',count)



layer_75 = L75()
try:
	try:
		layer_75_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_75_out = layer_75(layer_75_concat)
	except ValueError:
		layer_75_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_75_out = layer_75(layer_75_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_75 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
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
		layer_76_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_76_out = layer_76(layer_76_concat)
	except ValueError:
		layer_76_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_76_out = layer_76(layer_76_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_76 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
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
		layer_77_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_77_out = layer_77(layer_77_concat)
	except ValueError:
		layer_77_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_77_out = layer_77(layer_77_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_77 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
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
		layer_78_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_78_out = layer_78(layer_78_concat)
	except ValueError:
		layer_78_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_78_out = layer_78(layer_78_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_78 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
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
		layer_79_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_79_out = layer_79(layer_79_concat)
	except ValueError:
		layer_79_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_79_out = layer_79(layer_79_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_79 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
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
		layer_80_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out])
		layer_80_out = layer_80(layer_80_concat)
	except ValueError:
		layer_80_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_80_out = layer_80(layer_80_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_80 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_82_out, layer_83_out, layer_84_out, layer_85_out]:
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



layer_82 = L82()
try:
	try:
		layer_82_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_83_out, layer_84_out, layer_85_out])
		layer_82_out = layer_82(layer_82_concat)
	except ValueError:
		layer_82_concat = tf.keras.layers.concatenate([layer_1_out, layer_81_out, layer_83_out, layer_84_out, layer_85_out], axis=-2)
		layer_82_out = layer_82(layer_82_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_82 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_81_out, layer_83_out, layer_84_out, layer_85_out]:
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



layer_0_out = tf.keras.layers.Flatten()(layer_0_out)
layer_0_out = tf.keras.layers.Dense(10, activation='sigmoid')(layer_0_out)
model = tf.keras.Model(
    inputs=[layer_1_out],
    outputs=[layer_0_out],
)


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

	