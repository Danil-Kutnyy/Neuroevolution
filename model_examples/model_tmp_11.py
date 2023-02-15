
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

	
class L105(tf.keras.layers.Layer):
	def __init__(self):
		super(L105, self).__init__()
		self.units = 44

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
	
class L106(tf.keras.layers.Layer):
	def __init__(self):
		super(L106, self).__init__()
		self.units = 44

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
		return tf.keras.activations.swish( tf.matmul(inputs, self.w) ) 
			
class L107(tf.keras.layers.Layer):
	def __init__(self):
		super(L107, self).__init__()

	def call(self, inputs):
		return (tf.math.negative(inputs))
			
class L108(tf.keras.layers.Layer):
	def __init__(self):
		super(L108, self).__init__()
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
	
class L109(tf.keras.layers.Layer):
	def __init__(self):
		super(L109, self).__init__()
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
		return inputs
			
class L21(tf.keras.layers.Layer):
	def __init__(self):
		super(L21, self).__init__()
	def call(self, inputs):
		return inputs
			
class L22(tf.keras.layers.Layer):
	def __init__(self):
		super(L22, self).__init__()
	def call(self, inputs):
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
			
class L29(tf.keras.layers.Layer):
	def __init__(self):
		super(L29, self).__init__()

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
			
class L31(tf.keras.layers.Layer):
	def __init__(self):
		super(L31, self).__init__()

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
			
class L32(tf.keras.layers.Layer):
	def __init__(self):
		super(L32, self).__init__()

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
			
class L33(tf.keras.layers.Layer):
	def __init__(self):
		super(L33, self).__init__()

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
			
class L34(tf.keras.layers.Layer):
	def __init__(self):
		super(L34, self).__init__()

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
			
class L35(tf.keras.layers.Layer):
	def __init__(self):
		super(L35, self).__init__()

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
			
class L36(tf.keras.layers.Layer):
	def __init__(self):
		super(L36, self).__init__()

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
			
class L38(tf.keras.layers.Layer):
	def __init__(self):
		super(L38, self).__init__()

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
			
class L39(tf.keras.layers.Layer):
	def __init__(self):
		super(L39, self).__init__()

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
			
class L40(tf.keras.layers.Layer):
	def __init__(self):
		super(L40, self).__init__()

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
			
class L41(tf.keras.layers.Layer):
	def __init__(self):
		super(L41, self).__init__()

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
			
class L43(tf.keras.layers.Layer):
	def __init__(self):
		super(L43, self).__init__()

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
			
class L44(tf.keras.layers.Layer):
	def __init__(self):
		super(L44, self).__init__()
	def call(self, inputs):
		return inputs
			
class L45(tf.keras.layers.Layer):
	def __init__(self):
		super(L45, self).__init__()
	def call(self, inputs):
		return inputs
			
class L46(tf.keras.layers.Layer):
	def __init__(self):
		super(L46, self).__init__()
	def build(self, input_shape):
		self.i_layer = tf.keras.layers.Dense( units=18 )
	def call(self, inputs):
		return self.i_layer(inputs)
	
class L47(tf.keras.layers.Layer):
	def __init__(self):
		super(L47, self).__init__()
	def build(self, input_shape):
		self.i_layer = tf.keras.layers.Dense( units=18 )
	def call(self, inputs):
		return self.i_layer(inputs)
	
class L48(tf.keras.layers.Layer):
	def __init__(self):
		super(L48, self).__init__()
	def build(self, input_shape):
		self.i_layer = tf.keras.layers.LeakyReLU( alpha=0.352 )
	def call(self, inputs):
		return self.i_layer(inputs)
	
class L49(tf.keras.layers.Layer):
	def __init__(self):
		super(L49, self).__init__()
	def build(self, input_shape):
		self.i_layer = tf.keras.layers.LeakyReLU( alpha=0.352 )
	def call(self, inputs):
		return self.i_layer(inputs)
	
class L50(tf.keras.layers.Layer):
	def __init__(self):
		super(L50, self).__init__()
	def build(self, input_shape):
		self.i_layer = tf.keras.layers.LeakyReLU( alpha=0.352 )
	def call(self, inputs):
		return self.i_layer(inputs)
	
class L51(tf.keras.layers.Layer):
	def __init__(self):
		super(L51, self).__init__()
	def build(self, input_shape):
		self.i_layer = tf.keras.layers.LeakyReLU( alpha=0.352 )
	def call(self, inputs):
		return self.i_layer(inputs)
	
class L52(tf.keras.layers.Layer):
	def __init__(self):
		super(L52, self).__init__()

	def call(self, inputs):
		return ((inputs))
			
class L53(tf.keras.layers.Layer):
	def __init__(self):
		super(L53, self).__init__()

	def call(self, inputs):
		return ((inputs))
			
class L54(tf.keras.layers.Layer):
	def __init__(self):
		super(L54, self).__init__()

	def call(self, inputs):
		return ((inputs))
			
class L55(tf.keras.layers.Layer):
	def __init__(self):
		super(L55, self).__init__()

	def call(self, inputs):
		return ((inputs))
			
class L56(tf.keras.layers.Layer):
	def __init__(self):
		super(L56, self).__init__()

	def call(self, inputs):
		return ((inputs))
			
class L57(tf.keras.layers.Layer):
	def __init__(self):
		super(L57, self).__init__()

	def call(self, inputs):
		return ((inputs))
			
class L58(tf.keras.layers.Layer):
	def __init__(self):
		super(L58, self).__init__()

	def call(self, inputs):
		return ((inputs))
			
class L59(tf.keras.layers.Layer):
	def __init__(self):
		super(L59, self).__init__()

	def call(self, inputs):
		return ((inputs))
			
class L60(tf.keras.layers.Layer):
	def __init__(self):
		super(L60, self).__init__()

	def call(self, inputs):
		return ((inputs))
			
class L61(tf.keras.layers.Layer):
	def __init__(self):
		super(L61, self).__init__()
	def call(self, inputs):
		return inputs
			
class L62(tf.keras.layers.Layer):
	def __init__(self):
		super(L62, self).__init__()
	def call(self, inputs):
		return inputs
			
class L63(tf.keras.layers.Layer):
	def __init__(self):
		super(L63, self).__init__()
	def call(self, inputs):
		return inputs
			
class L64(tf.keras.layers.Layer):
	def __init__(self):
		super(L64, self).__init__()
	def call(self, inputs):
		return inputs
			
class L65(tf.keras.layers.Layer):
	def __init__(self):
		super(L65, self).__init__()
	def call(self, inputs):
		return inputs
			
class L66(tf.keras.layers.Layer):
	def __init__(self):
		super(L66, self).__init__()
	def call(self, inputs):
		return inputs
			
class L67(tf.keras.layers.Layer):
	def __init__(self):
		super(L67, self).__init__()
	def call(self, inputs):
		return inputs
			
class L68(tf.keras.layers.Layer):
	def __init__(self):
		super(L68, self).__init__()
	def call(self, inputs):
		return inputs
			
class L69(tf.keras.layers.Layer):
	def __init__(self):
		super(L69, self).__init__()
	def call(self, inputs):
		return inputs
			
class L70(tf.keras.layers.Layer):
	def __init__(self):
		super(L70, self).__init__()
	def call(self, inputs):
		return inputs
			
class L71(tf.keras.layers.Layer):
	def __init__(self):
		super(L71, self).__init__()
	def call(self, inputs):
		return inputs
			
class L72(tf.keras.layers.Layer):
	def __init__(self):
		super(L72, self).__init__()
	def call(self, inputs):
		return inputs
			
class L73(tf.keras.layers.Layer):
	def __init__(self):
		super(L73, self).__init__()
	def call(self, inputs):
		return inputs
			
class L74(tf.keras.layers.Layer):
	def __init__(self):
		super(L74, self).__init__()

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
			
class L75(tf.keras.layers.Layer):
	def __init__(self):
		super(L75, self).__init__()

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
			
class L76(tf.keras.layers.Layer):
	def __init__(self):
		super(L76, self).__init__()

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
			
class L77(tf.keras.layers.Layer):
	def __init__(self):
		super(L77, self).__init__()

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
			
class L78(tf.keras.layers.Layer):
	def __init__(self):
		super(L78, self).__init__()

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
			
class L79(tf.keras.layers.Layer):
	def __init__(self):
		super(L79, self).__init__()

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
			
class L80(tf.keras.layers.Layer):
	def __init__(self):
		super(L80, self).__init__()

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
			
class L81(tf.keras.layers.Layer):
	def __init__(self):
		super(L81, self).__init__()

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
			
class L82(tf.keras.layers.Layer):
	def __init__(self):
		super(L82, self).__init__()

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
			
class L83(tf.keras.layers.Layer):
	def __init__(self):
		super(L83, self).__init__()

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
			
class L84(tf.keras.layers.Layer):
	def __init__(self):
		super(L84, self).__init__()

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
			
class L85(tf.keras.layers.Layer):
	def __init__(self):
		super(L85, self).__init__()

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
			
class L86(tf.keras.layers.Layer):
	def __init__(self):
		super(L86, self).__init__()

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
			
class L87(tf.keras.layers.Layer):
	def __init__(self):
		super(L87, self).__init__()

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
			
class L88(tf.keras.layers.Layer):
	def __init__(self):
		super(L88, self).__init__()

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
			
class L89(tf.keras.layers.Layer):
	def __init__(self):
		super(L89, self).__init__()

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
			
class L90(tf.keras.layers.Layer):
	def __init__(self):
		super(L90, self).__init__()
	def call(self, inputs):
		return inputs
			
class L91(tf.keras.layers.Layer):
	def __init__(self):
		super(L91, self).__init__()
	def call(self, inputs):
		return inputs
			
class L92(tf.keras.layers.Layer):
	def __init__(self):
		super(L92, self).__init__()
	def build(self, input_shape):
		self.i_layer = tf.keras.layers.Dense( units=18 )
	def call(self, inputs):
		return self.i_layer(inputs)
	
class L93(tf.keras.layers.Layer):
	def __init__(self):
		super(L93, self).__init__()
	def build(self, input_shape):
		self.i_layer = tf.keras.layers.Dense( units=18 )
	def call(self, inputs):
		return self.i_layer(inputs)
	
class L94(tf.keras.layers.Layer):
	def __init__(self):
		super(L94, self).__init__()
	def build(self, input_shape):
		self.i_layer = tf.keras.layers.LeakyReLU( alpha=0.352 )
	def call(self, inputs):
		return self.i_layer(inputs)
	
class L95(tf.keras.layers.Layer):
	def __init__(self):
		super(L95, self).__init__()
	def build(self, input_shape):
		self.i_layer = tf.keras.layers.LeakyReLU( alpha=0.352 )
	def call(self, inputs):
		return self.i_layer(inputs)
	
class L96(tf.keras.layers.Layer):
	def __init__(self):
		super(L96, self).__init__()
	def build(self, input_shape):
		self.i_layer = tf.keras.layers.LeakyReLU( alpha=0.352 )
	def call(self, inputs):
		return self.i_layer(inputs)
	
class L97(tf.keras.layers.Layer):
	def __init__(self):
		super(L97, self).__init__()
	def build(self, input_shape):
		self.i_layer = tf.keras.layers.LeakyReLU( alpha=0.352 )
	def call(self, inputs):
		return self.i_layer(inputs)
	
class L98(tf.keras.layers.Layer):
	def __init__(self):
		super(L98, self).__init__()
	def call(self, inputs):
		return inputs
			
class L99(tf.keras.layers.Layer):
	def __init__(self):
		super(L99, self).__init__()
	def build(self, input_shape):
		self.i_layer = tf.keras.layers.ThresholdedReLU( theta=0.002 )
	def call(self, inputs):
		return self.i_layer(inputs)
	
class L100(tf.keras.layers.Layer):
	def __init__(self):
		super(L100, self).__init__()
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
	
class L101(tf.keras.layers.Layer):
	def __init__(self):
		super(L101, self).__init__()
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
	
class L102(tf.keras.layers.Layer):
	def __init__(self):
		super(L102, self).__init__()
	def build(self, input_shape):
		self.i_layer = tf.keras.layers.ThresholdedReLU( theta=0.002 )
	def call(self, inputs):
		return self.i_layer(inputs)
	
class L103(tf.keras.layers.Layer):
	def __init__(self):
		super(L103, self).__init__()
	def build(self, input_shape):
		self.i_layer = tf.keras.layers.ThresholdedReLU( theta=0.002 )
	def call(self, inputs):
		return self.i_layer(inputs)
	
class L104(tf.keras.layers.Layer):
	def __init__(self):
		super(L104, self).__init__()
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
	
class L110(tf.keras.layers.Layer):
	def __init__(self):
		super(L110, self).__init__()
	def call(self, inputs):
		return inputs
			
class L111(tf.keras.layers.Layer):
	def __init__(self):
		super(L111, self).__init__()
	def call(self, inputs):
		return inputs
			
class L112(tf.keras.layers.Layer):
	def __init__(self):
		super(L112, self).__init__()
	def call(self, inputs):
		return inputs
			
class L113(tf.keras.layers.Layer):
	def __init__(self):
		super(L113, self).__init__()
	def call(self, inputs):
		return inputs
			
class L114(tf.keras.layers.Layer):
	def __init__(self):
		super(L114, self).__init__()
	def call(self, inputs):
		return inputs
			
class L115(tf.keras.layers.Layer):
	def __init__(self):
		super(L115, self).__init__()
	def call(self, inputs):
		return inputs
			
class L116(tf.keras.layers.Layer):
	def __init__(self):
		super(L116, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L1L2(l1=0.0, l2=0.0)
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
		return (tf.math.divide_no_nan(inputs, self.w))
			
class L117(tf.keras.layers.Layer):
	def __init__(self):
		super(L117, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L1L2(l1=0.0, l2=0.0)
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
		return (tf.math.divide_no_nan(inputs, self.w))
			
class L118(tf.keras.layers.Layer):
	def __init__(self):
		super(L118, self).__init__()
	def call(self, inputs):
		return inputs
			
class L119(tf.keras.layers.Layer):
	def __init__(self):
		super(L119, self).__init__()

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
			
class L168(tf.keras.layers.Layer):
	def __init__(self):
		super(L168, self).__init__()

	def call(self, inputs):
		return ((inputs))
			
class L169(tf.keras.layers.Layer):
	def __init__(self):
		super(L169, self).__init__()

	def call(self, inputs):
		return ((inputs))
			
class L170(tf.keras.layers.Layer):
	def __init__(self):
		super(L170, self).__init__()

	def call(self, inputs):
		return ((inputs))
			
class L171(tf.keras.layers.Layer):
	def __init__(self):
		super(L171, self).__init__()

	def call(self, inputs):
		return ((inputs))
			
class L172(tf.keras.layers.Layer):
	def __init__(self):
		super(L172, self).__init__()

	def call(self, inputs):
		return ((inputs))
			
class L173(tf.keras.layers.Layer):
	def __init__(self):
		super(L173, self).__init__()

	def call(self, inputs):
		return ((inputs))
			
class L174(tf.keras.layers.Layer):
	def __init__(self):
		super(L174, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L1L2(l1=0.0, l2=0.0)
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
		return (tf.math.divide_no_nan(inputs, self.w))
			
class L175(tf.keras.layers.Layer):
	def __init__(self):
		super(L175, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L1L2(l1=0.0, l2=0.0)
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
		return (tf.math.divide_no_nan(inputs, self.w))
			
class L176(tf.keras.layers.Layer):
	def __init__(self):
		super(L176, self).__init__()
	def call(self, inputs):
		return inputs
			
class L177(tf.keras.layers.Layer):
	def __init__(self):
		super(L177, self).__init__()

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
			
class L226(tf.keras.layers.Layer):
	def __init__(self):
		super(L226, self).__init__()
	def call(self, inputs):
		return inputs
			
class L227(tf.keras.layers.Layer):
	def __init__(self):
		super(L227, self).__init__()
	def call(self, inputs):
		return inputs
			
class L228(tf.keras.layers.Layer):
	def __init__(self):
		super(L228, self).__init__()
	def call(self, inputs):
		return inputs
			
class L229(tf.keras.layers.Layer):
	def __init__(self):
		super(L229, self).__init__()
	def call(self, inputs):
		return inputs
			
class L230(tf.keras.layers.Layer):
	def __init__(self):
		super(L230, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L1L2(l1=0.0, l2=0.0)
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
		return (tf.math.divide_no_nan(inputs, self.w))
			
class L231(tf.keras.layers.Layer):
	def __init__(self):
		super(L231, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L1L2(l1=0.0, l2=0.0)
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
		return (tf.math.divide_no_nan(inputs, self.w))
			
class L232(tf.keras.layers.Layer):
	def __init__(self):
		super(L232, self).__init__()
	def call(self, inputs):
		return inputs
			
class L233(tf.keras.layers.Layer):
	def __init__(self):
		super(L233, self).__init__()

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
			
class L282(tf.keras.layers.Layer):
	def __init__(self):
		super(L282, self).__init__()

	def call(self, inputs):
		return ((inputs))
			
class L283(tf.keras.layers.Layer):
	def __init__(self):
		super(L283, self).__init__()

	def call(self, inputs):
		return ((inputs))
			
class L284(tf.keras.layers.Layer):
	def __init__(self):
		super(L284, self).__init__()

	def call(self, inputs):
		return ((inputs))
			
class L285(tf.keras.layers.Layer):
	def __init__(self):
		super(L285, self).__init__()

	def call(self, inputs):
		return ((inputs))
			
class L286(tf.keras.layers.Layer):
	def __init__(self):
		super(L286, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L1L2(l1=0.0, l2=0.0)
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
		return (tf.math.divide_no_nan(inputs, self.w))
			
class L287(tf.keras.layers.Layer):
	def __init__(self):
		super(L287, self).__init__()

	def build(self, input_shape):
		initializer = tf.keras.initializers.Ones()
		if initializer == None:
			initializer = "random_normal"
		regularizer = tf.keras.regularizers.L1L2(l1=0.0, l2=0.0)
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
		return (tf.math.divide_no_nan(inputs, self.w))
			
class L288(tf.keras.layers.Layer):
	def __init__(self):
		super(L288, self).__init__()
	def call(self, inputs):
		return inputs
			
class L289(tf.keras.layers.Layer):
	def __init__(self):
		super(L289, self).__init__()

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
		

layer_108 = L108()
layer_108_out = layer_108(layer_1_out)

layer_109 = L109()
try:
	try:
		layer_109_concat = tf.keras.layers.concatenate([layer_1_out, layer_108_out])
		layer_109_out = layer_109(layer_109_concat)
	except ValueError:
		layer_109_concat = tf.keras.layers.concatenate([layer_1_out, layer_108_out], axis=-2)
		layer_109_out = layer_109(layer_109_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_109 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_108_out]:
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



layer_107 = L107()
try:
	try:
		layer_107_concat = tf.keras.layers.concatenate([layer_1_out, layer_108_out, layer_109_out])
		layer_107_out = layer_107(layer_107_concat)
	except ValueError:
		layer_107_concat = tf.keras.layers.concatenate([layer_1_out, layer_108_out, layer_109_out], axis=-2)
		layer_107_out = layer_107(layer_107_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_107 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_108_out, layer_109_out]:
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



layer_109 = L109()
try:
	try:
		layer_109_concat = tf.keras.layers.concatenate([layer_1_out, layer_108_out])
		layer_109_out = layer_109(layer_109_concat)
	except ValueError:
		layer_109_concat = tf.keras.layers.concatenate([layer_1_out, layer_108_out], axis=-2)
		layer_109_out = layer_109(layer_109_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_109 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_108_out]:
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



layer_106 = L106()
try:
	try:
		layer_106_concat = tf.keras.layers.concatenate([layer_1_out, layer_107_out, layer_108_out, layer_109_out])
		layer_106_out = layer_106(layer_106_concat)
	except ValueError:
		layer_106_concat = tf.keras.layers.concatenate([layer_1_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_106_out = layer_106(layer_106_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_106 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_107_out, layer_108_out, layer_109_out]:
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



layer_110 = L110()
try:
	try:
		layer_110_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_110_out = layer_110(layer_110_concat)
	except ValueError:
		layer_110_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_110_out = layer_110(layer_110_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_110 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_111_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_111_out = layer_111(layer_111_concat)
	except ValueError:
		layer_111_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_111_out = layer_111(layer_111_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_111 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_112_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_112_out = layer_112(layer_112_concat)
	except ValueError:
		layer_112_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_112_out = layer_112(layer_112_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_112 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_113_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_113_out = layer_113(layer_113_concat)
	except ValueError:
		layer_113_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_113_out = layer_113(layer_113_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_113 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_114_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_114_out = layer_114(layer_114_concat)
	except ValueError:
		layer_114_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_114_out = layer_114(layer_114_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_114 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_115_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_115_out = layer_115(layer_115_concat)
	except ValueError:
		layer_115_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_115_out = layer_115(layer_115_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_115 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_116_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_116_out = layer_116(layer_116_concat)
	except ValueError:
		layer_116_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_116_out = layer_116(layer_116_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_116 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_117_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_117_out = layer_117(layer_117_concat)
	except ValueError:
		layer_117_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_117_out = layer_117(layer_117_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_117 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_118_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_118_out = layer_118(layer_118_concat)
	except ValueError:
		layer_118_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_118_out = layer_118(layer_118_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_118 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_119_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_119_out = layer_119(layer_119_concat)
	except ValueError:
		layer_119_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_119_out = layer_119(layer_119_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_119 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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



layer_168 = L168()
try:
	try:
		layer_168_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_168_out = layer_168(layer_168_concat)
	except ValueError:
		layer_168_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_168_out = layer_168(layer_168_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_168 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_169_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_169_out = layer_169(layer_169_concat)
	except ValueError:
		layer_169_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_169_out = layer_169(layer_169_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_169 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_170_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_170_out = layer_170(layer_170_concat)
	except ValueError:
		layer_170_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_170_out = layer_170(layer_170_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_170 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_171_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_171_out = layer_171(layer_171_concat)
	except ValueError:
		layer_171_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_171_out = layer_171(layer_171_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_171 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_172_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_172_out = layer_172(layer_172_concat)
	except ValueError:
		layer_172_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_172_out = layer_172(layer_172_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_172 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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



layer_173 = L173()
try:
	try:
		layer_173_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_173_out = layer_173(layer_173_concat)
	except ValueError:
		layer_173_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_173_out = layer_173(layer_173_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_173 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_174_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_174_out = layer_174(layer_174_concat)
	except ValueError:
		layer_174_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_174_out = layer_174(layer_174_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_174 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_175_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_175_out = layer_175(layer_175_concat)
	except ValueError:
		layer_175_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_175_out = layer_175(layer_175_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_175 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_176_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_176_out = layer_176(layer_176_concat)
	except ValueError:
		layer_176_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_176_out = layer_176(layer_176_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_176 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_177_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_177_out = layer_177(layer_177_concat)
	except ValueError:
		layer_177_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_177_out = layer_177(layer_177_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_177 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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



layer_226 = L226()
try:
	try:
		layer_226_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_226_out = layer_226(layer_226_concat)
	except ValueError:
		layer_226_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_226_out = layer_226(layer_226_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_226 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_226_concat = strong_concat_226(new_list_of_inputs)
	layer_226_out = layer_226(layer_226_concat)
if layer_226.count_params() != 0:
	if 226 in set_weigths:
		if len(layer_226.get_weights()) > 1:
			new_w = [get_new_weigths(layer_226, set_weigths[226])]+layer_226.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_226, set_weigths[226])]
		layer_226.set_weights(new_w)


try:
	count = 1
	for s in layer_226_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #226:',count)
except AttributeError:
	for i in layer_226_out:
		count = 1
		for s in layer_226_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #226:',count)



layer_227 = L227()
try:
	try:
		layer_227_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_227_out = layer_227(layer_227_concat)
	except ValueError:
		layer_227_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_227_out = layer_227(layer_227_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_227 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_227_concat = strong_concat_227(new_list_of_inputs)
	layer_227_out = layer_227(layer_227_concat)
if layer_227.count_params() != 0:
	if 227 in set_weigths:
		if len(layer_227.get_weights()) > 1:
			new_w = [get_new_weigths(layer_227, set_weigths[227])]+layer_227.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_227, set_weigths[227])]
		layer_227.set_weights(new_w)


try:
	count = 1
	for s in layer_227_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #227:',count)
except AttributeError:
	for i in layer_227_out:
		count = 1
		for s in layer_227_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #227:',count)



layer_228 = L228()
try:
	try:
		layer_228_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_228_out = layer_228(layer_228_concat)
	except ValueError:
		layer_228_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_228_out = layer_228(layer_228_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_228 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_228_concat = strong_concat_228(new_list_of_inputs)
	layer_228_out = layer_228(layer_228_concat)
if layer_228.count_params() != 0:
	if 228 in set_weigths:
		if len(layer_228.get_weights()) > 1:
			new_w = [get_new_weigths(layer_228, set_weigths[228])]+layer_228.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_228, set_weigths[228])]
		layer_228.set_weights(new_w)


try:
	count = 1
	for s in layer_228_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #228:',count)
except AttributeError:
	for i in layer_228_out:
		count = 1
		for s in layer_228_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #228:',count)



layer_229 = L229()
try:
	try:
		layer_229_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_229_out = layer_229(layer_229_concat)
	except ValueError:
		layer_229_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_229_out = layer_229(layer_229_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_229 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_229_concat = strong_concat_229(new_list_of_inputs)
	layer_229_out = layer_229(layer_229_concat)
if layer_229.count_params() != 0:
	if 229 in set_weigths:
		if len(layer_229.get_weights()) > 1:
			new_w = [get_new_weigths(layer_229, set_weigths[229])]+layer_229.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_229, set_weigths[229])]
		layer_229.set_weights(new_w)


try:
	count = 1
	for s in layer_229_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #229:',count)
except AttributeError:
	for i in layer_229_out:
		count = 1
		for s in layer_229_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #229:',count)



layer_230 = L230()
try:
	try:
		layer_230_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_230_out = layer_230(layer_230_concat)
	except ValueError:
		layer_230_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_230_out = layer_230(layer_230_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_230 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_230_concat = strong_concat_230(new_list_of_inputs)
	layer_230_out = layer_230(layer_230_concat)
if layer_230.count_params() != 0:
	if 230 in set_weigths:
		if len(layer_230.get_weights()) > 1:
			new_w = [get_new_weigths(layer_230, set_weigths[230])]+layer_230.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_230, set_weigths[230])]
		layer_230.set_weights(new_w)


try:
	count = 1
	for s in layer_230_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #230:',count)
except AttributeError:
	for i in layer_230_out:
		count = 1
		for s in layer_230_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #230:',count)



layer_231 = L231()
try:
	try:
		layer_231_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_231_out = layer_231(layer_231_concat)
	except ValueError:
		layer_231_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_231_out = layer_231(layer_231_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_231 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_231_concat = strong_concat_231(new_list_of_inputs)
	layer_231_out = layer_231(layer_231_concat)
if layer_231.count_params() != 0:
	if 231 in set_weigths:
		if len(layer_231.get_weights()) > 1:
			new_w = [get_new_weigths(layer_231, set_weigths[231])]+layer_231.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_231, set_weigths[231])]
		layer_231.set_weights(new_w)


try:
	count = 1
	for s in layer_231_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #231:',count)
except AttributeError:
	for i in layer_231_out:
		count = 1
		for s in layer_231_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #231:',count)



layer_232 = L232()
try:
	try:
		layer_232_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_232_out = layer_232(layer_232_concat)
	except ValueError:
		layer_232_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_232_out = layer_232(layer_232_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_232 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_232_concat = strong_concat_232(new_list_of_inputs)
	layer_232_out = layer_232(layer_232_concat)
if layer_232.count_params() != 0:
	if 232 in set_weigths:
		if len(layer_232.get_weights()) > 1:
			new_w = [get_new_weigths(layer_232, set_weigths[232])]+layer_232.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_232, set_weigths[232])]
		layer_232.set_weights(new_w)


try:
	count = 1
	for s in layer_232_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #232:',count)
except AttributeError:
	for i in layer_232_out:
		count = 1
		for s in layer_232_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #232:',count)



layer_233 = L233()
try:
	try:
		layer_233_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_233_out = layer_233(layer_233_concat)
	except ValueError:
		layer_233_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_233_out = layer_233(layer_233_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_233 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_233_concat = strong_concat_233(new_list_of_inputs)
	layer_233_out = layer_233(layer_233_concat)
if layer_233.count_params() != 0:
	if 233 in set_weigths:
		if len(layer_233.get_weights()) > 1:
			new_w = [get_new_weigths(layer_233, set_weigths[233])]+layer_233.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_233, set_weigths[233])]
		layer_233.set_weights(new_w)


try:
	count = 1
	for s in layer_233_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #233:',count)
except AttributeError:
	for i in layer_233_out:
		count = 1
		for s in layer_233_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #233:',count)



layer_282 = L282()
try:
	try:
		layer_282_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_282_out = layer_282(layer_282_concat)
	except ValueError:
		layer_282_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_282_out = layer_282(layer_282_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_282 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_283_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_283_out = layer_283(layer_283_concat)
	except ValueError:
		layer_283_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_283_out = layer_283(layer_283_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_283 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_284_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_284_out = layer_284(layer_284_concat)
	except ValueError:
		layer_284_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_284_out = layer_284(layer_284_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_284 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_285_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_285_out = layer_285(layer_285_concat)
	except ValueError:
		layer_285_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_285_out = layer_285(layer_285_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_285 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_286_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_286_out = layer_286(layer_286_concat)
	except ValueError:
		layer_286_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_286_out = layer_286(layer_286_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_286 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_287_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_287_out = layer_287(layer_287_concat)
	except ValueError:
		layer_287_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_287_out = layer_287(layer_287_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_287 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_288_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_288_out = layer_288(layer_288_concat)
	except ValueError:
		layer_288_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_288_out = layer_288(layer_288_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_288 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_289_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_289_out = layer_289(layer_289_concat)
	except ValueError:
		layer_289_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_289_out = layer_289(layer_289_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_289 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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



layer_0 = L0()
try:
	try:
		layer_0_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_0_out = layer_0(layer_0_concat)
	except ValueError:
		layer_0_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_0_out = layer_0(layer_0_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_0 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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



layer_10 = L10()
try:
	try:
		layer_10_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_10_out = layer_10(layer_10_concat)
	except ValueError:
		layer_10_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_10_out = layer_10(layer_10_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_10 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_11_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_11_out = layer_11(layer_11_concat)
	except ValueError:
		layer_11_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_11_out = layer_11(layer_11_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_11 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_12_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_12_out = layer_12(layer_12_concat)
	except ValueError:
		layer_12_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_12_out = layer_12(layer_12_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_12 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_13_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_13_out = layer_13(layer_13_concat)
	except ValueError:
		layer_13_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_13_out = layer_13(layer_13_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_13 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_14_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_14_out = layer_14(layer_14_concat)
	except ValueError:
		layer_14_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_14_out = layer_14(layer_14_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_14 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_15_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_15_out = layer_15(layer_15_concat)
	except ValueError:
		layer_15_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_15_out = layer_15(layer_15_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_15 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_16_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_16_out = layer_16(layer_16_concat)
	except ValueError:
		layer_16_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_16_out = layer_16(layer_16_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_16 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_17_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_17_out = layer_17(layer_17_concat)
	except ValueError:
		layer_17_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_17_out = layer_17(layer_17_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_17 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_18_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_18_out = layer_18(layer_18_concat)
	except ValueError:
		layer_18_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_18_out = layer_18(layer_18_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_18 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_19_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_19_out = layer_19(layer_19_concat)
	except ValueError:
		layer_19_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_19_out = layer_19(layer_19_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_19 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_20_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_20_out = layer_20(layer_20_concat)
	except ValueError:
		layer_20_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_20_out = layer_20(layer_20_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_20 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_21_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_21_out = layer_21(layer_21_concat)
	except ValueError:
		layer_21_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_21_out = layer_21(layer_21_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_21 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_22_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_22_out = layer_22(layer_22_concat)
	except ValueError:
		layer_22_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_22_out = layer_22(layer_22_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_22 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_23_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_23_out = layer_23(layer_23_concat)
	except ValueError:
		layer_23_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_23_out = layer_23(layer_23_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_23 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_24_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_24_out = layer_24(layer_24_concat)
	except ValueError:
		layer_24_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_24_out = layer_24(layer_24_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_24 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_25_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_25_out = layer_25(layer_25_concat)
	except ValueError:
		layer_25_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_25_out = layer_25(layer_25_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_25 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_26_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_26_out = layer_26(layer_26_concat)
	except ValueError:
		layer_26_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_26_out = layer_26(layer_26_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_26 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_27_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_27_out = layer_27(layer_27_concat)
	except ValueError:
		layer_27_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_27_out = layer_27(layer_27_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_27 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_28_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_28_out = layer_28(layer_28_concat)
	except ValueError:
		layer_28_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_28_out = layer_28(layer_28_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_28 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_29_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_29_out = layer_29(layer_29_concat)
	except ValueError:
		layer_29_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_29_out = layer_29(layer_29_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_29 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_30_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_30_out = layer_30(layer_30_concat)
	except ValueError:
		layer_30_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_30_out = layer_30(layer_30_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_30 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_31_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_31_out = layer_31(layer_31_concat)
	except ValueError:
		layer_31_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_31_out = layer_31(layer_31_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_31 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_32_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_32_out = layer_32(layer_32_concat)
	except ValueError:
		layer_32_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_32_out = layer_32(layer_32_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_32 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_33_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_33_out = layer_33(layer_33_concat)
	except ValueError:
		layer_33_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_33_out = layer_33(layer_33_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_33 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_34_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_34_out = layer_34(layer_34_concat)
	except ValueError:
		layer_34_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_34_out = layer_34(layer_34_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_34 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_35_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_35_out = layer_35(layer_35_concat)
	except ValueError:
		layer_35_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_35_out = layer_35(layer_35_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_35 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_36_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_36_out = layer_36(layer_36_concat)
	except ValueError:
		layer_36_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_36_out = layer_36(layer_36_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_36 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_37_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_37_out = layer_37(layer_37_concat)
	except ValueError:
		layer_37_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_37_out = layer_37(layer_37_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_37 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_38_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_38_out = layer_38(layer_38_concat)
	except ValueError:
		layer_38_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_38_out = layer_38(layer_38_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_38 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_39_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_39_out = layer_39(layer_39_concat)
	except ValueError:
		layer_39_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_39_out = layer_39(layer_39_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_39 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_40_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_40_out = layer_40(layer_40_concat)
	except ValueError:
		layer_40_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_40_out = layer_40(layer_40_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_40 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_41_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_41_out = layer_41(layer_41_concat)
	except ValueError:
		layer_41_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_41_out = layer_41(layer_41_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_41 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_42_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_42_out = layer_42(layer_42_concat)
	except ValueError:
		layer_42_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_42_out = layer_42(layer_42_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_42 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_43_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_43_out = layer_43(layer_43_concat)
	except ValueError:
		layer_43_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_43_out = layer_43(layer_43_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_43 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_44_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_44_out = layer_44(layer_44_concat)
	except ValueError:
		layer_44_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_44_out = layer_44(layer_44_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_44 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_45_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_45_out = layer_45(layer_45_concat)
	except ValueError:
		layer_45_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_45_out = layer_45(layer_45_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_45 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_46_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_46_out = layer_46(layer_46_concat)
	except ValueError:
		layer_46_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_46_out = layer_46(layer_46_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_46 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_47_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_47_out = layer_47(layer_47_concat)
	except ValueError:
		layer_47_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_47_out = layer_47(layer_47_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_47 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_48_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_48_out = layer_48(layer_48_concat)
	except ValueError:
		layer_48_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_48_out = layer_48(layer_48_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_48 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_49_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_49_out = layer_49(layer_49_concat)
	except ValueError:
		layer_49_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_49_out = layer_49(layer_49_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_49 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_50_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_50_out = layer_50(layer_50_concat)
	except ValueError:
		layer_50_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_50_out = layer_50(layer_50_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_50 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_51_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_51_out = layer_51(layer_51_concat)
	except ValueError:
		layer_51_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_51_out = layer_51(layer_51_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_51 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_52_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_52_out = layer_52(layer_52_concat)
	except ValueError:
		layer_52_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_52_out = layer_52(layer_52_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_52 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_53_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_53_out = layer_53(layer_53_concat)
	except ValueError:
		layer_53_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_53_out = layer_53(layer_53_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_53 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_54_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_54_out = layer_54(layer_54_concat)
	except ValueError:
		layer_54_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_54_out = layer_54(layer_54_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_54 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_55_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_55_out = layer_55(layer_55_concat)
	except ValueError:
		layer_55_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_55_out = layer_55(layer_55_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_55 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_56_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_56_out = layer_56(layer_56_concat)
	except ValueError:
		layer_56_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_56_out = layer_56(layer_56_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_56 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_57_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_57_out = layer_57(layer_57_concat)
	except ValueError:
		layer_57_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_57_out = layer_57(layer_57_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_57 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_58_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_58_out = layer_58(layer_58_concat)
	except ValueError:
		layer_58_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_58_out = layer_58(layer_58_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_58 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_59_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_59_out = layer_59(layer_59_concat)
	except ValueError:
		layer_59_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_59_out = layer_59(layer_59_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_59 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_60_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_60_out = layer_60(layer_60_concat)
	except ValueError:
		layer_60_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_60_out = layer_60(layer_60_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_60 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_61_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_61_out = layer_61(layer_61_concat)
	except ValueError:
		layer_61_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_61_out = layer_61(layer_61_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_61 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_62_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_62_out = layer_62(layer_62_concat)
	except ValueError:
		layer_62_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_62_out = layer_62(layer_62_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_62 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_63_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_63_out = layer_63(layer_63_concat)
	except ValueError:
		layer_63_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_63_out = layer_63(layer_63_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_63 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_64_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_64_out = layer_64(layer_64_concat)
	except ValueError:
		layer_64_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_64_out = layer_64(layer_64_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_64 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_65_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_65_out = layer_65(layer_65_concat)
	except ValueError:
		layer_65_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_65_out = layer_65(layer_65_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_65 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_66_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_66_out = layer_66(layer_66_concat)
	except ValueError:
		layer_66_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_66_out = layer_66(layer_66_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_66 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_67_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_67_out = layer_67(layer_67_concat)
	except ValueError:
		layer_67_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_67_out = layer_67(layer_67_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_67 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_68_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_68_out = layer_68(layer_68_concat)
	except ValueError:
		layer_68_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_68_out = layer_68(layer_68_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_68 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_69_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_69_out = layer_69(layer_69_concat)
	except ValueError:
		layer_69_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_69_out = layer_69(layer_69_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_69 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_70_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_70_out = layer_70(layer_70_concat)
	except ValueError:
		layer_70_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_70_out = layer_70(layer_70_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_70 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_71_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_71_out = layer_71(layer_71_concat)
	except ValueError:
		layer_71_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_71_out = layer_71(layer_71_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_71 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_72_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_72_out = layer_72(layer_72_concat)
	except ValueError:
		layer_72_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_72_out = layer_72(layer_72_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_72 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_73_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_73_out = layer_73(layer_73_concat)
	except ValueError:
		layer_73_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_73_out = layer_73(layer_73_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_73 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_74_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_74_out = layer_74(layer_74_concat)
	except ValueError:
		layer_74_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_74_out = layer_74(layer_74_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_74 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_75_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_75_out = layer_75(layer_75_concat)
	except ValueError:
		layer_75_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_75_out = layer_75(layer_75_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_75 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_76_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_76_out = layer_76(layer_76_concat)
	except ValueError:
		layer_76_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_76_out = layer_76(layer_76_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_76 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_77_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_77_out = layer_77(layer_77_concat)
	except ValueError:
		layer_77_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_77_out = layer_77(layer_77_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_77 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_78_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_78_out = layer_78(layer_78_concat)
	except ValueError:
		layer_78_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_78_out = layer_78(layer_78_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_78 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_79_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_79_out = layer_79(layer_79_concat)
	except ValueError:
		layer_79_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_79_out = layer_79(layer_79_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_79 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_80_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_80_out = layer_80(layer_80_concat)
	except ValueError:
		layer_80_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_80_out = layer_80(layer_80_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_80 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_81_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_81_out = layer_81(layer_81_concat)
	except ValueError:
		layer_81_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_81_out = layer_81(layer_81_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_81 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_82_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_82_out = layer_82(layer_82_concat)
	except ValueError:
		layer_82_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_82_out = layer_82(layer_82_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_82 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_83_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_83_out = layer_83(layer_83_concat)
	except ValueError:
		layer_83_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_83_out = layer_83(layer_83_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_83 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_84_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_84_out = layer_84(layer_84_concat)
	except ValueError:
		layer_84_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_84_out = layer_84(layer_84_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_84 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_85_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_85_out = layer_85(layer_85_concat)
	except ValueError:
		layer_85_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_85_out = layer_85(layer_85_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_85 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_86_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_86_out = layer_86(layer_86_concat)
	except ValueError:
		layer_86_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_86_out = layer_86(layer_86_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_86 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_87_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_87_out = layer_87(layer_87_concat)
	except ValueError:
		layer_87_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_87_out = layer_87(layer_87_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_87 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_88_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_88_out = layer_88(layer_88_concat)
	except ValueError:
		layer_88_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_88_out = layer_88(layer_88_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_88 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_89_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_89_out = layer_89(layer_89_concat)
	except ValueError:
		layer_89_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_89_out = layer_89(layer_89_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_89 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_90_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_90_out = layer_90(layer_90_concat)
	except ValueError:
		layer_90_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_90_out = layer_90(layer_90_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_90 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_91_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_91_out = layer_91(layer_91_concat)
	except ValueError:
		layer_91_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_91_out = layer_91(layer_91_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_91 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_92_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_92_out = layer_92(layer_92_concat)
	except ValueError:
		layer_92_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_92_out = layer_92(layer_92_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_92 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_93_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_93_out = layer_93(layer_93_concat)
	except ValueError:
		layer_93_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_93_out = layer_93(layer_93_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_93 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_94_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_94_out = layer_94(layer_94_concat)
	except ValueError:
		layer_94_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_94_out = layer_94(layer_94_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_94 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_95_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_95_out = layer_95(layer_95_concat)
	except ValueError:
		layer_95_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_95_out = layer_95(layer_95_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_95 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_96_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_96_out = layer_96(layer_96_concat)
	except ValueError:
		layer_96_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_96_out = layer_96(layer_96_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_96 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_97_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_97_out = layer_97(layer_97_concat)
	except ValueError:
		layer_97_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_97_out = layer_97(layer_97_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_97 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_98_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_98_out = layer_98(layer_98_concat)
	except ValueError:
		layer_98_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_98_out = layer_98(layer_98_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_98 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_99_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_99_out = layer_99(layer_99_concat)
	except ValueError:
		layer_99_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_99_out = layer_99(layer_99_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_99 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_100_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_100_out = layer_100(layer_100_concat)
	except ValueError:
		layer_100_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_100_out = layer_100(layer_100_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_100 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_101_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_101_out = layer_101(layer_101_concat)
	except ValueError:
		layer_101_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_101_out = layer_101(layer_101_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_101 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_102_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_102_out = layer_102(layer_102_concat)
	except ValueError:
		layer_102_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_102_out = layer_102(layer_102_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_102 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_103_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_103_out = layer_103(layer_103_concat)
	except ValueError:
		layer_103_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_103_out = layer_103(layer_103_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_103 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_104_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_104_out = layer_104(layer_104_concat)
	except ValueError:
		layer_104_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_104_out = layer_104(layer_104_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_104 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_105_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_105_out = layer_105(layer_105_concat)
	except ValueError:
		layer_105_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_105_out = layer_105(layer_105_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_105 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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



layer_110 = L110()
try:
	try:
		layer_110_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_110_out = layer_110(layer_110_concat)
	except ValueError:
		layer_110_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_110_out = layer_110(layer_110_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_110 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_111_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_111_out = layer_111(layer_111_concat)
	except ValueError:
		layer_111_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_111_out = layer_111(layer_111_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_111 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_112_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_112_out = layer_112(layer_112_concat)
	except ValueError:
		layer_112_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_112_out = layer_112(layer_112_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_112 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_113_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_113_out = layer_113(layer_113_concat)
	except ValueError:
		layer_113_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_113_out = layer_113(layer_113_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_113 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_114_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_114_out = layer_114(layer_114_concat)
	except ValueError:
		layer_114_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_114_out = layer_114(layer_114_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_114 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_115_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_115_out = layer_115(layer_115_concat)
	except ValueError:
		layer_115_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_115_out = layer_115(layer_115_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_115 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_116_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_116_out = layer_116(layer_116_concat)
	except ValueError:
		layer_116_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_116_out = layer_116(layer_116_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_116 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_117_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_117_out = layer_117(layer_117_concat)
	except ValueError:
		layer_117_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_117_out = layer_117(layer_117_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_117 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_118_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_118_out = layer_118(layer_118_concat)
	except ValueError:
		layer_118_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_118_out = layer_118(layer_118_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_118 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_119_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_119_out = layer_119(layer_119_concat)
	except ValueError:
		layer_119_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_119_out = layer_119(layer_119_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_119 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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



layer_168 = L168()
try:
	try:
		layer_168_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_168_out = layer_168(layer_168_concat)
	except ValueError:
		layer_168_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_168_out = layer_168(layer_168_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_168 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_169_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_169_out = layer_169(layer_169_concat)
	except ValueError:
		layer_169_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_169_out = layer_169(layer_169_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_169 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_170_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_170_out = layer_170(layer_170_concat)
	except ValueError:
		layer_170_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_170_out = layer_170(layer_170_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_170 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_171_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_171_out = layer_171(layer_171_concat)
	except ValueError:
		layer_171_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_171_out = layer_171(layer_171_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_171 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_172_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_172_out = layer_172(layer_172_concat)
	except ValueError:
		layer_172_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_172_out = layer_172(layer_172_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_172 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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



layer_173 = L173()
try:
	try:
		layer_173_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_173_out = layer_173(layer_173_concat)
	except ValueError:
		layer_173_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_173_out = layer_173(layer_173_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_173 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_174_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_174_out = layer_174(layer_174_concat)
	except ValueError:
		layer_174_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_174_out = layer_174(layer_174_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_174 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_175_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_175_out = layer_175(layer_175_concat)
	except ValueError:
		layer_175_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_175_out = layer_175(layer_175_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_175 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_176_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_176_out = layer_176(layer_176_concat)
	except ValueError:
		layer_176_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_176_out = layer_176(layer_176_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_176 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_177_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_177_out = layer_177(layer_177_concat)
	except ValueError:
		layer_177_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_177_out = layer_177(layer_177_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_177 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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



layer_226 = L226()
try:
	try:
		layer_226_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_226_out = layer_226(layer_226_concat)
	except ValueError:
		layer_226_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_226_out = layer_226(layer_226_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_226 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_226_concat = strong_concat_226(new_list_of_inputs)
	layer_226_out = layer_226(layer_226_concat)
if layer_226.count_params() != 0:
	if 226 in set_weigths:
		if len(layer_226.get_weights()) > 1:
			new_w = [get_new_weigths(layer_226, set_weigths[226])]+layer_226.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_226, set_weigths[226])]
		layer_226.set_weights(new_w)


try:
	count = 1
	for s in layer_226_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #226:',count)
except AttributeError:
	for i in layer_226_out:
		count = 1
		for s in layer_226_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #226:',count)



layer_227 = L227()
try:
	try:
		layer_227_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_227_out = layer_227(layer_227_concat)
	except ValueError:
		layer_227_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_227_out = layer_227(layer_227_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_227 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_227_concat = strong_concat_227(new_list_of_inputs)
	layer_227_out = layer_227(layer_227_concat)
if layer_227.count_params() != 0:
	if 227 in set_weigths:
		if len(layer_227.get_weights()) > 1:
			new_w = [get_new_weigths(layer_227, set_weigths[227])]+layer_227.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_227, set_weigths[227])]
		layer_227.set_weights(new_w)


try:
	count = 1
	for s in layer_227_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #227:',count)
except AttributeError:
	for i in layer_227_out:
		count = 1
		for s in layer_227_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #227:',count)



layer_228 = L228()
try:
	try:
		layer_228_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_228_out = layer_228(layer_228_concat)
	except ValueError:
		layer_228_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_228_out = layer_228(layer_228_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_228 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_228_concat = strong_concat_228(new_list_of_inputs)
	layer_228_out = layer_228(layer_228_concat)
if layer_228.count_params() != 0:
	if 228 in set_weigths:
		if len(layer_228.get_weights()) > 1:
			new_w = [get_new_weigths(layer_228, set_weigths[228])]+layer_228.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_228, set_weigths[228])]
		layer_228.set_weights(new_w)


try:
	count = 1
	for s in layer_228_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #228:',count)
except AttributeError:
	for i in layer_228_out:
		count = 1
		for s in layer_228_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #228:',count)



layer_229 = L229()
try:
	try:
		layer_229_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_229_out = layer_229(layer_229_concat)
	except ValueError:
		layer_229_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_229_out = layer_229(layer_229_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_229 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_229_concat = strong_concat_229(new_list_of_inputs)
	layer_229_out = layer_229(layer_229_concat)
if layer_229.count_params() != 0:
	if 229 in set_weigths:
		if len(layer_229.get_weights()) > 1:
			new_w = [get_new_weigths(layer_229, set_weigths[229])]+layer_229.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_229, set_weigths[229])]
		layer_229.set_weights(new_w)


try:
	count = 1
	for s in layer_229_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #229:',count)
except AttributeError:
	for i in layer_229_out:
		count = 1
		for s in layer_229_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #229:',count)



layer_230 = L230()
try:
	try:
		layer_230_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_230_out = layer_230(layer_230_concat)
	except ValueError:
		layer_230_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_230_out = layer_230(layer_230_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_230 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_230_concat = strong_concat_230(new_list_of_inputs)
	layer_230_out = layer_230(layer_230_concat)
if layer_230.count_params() != 0:
	if 230 in set_weigths:
		if len(layer_230.get_weights()) > 1:
			new_w = [get_new_weigths(layer_230, set_weigths[230])]+layer_230.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_230, set_weigths[230])]
		layer_230.set_weights(new_w)


try:
	count = 1
	for s in layer_230_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #230:',count)
except AttributeError:
	for i in layer_230_out:
		count = 1
		for s in layer_230_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #230:',count)



layer_231 = L231()
try:
	try:
		layer_231_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_231_out = layer_231(layer_231_concat)
	except ValueError:
		layer_231_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_231_out = layer_231(layer_231_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_231 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_231_concat = strong_concat_231(new_list_of_inputs)
	layer_231_out = layer_231(layer_231_concat)
if layer_231.count_params() != 0:
	if 231 in set_weigths:
		if len(layer_231.get_weights()) > 1:
			new_w = [get_new_weigths(layer_231, set_weigths[231])]+layer_231.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_231, set_weigths[231])]
		layer_231.set_weights(new_w)


try:
	count = 1
	for s in layer_231_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #231:',count)
except AttributeError:
	for i in layer_231_out:
		count = 1
		for s in layer_231_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #231:',count)



layer_232 = L232()
try:
	try:
		layer_232_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_232_out = layer_232(layer_232_concat)
	except ValueError:
		layer_232_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_232_out = layer_232(layer_232_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_232 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_232_concat = strong_concat_232(new_list_of_inputs)
	layer_232_out = layer_232(layer_232_concat)
if layer_232.count_params() != 0:
	if 232 in set_weigths:
		if len(layer_232.get_weights()) > 1:
			new_w = [get_new_weigths(layer_232, set_weigths[232])]+layer_232.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_232, set_weigths[232])]
		layer_232.set_weights(new_w)


try:
	count = 1
	for s in layer_232_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #232:',count)
except AttributeError:
	for i in layer_232_out:
		count = 1
		for s in layer_232_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #232:',count)



layer_233 = L233()
try:
	try:
		layer_233_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_233_out = layer_233(layer_233_concat)
	except ValueError:
		layer_233_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_233_out = layer_233(layer_233_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_233 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_233_concat = strong_concat_233(new_list_of_inputs)
	layer_233_out = layer_233(layer_233_concat)
if layer_233.count_params() != 0:
	if 233 in set_weigths:
		if len(layer_233.get_weights()) > 1:
			new_w = [get_new_weigths(layer_233, set_weigths[233])]+layer_233.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_233, set_weigths[233])]
		layer_233.set_weights(new_w)


try:
	count = 1
	for s in layer_233_out.shape:
		count = count*s
		if count > 10000000:
			raise NameError
	#print('do #233:',count)
except AttributeError:
	for i in layer_233_out:
		count = 1
		for s in layer_233_out.shape:
			count = count*s
			if count > 10000000:
				raise NameError
		#print('do #233:',count)



layer_282 = L282()
try:
	try:
		layer_282_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_282_out = layer_282(layer_282_concat)
	except ValueError:
		layer_282_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_282_out = layer_282(layer_282_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_282 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_283_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_283_out = layer_283(layer_283_concat)
	except ValueError:
		layer_283_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_283_out = layer_283(layer_283_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_283 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_284_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_284_out = layer_284(layer_284_concat)
	except ValueError:
		layer_284_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_284_out = layer_284(layer_284_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_284 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_285_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_285_out = layer_285(layer_285_concat)
	except ValueError:
		layer_285_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_285_out = layer_285(layer_285_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_285 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_286_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_286_out = layer_286(layer_286_concat)
	except ValueError:
		layer_286_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_286_out = layer_286(layer_286_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_286 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_287_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_287_out = layer_287(layer_287_concat)
	except ValueError:
		layer_287_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_287_out = layer_287(layer_287_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_287 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_288_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_288_out = layer_288(layer_288_concat)
	except ValueError:
		layer_288_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_288_out = layer_288(layer_288_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_288 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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
		layer_289_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out])
		layer_289_out = layer_289(layer_289_concat)
	except ValueError:
		layer_289_concat = tf.keras.layers.concatenate([layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out], axis=-2)
		layer_289_out = layer_289(layer_289_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_289 = Strong_concat()
	new_list_of_inputs = []
	for i in [layer_1_out, layer_106_out, layer_107_out, layer_108_out, layer_109_out]:
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

tf.keras.backend.set_value(model.optimizer.learning_rate, 0.005)
model.fit(train_images, train_labels, epochs=3, batch_size=32)
'''
tf.keras.backend.set_value(model.optimizer.learning_rate, 0.001)
model.fit(train_images, train_labels, epochs=3, batch_size=32)
'''
tf.keras.backend.set_value(model.optimizer.learning_rate, 0.0005)
model.fit(train_images, train_labels, epochs=4, batch_size=32)
'''
tf.keras.backend.set_value(model.optimizer.learning_rate, 0.0001)
model.fit(train_images, train_labels, epochs=5, batch_size=32)
'''
test_loss, result = model.evaluate(test_images,  test_labels, verbose=0)
print('res:',result)


	