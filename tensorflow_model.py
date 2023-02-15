#Compiler
from time import time
from random import randint
import random
import numpy as np
import re
import math
import networkx as nx
import tensorflow as tf
from copy import deepcopy
import os 
import signal
max_elements = 10000000
#max_elements = 10000000000
def signal_handler(signum, frame):
    raise RuntimeError
#evolve parameters
#broadcast_to
maximum_elemts = 1000000
maximum_nodes = 1000
locally_connected_implementation = 2
shape = [100,100]

#Functions for grapth
def nn_to_graph(nn):
	graph = {}
	for cell_indx, cell in enumerate(nn):
		graph[cell_indx] = []
		for con_indx, connect in enumerate(nn):
			if connect[2] in cell[3]:
				if cell_indx != con_indx:
					graph[cell_indx].append(con_indx)
	return graph

def nucl_to_number(sequence, max_n=None, stop=False, max_n_norm = False):
	number = 0
	if stop != False:
		stop_indx = sequence.find(stop)
		if stop_indx == -1:
			new_seq = sequence
		else:
			new_seq = sequence[:stop_indx]
	else:
		new_seq = sequence

	str_numb = 1
	if max_n_norm == False:
		if new_seq != '':
			for power, nucl in enumerate(new_seq):
				if nucl == 'A':
					numb = 0
				if nucl == 'T':
					numb = 0
				if nucl == 'C':
					numb = 1
				if nucl == 'G':
					numb = 2
				str_numb += numb*(3**power)
			if max_n != None:
				if str_numb > max_n:
					str_numb = max_n
			return str_numb
		else:
			return 1

	else:
		if new_seq != '':
			numb = 1
			for power, nucl in enumerate(new_seq):
				if nucl == 'A':
					numb = 0
				if nucl == 'T':
						numb = 0
				if nucl == 'C':
					numb = 1
				if nucl == 'G':
					numb = 2
				str_numb += numb*(3**power)
			num_rem = ( float(str_numb) % float(max_n_norm)) / float(max_n_norm)
			num_rem = num_rem * max_n_norm
			str_numb = int(num_rem)
			if str_numb == 0:
				str_numb=1
			return str_numb
		else:
			return 1
		

def create_grapth(nmbr_nodes, nmbr_conect):
	graph = {}
	for i in range(nmbr_nodes):
		graph[i] = []
		for i_2 in range(nmbr_nodes):
			random_num = randint(0,nmbr_conect)
			if random_num == 0:
				graph[i].append(i_2)

	for i in graph:
		if i in graph[i]:
			graph[i].remove(i)
	return graph

def find_start(graph):
	from_conections = set([])
	to_conections = set([])
	for elem in graph:
		if graph[elem] != []:
			from_conections.add(elem)
			for i in graph[elem]:
				if i not in to_conections:
					to_conections.add(i)

	start_nodes = []
	for i in (from_conections - to_conections):
		start_nodes.append(i)

	return start_nodes

def find_end(graph):
	from_conections = set([])
	to_conections = set([])
	for elem in graph:
		if graph[elem] != []:
			from_conections.add(elem)
			for i in graph[elem]:
				if i not in to_conections:
					to_conections.add(i)

	end_nodes = []
	for i in (to_conections - from_conections):
		end_nodes.append(i)

	return end_nodes

def create_normalized_graph(raw_grath, n_inputs = 1, n_outputs = 1):
	graph = raw_grath
	end_nodes = find_end(graph)
	end_nodes = end_nodes[-n_outputs:]


	#backward grath creation
	backward_graph = {}
	end = False
	all_checked = []
	current = []
	for i in end_nodes:
		current.append(i)
		all_checked.append(i)
		backward_graph[i] = graph[i]
	while end==False:
		new_current = []
		end = True
		for node in current:
			end = False
			for key in graph:
				if node in graph[key]:
					if key not in all_checked:
						all_checked.append(key)
						backward_graph[key] = graph[key]
						new_current.append(key)


		current = new_current[:]



	graph = backward_graph

	#remoove cylces from grapth
	G1 = nx.DiGraph()
	for i in graph:
		G1.add_node(i)
	existed = [[],[]]
	for n in graph:
		#print('----')
		#print(n)
		for e in graph[n]:
			G1.add_edge(n, e)
			#print(e)
	#for e in G1.edges:
	#	print(e)
	end = False
	remoove_edged = None
	#counter = 0
	while end == False:
		#print('#',counter)
		#counter+=1
		try:
			remoove_edged = []
			for cycle in nx.simple_cycles(G1):
				remoove_edged.append([cycle[-1], cycle[0]])
				#print('cycle',cycle)
				#if len(cycle)>2:
				#	remoove_edged_2 = [cycle[-2], cycle[-1]]
				#if len(cycle)>3:
				#	remoove_edged_2 = [cycle[-3], cycle[-2]]
				break
		except nx.exception.NetworkXError:
			G1 = nx.DiGraph()
			break
		if remoove_edged != []:
			#print("BEFORE!:")
			#print('nodes:',G1.nodes)
			#print('edges:',G1.edges)
			for a, b in remoove_edged:
				G1.remove_edge(a, b)
			#print(len(G1.edges))
			#print('remov:',remoove_edged[0], remoove_edged[1])
			#print("AFTER:")
			#print('nodes:',G1.nodes)
			#print('edges:',G1.edges)
		else:
			break

	#make new grapht, with no cylces
	acycl_grapth = {}
	for i in list(G1.nodes):
		acycl_grapth[i] = []
	for con_2 in list(G1.edges):
		acycl_grapth[con_2[0]].append(con_2[1])

	
	rev_G = nx.DiGraph.reverse(G1)
	rev_graph = {}
	for i in acycl_grapth:
		rev_graph[i] = []
	for con_2 in list(rev_G.edges):
		rev_graph[con_2[0]].append(con_2[1])



	input_layers = find_start(acycl_grapth)
	output_layers = find_end(acycl_grapth)

	if n_outputs < len(output_layers) and n_inputs < len(input_layers):
		recurent_input = input_layers[n_inputs:]
		recurent_output = output_layers[n_outputs:]
		last_index_tmp = min(len(recurent_input), len(recurent_output))
		recurent_input = recurent_input[:last_index_tmp]
		recurent_output = recurent_output[:last_index_tmp]

	else:
		recurent_input = []
		recurent_output = []



	return acycl_grapth, rev_graph, list(input_layers[:n_inputs]), list(output_layers[:n_outputs]), list(recurent_input), list(recurent_output)


def create_1024():
	list_a = []
	DNA_letters = 'ACGT'
	for a in range(4):
		for b in range(4):
			for c in range(4):
				for d in range(4):
					for e in range(4):
						tmp = ''
						tmp += DNA_letters[a]
						tmp += DNA_letters[b]
						tmp += DNA_letters[c]
						tmp += DNA_letters[d]
						tmp += DNA_letters[e]
						list_a.append(tmp)
	return list_a


list_seq5 = create_1024()
list_mat_mul = set(list_seq5[400:500])
list_train_par = set(list_seq5[6:70])
list_mat_op = set(list_seq5[70:134])
list_mat_op_ax = set(list_seq5[134:198])
list_shape = set(list_seq5[198:214])
list_broadc = set(list_seq5[214:220])
list_neg = set(list_seq5[220:226])
list_add = set(list_seq5[226:232])
list_roll = set(list_seq5[232:238])
list_tile = set(list_seq5[238:244])
list_transp = set(list_seq5[244:250])
list_redu = set(list_seq5[250:266])
list_ker_no_par = set(list_seq5[266:282])
list_14 = set(list_seq5[282:288])
list_15 = set(list_seq5[288:294])
list_16 = set(list_seq5[294:300])
list_17 = set(list_seq5[300:306])
list_18 = set(list_seq5[306:312])
list_19 = set(list_seq5[312:318])
list_20 = set(list_seq5[318:324])
list_21 = set(list_seq5[324:330])
list_22 = set(list_seq5[330:336])
list_23 = set(list_seq5[336:342])
list_24 = set(list_seq5[342:348])
list_25 = set(list_seq5[348:354])
list_26 = set(list_seq5[354:360])
list_27 = set(list_seq5[360:366])
list_28 = set(list_seq5[366:372])
list_29 = set(list_seq5[372:378])
list_30 = set(list_seq5[378:384])
list_31 = set(list_seq5[384:390])
list_activ = set(list_seq5[384:400])





























#create random neural network woth 100 nodes
'''
str_seq = 'QWERTYUIOPASDFGHJKLZXCVBNM'
dna_letter = 'ACGT'
nn = []

for i in range(60):
	random_name = str_seq[randint(0,len(str_seq)-1)]
	random_connection = [str_seq[randint(0,len(str_seq)-1)] for i in range(randint(0,3))]
	random_matrx = ''
	random_func = ''
	random_weights = ''
	for i in range(randint(1,200)):
		random_matrx += dna_letter[randint(0,3)]
		random_func += dna_letter[randint(0,3)]
		random_weights += dna_letter[randint(0,3)]
	nn.append([None, None, random_name, random_connection, random_matrx, random_func, random_weights])

nn[0][4] = 'GTCAA' + nn[0][4][5:]
'''
def nn_to_tensorflow_0(nn, shape=shape, maximum_elemts=maximum_elemts, maximum_nodes=maximum_nodes, locally_connected_implementation=locally_connected_implementation, batch_size=32, out_shape=10):
	graph = nn_to_graph(nn)
	end_graph, rev_graph, start_nodes, end_nodes, rec_in, rec_out = create_normalized_graph(graph)

	#end_graph = [0]
	test_model = '''
import tensorflow as tf
import numpy as np
import math
maximum_elemts = {maximum_elemts}
locally_connected_implementation = {locally_connected_implementation}

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

	'''.format(maximum_elemts=maximum_elemts, locally_connected_implementation=locally_connected_implementation)





	#creating all custom tensrflow layers from neural network
	set_weigths = {}
	if len(end_graph) == 0:
		return None
	for cell_indx in end_graph:
		#for cell_indx in [0]:
		#print('type:',nn[cell_indx][4][:30])
		#matmul layer
		activ_fun = ''
		if len(nn[cell_indx][5])>1:
			if nn[cell_indx][5][:2] == 'AA':
				activ_fun = 'tf.keras.activations.exponential'
			elif nn[cell_indx][5][:2] == 'AC':
				activ_fun = 'tf.keras.activations.relu'
			elif nn[cell_indx][5][:2] == 'AG':
				activ_fun = 'tf.keras.activations.softmax'
			elif nn[cell_indx][5][:2] == 'AT':
				activ_fun = 'tf.keras.activations.softplus'
			elif nn[cell_indx][5][:2] == 'CA':
				activ_fun = 'tf.keras.activations.swish'
			elif nn[cell_indx][5][:2] == 'CG':
				activ_fun = 'tf.keras.activations.tanh'
			elif nn[cell_indx][5][:2] == 'CC':
				activ_fun = 'tf.keras.activations.gelu'
			elif nn[cell_indx][5][:2] == 'CT':
				activ_fun = 'tf.keras.activations.hard_sigmoid'
			elif nn[cell_indx][5][:2] == 'GA':
				activ_fun = 'tf.keras.activations.selu'
			elif nn[cell_indx][5][:2] == 'GC':
				activ_fun = 'tf.keras.activations.sigmoid'





		trainbale_par = True
		initializer = None
		regularizer = None
		constrain = None

		try:
			if nn[cell_indx][6] != '':
				if nn[cell_indx][6][1] == 'A':
					trainbale_par = False
				if nn[cell_indx][6][0] == 'A' and len(nn[cell_indx][6])>2:
					tmp_seq = ''
					counter = 0
					weights_tmp = []
					for i in nn[cell_indx][6][3:]:
						if counter == 4:
							tmp_seq+=i
							tmp_seq = nucl_to_number(tmp_seq)
							weights_tmp.append(tmp_seq/1000)
							tmp_seq = ''
							counter=0
						else:
							tmp_seq+=i
							counter+=1
					if weights_tmp != []:
						if  nn[cell_indx][6][2] == "A":
							set_weigths[cell_indx] = [weights_tmp[0]]
						else:
							set_weigths[cell_indx] = weights_tmp
				else:
					numbers_cut = [m.start() for m in re.finditer('(?={})'.format('T'), nn[cell_indx][6][8:])]
					cuts = []
					last_i = 5
					for cut_i in numbers_cut:
						cuts.append([last_i, cut_i+8])
						last_i = cut_i+9
					numbers_weigths = []
					counter_par = 0
					for cut in cuts:
						if cut[0]-cut[1] == 0:
							pass
						else:
							numbers_weigths.append(nucl_to_number(nn[cell_indx][6][cut[0]:cut[1]],max_n_norm=1024))


					#initializer
					if nn[cell_indx][6][2:4] == 'AA':
						try:
							initializer = 'tf.keras.initializers.Constant(value={n_val})'.format(n_val = numbers_weigths[0]/1000)
						except IndexError:
							initializer = 'tf.keras.initializers.Constant(value=0.5)'
					elif nn[cell_indx][6][2:4] == 'AC':
						initializer = 'tf.keras.initializers.GlorotNormal()'
					elif nn[cell_indx][6][2:4] == 'AD':
						initializer = 'tf.keras.initializers.GlorotUniform()'
					elif nn[cell_indx][6][2:4] == 'AT':
						initializer = 'tf.keras.initializers.HeNormal()'
					elif nn[cell_indx][6][2:4] == 'CA':
						initializer = 'tf.keras.initializers.HeUniform()'
					elif nn[cell_indx][6][2:4] == 'CC':
						initializer = 'tf.keras.initializers.LecunNormal()'
					elif nn[cell_indx][6][2:4] == 'CD':
						initializer = 'tf.keras.initializers.LecunUniform()'
					elif nn[cell_indx][6][2:4] == 'CT':
						initializer = 'tf.keras.initializers.Ones()'
					elif nn[cell_indx][6][2:4] == 'DA':
						initializer = 'tf.keras.initializers.Zeros()'
					elif nn[cell_indx][6][2:4] == 'DC':
						try:
							initializer = 'tf.keras.initializers.Identity(gain={n_val})'.format(n_val = numbers_weigths[0]/500)
						except IndexError:
							initializer = 'tf.keras.initializers.Identity(gain=1)'
					elif nn[cell_indx][6][2:4] == 'DD':
						try:
							initializer = 'tf.keras.initializers.Orthogonal(gain={n_val})'.format(n_val = numbers_weigths[0]/500)
						except IndexError:
							initializer = 'tf.keras.initializers.Orthogonal(gain=1)'
					elif nn[cell_indx][6][2:4] == 'DT':
						try:
							initializer = 'tf.keras.initializers.RandomNormal(mean={n_val_0}, stddev={n_val_1})'.format(n_val_0 = numbers_weigths[0]/1000, n_val_1 = numbers_weigths[1]/1000)
						except IndexError:
							initializer = 'tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)'
					elif nn[cell_indx][6][2:4] == 'TA':
						try:
							initializer = 'tf.keras.initializers.RandomUniform(minval=-{n_val_0}, maxval={n_val_1})'.format(n_val_0 = numbers_weigths[0]/1000, n_val_1 = numbers_weigths[1]/1000)
						except IndexError:
							initializer = 'tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05)'
					elif nn[cell_indx][6][2:4] == 'TC':
						try:
							initializer = 'tf.keras.initializers.TruncatedNormal(mean={n_val_0}, stddev={n_val_1})'.format(n_val_0 = numbers_weigths[0]/1000, n_val_1 = numbers_weigths[1]/1000)
						except IndexError:
							initializer = 'tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05)'
					elif nn[cell_indx][6][2:4] == 'TT':
						try:
							initializer = 'tf.keras.initializers.VarianceScaling(scale={n_val})'.format(n_val = numbers_weigths[0]/500)
						except IndexError:
							initializer = 'tf.keras.initializers.VarianceScaling(scale=0.1)'



					#contrain
					if nn[cell_indx][6][4] == 'A':
						try:
							constrain = 'tf.keras.constraints.MaxNorm(max_value={n_val}, axis=0)'.format(n_val = numbers_weigths[2]/500)
						except IndexError:
							constrain = 'tf.keras.constraints.MaxNorm(max_value=2, axis=0)'
					elif nn[cell_indx][6][4] == 'C':
						try:
							a_1 = numbers_weigths[2]/500
							a_2 = numbers_weigths[3]/500
							if a_2 > a_1:
								a_1, a_2 = a_2, a_1
							elif a_2 == a_1:
								a_2 = a_2 + 0.01
							constrain = 'tf.keras.constraints.MinMaxNorm(min_value={n_val_0}, max_value={n_val_0}, rate=1.0, axis=0)'.format(n_val_0 = a_1, n_val_1 = a_2)
						except IndexError:
							constrain = 'tf.keras.constraints.MinMaxNorm(min_value=0.0, max_value=1.0, rate=1.0, axis=0)'
					elif nn[cell_indx][6][4] == 'D':
						constrain = 'tf.keras.constraints.NonNeg(axis=0)'
					elif nn[cell_indx][6][4] == 'T':
						constrain = 'tf.keras.constraints.UnitNorm(axis=0)'
					

					#regularizers
					if nn[cell_indx][6][5] == 'A':
						try:
							regularizer = 'tf.keras.regularizers.L1(l1={n_val})'.format(n_val = numbers_weigths[4]/1000)
						except IndexError:
							regularizer = 'tf.keras.regularizers.L1(l1=0.01)'
					elif nn[cell_indx][6][5] == 'C':
						try:
							regularizer = 'tf.keras.regularizers.L1L2(l1={n_val_0}, l2={n_val_1})'.format(n_val_0 = numbers_weigths[4]/1000, n_val_1 = numbers_weigths[5]/1000)
						except IndexError:
							regularizer = 'tf.keras.regularizers.L1L2(l1=0.0, l2=0.0)'
					elif nn[cell_indx][6][5] == 'D':
						regularizer = 'tf.keras.constraints.NonNeg(axis=0)'
					elif nn[cell_indx][6][5] == 'T':
						try:
							regularizer = 'tf.keras.regularizers.L2(l2={n_val})'.format(n_val = numbers_weigths[4]/1000)
						except IndexError:
							regularizer = 'tf.keras.regularizers.L2(l2=0.01)'
		
		except IndexError:
			pass










		#matmul layer
		if nn[cell_indx][4][:5] in list_mat_mul:
			units = 32
			if len(nn[cell_indx][4]) > 6:
				full_parameters_seq = nn[cell_indx][4][5:]
				units = nucl_to_number(full_parameters_seq[:7], max_n=maximum_nodes, stop='T')
			#numebr of units, other shapes after rank-2, concatination type
			layer_str = '''
class L{number}(tf.keras.layers.Layer):
	def __init__(self):
		super(L{number}, self).__init__()
		self.units = {units}

	def build(self, input_shape):
		initializer = {initializer}
		if initializer == None:
			initializer = "random_normal"
		regularizer = {regularizer}
		constrain = {constrain}
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable={trainable},
		)
		

	def call(self, inputs):
		return {activ_fun}( tf.matmul(inputs, self.w) ) 
			'''.format(number=cell_indx, units = units, trainable=trainbale_par, activ_fun=activ_fun, initializer=initializer, regularizer=regularizer, constrain=constrain)
			test_model = test_model + layer_str















		#trainable layers
		elif nn[cell_indx][4][:5] in list_train_par and len(nn[cell_indx][4]) > 5:
			trainable = 'True'
			matrix_op = ''
			check_1 = False
			shape_parameter = 'input_shape'
			if nn[cell_indx][6]!='':
				if nn[cell_indx][6][0] == 'C':
					shape_parameter = '1,'
					check_1 = True

			#numebr of units, other shapes after rank-2, concatination type
			if nn[cell_indx][4][3:5] == 'AA':
				matrix_op = 'tf.math.divide_no_nan'
			elif nn[cell_indx][4][3:5] == 'AC':
				matrix_op = 'tf.math.multiply_no_nan'
			elif nn[cell_indx][4][3:5] == 'AG':
				matrix_op = 'tf.add'
			elif nn[cell_indx][4][3:5] == 'AT':
				matrix_op = 'tf.math.pow'
			elif nn[cell_indx][4][3:5] == 'CA':
				matrix_op =  'tf.math.multiply_no_nan'
			elif nn[cell_indx][4][3:5] == 'CC':
				matrix_op = 'tf.math.maximum'
			elif nn[cell_indx][4][3:5] == 'CT':
				matrix_op = 'tf.math.minimum'
			if matrix_op == '':
				shape_parameter = 'input_shape[-1], 1'
				matrix_op = 'tf.matmul'
				check_1 = False

			layer_str = '''
class L{number}(tf.keras.layers.Layer):
	def __init__(self):
		super(L{number}, self).__init__()

	def build(self, input_shape):
		initializer = {initializer}
		if initializer == None:
			initializer = "random_normal"
		regularizer = {regularizer}
		constrain = {constrain}

		
		self.w = self.add_weight(
			shape=({shape_parameter}),
			initializer=initializer,
			regularizer=regularizer,
			constraint=constrain,
			trainable={trainable},
		)
		if {check_1}:
			self.w = tf.broadcast_to(self.w, shape=input_shape)
		

	def call(self, inputs):
		return {activ_fun}({matrix_op}(inputs, self.w))
			'''.format(number=cell_indx,  trainable=trainable, shape_parameter=shape_parameter, activ_fun=activ_fun, matrix_op=matrix_op, check_1=check_1, initializer=initializer, regularizer=regularizer, constrain=constrain)
			test_model = test_model + layer_str













		#matrix operations wiht no parameters
		elif nn[cell_indx][4][:5] in list_mat_op and len(nn[cell_indx][4]) > 5:
			matrix_op = ''
			#numebr of units, other shapes after rank-2, concatination type
			if nn[cell_indx][4][3:5] == 'AA':
				matrix_op = 'tf.math.abs'
			elif nn[cell_indx][4][3:5] == 'AC':
				matrix_op = 'tf.math.asin'
			elif nn[cell_indx][4][3:5] == 'AG':
				matrix_op = 'tf.math.atan'
			elif nn[cell_indx][4][3:5] == 'AT':
				matrix_op = 'tf.math.ceil'
			elif nn[cell_indx][4][3:5] == 'CA':
				matrix_op =  'tf.math.cos'
			elif nn[cell_indx][4][3:5] == 'CC':
				matrix_op = 'tf.math.floor'
			elif nn[cell_indx][4][3:5] == 'CT':
				matrix_op = 'tf.math.round'
			elif nn[cell_indx][4][3:5] == 'CT':
				matrix_op = 'sin'
			elif nn[cell_indx][4][3:5] == 'GC':
				matrix_op = 'tf.linalg.matrix_transpose'
			elif nn[cell_indx][4][3:5] == 'GG':
				matrix_op = 'tf.linalg.pinv'
			elif nn[cell_indx][4][3:5] == 'GT':
				matrix_op = 'tf.math.negative'
			elif nn[cell_indx][4][3:5] == 'TA':
				matrix_op = 'tf.ones_like'
			
			layer_str = '''
class L{number}(tf.keras.layers.Layer):
	def __init__(self):
		super(L{number}, self).__init__()

	def call(self, inputs):
		return {activ_fun}({matrix_op}(inputs))
			'''.format(number=cell_indx, activ_fun=activ_fun, matrix_op=matrix_op)

			test_model = test_model + layer_str














		#matrix opeartions with axis parameter
		#choose step 2
		elif nn[cell_indx][4][:5] in list_mat_op_ax and len(nn[cell_indx][4]) > 5:
			trainable = 'True'
			matrix_op = ''
			axis_parameter = ''

			special = False
			#numebr of units, other shapes after rank-2, concatination type
			#choose step 2
			if nn[cell_indx][4][3:5] == 'AA':
				matrix_op = 'tf.math.cumprod'
			elif nn[cell_indx][4][3:5] == 'AC':
				matrix_op = 'tf.math.cumsum'
			elif nn[cell_indx][4][3:5] == 'AG':
				matrix_op = 'tf.math.reduce_max'
			elif nn[cell_indx][4][3:5] == 'AT':
				matrix_op = 'tf.math.reduce_mean'
			elif nn[cell_indx][4][3:5] == 'CA':
				matrix_op =  'tf.math.reduce_min'
			elif nn[cell_indx][4][3:5] == 'CC':
				matrix_op = 'tf.math.reduce_prod'
			elif nn[cell_indx][4][3:5] == 'CT':
				matrix_op = 'tf.math.reduce_std'
			elif nn[cell_indx][4][3:5] == 'CG':
				matrix_op =  'tf.math.reduce_variance'
			elif nn[cell_indx][4][3:5] == 'TA':
				matrix_op = 'tf.math.reduce_prod'
			elif nn[cell_indx][4][3:5] == 'TC':
				special = True
				matrix_op = 'tf.reverse'
			elif nn[cell_indx][4][3:5] == 'TT':
				matrix_op = 'tf.sort'
			'''
			elif nn[cell_indx][4][3:5] == 'TG':
				matrix_op = 'tf.raw_ops.Prod'
			'''

			if nn[cell_indx][6]!='' and matrix_op!='' and special == False:
				if nn[cell_indx][6][0] == 'C':
					axis_parameter = ',axis=0'
				elif nn[cell_indx][6][0] == 'G':
					axis_parameter = ',axis=1'
				elif nn[cell_indx][6][0] == 'T':
					pass
					axis_parameter = ',axis=-2'
				elif nn[cell_indx][6][0] == 'A':
					axis_parameter = ',axis=-1'
			elif nn[cell_indx][6]!='' and matrix_op!='' and special == True:
				if nn[cell_indx][6][0] == 'C':
					axis_parameter = ',axis=[0]'
				elif nn[cell_indx][6][0] == 'G':
					axis_parameter = ',axis=[1]'
				elif nn[cell_indx][6][0] == 'T':
					pass
					axis_parameter = ',axis=[2]'
				elif nn[cell_indx][6][0] == 'A':
					axis_parameter = ',axis=[-1]'

			
			layer_str = '''
class L{number}(tf.keras.layers.Layer):
	def __init__(self):
		super(L{number}, self).__init__()

	def call(self, inputs):
		return {activ_fun}({matrix_op}(inputs{axis}))
			'''.format(number=cell_indx, activ_fun=activ_fun, matrix_op=matrix_op, axis=axis_parameter)

			test_model = test_model + layer_str













		#rehsape_by_number
		elif nn[cell_indx][4][:5] in list_shape and len(nn[cell_indx][4]) > 6:
			reverse_par = 'False'
			mulyiply_par = 'True'
			if nn[cell_indx][4][5] == 'A':
				reverse_par = 'True'
			if nn[cell_indx][4][6] == 'A':
				mulyiply_par = 'False'

			mulyiply_par = 'False'
			shape_num = nucl_to_number(nn[cell_indx][4][7:], stop='T')
			layer_str = '''
class L{number}(tf.keras.layers.Layer):
	def __init__(self):
		super(L{number}, self).__init__()
	def build(self, input_shape):
		n_elemts = 1
		for i in input_shape:
			n_elemts = n_elemts*i
		self.new_shape = input_shape
		reverse_l = {reverse}
		mulyiply_par = {mulyiply}
		shape_number = {shape_number}
		if shape_number < n_elemts:
			if mulyiply_par == False:
				if n_elemts % shape_number == 0:
					if reverse_l == False:
						self.new_shape = [shape_number, -1]
					else:
						self.new_shape = [-1, shape_number]
			else:
				if reverse_l == False:
					a = shape_number*input_shape[-1]
					if n_elemts % a == 0:
						self.new_shape = [-1,  a]
				else:
					a = shape_number*input_shape[-2]
					if n_elemts % (a) == 0:
						self.new_shape = [a, -1]

	def call(self, inputs):
		return tf.squeeze(tf.reshape(inputs, shape=self.new_shape))
			'''.format(number=cell_indx, shape_number=shape_num, reverse=reverse_par, mulyiply=mulyiply_par)
			test_model = test_model + layer_str













		#broadcast_t0
		#possibe lcatatopth, if cell withh divide multipe times
		elif nn[cell_indx][4][:5] in list_broadc and len(nn[cell_indx][4]) > 5:
			multiply_by = 0
			multiply_by = nucl_to_number(nn[cell_indx][4][5:], stop='T', max_n=maximum_nodes)

			layer_str = '''
class L{number}(tf.keras.layers.Layer):
	def __init__(self):
		super(L{number}, self).__init__()
	def build(self, input_shape):
		multiply = {multiply_by}
		maximum_elemts = {maximum_elemts}
		if multiply == 0:
			self.shape_full = input_shape
		else:
			n_counter = 1
			self.shape_full = [multiply]
			for i in input_shape:
				n_counter = n_counter*i
				self.shape_full.append(i) 
			if n_counter > maximum_elemts:
				self.shape_full = input_shape
	def call(self, inputs):
		return tf.squeeze(tf.broadcast_to(inputs, shape=self.shape_full))
			'''.format(number=cell_indx, multiply_by=multiply_by, maximum_elemts=maximum_elemts)
			test_model = test_model + layer_str












		#all negative
		#possibe lcatatopth, if cell withh divide multipe times
		elif nn[cell_indx][4][:5] in list_neg:


			nucl_to_number(nn[cell_indx][4][5:], stop='T', max_n=maximum_nodes)
			
			layer_str = '''
class L{number}(tf.keras.layers.Layer):
	def __init__(self):
		super(L{number}, self).__init__()
	def call(self, inputs):
		return tf.math.negative(inputs)
			'''.format(number=cell_indx)
			
			test_model = test_model + layer_str









		#padd
		elif nn[cell_indx][4][:5] in list_add and len(nn[cell_indx][4]) > 6:
			constant_values = 0
			
			pad_cut_list = [m.start() for m in re.finditer('(?={})'.format('T'), nn[cell_indx][4][6:])]
			pad_numbers = []
			if pad_cut_list != []:
				start = 0
				for i in pad_cut_list:
					pad_numbers.append( nucl_to_number(nn[cell_indx][4][6+start:i+6], max_n=int(maximum_elemts**(1/2)/2)) )
					start = i+1

				if pad_cut_list[-1] != len(nn[cell_indx][4][6:]):
					pad_numbers.append( nucl_to_number(nn[cell_indx][4][pad_cut_list[-1]+7:], max_n=int(maximum_elemts**(1/2)/2)) )
			else:
				pad_numbers = [0]

			
			if len(nn[cell_indx][6]) > 1:
				if nn[cell_indx][6][0] == 'A':
					num_tmp = nucl_to_number(nn[cell_indx][6][1:], stop='T')
					num_str = '-0.{}'.format(str(num_tmp)[:3])
				else:
					num_tmp = nucl_to_number(nn[cell_indx][6][1:], stop='T')
					num_str = '0.{}'.format(str(num_tmp)[:3])

				constant_values = float(num_str)

			if nn[cell_indx][4][5] == 'A':
				mode='CONSTANT'
				constant_values = 0.
			elif nn[cell_indx][4][5] == 'C':
				mode='CONSTANT'
			elif nn[cell_indx][4][5] == 'G':
				mode='REFLECT'
			elif nn[cell_indx][4][5] == 'T':
				mode='SYMMETRIC'

			layer_str = '''
class L{number}(tf.keras.layers.Layer):
	def __init__(self):
		super(L{number}, self).__init__()
	def build(self, input_shape):
		counter = 0
		self.padding = []
		pad_numbers = {pad_numbers}
		mode ='{mode}'

		if mode == 'CONSTANT':
			for i in input_shape:
				self.padding.append([])
				try:
					self.padding[-1].append( pad_numbers[counter] )
				except IndexError:
					self.padding[-1].append( pad_numbers[-1] )
				try:
					self.padding[-1].append( pad_numbers[counter+1] )
				except IndexError:
					self.padding[-1].append( pad_numbers[-1] )
				counter +=2
		else:
			for i in input_shape:
				self.padding.append([])
				try:
					self.padding[-1].append( min(pad_numbers[counter], i-1 ) )
				except IndexError:
					self.padding[-1].append(  min( pad_numbers[-1],  i-1 ) )
				try:
					self.padding[-1].append( min( pad_numbers[counter+1], i-1 ) )
				except IndexError:
					self.padding[-1].append( min( pad_numbers[-1],  i-1 ) )
				counter +=2




	def call(self, inputs):
		return tf.pad(inputs, paddings=self.padding, mode='{mode}', constant_values={constant_values})
			'''.format(number=cell_indx, pad_numbers=pad_numbers, mode=mode, constant_values=constant_values)
			test_model = test_model + layer_str
















		#roll
		elif nn[cell_indx][4][:5] in list_roll and len(nn[cell_indx][4]) > 5:
			axis_cut_list = [m.start() for m in re.finditer('(?={})'.format('T'), nn[cell_indx][4][5:])]
			axis_numbers = []
			if axis_cut_list != []:
				start = 0
				for i in axis_cut_list:
					axis_numbers.append( nucl_to_number(nn[cell_indx][4][5+start:i+5],max_n_norm=4) )
					start = i+1

				if axis_cut_list[-1] != len(nn[cell_indx][4][5:]):
					axis_numbers.append( nucl_to_number(nn[cell_indx][4][axis_cut_list[-1]+6:],max_n_norm=4) )
			else:
				axis_numbers = [0]

			


			shift_cut_list = [m.start() for m in re.finditer('(?={})'.format('T'), nn[cell_indx][6])]
			shift_numbers = []
			if shift_cut_list != []:
				start = 0
				for i in shift_cut_list:
					shift_numbers.append( nucl_to_number(nn[cell_indx][6][start:i]) )
					start = i+1

				if shift_cut_list[-1] != len(nn[cell_indx][6]):
					shift_numbers.append( nucl_to_number(nn[cell_indx][6][shift_cut_list[-1]+1:])-1 )

			else:
				shift_numbers = [0]

			
			layer_str = '''
class L{number}(tf.keras.layers.Layer):
	def __init__(self):
		super(L{number}, self).__init__()
	def build(self, input_shape):
		counter = 0
		axis_numbers={axis_numbers}
		shift_numbers={shift_numbers}
		self.axis = []
		self.shift = []
		max_n = len(input_shape)-1

		for i in input_shape:
			try:
				self.axis.append( min(axis_numbers[counter], max_n) )
			except IndexError:
				self.axis.append( min(axis_numbers[-1], max_n ) )

			try:
				self.shift.append( shift_numbers[counter] )
			except IndexError:
				self.shift.append( shift_numbers[-1] )

			counter +=1


	def call(self, inputs):
		return tf.roll(inputs, shift=self.shift, axis=self.axis)
			'''.format(number=cell_indx, axis_numbers=axis_numbers, shift_numbers=shift_numbers)
			test_model = test_model + layer_str














		#tile
		elif nn[cell_indx][4][:5] in list_tile:

			seq_cut = [m.start() for m in re.finditer('(?={})'.format('T'), nn[cell_indx][6])]
			multipler_axis = []
			if seq_cut != []:
				start = 0
				for i in seq_cut:
					multipler_axis.append( nucl_to_number(nn[cell_indx][6][start:i])+1 )
					start = i+1

				if seq_cut[-1] != len(nn[cell_indx][6]):
					multipler_axis.append( nucl_to_number(nn[cell_indx][6][seq_cut[-1]+1:])-1 )

			else:
				multipler_axis = [1]

			
			layer_str = '''
class L{number}(tf.keras.layers.Layer):
	def __init__(self):
		super(L{number}, self).__init__()
	def build(self, input_shape):
		counter = 0
		
		multipler_axis={multipler_axis}
		self.multipler_axis = []
		size_t = 1
		max_size = {maximum_elemts}
		
		for d_l in input_shape:
			size_t = size_t * d_l

		for i in input_shape:
			try:
				if multipler_axis[counter]*size_t > max_size:
					self.multipler_axis.append( 1 )
				else:
					self.multipler_axis.append( multipler_axis[counter] )
					size_t = size_t*multipler_axis[counter]
			except IndexError:
				self.multipler_axis.append( 1 )

			counter +=1



	def call(self, inputs):
		return tf.tile(inputs, self.multipler_axis)
			'''.format(number=cell_indx, multipler_axis=multipler_axis, maximum_elemts=maximum_elemts)
			test_model = test_model + layer_str









		#transpose
		elif nn[cell_indx][4][:5] in list_transp and len(nn[cell_indx][4])>5:
			perm_par = 'False'
			if nn[cell_indx][4][5] == 'A' or nn[cell_indx][4][5] == 'C':
				perm_par = 'True'

			seq_cut = [m.start() for m in re.finditer('(?={})'.format('T'), nn[cell_indx][6])]
			transpose_axis = []
			if seq_cut != []:
				start = 0
				for i in seq_cut:
					transpose_axis.append( nucl_to_number(nn[cell_indx][6][start:i])+1 )
					start = i+1

				if seq_cut[-1] != len(nn[cell_indx][6]):
					transpose_axis.append( nucl_to_number(nn[cell_indx][6][seq_cut[-1]+1:]) )

			else:
				transpose_axis = []

			
			layer_str = '''
class L{number}(tf.keras.layers.Layer):
	def __init__(self):
		super(L{number}, self).__init__()
	def build(self, input_shape):
		counter = 0
		

		transpose_axis={transpose_axis}
		perm_par = {perm_par}
		
		self.transpose_axis = []
		n_dims = len(input_shape)
		if len(transpose_axis) < n_dims or perm_par == False:
			for i in reversed( range(n_dims) ):
				self.transpose_axis.append(i)
		else:
			listofzeros = [0] * n_dims
			transpose_axis = transpose_axis[:n_dims]
			transpose_axis_tmp = transpose_axis[:]
			for i in transpose_axis:
				n_dims = n_dims-1
				value_tmp = max(transpose_axis_tmp)
				index_tmp = transpose_axis.index(value_tmp)
				transpose_axis[index_tmp] = -1
				transpose_axis_tmp.remove(value_tmp)
				listofzeros[index_tmp] = n_dims

			self.transpose_axis = listofzeros

	def call(self, inputs):
		return tf.transpose(inputs, perm=self.transpose_axis)
			'''.format(number=cell_indx, transpose_axis=transpose_axis, perm_par=perm_par)
			test_model = test_model + layer_str









		#reduce
		elif nn[cell_indx][4][:5] in list_redu and len(nn[cell_indx][4])>6:
			if nn[cell_indx][4][4] == 'A':
				matrix_reduce_op = 'tf.math.reduce_sum'
			elif nn[cell_indx][4][4] == 'C':
				matrix_reduce_op = 'tf.math.reduce_prod'
			elif nn[cell_indx][4][4] == 'G':
				if nn[cell_indx][4][5] == 'A' or nn[cell_indx][4][5] == 'C':
					matrix_reduce_op = 'tf.math.reduce_max'
				else:
					matrix_reduce_op = 'tf.math.reduce_min'
			else:
				if nn[cell_indx][4][5] == 'A' or nn[cell_indx][4][5] == 'C':
					matrix_reduce_op = 'tf.math.reduce_mean'
				else:
					matrix_reduce_op = 'tf.math.reduce_std'
			
			if nn[cell_indx][4][6] == 'A':
				axis_par = ', 0'
			elif nn[cell_indx][4][6] == 'C':
				axis_par = ', 1'
			elif nn[cell_indx][4][6] == 'G':
				axis_par = ', 2'
			else:
				axis_par = ', -1'
			
			layer_str = '''
class L{number}(tf.keras.layers.Layer):
	def __init__(self):
		super(L{number}, self).__init__()
	def call(self, inputs):
		return {matrix_reduce_op}(inputs{axis_par})
			'''.format(number=cell_indx, matrix_reduce_op=matrix_reduce_op, axis_par=axis_par)
			test_model = test_model + layer_str





		#out_of_the_box keras layers with no parameters
		elif nn[cell_indx][4][:5] in list_ker_no_par and len(nn[cell_indx][4])>4:
			if nn[cell_indx][4][4] == 'A':
				matrix_reduce_op = 'tf.keras.layers.Flatten'
			elif nn[cell_indx][4][4] == 'C':
				if nn[cell_indx][4][5] == 'A' or nn[cell_indx][4][5] == 'C':
					matrix_reduce_op = 'tf.keras.layers.GlobalAveragePooling1D'
				else:
					matrix_reduce_op = 'tf.keras.layers.GlobalAveragePooling2D'
			elif nn[cell_indx][4][4] == 'G':
				if nn[cell_indx][4][5] == 'A' or nn[cell_indx][4][5] == 'C':
					matrix_reduce_op = 'tf.keras.layers.GlobalMaxPool1D'
				else:
					matrix_reduce_op = 'tf.keras.layers.GlobalMaxPool2D'
			elif nn[cell_indx][4][4] == 'T':
				matrix_reduce_op = 'tf.keras.layers.PReLU'


			layer_str = '''
class L{number}(tf.keras.layers.Layer):
	def __init__(self):
		super(L{number}, self).__init__()
	def build(self, input_shape):
		if len(input_shape.as_list()) < 3:
			def i_o(x):
				return x
			self.m_op = i_o
		elif len(input_shape.as_list()) < 4:
			if {matrix_reduce_op} == tf.keras.layers.GlobalMaxPool2D:
				self.m_op =  tf.keras.layers.GlobalMaxPool1D()
			elif {matrix_reduce_op} == tf.keras.layers.GlobalAveragePooling2D:
				self.m_op =  tf.keras.layers.GlobalAveragePooling1D()
			else:
				self.m_op =  {matrix_reduce_op}()
		else:
			self.m_op =  {matrix_reduce_op}()

	def call(self, inputs):
		return self.m_op(inputs)
			'''.format(number=cell_indx, matrix_reduce_op=matrix_reduce_op)
			test_model = test_model + layer_str








		# Average_Max_Pooling_1D
		#GTCA
		elif nn[cell_indx][4][:5] in list_14 and len(nn[cell_indx][4]) > 7:
			pool_cut = nn[cell_indx][4][7:].find('T')
			if pool_cut == -1 or pool_cut ==len(nn[cell_indx][4][7:]):
				stride_val=None
				pool_val = 1+nucl_to_number(nn[cell_indx][4][7:], max_n_norm=8, stop='T')
			else:
				pool_val = 1+nucl_to_number(nn[cell_indx][4][7:pool_cut], max_n_norm=8, stop='T')
				stride_val = 1+nucl_to_number(nn[cell_indx][4][pool_cut:], max_n_norm=8, stop='T')


			if nn[cell_indx][4][5] == 'A':
				padding_val='valid'
				data_format = 'channels_last'
			elif nn[cell_indx][4][5] == 'C':
				padding_val='valid'
				data_format = 'channels_first'
			elif nn[cell_indx][4][5] == 'G':
				padding_val='same'
				data_format = 'channels_last'
			elif nn[cell_indx][4][5] == 'T':
				padding_val='same'
				data_format = 'channels_first'

			if nn[cell_indx][4][6] == 'A' or nn[cell_indx][4][6] == 'C':
				matrix_op = 'tf.keras.layers.MaxPooling1D'
			else:
				matrix_op = 'tf.keras.layers.AveragePooling1D'

			layer_str = '''
class L{number}(tf.keras.layers.Layer):
	def __init__(self):
		super(L{number}, self).__init__()
	def build(self, input_shape):
		n = len(input_shape.as_list())
		self.check_3 = False
		self.check_2 = False
		if n == 3:
			self.check_3 = True
		elif n == 2:
			self.check_2 = True
		if self.check_3 == True:
			self.i_layer = {matrix_op}(pool_size={pool_val}, strides={stride_val}, padding='{padding_val}', data_format='{data_format}')
		elif self.check_2 == True:
			self.i_layer = {matrix_op}(pool_size={pool_val}, strides={stride_val}, padding='{padding_val}', data_format='{data_format}')
	def call(self, inputs):
		if self.check_3 == True:
			return {matrix_op}(pool_size={pool_val}, strides={stride_val}, padding='{padding_val}', data_format='{data_format}')(inputs)
		elif self.check_2 == True:
			return tf.squeeze( self.i_layer( tf.expand_dims(inputs,-1) ), axis=-1)
		return inputs
			'''.format(number=cell_indx,pool_val=pool_val, stride_val=stride_val, padding_val=padding_val, data_format=data_format, matrix_op=matrix_op)
			test_model = test_model + layer_str



		# Average_Max_Pooling_2D
		#GTCA
		elif nn[cell_indx][4][:5] in list_15 and len(nn[cell_indx][4]) > 6:
			pool_cut = nn[cell_indx][4][6:].find('T')
			if pool_cut == -1 or pool_cut ==len(nn[cell_indx][4][6:]):
				pool_val = 1+nucl_to_number(nn[cell_indx][4][6:], max_n_norm=8, stop='T')
			else:
				pool_val = []
				pool_val.append( 1+nucl_to_number(nn[cell_indx][4][6:pool_cut], max_n_norm=8, stop='T') )
				pool_val.append(1+nucl_to_number(nn[cell_indx][4][pool_cut:], max_n_norm=8, stop='T') )

			if nn[cell_indx][6] != '':
				stride_cut = nn[cell_indx][6].find('T')
				if stride_cut == -1 or stride_cut == len(nn[cell_indx][6]):
					stride_val = 1+nucl_to_number(nn[cell_indx][6], max_n_norm=8, stop='T')
				else:
					stride_val = []
					stride_val.append( 1+nucl_to_number(nn[cell_indx][6][6:pool_cut], max_n_norm=8, stop='T') )
					stride_val.append( 1+nucl_to_number(nn[cell_indx][6][pool_cut:], max_n_norm=8, stop='T') )
			else:
				stride_val = None


			if nn[cell_indx][4][5] == 'A':
				padding_val='valid'
				matrix_op = 'tf.keras.layers.AveragePooling2D'
			elif nn[cell_indx][4][5] == 'C':
				padding_val='valid'
				matrix_op = 'tf.keras.layers.MaxPooling2D'
			elif nn[cell_indx][4][5] == 'G':
				padding_val='same'
				matrix_op = 'tf.keras.layers.AveragePooling2D'
			elif nn[cell_indx][4][5] == 'T':
				padding_val='same'
				matrix_op = 'tf.keras.layers.MaxPooling2D'

			layer_str = '''
class L{number}(tf.keras.layers.Layer):
	def __init__(self):
		super(L{number}, self).__init__()
	def build(self, input_shape):
		n = len(input_shape.as_list())
		self.check_4 = False
		self.check_3 = False
		if n == 4:
			self.check_4 = True
		elif n == 3:
			self.check_3 = True
		if self.check_4 == True:
			self.i_layer = {matrix_op}(pool_size={pool_val}, strides={stride_val}, padding='{padding_val}')
		elif self.check_3 == True:
			self.i_layer = {matrix_op}(pool_size={pool_val}, strides={stride_val}, padding='{padding_val}')
	def call(self, inputs):
		if self.check_4 == True:
			return self.i_layer(inputs)
		elif self.check_3 == True:
			return tf.squeeze(self.i_layer( tf.expand_dims(inputs, -1) ), axis=-1)
		return inputs
			'''.format(number=cell_indx,pool_val=tuple(pool_val), stride_val=tuple(stride_val), padding_val=padding_val, matrix_op=matrix_op)
			test_model = test_model + layer_str





		# Batch_Normlaziations
		#GTCA
		elif nn[cell_indx][4][:5] in list_16 and len(nn[cell_indx][4]) > 6:
			expand_last_dim = ''
			expand_last_dim_end = ''
			squeeze_start = ''
			squeeze_end = ''
			if nn[cell_indx][4][5]  == 'C':
				axis_par = '0'
			elif nn[cell_indx][4][5] == 'G':
				axis_par = '1'
			elif nn[cell_indx][4][5] == 'T':
				axis_par = '-2'
			elif nn[cell_indx][4][5] == 'A':
				axis_par = '-1'
			if nn[cell_indx][4][6]  == 'C' or  nn[cell_indx][4][6] == 'A':
				expand_last_dim = 'tf.expand_dims('
				expand_last_dim_end = ', -1)'
				squeeze_start = 'tf.squeeze('
				squeeze_end = ', axis=-1)'


			layer_str = '''
class L{number}(tf.keras.layers.Layer):
	def __init__(self):
		super(L{number}, self).__init__()
		self.i_layer = tf.keras.layers.BatchNormalization(axis={axis_par})
	def call(self, inputs):
		return {squeeze_start}self.i_layer({exp_dim_start}inputs{exp_dim_end}){squeeze_end}
			'''.format(number=cell_indx,axis_par=axis_par, exp_dim_start=expand_last_dim, exp_dim_end=expand_last_dim_end, squeeze_start=squeeze_start, squeeze_end=squeeze_end)
			test_model = test_model + layer_str









		# Conv1D
		#paramters - filters:int , kernel_size:int, strides:int, padding:("valid", "same" or "causal") , data_format:(channels_last, channels_first)
		#GTCA
		elif nn[cell_indx][4][:5] in list_17 and len(nn[cell_indx][4]) > 7:
			par_check = False
			seq_cut = [m.start() for m in re.finditer('(?={})'.format('T'), nn[cell_indx][4][6:])]
			filters_par = 0
			kernel_par = 0
			strides_par = 0
			if len(seq_cut)>2:
				par_check = True
				

				last_i = 6
				cuts = []
				for cut_i in seq_cut:
					cuts.append([last_i, cut_i+6])
					last_i = cut_i+7
				counter_par = 0
				for cut in cuts:
					start_i = cut[0]
					end_i = cut[1]
					if counter_par == 0:
						filters_par = nucl_to_number( nn[cell_indx][4][start_i:end_i])
					elif counter_par == 1:
						kernel_par = nucl_to_number( nn[cell_indx][4][start_i:end_i])
					elif counter_par == 2:
						strides_par = nucl_to_number( nn[cell_indx][4][start_i:end_i])	
					else:
						break
					counter_par+=1

			if nn[cell_indx][4][5] == 'A':
				padding_par='valid'
			elif nn[cell_indx][4][5] == 'C':
				padding_par='valid'
			elif nn[cell_indx][4][5] == 'G':
				padding_par='same'
			elif nn[cell_indx][4][5] == 'T':
				padding_par='causal'




			layer_str = '''
class L{number}(tf.keras.layers.Layer):
	def __init__(self):
		super(L{number}, self).__init__()
	def build(self, input_shape):
		inp_lsit = input_shape.as_list()
		n = len(inp_lsit)
		self.check_3 = False
		self.check_2 = False
		self.kernel_size = {kernel_par}
		self.strides = {strides_par}
		self.filters_par = {filters_par}
		if maximum_elemts < self.filters_par:
			self.filters_par = maximum_elemts
		if n == 3 and {par_check}:
			self.check_3 = True
			if inp_lsit[-2] < self.kernel_size:
				self.kernel_size = inp_lsit[-2]
			if inp_lsit[-2] < self.strides:
				self.strides = inp_lsit[-2]
		elif n == 2 and {par_check}:
			self.check_2 = True
			if inp_lsit[-1] < self.kernel_size:
				self.kernel_size = inp_lsit[-1]
			if inp_lsit[-1] < self.strides:
				self.strides = inp_lsit[-1]
		if self.check_3 == True:
			self.i_layer = tf.keras.layers.Conv1D(filters=self.filters_par, kernel_size=self.kernel_size, strides=self.strides, padding='{padding_par}')
		elif self.check_2 == True:
			self.i_layer = tf.keras.layers.Conv1D(filters=self.filters_par, kernel_size=self.kernel_size, strides=self.strides, padding='{padding_par}')
	def call(self, inputs):
		if self.check_3 == True:
			return self.i_layer(inputs)
		elif self.check_2 == True:
			return self.i_layer( tf.expand_dims(inputs, -1) )
		return inputs
	'''.format(number=cell_indx, par_check = par_check, filters_par=filters_par, kernel_par=kernel_par, strides_par=strides_par, padding_par=padding_par)
			test_model = test_model + layer_str




		# Conv1D transpose
		#paramters - filters:int , kernel_size:int, strides:int, padding:("valid", "same" or "causal") , data_format:(channels_last, channels_first)
		#GTCAA
		elif nn[cell_indx][4][:5] in list_18 and len(nn[cell_indx][4]) > 7:
			par_check = False
			seq_cut = [m.start() for m in re.finditer('(?={})'.format('T'), nn[cell_indx][4][6:])]
			filters_par = 0
			kernel_par = 0
			strides_par = 0
			output_padding_par = None
			if len(seq_cut)>2:
				par_check = True
				

				last_i = 6
				cuts = []
				for cut_i in seq_cut:
					cuts.append([last_i, cut_i+6])
					last_i = cut_i+7
				counter_par = 0
				for cut in cuts:
					start_i = cut[0]
					end_i = cut[1]
					if counter_par == 0:
						filters_par = nucl_to_number( nn[cell_indx][4][start_i:end_i])
					elif counter_par == 1:
						kernel_par = nucl_to_number( nn[cell_indx][4][start_i:end_i])
					elif counter_par == 2:
						strides_par = nucl_to_number( nn[cell_indx][4][start_i:end_i])	
					elif counter_par == 3:
						output_padding_par = nucl_to_number( nn[cell_indx][4][start_i:end_i])	
						if output_padding_par >= strides_par:
							output_padding_par = None

					else:
						break
					counter_par+=1

			if nn[cell_indx][4][5] == 'A':
				padding_par='valid'
			elif nn[cell_indx][4][5] == 'C':
				padding_par='valid'
			elif nn[cell_indx][4][5] == 'G':
				padding_par='same'
			elif nn[cell_indx][4][5] == 'T':
				padding_par='same'




			layer_str = '''
class L{number}(tf.keras.layers.Layer):
	def __init__(self):
		super(L{number}, self).__init__()
	def build(self, input_shape):
		inp_lsit = input_shape.as_list()
		n = len(inp_lsit)
		self.check_3 = False
		self.check_2 = False
		self.kernel_size = {kernel_par}
		self.strides = {strides_par}
		self.filters_par = {filters_par}
		if maximum_elemts < self.filters_par:
			self.filters_par = maximum_elemts
		if n == 3 and {par_check}:
			self.check_3 = True
			if inp_lsit[-2] < self.kernel_size:
				self.kernel_size = inp_lsit[-2]
			if inp_lsit[-2] < self.strides:
				self.strides = inp_lsit[-2]
		elif n == 2 and {par_check}:
			self.check_2 = True
			if inp_lsit[-1] < self.kernel_size:
				self.kernel_size = inp_lsit[-1]
			if inp_lsit[-1] < self.strides:
				self.strides = inp_lsit[-1]
		self.output_padding_par = {output_padding_par}
		if self.output_padding_par != None:
			if self.output_padding_par >= self.strides:
				self.output_padding_par=None
		
		if self.check_3 == True:
			self.i_layer = tf.keras.layers.Conv1DTranspose(filters=self.filters_par, kernel_size=self.kernel_size, strides=self.strides, padding='{padding_par}', output_padding=self.output_padding_par)
		elif self.check_2 == True:
			self.i_layer = tf.keras.layers.Conv1DTranspose(filters=self.filters_par, kernel_size=self.kernel_size, strides=self.strides, padding='{padding_par}', output_padding=self.output_padding_par)
	def call(self, inputs):
		if self.check_3 == True:
			return self.i_layer(inputs)
		elif self.check_2 == True:
			return self.i_layer( tf.expand_dims(inputs, -1) )
		return inputs
	'''.format(number=cell_indx, par_check = par_check, filters_par=filters_par, kernel_par=kernel_par, strides_par=strides_par, padding_par=padding_par, output_padding_par=output_padding_par)
			test_model = test_model + layer_str






		# Conv1D separable
		#paramters - filters:int , kernel_size:int, strides:int, padding:("valid", "same" or "causal") , data_format:(channels_last, channels_first)
		#GTCA
		elif nn[cell_indx][4][:5] in list_19 and len(nn[cell_indx][4]) > 7:
			par_check = False
			seq_cut = [m.start() for m in re.finditer('(?={})'.format('T'), nn[cell_indx][4][6:])]
			filters_par = 0
			kernel_par = 0
			strides_par = 0
			depth_multiplier_par = 0
			if len(seq_cut)>3:
				par_check = True
				

				last_i = 6
				cuts = []
				for cut_i in seq_cut:
					cuts.append([last_i, cut_i+6])
					last_i = cut_i+7
				counter_par = 0
				for cut in cuts:
					start_i = cut[0]
					end_i = cut[1]
					if counter_par == 0:
						filters_par = nucl_to_number( nn[cell_indx][4][start_i:end_i])
					elif counter_par == 1:
						kernel_par = nucl_to_number( nn[cell_indx][4][start_i:end_i])
					elif counter_par == 2:
						strides_par = nucl_to_number( nn[cell_indx][4][start_i:end_i])	
					elif counter_par == 3:
						depth_multiplier_par = nucl_to_number( nn[cell_indx][4][start_i:end_i])	
					else:
						break
					counter_par+=1

			if nn[cell_indx][4][5] == 'A':
				padding_par='valid'
			elif nn[cell_indx][4][5] == 'C':
				padding_par='valid'
			elif nn[cell_indx][4][5] == 'G':
				padding_par='same'
			elif nn[cell_indx][4][5] == 'T':
				padding_par='causal'




			layer_str = '''
class L{number}(tf.keras.layers.Layer):
	def __init__(self):
		super(L{number}, self).__init__()
	def build(self, input_shape):
		inp_lsit = input_shape.as_list()
		n = len(inp_lsit)
		self.check_3 = False
		self.check_2 = False
		self.kernel_size = {kernel_par}
		self.strides = {strides_par}
		self.filters_par = {filters_par}
		if maximum_elemts < self.filters_par:
			self.filters_par = maximum_elemts
		if n == 3 and {par_check}:
			self.check_3 = True
			if inp_lsit[-2] < self.kernel_size:
				self.kernel_size = inp_lsit[-2]
			if inp_lsit[-2] < self.strides:
				self.strides = inp_lsit[-2]
		elif n == 2 and {par_check}:
			self.check_2 = True
			if inp_lsit[-1] < self.kernel_size:
				self.kernel_size = inp_lsit[-1]
			if inp_lsit[-1] < self.strides:
				self.strides = inp_lsit[-1]

		if self.check_3 == True:
			self.i_layer =tf.keras.layers.SeparableConv1D(filters=self.filters_par, kernel_size=self.kernel_size, strides=self.strides, padding='{padding_par}', depth_multiplier={depth_multiplier})
		elif self.check_2 == True:
			self.i_layer = tf.keras.layers.SeparableConv1D(filters=self.filters_par, kernel_size=self.kernel_size, strides=self.strides, padding='{padding_par}', depth_multiplier={depth_multiplier})
	def call(self, inputs):
		if self.check_3 == True:
			return self.i_layer(inputs)
		elif self.check_2 == True:
			return self.i_layer( tf.expand_dims(inputs, -1) )
		return inputs
	'''.format(number=cell_indx, par_check = par_check, filters_par=filters_par, kernel_par=kernel_par, strides_par=strides_par, padding_par=padding_par, depth_multiplier=depth_multiplier_par)
			test_model = test_model + layer_str




		#Conv1D LocallyConnected
		#paramters - filters:int , kernel_size:int, strides:int, padding:("valid", "same" or "causal") , data_format:(channels_last, channels_first)
		#GTCA
		elif nn[cell_indx][4][:5] in list_20 and len(nn[cell_indx][4]) > 7:
			par_check = False
			seq_cut = [m.start() for m in re.finditer('(?={})'.format('T'), nn[cell_indx][4][6:])]
			filters_par = 0
			kernel_par = 0
			strides_par = 0
			if len(seq_cut)>2:
				par_check = True
				

				last_i = 6
				cuts = []
				for cut_i in seq_cut:
					cuts.append([last_i, cut_i+6])
					last_i = cut_i+7
				counter_par = 0
				for cut in cuts:
					start_i = cut[0]
					end_i = cut[1]
					if counter_par == 0:
						filters_par = nucl_to_number( nn[cell_indx][4][start_i:end_i])
					elif counter_par == 1:
						kernel_par = nucl_to_number( nn[cell_indx][4][start_i:end_i])
					elif counter_par == 2:
						strides_par = nucl_to_number( nn[cell_indx][4][start_i:end_i])	
					else:
						break
					counter_par+=1

			if nn[cell_indx][4][5] == 'A':
				padding_par='valid'
			elif nn[cell_indx][4][5] == 'C':
				padding_par='valid'
			elif nn[cell_indx][4][5] == 'G':
				padding_par='same'
			elif nn[cell_indx][4][5] == 'T':
				padding_par='same'




			layer_str = '''
class L{number}(tf.keras.layers.Layer):
	def __init__(self):
		super(L{number}, self).__init__()
	def build(self, input_shape):
		inp_lsit = input_shape.as_list()
		n = len(inp_lsit)
		self.check_3 = False
		self.check_2 = False
		self.kernel_size = {kernel_par}
		self.strides = {strides_par}
		self.filters_par = {filters_par}
		if maximum_elemts < self.filters_par:
			self.filters_par = maximum_elemts
		if n == 3 and {par_check}:
			self.check_3 = True
			if inp_lsit[-2] < self.kernel_size:
				self.kernel_size = inp_lsit[-2]
			if inp_lsit[-2] < self.strides:
				self.strides = inp_lsit[-2]
		elif n == 2 and {par_check}:
			self.check_2 = True
			if inp_lsit[-1] < self.kernel_size:
				self.kernel_size = inp_lsit[-1]
			if inp_lsit[-1] < self.strides:
				self.strides = inp_lsit[-1]

		if self.check_3 == True:
			self.i_layer = tf.keras.layers.LocallyConnected1D(filters=self.filters_par, kernel_size=self.kernel_size, strides=self.strides, padding='{padding_par}', implementation = locally_connected_implementation)
		elif self.check_2 == True:
			self.i_layer = tf.keras.layers.LocallyConnected1D(filters=self.filters_par, kernel_size=self.kernel_size, strides=self.strides, padding='{padding_par}', implementation = locally_connected_implementation)
	def call(self, inputs):
		if self.check_3 == True:
			return self.i_layer(inputs)
		elif self.check_2 == True:
			return self.i_layer( tf.expand_dims(inputs, -1) )
		return inputs
	'''.format(number=cell_indx, par_check = par_check, filters_par=filters_par, kernel_par=kernel_par, strides_par=strides_par, padding_par=padding_par)
			test_model = test_model + layer_str





		#Conv2D
		#paramters - filters:int , kernel_size:int, strides:int, padding:("valid", "same" or "causal") , data_format:(channels_last, channels_first)
		#GTCA
		elif nn[cell_indx][4][:5] in list_21 and len(nn[cell_indx][4]) > 7:
			par_check = False
			seq_cut = [m.start() for m in re.finditer('(?={})'.format('T'), nn[cell_indx][4][6:])]
			filters_par = 1
			kernel_par_1 = 1
			kernel_par_2 = 1
			strides_par = 1
			strides_par_1 = 1
			strides_par_2 = 1
			if len(seq_cut)>2:
				par_check = True
				last_i = 6
				cuts = []
				for cut_i in seq_cut:
					cuts.append([last_i, cut_i+6])
					last_i = cut_i+7
				counter_par = 0
				for cut in cuts:
					start_i = cut[0]
					end_i = cut[1]
					if counter_par == 0:
						filters_par = nucl_to_number( nn[cell_indx][4][start_i:end_i])
					elif counter_par == 1:
						kernel_par_1 = nucl_to_number( nn[cell_indx][4][start_i:end_i])
					elif counter_par == 2:
						kernel_par_2 = nucl_to_number( nn[cell_indx][4][start_i:end_i])
					elif counter_par == 3:
						strides_par_1 = nucl_to_number( nn[cell_indx][4][start_i:end_i])
					elif counter_par == 4:
						strides_par_2 = nucl_to_number( nn[cell_indx][4][start_i:end_i])	
					else:
						break
					counter_par+=1

			if nn[cell_indx][4][5] == 'A':
				padding_par='valid'
				matrix_op = 'LocallyConnected2D'
			elif nn[cell_indx][4][5] == 'C':
				padding_par='valid'
				matrix_op = 'Conv2D'
			elif nn[cell_indx][4][5] == 'G':
				padding_par='same'
				matrix_op = 'LocallyConnected2D'
			elif nn[cell_indx][4][5] == 'T':
				padding_par='same'
				matrix_op = 'Conv2D'


			layer_str = '''
class L{number}(tf.keras.layers.Layer):
	def __init__(self):
		super(L{number}, self).__init__()
	def build(self, input_shape):
		inp_lsit = input_shape.as_list()
		n = len(inp_lsit)
		self.check_4 = False
		self.check_3 = False
		self.kernel_1 = {kernel_1_par}
		self.kernel_2 = {kernel_2_par}
		self.strides_1 = {strides_par_1}
		self.strides_2 = {strides_par_2}
		self.filters_par = {filters_par}
		if maximum_elemts < self.filters_par:
			self.filters_par = maximum_elemts
		if n == 4 and {par_check}:
			self.check_4 = True
			if inp_lsit[-2] <self.kernel_1:
				self.kernel_1 = inp_lsit[-2]
			if inp_lsit[-2] < self.strides_1:
				self.strides_1 = inp_lsit[-2]

			if inp_lsit[-3] < self.kernel_2:
				self.kernel_2 = inp_lsit[-3]
			if inp_lsit[-3] < self.strides_2:
				self.strides_2 = inp_lsit[-3]
		elif n == 3 and {par_check}:
			self.check_3 = True
			if inp_lsit[-1] <self.kernel_1:
				self.kernel_1 = inp_lsit[-1]
			if inp_lsit[-1] < self.strides_1:
				self.strides_1 = inp_lsit[-1]

			if inp_lsit[-2] < self.kernel_2:
				self.kernel_2 = inp_lsit[-2]
			if inp_lsit[-2] < self.strides_2:
				self.strides_2 = inp_lsit[-2]
		if '{matrix_op}' == 'Conv2D':
			self.layer_i = tf.keras.layers.Conv2D(filters=self.filters_par, kernel_size=(self.kernel_2,self.kernel_1), strides=(self.strides_1,self.strides_2), padding='{padding_par}')
		else:
			self.layer_i = tf.keras.layers.LocallyConnected2D(filters=self.filters_par, kernel_size=(self.kernel_2,self.kernel_1), strides=(self.strides_1,self.strides_2), padding='{padding_par}', implementation = locally_connected_implementation)
	def call(self, inputs):
		if self.check_4 == True:
			return self.layer_i(inputs)
		elif self.check_3 == True:
			return self.layer_i(tf.expand_dims(inputs, -1))
		return inputs
	'''.format(number=cell_indx, par_check = par_check, filters_par=filters_par, kernel_1_par=kernel_par_1, kernel_2_par=kernel_par_2, strides_par_1=strides_par_1, strides_par_2=strides_par_2, padding_par=padding_par, matrix_op=matrix_op)
			test_model = test_model + layer_str




		#Conv2D transpose
		#paramters - filters:int , kernel_size:int, strides:int, padding:("valid", "same" or "causal") , data_format:(channels_last, channels_first)
		#GTCA
		elif nn[cell_indx][4][:5] in list_22 and len(nn[cell_indx][4]) > 7:
			par_check = False
			seq_cut = [m.start() for m in re.finditer('(?={})'.format('T'), nn[cell_indx][4][6:])]
			filters_par = 0
			kernel_par_1 = 0
			kernel_par_2 = 0
			strides_par = 0
			strides_par_1 = 1
			strides_par_2 = 1
			output_padding_1 = None
			output_padding_2 = None
			if len(seq_cut)>2:
				par_check = True
				last_i = 6
				cuts = []
				for cut_i in seq_cut:
					cuts.append([last_i, cut_i+6])
					last_i = cut_i+7
				counter_par = 0
				for cut in cuts:
					start_i = cut[0]
					end_i = cut[1]
					if counter_par == 0:
						filters_par = nucl_to_number( nn[cell_indx][4][start_i:end_i])
					elif counter_par == 1:
						kernel_par_1 = nucl_to_number( nn[cell_indx][4][start_i:end_i])
					elif counter_par == 2:
						kernel_par_2 = nucl_to_number( nn[cell_indx][4][start_i:end_i])
					elif counter_par == 3:
						strides_par_1 = nucl_to_number( nn[cell_indx][4][start_i:end_i])
					elif counter_par == 4:
						strides_par_2 = nucl_to_number( nn[cell_indx][4][start_i:end_i])
					elif counter_par == 5:
						output_padding_1 = nucl_to_number( nn[cell_indx][4][start_i:end_i])
					elif counter_par == 6:
						output_padding_2 = nucl_to_number( nn[cell_indx][4][start_i:end_i])		
					else:
						break
					counter_par+=1

			if nn[cell_indx][4][5] == 'A':
				padding_par='valid'
			elif nn[cell_indx][4][5] == 'C':
				padding_par='valid'
			elif nn[cell_indx][4][5] == 'G':
				padding_par='same'
			elif nn[cell_indx][4][5] == 'T':
				padding_par='same'




			layer_str = '''
class L{number}(tf.keras.layers.Layer):
	def __init__(self):
		super(L{number}, self).__init__()
	def build(self, input_shape):
		inp_lsit = input_shape.as_list()
		n = len(inp_lsit)
		self.check_4 = False
		self.check_3 = False
		self.kernel_1 = {kernel_1_par}
		self.kernel_2 = {kernel_2_par}
		self.strides_1 = {strides_par_1}
		self.strides_2 = {strides_par_2}
		self.filters_par = {filters_par}
		if {output_padding_2} == None:
			self.output_padding = None
		else:
			self.output_padding = None
			if {strides_par_1} > {output_padding_1}:
				if {strides_par_2} > {output_padding_2}:
					self.output_padding = [{output_padding_1}, {output_padding_2}]

		if maximum_elemts < self.filters_par:
			self.filters_par = maximum_elemts
		if n == 4 and {par_check}:
			self.check_4 = True
			if inp_lsit[-2] <self.kernel_1:
				self.kernel_1 = inp_lsit[-2]
			if inp_lsit[-2] < self.strides_1:
				self.strides_1 = inp_lsit[-2]

			if inp_lsit[-3] < self.kernel_2:
				self.kernel_2 = inp_lsit[-3]
			if inp_lsit[-3] < self.strides_2:
				self.strides_2 = inp_lsit[-3]
		elif n == 3 and {par_check}:
			self.check_3 = True
			if inp_lsit[-1] <self.kernel_1:
				self.kernel_1 = inp_lsit[-1]
			if inp_lsit[-1] < self.strides_1:
				self.strides_1 = inp_lsit[-1]

			if inp_lsit[-2] < self.kernel_2:
				self.kernel_2 = inp_lsit[-2]
			if inp_lsit[-2] < self.strides_2:
				self.strides_2 = inp_lsit[-2]

		if self.check_4 == True:
			self.i_layer = tf.keras.layers.Conv2DTranspose(filters=self.filters_par,  kernel_size=(self.kernel_2,self.kernel_1), strides=(self.strides_1,self.strides_2), padding='{padding_par}', output_padding=self.output_padding)
		elif self.check_3 == True:
			self.i_layer = tf.keras.layers.Conv2DTranspose(filters=self.filters_par,  kernel_size=(self.kernel_2,self.kernel_1), strides=(self.strides_1,self.strides_2), padding='{padding_par}', output_padding=self.output_padding)
	def call(self, inputs):
		if self.check_4 == True:
			return self.i_layer(inputs)
		elif self.check_3 == True:
			return self.i_layer( tf.expand_dims(inputs, -1) )
		return inputs
	'''.format(number=cell_indx, par_check = par_check, filters_par=filters_par, kernel_1_par=kernel_par_1, kernel_2_par=kernel_par_2, strides_par_1=strides_par_1, strides_par_2=strides_par_2, padding_par=padding_par, output_padding_1=output_padding_1, output_padding_2=output_padding_2)
			test_model = test_model + layer_str


		#Conv2D separable
		#paramters - filters:int , kernel_size:int, strides:int, padding:("valid", "same" or "causal") , data_format:(channels_last, channels_first)
		#GTCA
		elif nn[cell_indx][4][:5] in list_23 and len(nn[cell_indx][4]) > 7:
			par_check = False
			seq_cut = [m.start() for m in re.finditer('(?={})'.format('T'), nn[cell_indx][4][6:])]
			filters_par = 0
			kernel_par_1 = 0
			kernel_par_2 = 0
			strides_par = 0
			depth_multiplier_par = 1
			if len(seq_cut)>2:
				par_check = True
				last_i = 6
				cuts = []
				for cut_i in seq_cut:
					cuts.append([last_i, cut_i+6])
					last_i = cut_i+7
				counter_par = 0
				for cut in cuts:
					start_i = cut[0]
					end_i = cut[1]
					if counter_par == 0:
						filters_par = nucl_to_number( nn[cell_indx][4][start_i:end_i])
					elif counter_par == 1:
						kernel_par_1 = nucl_to_number( nn[cell_indx][4][start_i:end_i])
					elif counter_par == 2:
						kernel_par_2 = nucl_to_number( nn[cell_indx][4][start_i:end_i])
					elif counter_par == 3:
						strides_par = nucl_to_number( nn[cell_indx][4][start_i:end_i])
					elif counter_par == 4:
						depth_multiplier_par = nucl_to_number( nn[cell_indx][4][start_i:end_i])
					else:
						break
					counter_par+=1

			if nn[cell_indx][4][5] == 'A':
				padding_par='valid'
			elif nn[cell_indx][4][5] == 'C':
				padding_par='valid'
			elif nn[cell_indx][4][5] == 'G':
				padding_par='same'
			elif nn[cell_indx][4][5] == 'T':
				padding_par='same'




			layer_str = '''
class L{number}(tf.keras.layers.Layer):
	def __init__(self):
		super(L{number}, self).__init__()
	def build(self, input_shape):
		inp_lsit = input_shape.as_list()
		n = len(inp_lsit)
		self.check_4 = False
		self.check_3 = False
		self.kernel_1 = {kernel_1_par}
		self.kernel_2 = {kernel_2_par}
		self.strides_par = {strides_par}
		self.filters_par = {filters_par}
		self.depth_multiplier = {depth_multiplier}

		if maximum_elemts < self.filters_par * self.depth_multiplier:
			self.filters_par = int(sqrt( maximum_elemts ))
			self.depth_multiplie = math.int(sqrt( maximum_elemts ))

		if n == 4 and {par_check}:
			self.check_4 = True
			if inp_lsit[-2] <self.kernel_1:
				self.kernel_1 = inp_lsit[-2]
			if inp_lsit[-2] < self.strides_par:
				self.strides_par = inp_lsit[-2]

			if inp_lsit[-3] < self.kernel_2:
				self.kernel_2 = inp_lsit[-3]
			if inp_lsit[-3] < self.strides_par:
				self.strides_par = inp_lsit[-3]
		elif n == 3 and {par_check}:
			self.check_3 = True
			if inp_lsit[-1] <self.kernel_1:
				self.kernel_1 = inp_lsit[-1]
			if inp_lsit[-1] < self.strides_par:
				self.strides_par = inp_lsit[-1]

			if inp_lsit[-2] < self.kernel_2:
				self.kernel_2 = inp_lsit[-2]
			if inp_lsit[-2] < self.strides_par:
				self.strides_par = inp_lsit[-2]
		
		if self.check_4 == True:	
			self.i_layer = tf.keras.layers.SeparableConv2D(filters=self.filters_par, kernel_size=(self.kernel_2,self.kernel_1), strides=self.strides_par, padding='{padding_par}', depth_multiplier=self.depth_multiplier)
		elif self.check_3 == True:
			self.i_layer = tf.keras.layers.SeparableConv2D(filters=self.filters_par, kernel_size=(self.kernel_2,self.kernel_1), strides=self.strides_par, padding='{padding_par}', depth_multiplier=self.depth_multiplier)

	def call(self, inputs):
		if self.check_4 == True:
			return self.i_layer(inputs)
		elif self.check_3 == True:
			return self.i_layer( tf.expand_dims(inputs, -1) )
		return inputs
	'''.format(number=cell_indx, par_check = par_check, filters_par=filters_par, kernel_1_par=kernel_par_1, kernel_2_par=kernel_par_2, strides_par=strides_par, padding_par=padding_par, depth_multiplier=depth_multiplier_par)
			test_model = test_model + layer_str







		#Conv2D Depthwise
		#paramters - filters:int , kernel_size:int, strides:int, padding:("valid", "same" or "causal") , data_format:(channels_last, channels_first)
		#GTCA
		elif nn[cell_indx][4][:5] in list_24 and len(nn[cell_indx][4]) > 7:
			par_check = False
			seq_cut = [m.start() for m in re.finditer('(?={})'.format('T'), nn[cell_indx][4][6:])]
			filters_par = 0
			kernel_par_1 = 0
			kernel_par_2 = 0
			strides_par = 0
			depth_multiplier_par = 1
			if len(seq_cut)>2:
				par_check = True
				last_i = 6
				cuts = []
				for cut_i in seq_cut:
					cuts.append([last_i, cut_i+6])
					last_i = cut_i+7
				counter_par = 0
				for cut in cuts:
					start_i = cut[0]
					end_i = cut[1]
					if counter_par == 0:
						kernel_par_1 = nucl_to_number( nn[cell_indx][4][start_i:end_i])
					elif counter_par == 1:
						kernel_par_2 = nucl_to_number( nn[cell_indx][4][start_i:end_i])
					elif counter_par == 2:
						strides_par = nucl_to_number( nn[cell_indx][4][start_i:end_i])
					elif counter_par == 3:
						depth_multiplier_par = nucl_to_number( nn[cell_indx][4][start_i:end_i])
					else:
						break
					counter_par+=1

			if nn[cell_indx][4][5] == 'A':
				padding_par='valid'
			elif nn[cell_indx][4][5] == 'C':
				padding_par='valid'
			elif nn[cell_indx][4][5] == 'G':
				padding_par='same'
			elif nn[cell_indx][4][5] == 'T':
				padding_par='same'




			layer_str = '''
class L{number}(tf.keras.layers.Layer):
	def __init__(self):
		super(L{number}, self).__init__()
	def build(self, input_shape):
		inp_lsit = input_shape.as_list()
		n = len(inp_lsit)
		self.check_4 = False
		self.check_3 = False
		self.kernel_1 = {kernel_1_par}
		self.kernel_2 = {kernel_2_par}
		self.strides_par = {strides_par}
		self.depth_multiplier = {depth_multiplier}

		if maximum_elemts < self.depth_multiplier:
			self.depth_multiplie = maximum_elemts

		if n == 4 and {par_check}:
			self.check_4 = True
			if inp_lsit[-2] <self.kernel_1:
				self.kernel_1 = inp_lsit[-2]
			if inp_lsit[-2] < self.strides_par:
				self.strides_par = inp_lsit[-2]

			if inp_lsit[-3] < self.kernel_2:
				self.kernel_2 = inp_lsit[-3]
			if inp_lsit[-3] < self.strides_par:
				self.strides_par = inp_lsit[-3]
		elif n == 3 and {par_check}:
			self.check_3 = True
			if inp_lsit[-1] <self.kernel_1:
				self.kernel_1 = inp_lsit[-1]
			if inp_lsit[-1] < self.strides_par:
				self.strides_par = inp_lsit[-1]

			if inp_lsit[-2] < self.kernel_2:
				self.kernel_2 = inp_lsit[-2]
			if inp_lsit[-2] < self.strides_par:
				self.strides_par = inp_lsit[-2]
		
		if self.check_4 == True:
			self.i_layer = tf.keras.layers.DepthwiseConv2D(kernel_size=(self.kernel_2,self.kernel_1), strides=self.strides_par, padding='{padding_par}', depth_multiplier=self.depth_multiplier)
		elif self.check_3 == True:
			self.i_layer = tf.keras.layers.DepthwiseConv2D(kernel_size=(self.kernel_2,self.kernel_1), strides=self.strides_par, padding='{padding_par}', depth_multiplier=self.depth_multiplier)

	def call(self, inputs):
		if self.check_4 == True:
			return self.i_layer(inputs)
		elif self.check_3 == True:
			return self.i_layer( tf.expand_dims(inputs, -1) )
		return inputs
	'''.format(number=cell_indx, par_check = par_check,  kernel_1_par=kernel_par_1, kernel_2_par=kernel_par_2, strides_par=strides_par, padding_par=padding_par, depth_multiplier=depth_multiplier_par)
			test_model = test_model + layer_str





		#Permute
		#paramters - filters:int , kernel_size:int, strides:int, padding:("valid", "same" or "causal") , data_format:(channels_last, channels_first)
		#GTCA
		elif nn[cell_indx][4][:5] in list_25 and len(nn[cell_indx][4]) > 6:
			seq_cut = nn[cell_indx][4][5:].find('T')

			layer_str = '''
class L{number}(tf.keras.layers.Layer):
	def __init__(self):
		super(L{number}, self).__init__()
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
	'''.format(number=cell_indx)
			test_model = test_model + layer_str



		#Relu threshhold
		#paramters - filters:int , kernel_size:int, strides:int, padding:("valid", "same" or "causal") , data_format:(channels_last, channels_first)
		#GTCA
		elif nn[cell_indx][4][:5] in list_26 and len(nn[cell_indx][4]) > 7:
			threshhold_par = str(nucl_to_number( nn[cell_indx][4][5:]))
			if len(threshhold_par) >3:
				threshhold_par = threshhold_par[:3]
			threshhold_par = int(threshhold_par)/500
			layer_str = '''
class L{number}(tf.keras.layers.Layer):
	def __init__(self):
		super(L{number}, self).__init__()
	def build(self, input_shape):
		self.i_layer = tf.keras.layers.ThresholdedReLU( theta={threshhold_par} )
	def call(self, inputs):
		return self.i_layer(inputs)
	'''.format(number=cell_indx, threshhold_par = threshhold_par)
			test_model = test_model + layer_str


		#Leaky relu
		#paramters - filters:int , kernel_size:int, strides:int, padding:("valid", "same" or "causal") , data_format:(channels_last, channels_first)
		#GTCA
		elif nn[cell_indx][4][:5] in list_27 and len(nn[cell_indx][4]) > 7:
			threshhold_par = str(nucl_to_number( nn[cell_indx][4][5:]))
			if len(threshhold_par) >3:
				threshhold_par = threshhold_par[:3]
			threshhold_par = int(threshhold_par)/500
			layer_str = '''
class L{number}(tf.keras.layers.Layer):
	def __init__(self):
		super(L{number}, self).__init__()
	def build(self, input_shape):
		self.i_layer = tf.keras.layers.LeakyReLU( alpha={threshhold_par} )
	def call(self, inputs):
		return self.i_layer(inputs)
	'''.format(number=cell_indx, threshhold_par = threshhold_par)
			test_model = test_model + layer_str


		#Dense
		#paramters - filters:int , kernel_size:int, strides:int, padding:("valid", "same" or "causal") , data_format:(channels_last, channels_first)
		#GTCA
		elif nn[cell_indx][4][:5] in list_28 and len(nn[cell_indx][4]) > 5:
			units = nucl_to_number( nn[cell_indx][4][5:],stop="T")
			if units>maximum_nodes:
				units=maximum_nodes
			layer_str = '''
class L{number}(tf.keras.layers.Layer):
	def __init__(self):
		super(L{number}, self).__init__()
	def build(self, input_shape):
		self.i_layer = tf.keras.layers.Dense( units={units} )
	def call(self, inputs):
		return self.i_layer(inputs)
	'''.format(number=cell_indx, units =units)
			test_model = test_model + layer_str


		#squeeze
		#paramters - filters:int , kernel_size:int, strides:int, padding:("valid", "same" or "causal") , data_format:(channels_last, channels_first)
		#GTCA
		elif nn[cell_indx][4][:5] in list_29 and len(nn[cell_indx][4]) > 5:
			layer_str = '''
class L{number}(tf.keras.layers.Layer):
	def __init__(self):
		super(L{number}, self).__init__()
	def build(self, input_shape):
		inp_lsit = input_shape.as_list()
		self.expand_check = False
		if inp_lsit[0] == 1:
			self.expand_check = True
	def call(self, inputs):
		out = tf.squeeze(inputs)
		if self.expand_check:
			return tf.expand_dims(out, axis=0)
		return out
	'''.format(number=cell_indx)
			test_model = test_model + layer_str




		#Activation layer
		#paramters - filters:int , kernel_size:int, strides:int, padding:("valid", "same" or "causal") , data_format:(channels_last, channels_first)
		#GTCA
		elif nn[cell_indx][4][:5] in list_activ and len(nn[cell_indx][4]) > 5:
			if activ_fun == '':
				activ_fun_par = None
			else:
				activ_fun_par = activ_fun
			layer_str = '''
class L{number}(tf.keras.layers.Layer):
	def __init__(self):
		super(L{number}, self).__init__()
	def build(self, input_shape):
		if {activ_fun} == None:
			self.activ_fun = None
			self.check = False
		else:
			self.activ_fun = {activ_fun}
			self.check = True
		self.i_layer = tf.keras.layers.Activation(activation=self.activ_fun)
	def call(self, inputs):
		if self.check:
			return self.i_layer(inputs)
		return inputs
	'''.format(number=cell_indx, activ_fun = activ_fun_par)
			test_model = test_model + layer_str




		#shared weights
		#paramters - filters:int , kernel_size:int, strides:int, padding:("valid", "same" or "causal") , data_format:(channels_last, channels_first)
		#GTCA
		elif nn[cell_indx][4][:5] in list_30 and len(nn[cell_indx][4]) > 5:
			layer_copy_n = nucl_to_number( nn[cell_indx][4][5:])
			if layer_copy_n >= len(end_graph):
				layer_copy_n = ( math.floor(layer_copy_n/len(end_graph)) ) * len(end_graph)

			layer_str = '''
class L{number}(tf.keras.layers.Layer):
	def __init__(self):
		super(L{number}, self).__init__()
	def build(self, input_shape):
		try:
			self.shared_layer = layer_{layer_copy_n}
		except NameError:
			self.shared_layer = layer_none
	def call(self, inputs):
		return self.shared_layer(inputs)
	'''.format(number=cell_indx, layer_copy_n = layer_copy_n)
			test_model = test_model + layer_str






































		else:
			layer_str = '''
class L{number}(tf.keras.layers.Layer):
	def __init__(self):
		super(L{number}, self).__init__()
	def call(self, inputs):
		return inputs
			'''.format(number=cell_indx)
			test_model = test_model + layer_str















	'''
	with open('/Users/danilkutny/Desktop/ENN/layer_tmp.py', 'w') as file:
		file.write(test_model)
	from layer_tmp import L0

	l0 = L0()
	x = tf.constant([[[5.,5.,1.,7.,5.,2.,4.],[6.,2.,3.,4.,1.,7.,5.],[5.,1.,8.,4.,4.,1.,2.],[4.,6.,5.,8.,4.,7.,1.],[2.,3.,4.,6.,8.,1.,8.],[5.,3.,8.,1.,2.,6.,2.]],[[5.,5.,1.,7.,5.,2.,4.],[6.,2.,3.,4.,1.,7.,5.],[5.,1.,8.,4.,4.,1.,2.],[4.,6.,5.,8.,4.,7.,1.],[2.,3.,4.,6.,8.,1.,8.],[5.,3.,8.,1.,2.,6.,2.]],[[5.,5.,1.,7.,5.,2.,4.],[6.,2.,3.,4.,1.,7.,5.],[5.,1.,8.,4.,4.,1.,2.],[4.,6.,5.,8.,4.,7.,1.],[2.,3.,4.,6.,8.,1.,8.],[5.,3.,8.,1.,2.,6.,2.]]] )
	x = tf.fill([1,randint(1,100),randint(1,100)],0.5)
	print('INP:',x)
	output = l0(x)
	print('OUT:',output)
	print("----------------------------------DONE--------------------------------------")
	'''






	old_strong_concat = '''
	class Strong_concat(tf.keras.layers.Layer):
		def __init__(self):
			super(Strong_concat, self).__init__()
		def build(self, input_shape):
			self.name = 'Fucking Big concat'
			#print('input_shape:',input_shape)
			k_max = 1
			maximum_shape = []
			for layer_raw in input_shape:
				#print('layer_raw:',layer_raw)
				shape_i = layer_raw.as_list()
				if len(shape_i)>k_max:
					k_max = len(shape_i)
				for ax_ind, axi in enumerate(reversed(shape_i)):
					if len(maximum_shape) >= ax_ind+1:
						if axi > maximum_shape[ax_ind]:
							 maximum_shape[ax_ind] = axi
					else:
						maximum_shape.append(axi)

			self.maximum_shape = list(reversed(maximum_shape))
			self.paddings = []
			for layer_raw in input_shape:
				shape_i = layer_raw.as_list()
				self.paddings.append([])
				maximum_shape_i = self.maximum_shape[-len(shape_i):]
				for axis_indx, axi in enumerate(shape_i):
					self.paddings[-1].append([0, maximum_shape_i[axis_indx]-axi])
				if len(shape_i) == len(self.maximum_shape):
					self.paddings[-1][0] = [0,0]
			#print('paddings:',self.paddings)
			print('maximum_shape:',self.maximum_shape)
		def call(self, inputs):
			new_concat_input = []
			for layer_raw, pad in zip(inputs, self.paddings):
				print('layer_raw:',layer_raw)
				print('paddings:',pad)
				#print('padding out:',layer_pad_out.shape)
				if tf.size(layer_raw)== 1:
					print("DOING FILL:",self.maximum_shape, tf.shape(layer_raw))
					new_concat_input.append( tf.fill( self.maximum_shape, tf.squeeze(layer_raw) ) )
				elif len(layer_pad_out.shape) == len(self.maximum_shape):
					layer_pad_out = tf.pad(layer_raw, pad)
					new_concat_input.append( layer_pad_out )
				else:
					layer_pad_out = tf.pad(layer_raw, pad)
					max_shape_i = self.maximum_shape[1:]
					max_shape_i.insert(0, 1)
					new_concat_input.append( tf.broadcast_to(layer_pad_out, max_shape_i) )
			layer_strong_concat = tf.keras.layers.concatenate(new_concat_input, 0)
			return layer_strong_concat
	'''

	strong_concat = '''
class Strong_concat(tf.keras.layers.Layer):
	def __init__(self):
		super(Strong_concat, self).__init__()
	def build(self, input_shape):
		k_max = 1
		maximum_shape = []
		for layer_raw in input_shape:
			shape_i = layer_raw.as_list()
			if len(shape_i)>k_max:
				k_max = len(shape_i)
			for ax_ind, axi in enumerate(reversed(shape_i)):
				if len(maximum_shape) >= ax_ind+1:
					if axi > maximum_shape[ax_ind]:
						 maximum_shape[ax_ind] = axi
				else:
					maximum_shape.append(axi)

		self.maximum_shape = list(reversed(maximum_shape))
		self.paddings = []
		for layer_raw in input_shape:
			shape_i = layer_raw.as_list()
			self.paddings.append([])
			if shape_i == [1]:
				self.paddings[-1].append(0)
			else:
				maximum_shape_i = self.maximum_shape[-len(shape_i):]
				for axis_indx, axi in enumerate(shape_i):
					self.paddings[-1].append([0, maximum_shape_i[axis_indx]-axi])
				if len(shape_i) == len(self.maximum_shape):
					self.paddings[-1][0] = [0,0]


	def call(self, inputs):
		new_concat_input = []
		for layer_raw, pad in zip(inputs, self.paddings):
			if pad == [0]:
				max_shape_i = self.maximum_shape[1:]
				max_shape_i.insert(0, 1)
				new_concat_input.append( tf.fill( max_shape_i, tf.squeeze(layer_raw) ) )
			else:
				layer_pad_out = tf.pad(layer_raw, pad)
				if len(layer_pad_out.shape) == len(self.maximum_shape):
					new_concat_input.append( layer_pad_out )
				else:
					max_shape_i = self.maximum_shape[1:]
					max_shape_i.insert(0, 1)
					new_concat_input.append( tf.broadcast_to(layer_pad_out, max_shape_i) )
		layer_strong_concat = tf.keras.layers.concatenate(new_concat_input, )
		return layer_strong_concat
	'''



	strong_concat_batch_wise = '''
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
			self.shape_len.append(len(layer_raw.as_list()))
			self.add_shapes.append(self.max_shape_len-self.shape_len[-1])
			self.paddings.append([])
			if shape_i == [1]:
				self.paddings[-1].append([0, self.maximum_shape[-1]-1])
			else:
				maximum_shape_i = self.maximum_shape[-len(shape_i):]
				for axis_indx, axi in enumerate(shape_i):
					self.paddings[-1].append([0, maximum_shape_i[axis_indx]-axi])
				#if len(shape_i) == len(self.maximum_shape):
					#self.paddings[-1][0] = [0,0]
			self.paddings[-1].insert(0, [0, 0])
	def call(self, inputs):
		new_concat_input = []
		for layer_raw, pad, shp, shp_l, ad_shp in zip(inputs, self.paddings, self.shapes, self.shape_len, self.add_shapes):
			if pad == [0]:
				max_shape_i = self.maximum_shape
				max_shape_i.insert(0, 1)
				max_shape_i.insert(0, 0)
				new_concat_input.append( tf.fill( max_shape_i, tf.squeeze(layer_raw) ) )
			else:
				layer_pad_out = tf.pad(layer_raw, pad)
				if shp_l == self.max_shape_len:
					new_concat_input.append( layer_pad_out )
				else:
					for i in range(ad_shp):
						layer_pad_out = tf.expand_dims(layer_pad_out, axis=1)
						if i != ad_shp:
							layer_pad_out = tf.repeat(layer_pad_out, repeats=self.maximum_shape_with_bacth[-i-2], axis=1)
					new_concat_input.append( layer_pad_out)
		layer_strong_concat = tf.keras.layers.concatenate(new_concat_input, 1)
		return layer_strong_concat
	'''




	strong_add_concat = '''
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
	'''

	test_model = test_model + strong_add_concat
	test_model += '''
set_weigths = {set_weigths}
	'''.format(set_weigths=set_weigths)
















	input_layer = start_nodes+rec_in
	#print('n of inputs::',len(input_layer))
	outlut_layer = end_nodes+rec_out
	done_nodes = deepcopy(start_nodes)
	curent_nodes = []
	delete_nodes = []
	new_current_nodes = []

	for i in input_layer:
		for i_2 in end_graph[i]:
			curent_nodes.append(i_2)
	curent_nodes = list(set(curent_nodes))
	#print('graph:',end_graph)
	#print('input_layer:',input_layer)
	#print('output layer', outlut_layer)
	#print('curent_nodes:',curent_nodes)

	#print('SHAPE!:',shape)
	if type(shape[0]) == int:
		shapes = iter([shape])
	else:
		shapes = iter(shape)

	output_shapes = []
	for i_l in input_layer:
		try:
			shape_tmp = next(shapes)
			test_model += '''
layer_{cell_number}_out = tf.keras.Input(shape={shape}, name="{cell_number}, Input", batch_size={batch_size})
		'''.format(cell_number=i_l,shape=shape_tmp, batch_size=batch_size)
			output_shapes.append(shape_tmp)
		except StopIteration:
			if len(nn[i_l][4]) > 1:
				if nn[i_l][4][0] == 'A' or nn[i_l][4][0] == 'C':
					tmp_ind = nn[i_l][4][1:].find('T')
					if tmp_ind!= -1 and tmp_ind!=len(nn[i_l][4][1:]):
						shape_tmp = [nucl_to_number(nn[i_l][4][1:], max_n_norm=int(maximum_nodes*(1/2)), stop='T'), nucl_to_number(nn[i_l][4][2+tmp_ind:], max_n_norm=int(maximum_nodes*(1/2)), stop='T')]
					else:
						shape_tmp = [nucl_to_number(nn[i_l][4][1:], max_n_norm=int(maximum_nodes*(1/2)), stop='T')]
				else:
					shape_tmp = [nucl_to_number(nn[i_l][4][1:], max_n_norm=int(maximum_nodes*(1/2)), stop='T')]
			else:
				shape_tmp=[100,100]
			output_shapes.append(shape_tmp)
			test_model += '''
layer_{cell_number}_out = tf.keras.Input(shape={shape}, name="{cell_number}, Input", batch_size={batch_size})
		'''.format(cell_number=i_l,shape=shape_tmp, batch_size=batch_size)
		done_nodes.append(i_l)





	'''
	print('Graph:',end_graph)
	print('done:')
	print('input_layer:',input_layer)
	'''
	counet_2 = 0
	while curent_nodes != []:
		#print('whl:',counet_2)
		if counet_2 > 100:
			counet_2 = 0
			break
		#print('#{}'.format(counet_2), curent_nodes)
		counet_2+=1
		con = []
		for cur_node in curent_nodes:
			con = con+rev_graph[cur_node]
			'''
			if cur_node in done_nodes:
				print("HERE!:",cur_node)
				curent_nodes = []
				print('done_nodes:',done_nodes)
				print('rev_graph node:',rev_graph[cur_node])
				break
			'''
			if all(elem in done_nodes for elem in rev_graph[cur_node]):
				#print('EXIST!!!!')
				delete_nodes.append(cur_node)
				done_nodes.append(cur_node)
				#print('append done node by:',cur_node)
				for i in end_graph[cur_node]:
					if i not in done_nodes:
						new_current_nodes.append(i)
				if len(rev_graph[cur_node]) > 0:
					prev_layers = '['
					for l_input in rev_graph[cur_node]:
						prev_layers+='layer_{l_input}_out, '.format(l_input=l_input)
					prev_layers = prev_layers[:-2]
					prev_layers+=']'

					if len(rev_graph[cur_node]) == 1:
						test_model +='''

layer_{cur_node} = L{cur_node}()
layer_{cur_node}_out = layer_{cur_node}(layer_{l_input}_out)
'''.format(cur_node=cur_node, l_input=rev_graph[cur_node][0])
					else:
						test_model += '''
layer_{cur_node} = L{cur_node}()
try:
	try:
		layer_{cur_node}_concat = tf.keras.layers.concatenate({prev_layer})
		layer_{cur_node}_out = layer_{cur_node}(layer_{cur_node}_concat)
	except ValueError:
		layer_{cur_node}_concat = tf.keras.layers.concatenate({prev_layer}, axis=-2)
		layer_{cur_node}_out = layer_{cur_node}(layer_{cur_node}_concat)
except ValueError:
	#print('do strong concat!')
	strong_concat_{cur_node} = Strong_concat()
	new_list_of_inputs = []
	for i in {prev_layer}:
		if type(i) == tuple:
			for i_2 in i:
				new_list_of_inputs.append(i_2)
		else:
			new_list_of_inputs.append(i)

	layer_{cur_node}_concat = strong_concat_{cur_node}(new_list_of_inputs)
	layer_{cur_node}_out = layer_{cur_node}(layer_{cur_node}_concat)
if layer_{cur_node}.count_params() != 0:
	if {cur_node} in set_weigths:
		if len(layer_{cur_node}.get_weights()) > 1:
			new_w = [get_new_weigths(layer_{cur_node}, set_weigths[{cur_node}])]+layer_{cur_node}.get_weights()[1:]
		else:
			new_w = [get_new_weigths(layer_{cur_node}, set_weigths[{cur_node}])]
		layer_{cur_node}.set_weights(new_w)


try:
	count = 1
	for s in layer_{cur_node}_out.shape:
		count = count*s
		if count > {max_elements}:
			raise NameError
	#print('do #{cur_node}:',count)
except AttributeError:
	for i in layer_{cur_node}_out:
		count = 1
		for s in layer_{cur_node}_out.shape:
			count = count*s
			if count > {max_elements}:
				raise NameError
		#print('do #{cur_node}:',count)


'''.format(cur_node=cur_node, prev_layer=prev_layers, max_elements=max_elements)
		#print('connections:',res)
		#print('done_nodes:',done_nodes)
		for i_d in delete_nodes:
			curent_nodes.remove(i_d)
		for i_n in new_current_nodes:
			curent_nodes.append(i_n)


		curent_nodes = list(set(curent_nodes))
		
		res = []
		for i in curent_nodes:
		    if i not in res:
		        res.append(i)

		curent_nodes = res
		
		delete_nodes = []
		new_current_nodes = []

	for i in outlut_layer:
		test_model+='''
layer_{cell_number}_out = tf.keras.layers.Flatten()(layer_{cell_number}_out)
layer_{cell_number}_out = tf.keras.layers.Dense({out_shape}, activation='sigmoid')(layer_{cell_number}_out)'''.format(cell_number=i, out_shape=out_shape)

	input_layers_list_str = '['
	output_layers_list_str = '['
	for i in input_layer:
		input_layers_list_str+='layer_{cell_number}_out, '.format(cell_number=i)
	input_layers_list_str = input_layers_list_str[:-2]
	input_layers_list_str+=']'

	for i in outlut_layer:
		output_layers_list_str+='layer_{cell_number}_out, '.format(cell_number=i)
	output_layers_list_str = output_layers_list_str[:-2]
	output_layers_list_str+=']'

	test_model += '''
model = tf.keras.Model(
    inputs={inputs},
    outputs={outputs},
)

\'\'\'
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
\'\'\'
	'''.format(inputs=input_layers_list_str,outputs=output_layers_list_str)
	dir_path = os.path.dirname(os.path.realpath(__file__))
	with open('{url}/model_tmp.py'.format(url=dir_path), 'w') as file:
		file.write(test_model)
	
	#from model_tmp import model
	return output_shapes, len(outlut_layer), input_layer, counet_2
def nn_to_tensorflow(nn, shape=shape, maximum_elemts=maximum_elemts, maximum_nodes=maximum_nodes, locally_connected_implementation=locally_connected_implementation, time_limit=4):
	signal.signal(signal.SIGALRM, signal_handler)
	signal.alarm(time_limit)
	try:
		y, z, input_layer, counet_2 = nn_to_tensorflow_0(nn, shape=shape, maximum_elemts=maximum_elemts, maximum_nodes=maximum_nodes, locally_connected_implementation=locally_connected_implementation)
		print('nn_to_tf done')
		signal.alarm(0)
		return y, z, counet_2
	except RuntimeError:
		signal.alarm(0)
		return None
	except TypeError:
		signal.alarm(0)
		return None
'''
str_seq = 'QWERTYUIOPASDFGHJKLZXCVBNM'
dna_letter = 'ACGT'
nn = []

for i in range(40):
	random_name = str_seq[randint(0,len(str_seq)-1)]
	random_connection = [str_seq[randint(0,len(str_seq)-1)] for i in range(randint(0,3))]
	random_matrx = ''
	random_func = ''
	random_weights = ''
	for i in range(randint(1,200)):
		random_matrx += dna_letter[randint(0,3)]
		random_func += dna_letter[randint(0,3)]
		random_weights += dna_letter[randint(0,3)]
	nn.append([None, None, random_name, random_connection, random_matrx, random_func, random_weights])
nn.append([None, None, random_name, random_connection, random_matrx, random_func, random_weights])

#print('nn:',nn)
graph = nn_to_graph(nn)
end_graph, rev_graph, start_nodes, end_nodes, rec_in, rec_out = create_normalized_graph(graph)
in_l = start_nodes+rec_in
#a, b, in_l = nn_to_tensorflow(nn)
nn.append([None, None, '!!!!!!!', [], '!!!!!!!', '!!!!!!!', '!!!!!!!'])
for i in in_l:
	for cell_indx, cell in enumerate(nn):
		if cell_indx == i:
			nn[-1][3].append(cell[2])
#print('last:',nn[-1])


a, b, in_l = nn_to_tensorflow(nn, shape=[28,28])
from model_tmp import model
model.summary()
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

'''
nn = [ ['',{'A':1},'AAA',['CCC'],'TTTTTT','', ''], ['',{'A':1},'CCC',['DDD'],'TTTTTT','', ''], ['',{'A':1},'DDD',[],'TTTTTT','', ''] ]
model, a, b = nn_to_tensorflow(nn)

model.summary()

tf.keras.utils.plot_model(model, "random_model_100.png",show_shapes=True)
#tnsfl_model = model
'''



























