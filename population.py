#Populaation
import tensorflow as tf
from tensorflow import keras
from GenPile_2 import NN
from tensorflow_model import nn_to_tensorflow
import random
import math
import statistics
from random import randint
import re
import os
from time import sleep
from time import time as tme
import json
import concurrent.futures
import signal


training_time = 420
number_nns = 10
dir_path = os.path.dirname(os.path.realpath(__file__))
start_pop = '{url}/boost_perfomnc_gen/gen_2.json'.format(url=dir_path)
save_folder = "boost_perfomnc_gen"


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

def signal_handler(signum, frame):
    raise RuntimeError

def pairwise(iterable):
	iter_tmp = iter(iterable)
	paiwise_list = []
	for i in range(math.floor(len(iterable)/2)):
		a = next(iter_tmp)
		b = next(iter_tmp)
		paiwise_list.append([a,b])
	return paiwise_list

def find_simular(list_1, list_2):
	len1  = len(list_1)
	len2  = len(list_2)
	simualrity_l1 = {}
	for elem_ind, elem in enumerate(list_1):
		simualrity_l1[elem_ind] = []
		elem_len = len(elem)
		sim_n1 = randint(0,elem_len-8)
		sim_n2 = randint(0,elem_len-8)
		sim_n3 = randint(0,elem_len-8)
		sim_1 = elem[sim_n1:sim_n1+8]
		sim_2 = elem[sim_n2:sim_n2+8]
		sim_3 = elem[sim_n3:sim_n3+8]
		for seq_2_ind, seq_2 in enumerate(list_2):
			true_n = 0
			a = seq_2.find(sim_1)
			b = seq_2.find(sim_2)
			c = seq_2.find(sim_3)
			if a != -1:
				true_n+=1
			if b != -1:
				true_n+=1
			if c != -1:
				true_n+=1
			if true_n == 3:
				simualrity_l1[elem_ind].append(seq_2_ind)
	return simualrity_l1

def reverse_grapth(d, l2):
	new_dic = {}
	for key, value in d.items():
		if value == []:
			pass
		else:
			for val in value:
				if val in new_dic:
					if key not in new_dic[val]:
						new_dic[val].append(key)
				else:
					new_dic[val] = [key]
	for k in range(len(l2)):
		if k not in new_dic:
			new_dic[k] = []

	sorted_dic = {}
	for k in sorted(new_dic.keys()):
		sorted_dic[k]=new_dic[k]
	return sorted_dic


def find_simular(list_1, list_2):
	len1  = len(list_1)
	len2  = len(list_2)
	simualrity_l1 = {}
	for elem_ind, elem in enumerate(list_1):
		simualrity_l1[elem_ind] = []
		elem_len = len(elem)
		#print('elem_len:',elem_len)
		sim_n1 = randint(0,elem_len-8)
		sim_n2 = randint(0,elem_len-8)
		sim_n3 = randint(0,elem_len-8)
		sim_1 = elem[sim_n1:sim_n1+8]
		sim_2 = elem[sim_n2:sim_n2+8]
		sim_3 = elem[sim_n3:sim_n3+8]
		for seq_2_ind, seq_2 in enumerate(list_2):
			true_n = 0
			a = seq_2.find(sim_1)
			b = seq_2.find(sim_2)
			c = seq_2.find(sim_3)
			if a != -1:
				true_n+=1
			if b != -1:
				true_n+=1
			if c != -1:
				true_n+=1
			if true_n > 1:
				simualrity_l1[elem_ind].append(seq_2_ind)
	return simualrity_l1
def reverse_grapth(d, l2):
	new_dic = {}
	for key, value in d.items():
		if value == []:
			pass
		else:
			for val in value:
				if val in new_dic:
					if key not in new_dic[val]:
						new_dic[val].append(key)
				else:
					new_dic[val] = [key]
	for k in range(len(l2)):
		if k not in new_dic:
			new_dic[k] = []

	sorted_dic = {}
	for k in sorted(new_dic.keys()):
		sorted_dic[k]=new_dic[k]
	return sorted_dic

def recombine_seq(seq1, seq2, recombination_site):
	#find all appearance of recombinatio nsite in the genomes(both seq1 and seq2)
	#1
	recomb1_start_indxs = [m.start() for m in re.finditer('(?={})'.format(recombination_site), seq1)]
	#2
	recomb2_start_indxs = [m.start() for m in re.finditer('(?={})'.format(recombination_site), seq2)]

	#adding each piece of recombination dna into seaprate list of dan pieces, according to list of recobination sites for further recombination !!!!   Paramter - 4
	recomb_seqs_1 = []
	recomb_seqs_2 = []
	start_seq = 0
	length = 4
	#1
	for ind in recomb1_start_indxs:
		if ind-start_seq>len(recombination_site)+length:
			add_seq_tmp = seq1[start_seq:ind]
			recomb_seqs_1.append(seq1[start_seq:ind])
			start_seq = ind
	if len(seq1[start_seq:]) > len(recombination_site)+length:
		recomb_seqs_1.append(seq1[start_seq:])
	else:
		recomb_seqs_1[-1] = recomb_seqs_1[-1]+seq1[start_seq:]
	#2
	start_seq = 0
	for ind in recomb2_start_indxs:
		if ind-start_seq>len(recombination_site)+length:
			recomb_seqs_2.append(seq2[start_seq:ind])
			start_seq = ind
	if len(seq2[start_seq:]) > len(recombination_site)+length:
		recomb_seqs_2.append(seq2[start_seq:])
	else:
		recomb_seqs_2[-1] = recomb_seqs_2[-1]+seq2[start_seq:]
	recomb_seqs_2 = [x for x in recomb_seqs_2 if x != '']

	#find all simular of one list and other one, reproduced as a graph
	simular_dic1 = find_simular(recomb_seqs_1, recomb_seqs_2)
	
	#Do recombination of simular piecs 
	new_list_1 = []
	new_lsit_2 = []

	#1
	counter = 0
	add_to_seqcond_list_1 = []
	for seq_1, seq_2_list in simular_dic1.items():
		if seq_2_list != []:
			if random.randint(0,1)==1:
				new_list_1.append(recomb_seqs_1[seq_1])
			else:
				new_list_1.append(recomb_seqs_2[seq_2_list[0]])
		else:
			new_list_1.append(recomb_seqs_1[seq_1])
			add_to_seqcond_list_1.append([counter,seq_1])
		counter+=1
	
	#2
	simular_dic2 = reverse_grapth(simular_dic1, recomb_seqs_2)
	add_to_seqcond_list_2 = []
	counter = 0
	for seq_2, seq_1_list in simular_dic2.items():
		if seq_1_list != []:
			if random.randint(0,1)==1:
				new_lsit_2.append(recomb_seqs_2[seq_2])
			else:
				new_lsit_2.append(recomb_seqs_1[seq_1_list[0]])
		else:
			new_lsit_2.append(recomb_seqs_2[seq_2])
			add_to_seqcond_list_2.append([counter, seq_2])
		counter+=1
	
	#with probability of 50% - add strange dna piece, whoch is non-honologis to this dna
	#counter make sure, that piece is inserted in the rigth place, according to another dna 
	counter = 0
	for x in add_to_seqcond_list_1:
		indx, seq_n = x
		if random.randint(0,1)==1:
			new_lsit_2.insert(indx+counter, recomb_seqs_1[seq_n])
		counter+=1

	counter = 0
	for x in add_to_seqcond_list_2:
		indx, seq_n = x
		if random.randint(0,1)==1:
			new_list_1.insert(indx+counter, recomb_seqs_2[seq_n])

	#glue all pieces together 
	new_seq_1 = ''
	for i in new_list_1:
		new_seq_1+=i
	new_seq_2 = ''
	for i in new_lsit_2:
		new_seq_2+=i

	return new_seq_1, new_seq_2

def mutation(new_seq_1, mutation_rate=[None, None, None, None, None, None, None]):
	dna_letters = 'ACGT'
	mutation_rate=[None, None, None, None, None, None, None]
	if mutation_rate[0] == None:
		mutation_rate[0] = int(len(new_seq_1)/200)
	if mutation_rate[1] == None:
		mutation_rate[1] = int(len(new_seq_1)/100)
	if mutation_rate[2] == None:
		mutation_rate[2] = int(len(new_seq_1)/100)
	if mutation_rate[3] == None:
		mutation_rate[3] = int(len(new_seq_1)/100)
	if mutation_rate[4] == None:
		mutation_rate[4] = int(len(new_seq_1)/100)
	if mutation_rate[5] == None:
		mutation_rate[5] = int(len(new_seq_1)/2)
	if mutation_rate[6] == None:
		mutation_rate[6] = int(len(new_seq_1))
	#1 - single letter mutation
	seq_length = len(new_seq_1)
	n = int(3*seq_length/mutation_rate[0])
	for i in range(n):
		random_indx = []
		if random.randint(0, 2) == 0:
			random_indx.append(random.randint(1,seq_length))
		random_indx.sort()
		new_seq_2 = ''
		start=0
		for indx_i in random_indx:
			new_seq_2+=new_seq_1[start:indx_i-1] + dna_letters[random.randint(0,3)]
			start = indx_i
		if start!= len(new_seq_1):
			new_seq_2 += new_seq_1[start:]
		new_seq_1 = str(new_seq_2)
			#new_seq_1 = new_seq_1[:random_indx-1] + dna_letters[random.randint(0,3)] + new_seq_1[random_indx:]
	n = int(3*seq_length/mutation_rate[1])
	for i in range(n):
		random_indx = []
		if random.randint(0, 2) == 0:
			random_indx.append(random.randint(0,seq_length))
		random_indx.sort()
		new_seq_2 = ''
		start=0
		for indx_i in random_indx:
			new_seq_2 += new_seq_1[start:indx_i] + dna_letters[random.randint(0,3)]
			start = indx_i
		if start!= len(new_seq_1):
			new_seq_2 += new_seq_1[start:]
		new_seq_1 = str(new_seq_2)
	n = int(3*seq_length/mutation_rate[2])
	for i in range(n):
		random_indx = []
		if random.randint(0, 2) == 0:
			random_indx.append(random.randint(1,seq_length))
		random_indx.sort()
		new_seq_2 = ''
		start=0
		for indx_i in random_indx:
			new_seq_2 += new_seq_1[start:indx_i-1]
			start = indx_i
		if start!= len(new_seq_1):
			new_seq_2 += new_seq_1[start:]
		new_seq_1 = str(new_seq_2)
	n = int(3*seq_length/mutation_rate[3])
	for i in range(n):
		random_indx = []
		if random.randint(0, 2) == 0:
			random_indx.append(random.randint(0,seq_length))
		random_indx.sort()
		new_seq_2 = ''
		start=0
		for indx_i in random_indx:
			new_seq_2 += new_seq_1[start:indx_i] + dna_letters[random.randint(0,3)] + dna_letters[random.randint(0,3)] + dna_letters[random.randint(0,3)]
			start = indx_i
		if start!= len(new_seq_1):
			new_seq_2 += new_seq_1[start:]
		new_seq_1 = str(new_seq_2)
	n = int(3*seq_length/mutation_rate[4])
	for i in range(n):
		random_indx = []
		if random.randint(0, 2) == 0:
			random_indx.append(random.randint(3,seq_length))
		random_indx.sort()
		new_seq_2 = ''
		start=0
		for indx_i in random_indx:
			new_seq_2 += new_seq_1[start:indx_i-3]
			start = indx_i
		if start!= len(new_seq_1):
			new_seq_2 += new_seq_1[start:]
		new_seq_1 = str(new_seq_2)
	n = int(3*seq_length/mutation_rate[5])
	for i in range(n):
		'''
		#6 - piece of dna removal
		if random.randint(0, 2) == 0:
			random_length = random.randint(0,8096)
			random_indx = random.randint(random_length,seq_length)
			new_seq_1 = new_seq_1[:random_indx-random_length] + new_seq_1[random_indx:]
		'''
		random_indx_dic = {}
		random_indx = []
		if random.randint(0, 2) == 0:
			random_length = random.randint(0,8096)
			r_in = random.randint(random_length,seq_length)
			random_indx.append(r_in)
			random_indx_dic[r_in] = random_length
			random_indx.append(r_in)

		random_indx.sort()
		new_seq_2 = ''
		start=0
		for indx_i in random_indx:
			new_seq_2 += new_seq_1[start:indx_i-random_indx_dic[indx_i]]
			start = indx_i
		if start!= len(new_seq_1):
			new_seq_2 += new_seq_1[start:]
		new_seq_1 = str(new_seq_2)
	n = int(3*seq_length/mutation_rate[6])
	for i in range(n):
		'''
		#7 - piece of dna duplication
		if random.randint(0, 2) == 0:
			random_length = random.randint(0,8096)
			random_indx = random.randint(random_length,seq_length)
			new_seq_1 = new_seq_1[:random_indx] + new_seq_1[random_indx-random_length:random_indx] + new_seq_1[random_indx:]
		'''
		random_indx_dic = {}
		random_indx = []
		if random.randint(0, 2) == 0:
			random_length = random.randint(0,8096)
			r_in = random.randint(random_length,seq_length)
			random_indx.append(r_in)
			random_indx_dic[r_in] = random_length
			random_indx.append(r_in)

		random_indx.sort()
		new_seq_2 = ''
		start=0
		for indx_i in random_indx:
			new_seq_2 += new_seq_1[start:indx_i]+new_seq_1[indx_i-random_indx_dic[indx_i]:indx_i]
			start = indx_i
		if start != len(new_seq_1):
			new_seq_2 += new_seq_1[start:]
		new_seq_1 = str(new_seq_2)
	return new_seq_1

def recombine_seq2(seq1, seq2, recombination_site):
	chanse_const = 256
	n1_random = random.randint(0,len(seq1)-chanse_const)
	n2_random = random.randint(0,len(seq2)-chanse_const)
	if n1_random > n2_random:
		n1_random = n2_random	

	new_seq1 = seq1[:n1_random]+seq2[n1_random:]
	new_seq2 = seq2[:n1_random]+seq1[n1_random:]
	return new_seq1, new_seq2

def recombine_seq2_2(seq1, seq2, recombination_site):
	chanse_const = 1024
	n_random = random.randint(0,len(seq1)-chanse_const)
	find_site = seq1[n_random:n_random+chanse_const]
	indx_find = seq2.find(find_site)
	if indx_find == -1:
		return seq1, seq2
	else:
		new_seq1 = seq1[:n_random]+seq2[indx_find:]
		new_seq2 = seq2[:indx_find]+seq1[n_random:]
		return new_seq1, new_seq2
@tf.function
def batch_train(imgs, lbls, dummy, rnge1, rnge2):
	loss_sum = []
	with tf.GradientTape(watch_accessed_variables=False) as tape:
		tape.watch(model.trainable_weights)
		for step in range(rnge1, rnge2):
			input_tensor = [tf.expand_dims(imgs[step], axis=0)]
			input_tensor = input_tensor + dummy
			y_pred_raw = model(input_tensor)
			y_true = train_labels[step]
			y_pred = tf.reshape(y_pred_raw[0], [-1])[:10]
			loss_sum.append(loss_func(tf.one_hot(y_true, depth), y_pred))
		grads = tape.gradient(sum(loss_sum), model.trainable_variables)
		optimizer.apply_gradients(zip(grads, model.trainable_variables))
#max_layers=self.max_layers, max_time=self.max_dev_time, maximum_elemts=self.maximum_elemts,  maximum_nodes=self.maximum_nodes, 
#locally_connected_implementation=self.locally_connected_implementation, shape=self.shape
def develop_funct(genome, max_layers, max_time, maximum_elemts, maximum_nodes, locally_connected_implementation, shape):
	tmp_nn = NN(genome, max_layers=max_layers, max_time=max_time)
	#t1 = time()
	#try:
	tmp_nn.develop()
	#except Exception:
	#	pass
	#t2 = time()
	#print('stop develop:',t2-t1)
	number_cells = 0
	number_divergent_names = []
	number_of_connections = 0
	mutation = tmp_nn.mutation_rate
	unique_matrix_op = []
	for cell in tmp_nn.nn:
		number_cells+=1
		if len(cell[2]) >6:
			r_n_c = random.randint(0, len(cell[2])-6)
			check=False
			for seq_tmp in number_divergent_names:
				if cell[2][r_n_c:r_n_c+6] in seq_tmp:
					check = True
		else:
			check=False
			for seq_tmp in number_divergent_names:
				if cell[2] in seq_tmp:
					check=True
		if check == False:
			number_divergent_names.append(cell[2])
		for conect in cell[3]:
			number_of_connections+=1
		
		add_check = True
		list_some = create_1024()
		if len(cell[4]) < 5:
			add_check = False
		if cell[4][:5] in set(list_some[500:]):
			add_check = False
		if add_check == True:
			for sing_op in unique_matrix_op:
				if cell[4][:5] in sing_op[:5] or sing_op[:5] in cell[4][:5]:
					add_check = False
		if add_check == True:
			unique_matrix_op.append(cell[4])

	#try:
		#model = nn_to_tensorflow(tmp_nn.nn, maximum_elemts=maximum_elemts, maximum_nodes=maximum_nodes, 
		#locally_connected_implementation=locally_connected_implementation, shape=shape, time_limit=max_time)
	#except Exception:
		#population.append(None)
	#print('mutation_rate:',tmp_nn.mutation_rate)
	return [number_cells, len(number_divergent_names), number_of_connections, genome], tmp_nn.nn, mutation, len(unique_matrix_op), tmp_nn.learning_rate

def develop_funct_wrapper(arg):
	genome, max_layers, max_time, maximum_elemts, maximum_nodes, locally_connected_implementation, shape = arg
	return develop_funct(genome, max_layers, max_time, maximum_elemts, maximum_nodes, locally_connected_implementation, shape)

class Population():
	#maximum genes, number of individual, type of selection (roulette wheel), mutation rate (constant or not), selection, fintenss function, max naumber of geenerationions,
	#
	def __init__(self, max_individuals=300, genome_max_size = 8000000, mutatation_rate=None, selection_strategy='RL', selection_rate=0.5, max_layers=1000,  max_dev_time=1, max_tf_time = 5,
		maximum_elemts=300000000, maximum_nodes=1000, locally_connected_implementation=2, shape=[[100, 100]]):
		self.genome_max = genome_max_size
		self.mut_rate = mutatation_rate
		self.select_strategy =  selection_strategy
		self.select_rate = selection_rate
		self.max_ind = max_individuals
		self.max_layers = max_layers
		self.max_dev_time = max_dev_time
		self.maximum_elemnts = maximum_elemts
		self.maximum_nodes = maximum_nodes
		self.locally_connected_implementation = locally_connected_implementation
		self.shape = shape
		self.population_geneome = []
		self.split = 0.2
		self.recombination_site = 'TTTTTT'
		self.maximum_parameters = 4000000
		self.mutation = []

	def make_random_pop(self, individuals=300, chromatin_var=False):
		self.individuals = individuals
		DNA_letters = 'ACGT'
		for i in range(individuals):
			tmp_genome = ''
			chromatin=False
			for i in range(random.randint(2048,self.genome_max)):
				if chromatin_var == True:
					n_rand = random.randint(0,2048)
					if n_rand == 0:
						if chromatin == True:
							chromatin = False
					else:
						chromatin = True
					if chromatin == False:
						tmp_genome+=DNA_letters[random.randint(0,3)]
					else:
						tmp_genome+=DNA_letters.lower()[random.randint(0,3)]
				else:
					tmp_genome+=DNA_letters[random.randint(0,3)]
				
			self.population_geneome.append(tmp_genome)
			self.mutation.append([None, None, None, None, None, None, None])

	def add_individuals(self, genomes, individuals):
		if type(genomes) == list:
			if len(genomes) >= self.max_ind:
				self.population_geneome = []
				for ind in genomes[:self.max_ind]:
					self.population_geneome.append(mutation(ind))
					self.mutation.append([None, None, None, None, None, None, None])
			else:
				multiplier = math.ceil(self.max_ind/len(genomes))
				tmp_list = genomes * multiplier
				for ind in tmp_list[:self.max_ind]:
					self.population_geneome.append(mutation(ind))
					self.mutation.append([None, None, None, None, None, None, None])

		elif type(genomes) == str:
			self.population_geneome = []
			for i in range(self.max_ind):
				self.population_geneome.append(mutation(genomes[:self.genome_max]))
				self.mutation.append([None, None, None, None, None, None, None])

		else:
			raise ValueError("Argument passed is {}, whereas argument should be eather list or string.".format(type(genomes)))

	def develop_mult(self, print_var=False):
		if self.population_geneome == []:
			print('Was not able to proceed. First add genomes to the populaton or create random genomes')
			raise ValueError()
		else:

			self.population = []
			logs = []
			counter = 0
			self.mutation = []
			indexes = []
			new_genomes = []
			print('start developing')
			with concurrent.futures.ProcessPoolExecutor() as executor:
				result = [executor.submit(develop_funct_wrapper, [genome, self.max_layers, self.max_dev_time, self.maximum_elemnts,  self.maximum_nodes, self.locally_connected_implementation, self.shape]) for genome in self.population_geneome]
				for f in concurrent.futures.as_completed(result):
					log, model, mut  = f.result()
					self.mutation.append(mut)
					logs.append(log)
					self.population.append(model)
					#self.population.append(nn_to_tensorflow(model, maximum_elemts=self.maximum_elemnts, maximum_nodes=self.maximum_nodes, 
					#locally_connected_implementation=self.locally_connected_implementation, shape=self.shape, time_limit=self.max_dev_time))
					new_genomes.append(log[-1])
					indexes.append(counter)
					counter+=1
			print('done developing')
			self.population_geneome = new_genomes
				#individ in pop - model, number_of_cells, number of divergent names, number_of_connections
				#logs.append([number_cells, len(number_divergent_names), number_of_connections, genome])
			print('logs!')
			print('indexes:',indexes)
			print(counter)
		return self.population, logs

	def develop_singl(self, print_var=False, def_lr=0.005):
		if self.population_geneome == []:
			print('Was not able to proceed. First add genomes to the populaton or create random genomes')
			raise ValueError()
		else:

			self.population = []
			logs = []
			counter = 0
			self.mutation = []
			indexes = []
			new_genomes = []
			print('start developing')
			dev_count = 0
			uniq_ops_list = []
			lrs = []
			for genome_i in self.population_geneome:
				print('#:',dev_count)
				dev_count+=1
				log, model, mut, n_uniq_ops, lr = develop_funct_wrapper([genome_i, self.max_layers, self.max_dev_time, self.maximum_elemnts,  self.maximum_nodes, self.locally_connected_implementation, self.shape])
				self.mutation.append(mut)
				logs.append(log)
				self.population.append(model)
				#self.population.append(nn_to_tensorflow(model, maximum_elemts=self.maximum_elemnts, maximum_nodes=self.maximum_nodes, 
				#locally_connected_implementation=self.locally_connected_implementation, shape=self.shape, time_limit=self.max_dev_time))
				new_genomes.append(log[-1])
				indexes.append(counter)
				counter+=1
				uniq_ops_list.append(n_uniq_ops)
				if lr == []:
					lrs.append([def_lr])
				else:
					lrs.append(lr)
			print('done developing')
			self.population_geneome = new_genomes
				#individ in pop - model, number_of_cells, number of divergent names, number_of_connections
				#logs.append([number_cells, len(number_divergent_names), number_of_connections, genome])
			print('logs!')
			print('indexes:',indexes)
		return self.population, logs, uniq_ops_list, lrs
	

	def new_generation(self, results):
		total_res = sum(results)
		prob_dist = [i/total_res for i in results]
		if self.max_ind % 2 != 0:
			selected_genomes = random.choices(population=self.population_geneome, weights=prob_dist, k=self.max_ind+1)
		else:
			selected_genomes = random.choices(population=self.population_geneome, weights=prob_dist, k=self.max_ind)
		indexes_choosen = []
		for gen_i in selected_genomes:
			indexes_choosen.append(self.population_geneome.index(gen_i))
		print('chosne indexes:',indexes_choosen)
		new_genome_pop = []
		#print('total:',len(pairwise(selected_genomes)))
		counter = 0
		print('start pairwise')
		for genome_1, genome_2 in pairwise(selected_genomes):
			print('new pariwise iter')
			#print('len before:',len(genome_1),len(genome_2))
			g1_indx = self.population_geneome.index(genome_1)
			g2_indx = self.population_geneome.index(genome_2)
			print('start recombinatio nof sequences')
			new_genome_1, new_genome_2 = recombine_seq2(genome_1[:self.genome_max:], genome_2[:self.genome_max], recombination_site=self.recombination_site)
			print('do mutations')
			new_genome_pop.append(mutation(new_genome_1, mutation_rate=self.mutation[counter]))
			counter+=1
			new_genome_pop.append(mutation(new_genome_2, mutation_rate=self.mutation[counter]))
			counter+=1
			#print('len after:',len(new_genome_1),len(new_genome_2))
		print('done pairwise')
		self.population_geneome = new_genome_pop




#make populaton
if __name__ == "__main__":
	import sys
	#number_nns = 10
	batch_size = 32
	#dir_path = os.path.dirname(os.path.realpath(__file__))
	pop = Population(max_individuals=number_nns, max_dev_time=12, genome_max_size=5000000, shape=[[28,28]])
	print('population created')
	with open(start_pop, 'r') as file:
		list_i = json.load(file)
	print('done loading')
	genome = []
	#print(list_i[0])
	for i in list_i:
		#print(i[1])
		#if i[1] > 6:
		genome.append(i[-1][:pop.genome_max])
	print('adding individuals')
	pop.add_individuals(genome, number_nns)
	#pop.make_random_pop(30)
	print('individuals added')

	fashion_mnist = tf.keras.datasets.fashion_mnist
	(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
	loss_func = tf.losses.CategoricalCrossentropy()
	depth = 10
	#develop 
	for n in range(0,10000) :
		nn_counter = 0
		print('--------- gen#{n} ---------'.format(n=n))
		genomes_and_res = []
		all_res = []
		models, logs, list_n_uniq_ops, lrs = pop.develop_singl()
		all_genomes = []
		for nn, z, n_unq_op, lr in zip(models, logs, list_n_uniq_ops, lrs):
			print('nn#:',nn_counter)
			print('lr:',lr)
			nn_counter+=1
			result = 0.000000000000001
			#print(z[:3])
			
			try:
				del sys.modules["model_tmp"]
			except KeyError:
				pass
			if nn != None:
				tensorflow_model = nn_to_tensorflow(nn, maximum_elemts=pop.maximum_elemnts, maximum_nodes=pop.maximum_nodes, 
				locally_connected_implementation=pop.locally_connected_implementation, shape=pop.shape, time_limit=pop.max_dev_time)
				print('model done')
				#sleep(5)
			elif len(z[-1])>pop.genome_max:
				tensorflow_model = None
			else:
				tensorflow_model = None
			if tensorflow_model != None:
				#print('model exist!')
				x, y, while_count = tensorflow_model
				model_safe = True
				try:
					from model_tmp import model
					print('model imported')
				except NameError:
					print('bad model (NameError)')
					sleep(5)
					model_safe = False
				except Exception:
					print('unknown erorr during model importing')
					sleep(5)
				except AttributeError:
					model_safe=False
					print('batch unwise concatination')
				try:
					model.count_params()
				except AttributeError:
					print('batch unwise concatination')
					model_safe=False
				except Exception:
					print('strong other error:')
					model_safe=False

				#model.summary()
				
				#for cell in nn:
					#if cell[4:] != ['TTTTTT', '', '']:
						#print(cell[4:])



				'''
			max_elems_check = False
			big_total = 0
			for lay in model.layers:
				result+=1
				total = 1
				for d in lay.output.get_shape():
					total = d*total

				if total > pop.maximum_elemnts:
					max_elems_check=True
				big_total = max(big_total, total)
			
			print('Total',big_total)
			#print('max_elems_check:',max_elems_check)
			if max_elems_check == True:
				result=result/big_total
			if model.count_params() > 0:
				result+=result*10
				'''



			
				if model_safe == False:
					print('to much elmts or batch unwise!')
				elif model.count_params() > pop.maximum_parameters:
					print('to much params:',model.count_params())
				elif model.count_params() == 0:
					print('No trainbale params:',model.count_params())
				elif model.layers[-1].count_params() > pop.maximum_parameters/4:
					print('last layer has too much parameters:',model.count_params())
				else:
					current_step = 0
					optimizer = keras.optimizers.Adam(learning_rate=0.01)
					dummy_tensor = []
					try:
						model.compile(optimizer='adam',
			              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
			              metrics=['accuracy'])
						print('model compiled, do training')

						res_incr = True
						res_old = 0.1
						result = 0.1
						lr = iter(lr)
						while res_incr:
							signal.signal(signal.SIGALRM, signal_handler)
							signal.alarm(training_time)
							try:
								lr_current = next(lr)
							except Exception:
								pass
							try:
								tf.config.run_functions_eagerly(False)
								tf.keras.backend.set_value(model.optimizer.learning_rate, lr_current)
								model.fit(train_images, train_labels, epochs=1, batch_size=32, verbose=1)
								test_loss, result = model.evaluate(test_images,  test_labels, verbose=0)
								#print(result)
								if result > res_old+0.01:
									pass
								else:
									res_incr = False
									result = max(result, res_old)
									#print('last res:',result, result, res_old)
									break
								res_old = 0+result
								'''
								tf.keras.backend.set_value(model.optimizer.learning_rate, 0.0001)
								model.fit(train_images, train_labels, epochs=1, batch_size=32, verbose=1)
								'''
							except ValueError:
								tf.config.run_functions_eagerly(True)
								tf.keras.backend.set_value(model.optimizer.learning_rate, lr_current)
								model.fit(train_images, train_labels, epochs=1, batch_size=32, verbose=1)
								test_loss, result = model.evaluate(test_images,  test_labels, verbose=0)
								#print(result)
								if result > res_old+0.01:
									pass
								else:
									res_incr = False
									result = max(result, res_old)
									#print('last res:',result, result, res_old)
									break
								res_old = 0+result
							signal.alarm(0)
							'''
							tf.keras.backend.set_value(model.optimizer.learning_rate, 0.0001)
							model.fit(train_images, train_labels, epochs=1, batch_size=32, verbose=1)
							'''
						
						#tf.keras.backend.set_value(model.optimizer.learning_rate, 0.000001)
						#model.fit(train_images, train_labels, epochs=5, batch_size=32, verbose=0)

						#test_loss, result = model.evaluate(test_images,  test_labels, verbose=0)
						'''
						if result > 0.4:
							result += n_unq_op/10
							print('add unqie op res:',n_unq_op/10)
						if result>2:	
							print('add whiles:',while_count/10)
							result += while_count/10
						'''
						print('res:',result)
						result = (1+result) ** 128
					except RuntimeError:
						'to muxh time training'
						signal.alarm(0)
					except Exception:
						print('bad model on fit')
						signal.alarm(0)
						sleep(5)
					signal.alarm(0)
			

					'''
					for shape_i in x[1:]:
						dummy_tensor.append(tf.expand_dims(tf.zeros(shape_i), axis=0))
					for batch_step in range(100):
						with tf.GradientTape(watch_accessed_variables=False) as tape:
							tape.watch(model.trainable_weights)

							loss_sum = []
							for mini_step in range(batch_size):
								#print('shapes:',x)
								input_tensor = [tf.expand_dims(train_images[current_step], axis=0)]
								#input_tensor.append(tf.expand_dims(train_images[current_step], axis=0))
								input_tensor = input_tensor + dummy_tensor
								#for shape_i in x[1:]:
									#input_tensor.append(tf.expand_dims(tf.zeros(shape_i), axis=0))
								#print('shapes loop:')
								#for f in input_tensor:
									#print(f.shape)
								y_pred_raw = model(input_tensor)
								
								y_true = train_labels[current_step]
								if type(y_pred_raw) == list:
									y_pred = tf.reshape(y_pred_raw[0], [-1])[:10]
								else:
									y_pred = tf.reshape(y_pred_raw, [-1])[:10]
								#print('bef y_pred:',y_pred)
								y_pred = tf.concat([y_pred, tf.zeros([10])],axis=-1)[:10]
								#print('aft y_pred:',y_pred)
								loss_sum.append(loss_func(tf.one_hot(y_true, depth), y_pred))
								#if use all datasets
								#if current_step == len(train_labels):
									#break
								current_step+=1
							grads = tape.gradient(sum(loss_sum), model.trainable_variables)
							optimizer.apply_gradients(zip(grads, model.trainable_variables))
						#print(batch_step)

					all_batch_res = []
					counter = 0
					
					for image, label in zip(test_images, test_labels):
						if counter == 100:
							break
						input_tensor = []
						input_tensor.append(tf.expand_dims(image, axis=0))
						for shape_i in x[1:]:
							input_tensor.append(tf.expand_dims(tf.zeros(shape_i), axis=0))
						y_pred_raw = model(input_tensor)
						#print('label:',label)
						if type(y_pred_raw) == list:
							y_pred = tf.reshape(y_pred_raw[0], [-1])[:10]
						else:
							y_pred = tf.reshape(y_pred_raw, [-1])[:10]
						#print('pred:',tf.math.argmax(y_pred))
						if tf.math.argmax(y_pred) == label:
							all_batch_res.append(1.)
						else:
							all_batch_res.append(0.)
						counter+=1
					result = ( 1+(sum(all_batch_res)/len(all_batch_res)) ) ** 64
					
					

					#print('result:',sum(all_batch_res)/len(all_batch_res))
					'''
					
			else:	
				print('bad model')
			


				#print('trainable pars:',trainable_count)
				#model.summary()
			#else:
				#print('mdl:',tensorflow_model)

			'''
			if z[0] > 100:
				result = min(10+(z[1]*10)**3, 1000000000)
			else:
				result = min([z[0]/10,10])
			if z[2] > 0:
				result += 10000000000
			result+=(1300*z[2])**3
			try:
				del sys.modules["model_tmp"]
			except KeyError:
				pass
			try:
				tensorflow_model = nn_to_tensorflow(nn, maximum_elemts=pop.maximum_elemnts, maximum_nodes=pop.maximum_nodes, 
					locally_connected_implementation=pop.locally_connected_implementation, shape=pop.shape, time_limit=pop.max_dev_time)
				from model_tmp import model
				result = 9993806416064010
			except Exception:
				pass
			'''
			'''
			try:
				if result > max(all_res):
					with open('{url}/model_tmp.py'.format(url=dir_path), 'r') as file_r:
						str_model = file_r.readlines()
						with open('{url}/best_model_yet.py'.format(url=dir_path), 'w') as file_w:
							file_w.write(str_model)

			except ValueError:
				pass
			'''
			all_res.append(result)
			genomes_and_res.append([z[3],result])
			all_genomes.append(z)
			model = None
		with open('{url}/{folder}/gen_{i}.json'.format(url=dir_path, i=n, folder=save_folder), 'w') as file:
			json.dump(all_genomes, file)
		#for i in all_genomes:
			#print(len(i[-1]))
		print_res = [x**(1/128)-1 for x in all_res]
		print(print_res)
		print('----------')
		print(all_res)
		#print('mut rate:',pop.mutation)
		#print('do new gen')
		#print('start new gen')
		#print('all res length:',all_res)
		pop.new_generation(all_res)
		#print('done new gen')






'''
with open('{url}/best_genome.txt'.format(url=dir_path), 'w') as file:
		file.write(pop.population_geneome[best_indx])
'''



































































