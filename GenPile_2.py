#Compiler
from time import time
import random
from random import randint
#import numpy as np
import re
import math
from copy import deepcopy
import signal

#Genetic code
DNA_letters = 'ACGT'

TATA_box = 'TATA'
termination_site = 'AAAAA'

start_codon = 'ACG'
stop_codon = 'CAG'
protein_and_codon = 'GTAA'
protein_cut_codon = 'GTCA'

protein_function_start_codon = 'CGC'
protein_function_stop_codon = 'TAG'


exon_cut = 'GC' 
exon_start_site = 'AT' 
exon_end_site = 'GC'






activation_strength = 128
concentration_maximum = 4000

def signal_handler(signum, frame):
    raise RuntimeError

#create a list of all possible 6 letter words, then divide into N nunebr of chuuncs, similar size
#used to assigne different each chunk of words inbto a specific protein category
def protein_types_code(n_types):
	dna_letters = 'AGCT'
	all_types = []
	for a in dna_letters:
		for b in dna_letters:
			for c in dna_letters:
				for d in dna_letters:
					for e in dna_letters:
						for f in dna_letters:
							seq = a+b+c+d+e+f
							all_types.append(seq)

	protein_types = []
	division = math.floor(len(all_types)/n_types)
	start = 0
	for i in range(int(division), len(all_types), int(division)):
		protein_types.append( set(all_types[start:i]) )
		start = i
	return protein_types

protein_types =  protein_types_code(30)#old -22

#codon - 3 letter word
#to do some function, proteins need to divide all posssible codon combination into chunks.
#for example, one protein use codon "ATA" to "GTA" to add a specific sequence to gencode, 
#while 'GTC' to 'TTC' used to delete gene code at a specific place

def exon_types(number_of_types):
	possibel_codons = ['ATA', 'ATC', 'ATG', 'GTA', 'GTC', 'GTG', 'TTA', 'TTC', 'TTG', 'CTA', 'CTC', 'CTG', 'CAA', 'CAC', 'CAG', 'CAT', 'CCA', 'CCC', 'CCG', 'CCT', 'CGA', 'CGC', 'CGG', 'CGT', 'AAA', 'AAC', 'AAG', 'AAT', 'ACA', 'ACC', 'ACG', 'ACT', 'AGA', 'AGC', 'AGG', 'AGT', 'GAA', 'GAC', 'GAG', 'GAT', 'GCA', 'GCC', 'GCG', 'GCT', 'GGA', 'GGC', 'GGG', 'GGT', 'TAA', 'TAC', 'TAG', 'TAT', 'TCA', 'TCC', 'TCG', 'TCT', 'TGA', 'TGC', 'TGG', 'TGT']
	codon_none = ['ATT', 'GTT', 'TTT', 'CTT']
	exon_divide_by = math.floor( len(possibel_codons) / number_of_types )
	additional = len(possibel_codons) - exon_divide_by *number_of_types
	exons = []
	for i in range( number_of_types ):
		exons.append( set(possibel_codons[i*exon_divide_by:(i+1)*exon_divide_by]) )
	if additional != 0:
		exons[0] = exons[0].union(set(possibel_codons[-additional:]))
	exons.append(set(codon_none))
	return exons

#CHEKED!
#divide seq of letters into codons
def codon_reader(seq):
	codons = []
	codon = ''
	for counter, nucl in enumerate(seq):
		codon+=nucl
		if (counter+1) % 3 == 0:
			codons.append(codon)
			codon = ''
	return codons

#CHEKED!
#transcribe codons into number
def codons_to_number(codons, seq_length = None):
	number_code = {'AAA': 0, 'AAC': 1, 'AAG': 2, 'AAT': 3, 'ACA': 4, 'ACC': 5, 'ACG': 6, 'ACT': 7, 'AGA': 8, 'AGC': 9, 'AGG': 0, 'AGT': 1, 'ATA': 2, 'ATC': 3, 'ATG': 4, 'ATT': 5, 'CAA': 6, 'CAC': 7, 'CAG': 8, 'CAT': 9, 'CCA': 0, 'CCC': 1, 'CCG': 2, 'CCT': 3, 'CGA': 4, 'CGC': 5, 'CGG': 6, 'CGT': 7, 'CTA': 8, 'CTC': 9, 'CTG': 0, 'CTT': 1, 'GAA': 2, 'GAC': 3, 'GAG': 4, 'GAT': 5, 'GCA': 6, 'GCC': 7, 'GCG': 8, 'GCT': 9, 'GGA': 0, 'GGC': 1, 'GGG': 2, 'GGT': 3, 'GTA': 4, 'GTC': 5, 'GTG': 6, 'GTT': 7, 'TAA': 8, 'TAC': 9, 'TAG': 0, 'TAT': 1, 'TCA': 2, 'TCC': 3, 'TCG': 4, 'TCT': 5, 'TGA': 6, 'TGC': 7, 'TGG': 8, 'TGT': 9, 'TTA': 0, 'TTC': 1, 'TTG': 2, 'TTT': 3}
	number = ''
	if seq_length == None:
		for codon in codons:
			number+= str(number_code[codon])

		if len(number) > 3:
			return int(number[-3:])+1
		else:
			return int(number)+1
	else:
		for codon in codons:
			number += str(number_code[codon])

		if len(number) > seq_length:
			return int(number[-seq_length:])+1
		else:
			return int(number)+1


# TAA TAA TAA AAG     ATT   ACA ATA AAC  AAC  AGA  ACT
#      AAAAT           TG   CA  TA   AC  AC   GA    CG 

#transcribe codons into a seq
def codons_to_nucl(codons): 
	nucl_code = {'AAA': 'AA', 'AAC': 'AC', 'AAG': 'AT', 'AAT': 'AG', 'ACA': 'CA', 'ACC': 'CC', 'ACG': 'CT', 'ACT': 'CG', 'AGA': 'GA', 'AGC': 'GC', 'AGG': 'GT', 'AGT': 'GG', 'ATA': 'TA', 'ATC': 'TC', 'ATG': 'TT', 'ATT': 'TG', 'CAA': 'AA', 'CAC': 'AC', 'CAG': 'AT', 'CAT': 'AG', 'CCA': 'CA', 'CCC': 'CC', 'CCG': 'CT', 'CCT': 'CG', 'CGA': 'GA', 'CGC': 'GC', 'CGG': 'GT', 'CGT': 'GG', 'CTA': 'TA', 'CTC': 'TC', 'CTG': 'TT', 'CTT': 'TG', 'GAA': 'AA', 'GAC': 'AC', 'GAG': 'AT', 'GAT': 'AG', 'GCA': 'CA', 'GCC': 'CC', 'GCG': 'CT', 'GCT': 'CG', 'GGA': 'GA', 'GGC': 'GC', 'GGG': 'GT', 'GGT': 'GG', 'GTA': 'TA', 'GTC': 'TC', 'GTG': 'TT', 'GTT': 'TG', 'TAA': 'A', 'TAC': 'C', 'TAG': 'G', 'TAT': 'T', 'TCA': 'A', 'TCC': 'C', 'TCG': 'G', 'TCT': 'T', 'TGA': 'A', 'TGC': 'C', 'TGG': 'G', 'TGT': 'T', 'TTA': 'A', 'TTC': 'C', 'TTG': 'G', 'TTT': 'T'}
	seq = ''
	for codon in codons:
		seq += nucl_code[codon]
	return seq


#CHEKED
#this function help proteins to check cell or multiceell proteins condition,
#for example it can do: if protein with sequence "ACTAGTCGATGCAT" is exists in the cell, return True
#IF GAG AAA AAA AAA ACT CAG AAT AGG AGG
#	  ACA        TAC TAC TAC           ACT          AAG    CTG   
#if any_seq      CCC(TAC TAC TAC)    end of seq      in   cell type
#CHEKC FOR AAA protein in common, rate shoudl be > 300
#
def boolian_logic(cell,common_proteins, codons):
	if len(codons) < 5:
		return None

	else:
		#variable types
		seq_any_code = ('AAA', 'AAC', 'AAG', 'AAT', 'ACA', 'ACC', 'ACG', 'ACT', 'AGA', 'AGC', 'AGG', 'AGT', 'ATA', 'ATC', 'ATG', 'ATT', 'CAA', 'CAC', 'CAG', 'CAT', 'CCA', 'CCC', 'CCG', 'CCT', 'CGA', 'CGC', 'CGG', 'CGT', 'CTA', 'CTC', 'CTG', 'CTT')
		seq_common_code = ('GAA', 'GAC', 'GAG', 'GAT', 'GCA', 'GCC', 'GCG', 'GCT', 'GGA', 'GGC', 'GGG', 'GGT', 'GTA', 'GTC', 'GTG', 'GTT')
		seq_cell_prot_code = ('TAA', 'TAC', 'TAG', 'TAT', 'TCA', 'TCC', 'TCG', 'TCT', 'TGA', 'TGC', 'TGG', 'TGT', 'TTA', 'TTC', 'TTG', 'TTT')
		
		stop_seq_codons = ('ACG', 'ACT', 'GCT', 'GCG')

		#variable 1 type operations
		seq_in_code = ('AAA', 'AAC', 'AAG', 'AAT', 'ACA', 'ACC', 'ACG', 'ACT', 'AGA', 'AGC', 'AGG', 'AGT', 'ATA', 'ATC', 'ATG', 'ATT', 'CAA', 'CAC', 'CAG', 'CAT', 'CCA', 'CCC', 'CCG', 'CCT', 'CGA', 'CGC', 'CGG', 'CGT', 'CTA', 'CTC', 'CTG', 'CTT')
		seq_not_in_code = ('GAA', 'GAC', 'GAG', 'GAT', 'GCA', 'GCC', 'GCG', 'GCT', 'GGA', 'GGC', 'GGG', 'GGT', 'GTA', 'GTC', 'GTG', 'GTT', 'TAA', 'TAC', 'TAG', 'TAT', 'TCA', 'TCC', 'TCG', 'TCT', 'TGA', 'TGC', 'TGG', 'TGT', 'TTA', 'TTC', 'TTG', 'TTT')

		#second variable woht ontl first type operation
		common_code = ( 'AGC', 'AGG', 'AGT', 'ATA', 'ATC', 'ATG', 'ATT', 'CAA', 'CAC')
		protein_cell_code = ('CCA', 'CCC', 'CCG', 'CCT', 'CGA', 'CGC', 'CGG', 'CGT', 'CTA', 'CTC')
		cell_type_code = ('CAG', 'CAT', 'CTG', 'GGA', 'GAA', 'GAC', 'GAG', 'GAT', 'GCA')
		connections_code = ('GCC', 'GCG', 'GCT', 'CTT', 'GGC', 'GGG', 'GGT', 'GTA', 'GTC')
		matrix_operations_code = ('GTG', 'GTT', 'TAA', 'TAC', 'TAG', 'TAT', 'TCA', 'TCC', 'TCG')
		activation_function_code = ('TCT', 'TGA', 'TGC', 'TGG', 'TGT', 'TTA', 'TTC', 'TTG', 'TTT')
		weights_code  = ('AAA', 'AAC', 'AAG', 'AAT', 'ACA', 'ACC', 'ACG', 'ACT', 'AGA')
		
		#variable 2 operations
		seq_in_common_code = ('AAA', 'AAC', 'AAG', 'AAT', 'ACA', 'ACC', 'ACG', 'ACT', 'AGA', 'AGC', 'AGG', 'AGT', 'ATA', 'ATC', 'ATG', 'ATT', 'CAA', 'CAC', 'CAG', 'CAT', 'CCA', 'CCC', 'CCG', 'CCT', 'CGA', 'CGC', 'CGG', 'CGT', 'CTA', 'CTC', 'CTG', 'CTT')
		seq_in_cell_code = ('GAA', 'GAC', 'GAG', 'GAT', 'GCA', 'GCC', 'GCG', 'GCT', 'GGA', 'GGC', 'GGG', 'GGT', 'GTA', 'GTC', 'GTG', 'GTT', 'TAA', 'TAC', 'TAG', 'TAT', 'TCA', 'TCC', 'TCG', 'TCT', 'TGA', 'TGC', 'TGG', 'TGT', 'TTA', 'TTC', 'TTG', 'TTT')

		#second operation
		bigger_oper_code = ('AAA', 'AAC', 'AAG', 'AAT', 'ACA', 'ACC', 'ACG', 'ACT', 'AGA', 'AGC', 'AGG', 'AGT', 'ATA', 'ATC', 'ATG', 'ATT', 'CAA', 'CAC', 'CAG', 'CAT', 'CCA', 'CCC', 'CCG', 'CCT', 'CGA', 'CGC', 'CGG', 'CGT', 'CTA', 'CTC', 'CTG', 'CTT')
		smaller_oper_code = ('GAA', 'GAC', 'GAG', 'GAT', 'GCA', 'GCC', 'GCG', 'GCT', 'GGA', 'GGC', 'GGG', 'GGT', 'GTA', 'GTC', 'GTG', 'GTT', 'TAA', 'TAC', 'TAG', 'TAT', 'TCA', 'TCC', 'TCG', 'TCT', 'TGA', 'TGC', 'TGG', 'TGT', 'TTA', 'TTC', 'TTG', 'TTT')
		equal_oper_code = ('CTC', 'CTG', 'CTT', 'TTC', 'TTG', 'TTT')

		#looking for sequnce
		if codons[0] in seq_any_code:
			seq = None
			operation = None
			where_to_find = None
			stop_index = None
			for stop_codon_test in stop_seq_codons:
				try:
					stop_index = codons[2:].index(stop_codon_test)
					stop_index = stop_index+2
				except ValueError: 
					pass
			if stop_index == None:
				return None
			else:
				if len(codons[stop_index+1:]) < 2:
					return None
				else:
					seq = codons_to_nucl(codons[1:stop_index])
					if codons[stop_index+1] in seq_in_code:
						operation = 'in'
					else:
						operation = 'not in' 
					
					if codons[stop_index+2] in common_code:
						where_to_find = 'self.common_proteins'
						element_exist_check = False
						for prot_elem in common_proteins:
							if seq in prot_elem:
								seq = prot_elem
								element_exist_check = True
								break
						if element_exist_check != True:
							return 'condition = False'

					elif codons[stop_index+2] in protein_cell_code:
						where_to_find = 'cell[1]'
						element_exist_check = False
						for prot_elem in cell[1]:
							if seq in prot_elem:
								seq = prot_elem
								element_exist_check = True
								break
						if element_exist_check != True:
							return 'condition = False'
					elif codons[stop_index+2] in cell_type_code:
						where_to_find = 'cell[2]'
					elif codons[stop_index+2] in connections_code:
						where_to_find = 'cell[3]'
					elif codons[stop_index+2] in matrix_operations_code:
						where_to_find = 'cell[4]'
					elif codons[stop_index+2] in activation_function_code:
						where_to_find = 'cell[5]'
					elif codons[stop_index+2] in weights_code:
						where_to_find = 'cell[6]'

					return "if '{}' {} {}: condition = True".format(seq, operation, where_to_find)




		#looking for specific protein concentration
		if codons[0] in seq_cell_prot_code:
			seq = None
			operation = None
			number = None
			stop_index = None

			for stop_codon_test in stop_seq_codons:
				try:
					stop_index = codons[2:].index(stop_codon_test)
					stop_index = stop_index+2
				except ValueError: 
					pass
			if stop_index == None:
				return None
			else:
				if len(codons[stop_index+1:]) < 2:
					return None
				else:
					seq = codons_to_nucl(codons[1:stop_index])
					element_exist_check = False
					for prot_elem in cell[1]:
						if seq in prot_elem:
							seq = prot_elem
							element_exist_check = True 
							break
					if element_exist_check != True:
						return 'condition = False'
					if codons[stop_index+1] in bigger_oper_code:
						operation = '>'
					if codons[stop_index+1] in smaller_oper_code:
						operation = '<'
					if codons[stop_index+1] in equal_oper_code:
						operation = '=='

					if len(codons[stop_index+2:]) > 3:
						number = codons_to_number(codons[-3:])
					else:
						number = codons_to_number(codons[stop_index+2:])

					return "if cell[1]['{}'] {} {}: condition = True".format(seq, operation, number)

		if codons[0] in seq_common_code:
			seq = None
			operation = None
			number = None
			stop_index = None
			commmon_seq = False
			cell_seq = False

			for stop_codon_test in stop_seq_codons:
				try:
					stop_index = codons[2:].index(stop_codon_test)
					stop_index = stop_index+2
				except ValueError: 
					pass
			if stop_index == None:
				return None
			else:
				if len(codons[stop_index+1:]) < 2:
					return None
				else:
					seq = codons_to_nucl(codons[1:stop_index])
					element_exist_check = False
					for prot_elem in common_proteins:
						if seq in prot_elem:
							seq = prot_elem
							element_exist_check = True
							commmon_seq = True
							break
					for prot_elem in cell[1]:
						if seq in prot_elem:
							seq = prot_elem
							element_exist_check = True 
							cell_seq = True
							break

					if element_exist_check != True:
						return 'condition = False'
					if codons[stop_index+1] in bigger_oper_code:
						operation = '>'
					if codons[stop_index+1] in smaller_oper_code:
						operation = '<'
					if codons[stop_index+1] in equal_oper_code:
						operation = '=='

					if len(codons[stop_index+2:]) > 3:
						number = codons_to_number(codons[-3:])
					else:
						number = codons_to_number(codons[stop_index+2:])

					if commmon_seq == True:
						return "if self.common_proteins['{}'] {} {}: condition = True".format(seq, operation, number)
					elif cell_seq == True:
						return "if cell[1]['{}'] {} {}: condition = True".format(seq, operation, number)
					else:
						return "condition = False"













class NN():
	def __init__(self, genome, mutatation=[0.001, 0.001, 0.001, 0.001, True], max_time=4, max_layers=1000):
		if mutatation[4]!=False:
			self.mutatation = mutatation[:4]
		self.time_limit = max_time
		self.layer_limit = max_layers
		self.full_genome = genome
		self.common_proteins = {}
		self.transcription = {} #{'where to start':'how many to add'}
		self.silence = {}
		self.splicing = {} #{'proteins':'how much'}
		self.splice_regul = [{},{}]#{'seq':[0,10],'seq':[1,10],'seq':[2,10]}
		self.transposone = []
		self.conentration_maximum = 1000
		self.learning_rate = []
		self.mutation_rate = [None, None, None, None, None, None, None]
		self.counter = 0
		#self.selective_common = {'AAAA':{},'BBBBB':{}} for future

		#parameters

		#neural network
		#           ['genome', 'proteins', 'cell type', 'cell conections', 'matrix operation', 'activation function', 'weiights']
		#self.nn = [ [genome,       {} ,       'AAA',              [],           '',                      '',             ''] ]	
		self.nn = [ [genome, {'AAAATTGCATAACGACGACGGC':1,'GGAGTTGCGTCTAATAATAAAAGATTACAATAAACTGGAACAGAACTGC':1} , 'AAA', [], 'TTTTTT','', ''] ]		


	#this method develops neural netwrok, from onee cell up to fully developed network
	#--------------------------------------------------------------------------------------------------------------------------------
	#--------------------------------------------------------------------------------------------------------------------------------
	#--------------------------------------------------------------------------------------------------------------------------------
	
	def develop(self):
		development = True
		chekc_dev = False
		dev_cointer = 0

		signal.signal(signal.SIGALRM, signal_handler)
		signal.alarm(self.time_limit)
		#t_start = time.time()
		try:
			while development == True:
				#print('')
				#print('do while',self.counter)
				self.counter+=1
				#print('c:',self.counter)
				#print(t_start)
				#print(time.time())
				#if time.time() - t_start > 5:
					#print("HERE!")
				#while development programm is workin - looping over each cell embrio and 
				#doing all the actions, according to dna-propgramm in each cell
				apoptosis_list = []
				counter_cell_n = 0
				current_cell_number = len(self.nn)
				for cell in self.nn:
					if counter_cell_n >= current_cell_number:
						break
					counter_cell_n +=1
					self.activation_strength = 128
					self.silence_strength = 128
					
					'''
					if len(self.nn) > 2 and chekc_dev == False:
						for ind, i in enumerate(self.nn):
							if ind == 1:
								i[2] = 'CCC'
							elif ind == 2:
								i[2] = 'GGG'
								self.nn[ind][1]['ACTACTACTACT']=222
								self.nn[ind][1]['AGTAGTAGTAGT']=747
							else:
								i[2] = 'AAA'
								self.nn[ind][1]['ACTACTACTACT']=222
								self.nn[ind][1]['AGTAGTAGTAGT']=747
								
						chekc_dev = True
					'''
					self.transcription = {}
					self.splicing = {}
					self.transposone = []
					self.silence = {}
					self.splice_regul = [{},{}]
					shuffle_var = False
					apoptosis = False
					
					

					proteins_full = []
					delete_list = []
					#print('doing proteins:')
					#print('start protein preapre')
					for protein, rate in cell[1].items():
						#print('rate before start:',rate)
						if rate > concentration_maximum:
							rate = concentration_maximum
							cell[1][protein] = rate


						if rate < 1:
							delete_list.append(protein)
						elif protein == '':
							delete_list.append(protein)
						else:

							#findng AND codons in protein(one protein can do multipel functions. If tis is divided into two fucntional units,
							#AND codon is used) 
							prot_ands_first_test = [m.start() for m in re.finditer('(?={})'.format(protein_and_codon), protein)]
							prot_ands_second_test = []
							#if no AND codon found
							if prot_ands_first_test == []:
								proteins_full.append((protein, rate, protein))
							else:
								for and_codon in prot_ands_first_test:
									#each fucntion protein part is coded with 6 letters, so if after AND codon,
									#there is no 6 letter - there will be no function
									if len(protein[and_codon:]) > 9:
										if exon_cut not in protein[and_codon:and_codon+10]:
											prot_ands_second_test.append(and_codon)
								#adding each functional unit of single protein into list of proteins_full, which
								#contains each functional unit of single protein 
								start_proteins_active_center = -4
								for protein_and in prot_ands_second_test:
									proteins_full.append((protein[start_proteins_active_center+4:protein_and], rate, protein))
									start_proteins_active_center = protein_and
								

								proteins_full.append((protein[start_proteins_active_center+4:], rate, protein))


					for del_prot in delete_list:
						del cell[1][del_prot]
						#print('delete')

					#for each functional unit of a proteinm isolae exons and type of protein unit
					for protein, rate, full_protein_seq in proteins_full:
						#print('protein preapre 2')
						#print('rate start:',rate)
						protein_termination_codon = False
						#6 first latter - type of protein
						protein_type = protein[:6]
						protein_exons = protein[6:]
						#finding all exons in a protein
						exons_index = [m.start() for m in re.finditer('(?={})'.format(exon_cut), protein_exons)]
						exons = []
						exon_start = 0
						#make shure there are exons in protein
						if exons_index != []:
							#it is easier to have even numebr of exons cuts, so this pece if code deals with off umebr of exons cuts
							if len(exons_index) == 1:
								#if there is only one exon cuts and,after the exon cut there is some letter, this pice of code add
								#another exon cut sequence in the end, so that there be even number
								if len(protein_exons) > exons_index[0]:
									exons_index.append(len(protein_exons))

								#however, if the exon cut sequence is in end, make sure just delete it
								else:
									exons = []
							#adding exon cut in the end, if exosn cuts is uneven, andthere is meaningul sequence after lust exon cut
							if len(exons_index) % 2 != 0:
								if len(protein_exons) > exons_index[-1]:
									exons_index.append(len(protein_exons))
								else:
									exons = []
							
							#after applying the rules to exon cuts, now actual exon cuts are made
							exon_start = exons_index[0]
							for exon_ind in exons_index[1:]:
								exons.append(protein_exons[exon_start+2:exon_ind])
								exon_start = exon_ind

						else:
							working_exons = []
						#as information is in codonds, if there is no 3 letter in exons, remooves it from exons list
						for i in exons:
							working_exons = [codon_reader(exon) for exon in exons if len(exon) > 5]
						
						for ex_ind, ex_tmp in enumerate(working_exons):
							if stop_codon in ex_tmp:
								protein_termination_codon = True
								termination_ex_ind = ex_ind+1
								termination_cod_ind = ex_tmp.index(stop_codon)
								
						if protein_termination_codon == True:
							working_exons = working_exons[:termination_ex_ind]
							working_exons[-1] = working_exons[-1][:termination_cod_ind]
							if len(working_exons[-1]) < 2:
								working_exons = working_exons[:-1]




						
						

						'''
						TATA_box = 'TATA'
						termination_site = 'AAAAA'

						start_codon = 'ACG'
						stop_codon = 'CAG'
						protein_and_codon = 'GTAA'
						protein_cut_codon = 'GTCA'

						protein_function_start_codon = 'CGC'
						protein_function_stop_codon = 'TAG'

						exon_cut = 'GC' 

						exon_start = 'AT' 
						exon_end = 'GC'

						NUCL CODE = 'ACG': 'CT', 'ACT': 'CG', 'AGA': 'GA'
						'''


						#Gene transcription activation
						#--------------------------------
						
						
						if protein_type in protein_types[0]:
							#print('Gene transcription')
							exon_type_1, exon_type_2, exon_type_3, exon_type_4, exon_type_5, exon_none = exon_types(5)
							protein_express=[]
							factor_rate = rate
							rate=1
							common=False
							condition=True
							if working_exons != []:
								for exon in working_exons:
									exon_codons = exon
									
								
									if exon_codons[0] in exon_type_1:

										seq_find = codons_to_nucl(exon_codons[1:])
										if len(seq_find)>2:
											add_dna_index = [m.start() for m in re.finditer('(?={})'.format(seq_find), cell[0])]
											protein_express.extend(add_dna_index)
									elif exon_codons[0] in exon_type_2:
										rate = codons_to_number(exon_codons[1:]) * factor_rate
										if rate > concentration_maximum:
											rate = concentration_maximum
									elif exon_codons[0] in exon_type_3:
										common = True
										
									elif exon_codons[0] in exon_type_4:
								
										condition_code = boolian_logic(cell, self.common_proteins, exon_codons[1:])

										if condition_code != None:
											
											condition = False
											condition_code = condition_code.replace('self.common_proteins','common_proteins')
											loc = {'common_proteins':self.common_proteins, 'cell':cell}
											common_proteins = self.common_proteins
											try:
												exec(condition_code, globals(), loc)
											except KeyError:
												pass
											
											try:
												condition = loc['condition']
											except KeyError:
												pass
											
									elif exon_codons[0] in exon_type_5:
										if len(exon_codons) > 3:
											random_number = codons_to_number(exon_codons[1:])
											rate = randint(1, random_number)




							else:			
								pass

							if common == True:
							
								if full_protein_seq in self.common_proteins:
									self.common_proteins[full_protein_seq] = self.common_proteins[full_protein_seq]+rate
								else:
									self.common_proteins[full_protein_seq] = rate

							if condition == True:
								for protein_index in protein_express:
									if protein_index in self.transcription:
										self.transcription[protein_index]=self.transcription[protein_index]+rate
									else:
										self.transcription[protein_index]=rate
						#--------------------------------
							
						
						
						#Gene repressor
						#--------------------------------
						elif protein_type in protein_types[1]:
							#print('Gene repressor')
							exon_type_1, exon_type_2, exon_type_3, exon_type_4, exon_type_5, exon_none = exon_types(5)
							protein_repress=[]
							factor_rate = rate
							rate=1
							common=False
							condition=True
							if working_exons != []:
								for exon in working_exons:
									exon_codons = exon
									if exon_codons[0] in exon_type_1:
										seq_find = codons_to_nucl(exon_codons[1:])
										if len(seq_find)>2:
											add_dna_index = [m.start() for m in re.finditer('(?={})'.format(seq_find), cell[0])]
											protein_repress.extend(add_dna_index)
									elif exon_codons[0] in exon_type_2:
										rate = codons_to_number(exon_codons[1:]) * factor_rate
										if rate > concentration_maximum:
											rate = concentration_maximum
									elif exon_codons[0] in exon_type_3:
										common = True
										
									elif exon_codons[0] in exon_type_4:

										condition_code = boolian_logic(cell, self.common_proteins, exon_codons[1:])

										if condition_code != None:
											condition = False
											condition_code = condition_code.replace('self.common_proteins','common_proteins')
											loc = {'common_proteins':self.common_proteins, 'cell':cell}
											common_proteins = self.common_proteins
											try:
												exec(condition_code, globals(), loc)
											except KeyError:
												pass
											try:
												condition = loc['condition']
											except KeyError:
												pass
									elif exon_codons[0] in exon_type_5:
										if len(exon_codons) > 3:
											random_number = codons_to_number(exon_codons[1:])
											rate = randint(1, random_number)
							else:			
								pass


							if common == True:
								if full_protein_seq in self.common_proteins:
									self.common_proteins[full_protein_seq] = self.common_proteins[full_protein_seq]+rate
								else:
									self.common_proteins[full_protein_seq] = rate

							if condition == True:
								for protein_index in protein_repress:
									if protein_index in self.silence:
										self.silence[protein_index]=self.silence[protein_index]+rate
									else:
										self.silence[protein_index]=rate
						#--------------------------------

						
			
						#Gene shaperon add
						#--------------------------------

						elif protein_type in protein_types[2]:
							#print('shaperon add')
							exon_type_1, exon_type_2, exon_type_3, exon_type_4, exon_type_5, exon_none = exon_types(5)
							protein_search=[]
							seq_to_add = ''
							common=False
							condition=True
							#add condition - left:1, right:2, center:3, delete:4
							add_condition = 1
							
							if working_exons != []:
								for exon in working_exons:
									exon_codons = exon
									if exon_codons[0] in exon_type_1:

										seq_find = codons_to_nucl(exon_codons[1:])
										for prot, prot_rate in cell[1].items():
											if seq_find in prot:
												protein_search.append((prot, prot_rate, seq_find))
									
									elif exon_codons[0] in exon_type_2:
										seq_add = codons_to_nucl(exon_codons[1:])
										seq_to_add+=seq_add
									elif exon_codons[0] in exon_type_3:
										common = True
										
									elif exon_codons[0] in exon_type_4:

										condition_code = boolian_logic(cell, self.common_proteins, exon_codons[1:])

										if condition_code != None:
											condition = False
											condition_code = condition_code.replace('self.common_proteins','common_proteins')
											loc = {'common_proteins':self.common_proteins, 'cell':cell}
											common_proteins = self.common_proteins
											try:
												exec(condition_code, globals(), loc)
											except KeyError:
												pass
											try:
												condition = loc['condition']
											except KeyError:
												pass
									elif exon_codons[0] in exon_type_5:
					
										if exon_codons[1] in ('AAA', 'AAC', 'AAG', 'AAT', 'ACA', 'ACC', 'ACG', 'ACT', 'AGA', 'AGC', 'AGG', 'AGT', 'ATA', 'ATC', 'ATG','ATT'):
											add_condition = 1
										if exon_codons[1] in ('GAA', 'GAC', 'GAG', 'GAT', 'GCA', 'GCC', 'GCG', 'GCT', 'GGA', 'GGC', 'GGG', 'GGT', 'GTA', 'GTC', 'GTG', 'GTT'):
											add_condition = 2
										if exon_codons[1] in ('TAA', 'TAC', 'TAG', 'TAT', 'TCA', 'TCC', 'TCG', 'TCT', 'TGA', 'TGC', 'TGG', 'TGT', 'TTA', 'TTC', 'TTG', 'TTT'):
											add_condition = 3
										if exon_codons[1] in ('CAA', 'CAC', 'CAG', 'CAT', 'CCA', 'CCC', 'CCG', 'CCT', 'CGA', 'CGC', 'CGG', 'CGT', 'CTA', 'CTC', 'CTG', 'CTT'):
											add_condition = 4
							
							else:			
								pass


							if common == True:

								if full_protein_seq in self.common_proteins:
									if self.common_proteins[full_protein_seq] < rate:
										self.common_proteins[full_protein_seq] = rate
								else:
									self.common_proteins[full_protein_seq] = rate

							if condition == True:
								
								for prot, prot_rate, seq_find in protein_search:
									
									if seq_to_add not in prot:
										if add_condition == 1:
											new_value = prot+seq_to_add
										elif add_condition == 2:
											new_value = seq_to_add+prot
										elif add_condition == 3:
											index_tmp = prot.find(seq_find)
											new_value = prot[:index_tmp] + seq_to_add + prot[index_tmp:]

										elif add_condition == 4:
											index_tmp = prot.find(seq_find)
											new_value = prot[:index_tmp] + seq_to_add +  prot[index_tmp+len(seq_find):]																		
										
										try:
											if prot_rate < rate:
												cell[1][new_value] = cell[1][new_value]+prot_rate
												del cell[1][prot]
											else:
												cell[1][new_value] = cell[1][new_value] + rate
												cell[1][prot] = prot_rate - rate

										except KeyError:
											try:
												if prot_rate < rate:
													cell[1][new_value] = cell[1].pop(prot)
												else:
													cell[1][new_value] = rate
													cell[1][prot] = prot_rate - rate
											except KeyError:#STRANGE THING
												pass
								

								
						#--------------------------------


						#Gene shaperon remoove
						#--------------------------------
						elif protein_type in protein_types[3]:
							#print('shaperon remoove')
							exon_type_1, exon_type_2, exon_type_3, exon_type_4, exon_type_5, exon_type_6, exon_none = exon_types(6)
							protein_search=[]
							seq_to_add = ''
							common=False
							condition=True
							splice_range = 16
							#add condition - left:1, right:2, center:3, delete:4
							remoove_condition = True
					
							if working_exons != []:
								for exon in working_exons:
									exon_codons = exon
									if exon_codons[0] in exon_type_1:

										seq_find = codons_to_nucl(exon_codons[1:])
									
										for prot, prot_rate in cell[1].items():
											if seq_find in prot:
									
												protein_search.append((prot, prot_rate, seq_find))
									
									elif exon_codons[0] in exon_type_2:
										seq_add = codons_to_nucl(exon_codons[1:])
										seq_to_add+=seq_add

									elif exon_codons[0] in exon_type_3:
										common = True
										
									elif exon_codons[0] in exon_type_4:

										condition_code = boolian_logic(cell, self.common_proteins, exon_codons[1:])

										if condition_code != None:
											condition = False
											condition_code = condition_code.replace('self.common_proteins','common_proteins')
											loc = {'common_proteins':self.common_proteins, 'cell':cell}
											common_proteins = self.common_proteins
											try:
												exec(condition_code, globals(), loc)
											except KeyError:
												pass
											try:
												condition = loc['condition']
											except KeyError:
												pass
									elif exon_codons[0] in exon_type_5:
										remoove_condition = False
										
									
									elif exon_codons[0] in exon_type_6:
										if len(exon_codons) > 2:
											splice_range = codons_to_number(exon_codons[-2:])
										else:
											splice_range = codons_to_number([exon_codons[1]])
							else:		
								pass


							if common == True:
								if full_protein_seq in self.common_proteins:
									if self.common_proteins[full_protein_seq] < rate:
										self.common_proteins[full_protein_seq] = rate
								else:
									self.common_proteins[full_protein_seq] = rate

							if condition == True:
								if seq_to_add == '':
									for prot, prot_rate, seq_find in protein_search:
										if prot not in cell[1]:
											pass
										else:
											if remoove_condition == True:
												index_tmp = prot.find(seq_find)
												range_right = min(len(prot[index_tmp+len(seq_find):]), splice_range)
												new_value = prot[:index_tmp] + prot[index_tmp+len(seq_find)+range_right:]

											else:
												
												index_tmp = prot.find(seq_find)

												range_left = min(len(prot[:index_tmp]), splice_range)
			
												new_value = prot[:index_tmp-range_left]+prot[index_tmp+len(seq_find):]

											try:
												if prot_rate <= rate:

													cell[1][new_value] = cell[1][new_value]+cell[1][prot]

											
													del cell[1][prot]
			


												else:
													
													cell[1][new_value] = cell[1][new_value] + rate
													cell[1][prot] = prot_rate - rate
													

											except KeyError:
												if prot_rate <= rate:
													
													cell[1][new_value] = cell[1].pop(prot)
													
												else:
													
													cell[1][new_value] = rate
													cell[1][prot] = prot_rate - rate
													
									
								else:
									for prot, prot_rate, seq_find in protein_search:
										if prot not in cell[1]:
											pass
										else:
											if seq_to_add in prot:
												seq_find = seq_to_add
												if remoove_condition == True:
													index_tmp = prot.find(seq_find)
													range_right = min(len(prot[index_tmp+len(seq_find):]), splice_range)
													new_value = prot[:index_tmp] + prot[index_tmp+len(seq_find)+range_right:]

												else:
													index_tmp = prot.find(seq_find)
													range_left = min(len(prot[:index_tmp]), splice_range)
													
													new_value = prot[:index_tmp-range_left]+prot[index_tmp+len(seq_find):]

																
												
												try:
													if prot_rate <= rate:
														cell[1][new_value] = cell[1][new_value]+cell[1][prot]
														del cell[1][prot]
													else:
														cell[1][new_value] = cell[1][new_value] + rate
														cell[1][prot] = prot_rate - rate

												except KeyError:
													if prot_rate <= rate:
														cell[1][new_value] = cell[1].pop(prot)
													else:
														cell[1][new_value] = rate
														cell[1][prot] = prot_rate - rate
						
							
								
						#--------------------------------

						


						#Cell division activator
						#--------------------------------
					
						elif protein_type in protein_types[4]:
							#print('division')
							exon_type_1, exon_type_2, exon_type_3, exon_type_4, exon_type_5, exon_type_6, exon_none = exon_types(6)
							search_cut_seq = ''
							common=False
							condition=True
							cell_divisions = 1
							self_rate_down = 1
							self_rate_up = 1

							if working_exons != []:
								for exon in working_exons:
									exon_codons = exon
									if exon_codons[0] in exon_type_1:
										self_rate_down = codons_to_number(exon_codons[1:])
										
									elif exon_codons[0] in exon_type_2:
										cell_divisions = codons_to_number(exon_codons[1:], 2)
									elif exon_codons[0] in exon_type_3:
										self_rate_up = codons_to_number(exon_codons[1:])
										
									elif exon_codons[0] in exon_type_4:
										common = True
										
									elif exon_codons[0] in exon_type_5:

										condition_code = boolian_logic(cell, self.common_proteins, exon_codons[1:])

										if condition_code != None:
											condition = False
											condition_code = condition_code.replace('self.common_proteins','common_proteins')
											loc = {'common_proteins':self.common_proteins, 'cell':cell}
											common_proteins = self.common_proteins
											try:
												exec(condition_code, globals(), loc)
											except KeyError:
												pass
											try:
												condition = loc['condition']
											except KeyError:
												pass
									elif exon_codons[0] in exon_type_6:
										search_cut_seq += codons_to_nucl(exon_codons[1:])

							
							else:			
								pass
							
							if common == True:

								if full_protein_seq in self.common_proteins:
									if self.common_proteins[full_protein_seq] < rate:
										self.common_proteins[full_protein_seq] = rate
								else:
									self.common_proteins[full_protein_seq] = rate

							if condition == True:
								if self_rate_down <= rate:
									if self_rate_up >= rate:
										try:
											cell[1][full_protein_seq] = cell[1][full_protein_seq]+1
										except KeyError:
											pass
										for i in range(cell_divisions):
											self.nn.append(deepcopy(cell))
											try:
												self.nn[-1][1][full_protein_seq] = self.nn[-1][1][full_protein_seq]+1
											except KeyError:
												pass
										

						#--------------------------------	

						



						#Cell protein shuffle
						#--------------------------------

						elif protein_type in protein_types[5]:
							#print('protein shuffle')
							exon_type_1, exon_type_2, exon_type_3, exon_type_4, exon_type_5, exon_none = exon_types(5)
							condition=True
							self_rate_down = 1
							self_rate_up = 5000
							#add condition - left:1, right:2, center:3, delete:4

							if working_exons != []:
								for exon in working_exons:
									exon_codons = exon
									if exon_codons[0] in exon_type_1:
										self_rate_down = codons_to_number(exon_codons[1:])
									
									elif exon_codons[0] in exon_type_3:
										self_rate_up = codons_to_number(exon_codons[1:])

									elif exon_codons[0] in exon_type_4:
										common = True
										
									elif exon_codons[0] in exon_type_5:

										condition_code = boolian_logic(cell, self.common_proteins, exon_codons[1:])

										if condition_code != None:
											condition = False
											condition_code = condition_code.replace('self.common_proteins','common_proteins')
											loc = {'common_proteins':self.common_proteins, 'cell':cell}
											common_proteins = self.common_proteins
											try:
												exec(condition_code, globals(), loc)
											except KeyError:
												pass
											try:
												condition = loc['condition']
											except KeyError:
												pass
							
							else:			
								pass
							
							if common == True:

								if full_protein_seq in self.common_proteins:
									if self.common_proteins[full_protein_seq] < rate:
										self.common_proteins[full_protein_seq] = rate
								else:
									self.common_proteins[full_protein_seq] = rate

							if condition == True:
								if self_rate_down <= rate:
									if self_rate_up >= rate:
										shuffle_var = True


						#--------------------------------	





						#Cell trsnaposone
						#--------------------------------

						elif protein_type in protein_types[6]:
							#print('Cell trsnaposone')
							exon_type_1, exon_type_2, exon_type_3, exon_type_4, exon_type_5, exon_type_6, exon_type_7, exon_none = exon_types(7)
							common=False
							condition=True
							self_rate_down = 1
							self_rate_up = 5000
							search_site = ''
							paste_site = ''
							remoove_self = True
					
							
							if working_exons != []:
								for exon in working_exons:
									exon_codons = exon

									if exon_codons[0] in exon_type_1:
										paste_site = codons_to_nucl(exon_codons[1:])
									
									if exon_codons[0] in exon_type_2:
										self_rate_down = codons_to_number(exon_codons[1:])
									
									elif exon_codons[0] in exon_type_3:
										self_rate_up = codons_to_number(exon_codons[1:])

									elif exon_codons[0] in exon_type_4:
										common = True
										
									elif exon_codons[0] in exon_type_5:

										condition_code = boolian_logic(cell, self.common_proteins, exon_codons[1:])

										if condition_code != None:
											condition = False
											condition_code = condition_code.replace('self.common_proteins','common_proteins')
											loc = {'common_proteins':self.common_proteins, 'cell':cell}
											common_proteins = self.common_proteins
											try:
												exec(condition_code, globals(), loc)
											except KeyError:
												pass
											try:
												condition = loc['condition']
											except KeyError:
												pass
									elif exon_codons[0] in exon_type_6:
										search_site+=codons_to_nucl(exon_codons[1:])
										
									elif exon_codons[0] in exon_type_7:
										remoove_self = False
							else:			
								pass
							
							if common == True:

								if full_protein_seq in self.common_proteins:
									if self.common_proteins[full_protein_seq] < rate:
										self.common_proteins[full_protein_seq] = rate
								else:
									self.common_proteins[full_protein_seq] = rate

							if condition == True:

								if search_site != '':
					
									if self_rate_down <= rate:
							
										if self_rate_up >= rate:
					
											start_site_tmp = cell[0].find(search_site)
											if start_site_tmp != -1:
												
												end_site_tmp = cell[0][start_site_tmp+len(search_site):start_site_tmp+len(search_site)+8000].find(search_site)
												if end_site_tmp != -1:

													end_site_tmp = start_site_tmp + end_site_tmp + len(search_site) * 2

													if paste_site == '':
														for i in range(randint(0,100)):
															paste_site += 'A'
														paste_site_ind = randint(10,len(cell[0])-10)
													else:
														paste_site_ind = cell[0].find(paste_site)
													
													if paste_site_ind != -1:
														paste_site_ind = paste_site_ind + math.floor(len(paste_site)/2)

														self.transposone = [start_site_tmp, end_site_tmp, paste_site_ind, full_protein_seq]


						#--------------------------------


						#Cell chromotin pack
						#--------------------------------
						elif protein_type in protein_types[7]:
							#print('chromotin pack')
							exon_type_1, exon_type_2, exon_type_3, exon_type_4, exon_type_5, exon_type_6, exon_none = exon_types(6)
							common=False
							condition=True
							self_rate_down = 1
							self_rate_up = 5000
							search_site = ''
							direction = 'r'
							self_power = 2048 #l - left, r-right, or b - both
							#add condition - left:1, right:2, center:3, delete:4
							if working_exons != []:
								for exon in working_exons:
									exon_codons = exon

									if exon_codons[0] in exon_type_1:
										search_site+=codons_to_nucl(exon_codons[1:])
									if exon_codons[0] in exon_type_2:
										self_rate_down = codons_to_number(exon_codons[1:])
									elif exon_codons[0] in exon_type_3:
										self_rate_up = codons_to_number(exon_codons[1:])
									elif exon_codons[0] in exon_type_4:
										common = True
									elif exon_codons[0] in exon_type_5:
										condition_code = boolian_logic(cell, self.common_proteins, exon_codons[1:])

										if condition_code != None:
											condition = False
											condition_code = condition_code.replace('self.common_proteins','common_proteins')
											loc = {'common_proteins':self.common_proteins, 'cell':cell}
											common_proteins = self.common_proteins
											try:
												exec(condition_code, globals(), loc)
											except KeyError:
												pass
											try:
												condition = loc['condition']
											except KeyError:
												pass
									elif exon_codons[0] in exon_type_6:
					
										left_ex, right_ex, center_ex, n_ex = exon_types(3)
					
										if exon_codons[1] in left_ex:
				
											direction = 'l'
										if exon_codons[1] in right_ex or exon_codons[1] in n_ex:
			
											pass
										if exon_codons[1] in center_ex:
			
											direction = 'b'
										if len(exon_codons) > 2:
											self_power = codons_to_number(exon_codons[2:], 5)


							else:			
								pass
							
							
							if common == True:

								if full_protein_seq in self.common_proteins:
									if self.common_proteins[full_protein_seq] < rate:
										self.common_proteins[full_protein_seq] = rate
								else:
									self.common_proteins[full_protein_seq] = rate

							if condition == True:
						
								if search_site != '':
									
								
									if self_rate_down <= rate:
								
										if self_rate_up >= rate:
													
												histone_idexes = [m.start() for m in re.finditer('(?={})'.format(search_site), cell[0])]
							
												if direction == 'r':

													for histone_index in histone_idexes:
											
														cell[0] = cell[0][:histone_index] + cell[0][histone_index:histone_index+self_power].lower() + cell[0][histone_index+self_power:]
												elif direction == 'l':

													for histone_index in histone_idexes:
										
														tmp = len(cell[0][:histone_index])
														if tmp > self_power:
															tmp = self_power
														else:
															pass
														cell[0] = cell[0][:histone_index-tmp] + cell[0][histone_index-tmp:histone_index+len(search_site)].lower() + cell[0][histone_index+len(search_site):]
											
												elif direction == 'b':


													for histone_index in histone_idexes:
									
														tmp = len(cell[0][:histone_index])
														if tmp > self_power:
															tmp = self_power
														else:
															pass
														cell[0] = cell[0][:histone_index-tmp] + cell[0][histone_index-tmp:histone_index+self_power].lower() + cell[0][histone_index+self_power:]
												

						#--------------------------------



						#Cell chromotin unpack
						#--------------------------------
						elif protein_type in protein_types[8]:
							#print('chromotin unpack')
							exon_type_1, exon_type_2, exon_type_3, exon_type_4, exon_type_5, exon_type_6, exon_none = exon_types(6)
							

							common=False
							condition=True
							self_rate_down = 1
							self_rate_up = 5000
							search_site = ''
							direction = 'r'
							self_power = 2048 #l - left, r-right, or b - both
							#add condition - left:1, right:2, center:3, delete:4
							if working_exons != []:
								for exon in working_exons:
									exon_codons = exon

									if exon_codons[0] in exon_type_1:
										search_site+=codons_to_nucl(exon_codons[1:])
									
									if exon_codons[0] in exon_type_2:
										self_rate_down = codons_to_number(exon_codons[1:])
										
									elif exon_codons[0] in exon_type_3:
										self_rate_up = codons_to_number(exon_codons[1:])
										
									elif exon_codons[0] in exon_type_4:
										common = True
										
									elif exon_codons[0] in exon_type_5:
										
										condition_code = boolian_logic(cell, self.common_proteins, exon_codons[1:])

										if condition_code != None:
											condition = False
											condition_code = condition_code.replace('self.common_proteins','common_proteins')
											loc = {'common_proteins':self.common_proteins, 'cell':cell}
											common_proteins = self.common_proteins
											try:
												exec(condition_code, globals(), loc)
											except KeyError:
												pass
											try:
												condition = loc['condition']
											except KeyError:
												pass
									elif exon_codons[0] in exon_type_6:
										
										left_ex, right_ex, center_ex, n_ex = exon_types(3)

										if exon_codons[1] in left_ex:
											direction = 'l'
										if exon_codons[1] in right_ex or exon_codons[1] in n_ex:
											pass
										if exon_codons[1] in center_ex:
											direction = 'b'
										if len(exon_codons) > 2:
											self_power = codons_to_number(exon_codons[2:], 5)



							else:			
								pass
							
							
							if common == True:

								if full_protein_seq in self.common_proteins:
									if self.common_proteins[full_protein_seq] < rate:
										self.common_proteins[full_protein_seq] = rate
								else:
									self.common_proteins[full_protein_seq] = rate

							if condition == True:
							
								if search_site != '':
									search_site = search_site.lower()
						
									if self_rate_down <= rate:
							
										if self_rate_up >= rate:
												
												histone_idexes = [m.start() for m in re.finditer('(?={})'.format(search_site), cell[0])]
												
												if histone_idexes != []:
													if len(histone_idexes) < rate:
														histone_idexes = histone_idexes[:rate]

												if direction == 'r':
													for histone_index in histone_idexes:
														
														cell[0] = cell[0][:histone_index] + cell[0][histone_index:histone_index+self_power].upper() + cell[0][histone_index+self_power:]
												elif direction == 'l':
													for histone_index in histone_idexes:
										
														tmp = len(cell[0][:histone_index])
														if tmp > self_power:
															tmp = self_power
														else:
															pass
														cell[0] = cell[0][:histone_index-tmp] + cell[0][histone_index-tmp:histone_index+len(search_site)].upper() + cell[0][histone_index+len(search_site):]
										
												elif direction == 'b':
													for histone_index in histone_idexes:
														tmp = len(cell[0][:histone_index])
														if tmp > self_power:
															tmp = self_power
														else:
															pass
														cell[0] = cell[0][:histone_index-tmp] + cell[0][histone_index-tmp:histone_index+self_power].upper() + cell[0][histone_index+self_power:]


						#--------------------------------




						#Cell protein deletion
						#--------------------------------

						elif protein_type in protein_types[9]:
							#print('protein deletion')
							exon_type_1, exon_type_2, exon_type_3, exon_type_4, exon_type_5, exon_none = exon_types(5)
							common=False
							condition=True
							self_rate_down = 1
							self_rate_up = 5000
							search_site = ''
							#add condition - left:1, right:2, center:3, delete:4
							#GGAGTTGCTAATAATAAAAGATTACAATAAACAA CAG AAC TGC
							#AAAATT GC _ATA ACG ACG ACG_ GC  GC CAT ACT GC

							if working_exons != []:
								for exon in working_exons:

									exon_codons = exon

									if exon_codons[0] in exon_type_1:
										search_site+=codons_to_nucl(exon_codons[1:])
									if exon_codons[0] in exon_type_2:
										self_rate_down = codons_to_number(exon_codons[1:])
									
									elif exon_codons[0] in exon_type_3:
										self_rate_up = codons_to_number(exon_codons[1:])

									elif exon_codons[0] in exon_type_4:
										common = True
										
									elif exon_codons[0] in exon_type_5:

										condition_code = boolian_logic(cell, self.common_proteins, exon_codons[1:])

										if condition_code != None:
											condition = False
											condition_code = condition_code.replace('self.common_proteins','common_proteins')
											loc = {'common_proteins':self.common_proteins, 'cell':cell}
											common_proteins = self.common_proteins
											try:
												exec(condition_code, globals(), loc)
											except KeyError:
												pass
											try:
												condition = loc['condition']
											except KeyError:
												pass
							else:			
								pass
							
							if common == True:

								if full_protein_seq in self.common_proteins:
									if self.common_proteins[full_protein_seq] < rate:
										self.common_proteins[full_protein_seq] = rate
								else:
									self.common_proteins[full_protein_seq] = rate

							if condition == True:
						
								if search_site != '':
								
									if self_rate_down <= rate:
										
										if self_rate_up >= rate:
									
											proteins_list = []
											for prot in cell[1]:
												if search_site in prot:
													proteins_list.append([prot, cell[1][prot]])
											for prot, prot_rate in proteins_list:
												
												if rate >= prot_rate:
													del cell[1][prot]
												else:
													cell[1][prot] = cell[1][prot]-rate
						

						#--------------------------------





						#Cell channels passive
						#--------------------------------
					

												

						elif protein_type in protein_types[10]:
							#print('channels passive')
					
							exon_type_1, exon_type_2, exon_type_3, exon_type_4, exon_type_5, exon_type_6, exon_type_7, exon_none = exon_types(7)
							common=False
							condition=True
							self_rate_down = 1
							self_rate_up = 5000
							search_cell_type = []
							prot_search_seq = []
							all_prot = False
							self_add = False
							way_dir = 't'# t - to, f-from, b-both
							#add condition - left:1, right:2, center:3, delete:4

							#GGA AAT TAT AAT TAT AAT TAT AAT TAT 
					
							if working_exons != []:
								for exon in working_exons:

									exon_codons = exon

									if exon_codons[0] in exon_type_1:
										search_cell_type.append(codons_to_nucl(exon_codons[1:]))

									if exon_codons[0] in exon_type_2:
										self_rate_down = codons_to_number(exon_codons[1:])
									
									elif exon_codons[0] in exon_type_3:
										self_rate_up = codons_to_number(exon_codons[1:])

									elif exon_codons[0] in exon_type_4:
										common = True
										
									elif exon_codons[0] in exon_type_5:

										condition_code = boolian_logic(cell, self.common_proteins, exon_codons[1:])

										if condition_code != None:
											condition = False
											condition_code = condition_code.replace('self.common_proteins','common_proteins')
											loc = {'common_proteins':self.common_proteins, 'cell':cell}
											common_proteins = self.common_proteins
											try:
												exec(condition_code, globals(), loc)
											except KeyError:
												pass
											try:
												condition = loc['condition']
											except KeyError:
												pass
									elif exon_codons[0] in exon_type_6:
										prot_search_seq.append(codons_to_nucl(exon_codons[1:]))


									elif exon_codons[0] in exon_type_7:

										
										ex1, ex2, ex3, ex4, exon_none = exon_types(4)
							
										if exon_codons[1] in ex1:
											way_dir = 'f'

										elif exon_codons[1] in ex2:
											way_dir = 'b'
										
										elif exon_codons[1] in ex3:
											all_prot = True

										elif exon_codons[1] in ex4:
											self_add = True

										

							

							else:			
								pass
							
							if common == True:

								if full_protein_seq in self.common_proteins:
									if self.common_proteins[full_protein_seq] < rate:
										self.common_proteins[full_protein_seq] = rate
								else:
									self.common_proteins[full_protein_seq] = rate



							if condition == True:
						
									
									if self_rate_down <= rate:
									
										if self_rate_up >= rate:
									

											cells_indxs = []
											for cell_index, cell_n in enumerate(self.nn):
												for search_seq in search_cell_type:
													if search_seq in cell_n[2]:
														cells_indxs.append(cell_index)

											
											if cells_indxs != []:	
												
												if all_prot == True:

													if way_dir == 't':
												
														for prot, rte in cell[1].items():
															
															if self_add != True:
																if prot == full_protein_seq:
																	continue
															for cell_indx in cells_indxs:
																if prot in self.nn[cell_indx][1]:
																	if self.nn[cell_indx][1][prot] >= rte:
																		pass
																	else:
																		self.nn[cell_indx][1][prot] = math.ceil( (rte + self.nn[cell_indx][1][prot]) / 2 )
																		cell[1][prot] = math.ceil( (rte + self.nn[cell_indx][1][prot]) / 2 )
																else:
																	self.nn[cell_indx][1][prot] = math.ceil( rte / 2)
																	cell[1][prot] = math.ceil( rte / 2 )
																
													elif way_dir == 'f':
													
														for cell_indx in cells_indxs:
															for prot, rte in self.nn[cell_indx][1].items():
								
																if self_add != True:
																	if prot == full_protein_seq:
																		continue
				
																if prot in cell[1]:
																	if cell[1][prot] >= rte:
																		pass
																	else:
																		cell[1][prot] = math.ceil( (rte + cell[1][prot]) / 2)
																		self.nn[cell_indx][1][prot] = math.ceil( (rte + cell[1][prot]) / 2 )
																else:
																	cell[1][prot] = math.ceil(rte /2 )
																	self.nn[cell_indx][1][prot] = math.ceil( rte / 2)



													elif way_dir == 'b':
														
														for prot, rte in cell[1].items():
													
															if self_add != True:
																if prot == full_protein_seq:
																	continue
															
															for cell_indx in cells_indxs:
																if prot in self.nn[cell_indx][1]:
																	if self.nn[cell_indx][1][prot] >= rte:
																		pass
																	else:

																		self.nn[cell_indx][1][prot] = math.ceil( (rte + self.nn[cell_indx][1][prot]) / 2 )
																		cell[1][prot] = math.ceil( (rte + self.nn[cell_indx][1][prot]) / 2 )

																else:

																	self.nn[cell_indx][1][prot] = math.ceil( rte / 2 )
																	cell[1][prot] = math.ceil( rte / 2 )


														for cell_indx in cells_indxs:
															for prot, rte in self.nn[cell_indx][1].items():
														
																if self_add != True:
																	if prot == full_protein_seq:
																		continue
														
																if prot in cell[1]:

																	if cell[1][prot] >= rte:
																		pass
																	else:

																		cell[1][prot] = math.ceil( (rte + cell[1][prot]) / 2)
																		self.nn[cell_indx][1][prot] = math.ceil( (rte + cell[1][prot]) / 2 )

																else:

																	cell[1][prot] = math.ceil( rte /2 )
																	self.nn[cell_indx][1][prot] = math.ceil( rte / 2 )

												

												elif all_prot == False:
													
													if prot_search_seq != []:
														if way_dir == 't':
															for prot, rte in cell[1].items():
																
													
																
																if [i for i in prot_search_seq if i in prot] != []:
																	for cell_indx in cells_indxs:
											
																		if prot in self.nn[cell_indx][1]:
																			if self.nn[cell_indx][1][prot] >= rte:
																				pass
																			else:
																				self.nn[cell_indx][1][prot] = math.ceil( (rte + self.nn[cell_indx][1][prot]) / 2 )
																				cell[1][prot] = math.ceil( (rte + self.nn[cell_indx][1][prot]) / 2)
																		else:
																			self.nn[cell_indx][1][prot] =math.ceil( rte / 2)
																			cell[1][prot] = math.ceil( rte / 2 )
														
														elif way_dir == 'f':
															for cell_indx in cells_indxs:
																for prot, rte in self.nn[cell_indx][1].items():
																	if [i for i in prot_search_seq if i in prot] != []:
																		if prot in cell[1]:
																			if cell[1][prot] >= rte:
																				pass
																			else:
																				cell[1][prot] = math.ceil( (rte + cell[1][prot]) / 2 )
																				self.nn[cell_indx][1][prot] =  math.ceil( (rte + cell[1][prot]) / 2)
																		else:
																			cell[1][prot] = math.ceil( rte /2 )
																			self.nn[cell_indx][1][prot] = math.ceil( rte / 2 )
														
														elif way_dir == 'b':
															for prot, rte in cell[1].items():
																if [i for i in prot_search_seq if i in prot] != []:
																	for cell_indx in cells_indxs:
																		if prot in self.nn[cell_indx][1]:
																			if self.nn[cell_indx][1][prot] >= rte:
																				pass
																			else:
																				self.nn[cell_indx][1][prot] = math.ceil( (rte + self.nn[cell_indx][1][prot]) / 2 )
																				cell[1][prot] = math.ceil( (rte + self.nn[cell_indx][1][prot]) / 2 )
																		else:
																			self.nn[cell_indx][1][prot] = math.ceil( rte / 2)
																			cell[1][prot] = math.ceil( rte / 2)

															for cell_indx in cells_indxs:
																for prot, rte in self.nn[cell_indx][1].items():
																	if [i for i in prot_search_seq if i in prot] != []:
																		if prot in cell[1]:
																			if cell[1][prot] >= rte:
																				pass
																			else:
																				cell[1][prot] = math.ceil( (rte + cell[1][prot]) / 2)
																				self.nn[cell_indx][1][prot] = math.ceil((rte + cell[1][prot]) / 2)
																		else:
																			cell[1][prot] = math.ceil(rte /2 )
																			self.nn[cell_indx][1][prot] = math.ceil(rte / 2)

						#--------------------------------



						#Cell channels active
						#--------------------------------
					

				
						elif protein_type in protein_types[11]:
							#print('channels active')
							exon_type_1, exon_type_2, exon_type_3, exon_type_4, exon_type_5, exon_type_6, exon_type_7, exon_none = exon_types(7)
							common=False
							condition=True
							self_rate_down = 1
							self_rate_up = 5000
							search_cell_type = []
							prot_search_seq = []
							self_add = False
							way_dir = 't'# t - to, f-from, b-both
							#add condition - left:1, right:2, center:3, delete:4

							#GGA AAT TAT AAT TAT AAT TAT AAT TAT 
				
							if working_exons != []:
								for exon in working_exons:

									exon_codons = exon

									if exon_codons[0] in exon_type_1:
										search_cell_type.append(codons_to_nucl(exon_codons[1:]))

									if exon_codons[0] in exon_type_2:
										self_rate_down = codons_to_number(exon_codons[1:])
									
									elif exon_codons[0] in exon_type_3:
										self_rate_up = codons_to_number(exon_codons[1:])

									elif exon_codons[0] in exon_type_4:
										common = True
										
									elif exon_codons[0] in exon_type_5:

										condition_code = boolian_logic(cell, self.common_proteins, exon_codons[1:])

										if condition_code != None:
											condition = False
											condition_code = condition_code.replace('self.common_proteins','common_proteins')
											loc = {'common_proteins':self.common_proteins, 'cell':cell}
											common_proteins = self.common_proteins
											try:
												exec(condition_code, globals(), loc)
											except KeyError:
												pass
											try:
												condition = loc['condition']
											except KeyError:
												pass
									elif exon_codons[0] in exon_type_6:
										prot_search_seq.append(codons_to_nucl(exon_codons[1:]))


									elif exon_codons[0] in exon_type_7:

										
										ex1, ex2, exon_none = exon_types(2)
										
										if exon_codons[1] in ex1:
											way_dir = 'f'
										elif exon_codons[1] in ex2:
											self_add = True

							else:			
								pass
							
							if common == True:

								if full_protein_seq in self.common_proteins:
									if self.common_proteins[full_protein_seq] < rate:
										self.common_proteins[full_protein_seq] = rate
								else:
									self.common_proteins[full_protein_seq] = rate



							if condition == True:
								if self_rate_down <= rate:
								
									if self_rate_up >= rate:
								

										cells_indxs = []
										for cell_index, cell_n in enumerate(self.nn):
											for search_seq in search_cell_type:
												if search_seq in cell_n[2]:
													cells_indxs.append(cell_index)

										
										if cells_indxs != []:			
											if prot_search_seq != []:
												if way_dir == 't':
													self_chnnel_deletion = []
			
													for prot, rte in cell[1].items():
														
					
												
														if [i for i in prot_search_seq if i in prot] != []:
															for cell_indx in cells_indxs:

																if prot in self.nn[cell_indx][1]:
																	self.nn[cell_indx][1][prot] = self.nn[cell_indx][1][prot] + rte
																	if prot not in self_chnnel_deletion:
																		self_chnnel_deletion.append(prot)

																else:
																	self.nn[cell_indx][1][prot] = rte
																	if prot not in self_chnnel_deletion:
																		self_chnnel_deletion.append(prot)

							

													for prot in self_chnnel_deletion:

														del cell[1][prot]


												elif way_dir == 'f':
													others_chnnel_deletion = []
													for cell_indx in cells_indxs:
														for prot, rte in self.nn[cell_indx][1].items():
															if [i for i in prot_search_seq if i in prot] != []:
																if prot in cell[1]:
																	cell[1][prot] = cell[1][prot]+rte
																	others_chnnel_deletion.append([prot, cell_indx])

																else:
																	cell[1][prot] = rte
																	others_chnnel_deletion.append([prot, cell_indx])

													for prot, ind in others_chnnel_deletion:
														if prot in self.nn[ind][1]:
															del self.nn[ind][1][prot]



						#--------------------------------


						#cell apoptosis
						#--------------------------------
						elif protein_type in protein_types[12]:
							#print(' apoptosis')
							exon_type_1, exon_type_2, exon_type_3, exon_type_4, exon_none = exon_types(4)
							common=False
							condition=True
							self_rate_down = 1
							self_rate_up = 5000

							#GGA AAT TAT AAT TAT AAT TAT AAT TAT 
							
							if working_exons != []:
								for exon in working_exons:

									exon_codons = exon

									if exon_codons[0] in exon_type_1:
										self_rate_down = codons_to_number(exon_codons[1:])
									
									elif exon_codons[0] in exon_type_2:
										self_rate_up = codons_to_number(exon_codons[1:])

									elif exon_codons[0] in exon_type_3:
										common = True
										
									elif exon_codons[0] in exon_type_4:

										condition_code = boolian_logic(cell, self.common_proteins, exon_codons[1:])

										if condition_code != None:
											condition = False
											condition_code = condition_code.replace('self.common_proteins','common_proteins')
											loc = {'common_proteins':self.common_proteins, 'cell':cell}
											common_proteins = self.common_proteins
											try:
												exec(condition_code, globals(), loc)
											except KeyError:
												pass
											try:
												condition = loc['condition']
											except KeyError:
												pass
									
							else:			
								pass
							
							if common == True:

								if full_protein_seq in self.common_proteins:
									if self.common_proteins[full_protein_seq] < rate:
										self.common_proteins[full_protein_seq] = rate
								else:
									self.common_proteins[full_protein_seq] = rate


							
							if condition == True:
								if self_rate_down <= rate:
								
									if self_rate_up >= rate:
								
										apoptosis = True

						#--------------------------------






						#cell secrete
						#--------------------------------
						elif protein_type in protein_types[13]:
							#print(' secrete')
							exon_type_1, exon_type_2, exon_type_3, exon_type_4, exon_type_5, exon_type_6, exon_type_7, exon_none = exon_types(7)
							common=False
							condition=True
							self_rate_down = 1
							self_rate_up = 5000
							produce = []
							super_prot = ''
							produce_rate = 1

							
							
							if working_exons != []:
								for exon in working_exons:

									exon_codons = exon

									if exon_codons[0] in exon_type_1:
										self_rate_down = codons_to_number(exon_codons[1:])
									
									elif exon_codons[0] in exon_type_2:
										self_rate_up = codons_to_number(exon_codons[1:])

									elif exon_codons[0] in exon_type_3:
										common = True
										
									elif exon_codons[0] in exon_type_4:

										condition_code = boolian_logic(cell, self.common_proteins, exon_codons[1:])

										if condition_code != None:
											condition = False
											condition_code = condition_code.replace('self.common_proteins','common_proteins')
											loc = {'common_proteins':self.common_proteins, 'cell':cell}
											common_proteins = self.common_proteins
											try:
												exec(condition_code, globals(), loc)
											except KeyError:
												pass
											try:
												condition = loc['condition']
											except KeyError:
												pass
									elif exon_codons[0] in exon_type_5:
										produce.append(codons_to_nucl(exon_codons[1:]))
									elif exon_codons[0] in exon_type_6:
										super_prot += codons_to_nucl(exon_codons[1:])
									elif exon_codons[0] in exon_type_7:
										produce_rate = codons_to_number(exon_codons[1:])
										
							else:			
								pass
							
							if common == True:

								if full_protein_seq in self.common_proteins:
									if self.common_proteins[full_protein_seq] < rate:
										self.common_proteins[full_protein_seq] = rate
								else:
									self.common_proteins[full_protein_seq] = rate


							if condition == True:
								if self_rate_down <= rate:
								
									if self_rate_up >= rate:
								
										if super_prot != '':
											if super_prot in cell[1]:
												cell[1][super_prot] = cell[1][super_prot] + (rate * produce_rate)
											else:
												cell[1][super_prot] = rate * produce_rate
											
										for sec_prod in produce:
											if sec_prod in cell[1]:
												cell[1][sec_prod] = cell[1][sec_prod] + (rate * produce_rate)
											else:
												cell[1][sec_prod] = rate * produce_rate

						#--------------------------------









						#cell activation and sielnce strenth
						#--------------------------------
						elif protein_type in protein_types[14]:
							#print('activation and sielnce strenth')
							exon_type_1, exon_type_2, exon_type_3, exon_type_4, exon_type_5, exon_type_6, exon_none = exon_types(6)
							common=False
							condition=True
							self_rate_down = 1
							self_rate_up = 5000
							tmp_act_str = None
							tmp_sile_str = None
							produce_rate = 1

							
							
							if working_exons != []:
								for exon in working_exons:

									exon_codons = exon

									if exon_codons[0] in exon_type_1:
										self_rate_down = codons_to_number(exon_codons[1:])
									
									elif exon_codons[0] in exon_type_2:
										self_rate_up = codons_to_number(exon_codons[1:])

									elif exon_codons[0] in exon_type_3:
										common = True
										
									elif exon_codons[0] in exon_type_4:

										condition_code = boolian_logic(cell, self.common_proteins, exon_codons[1:])

										if condition_code != None:
											condition = False
											condition_code = condition_code.replace('self.common_proteins','common_proteins')
											loc = {'common_proteins':self.common_proteins, 'cell':cell}
											common_proteins = self.common_proteins
											try:
												exec(condition_code, globals(), loc)
											except KeyError:
												pass
											try:
												condition = loc['condition']
											except KeyError:
												pass
									elif exon_codons[0] in exon_type_5:
										tmp_act_str = codons_to_number(exon_codons[1:], 4)
									elif exon_codons[0] in exon_type_6:
										tmp_sile_str = codons_to_number(exon_codons[1:], 4)

							else:			
								pass
							
							
							if common == True:

								if full_protein_seq in self.common_proteins:
									if self.common_proteins[full_protein_seq] < rate:
										self.common_proteins[full_protein_seq] = rate
								else:
									self.common_proteins[full_protein_seq] = rate

							if condition == True:
								if self_rate_down <= rate:
								
									if self_rate_up >= rate:
									
										if tmp_act_str != None:
											self.activation_strength = tmp_act_str
										
										if tmp_sile_str != None:
											self.silence_strength = tmp_sile_str
						#--------------------------------




						#signaling
						#--------------------------------
						elif protein_type in protein_types[15]:
							#print('signaling')
							exon_type_1, exon_type_2, exon_type_3, exon_type_4, exon_type_5, exon_none = exon_types(5)
							common=False
							condition=True
							self_rate_down = 1
							self_rate_up = 5000
							random_number = None
							produce_rate = 1
							
							if working_exons != []:
								for exon in working_exons:

									exon_codons = exon

									if exon_codons[0] in exon_type_1:
										self_rate_down = codons_to_number(exon_codons[1:])
									
									elif exon_codons[0] in exon_type_2:
										self_rate_up = codons_to_number(exon_codons[1:])

									elif exon_codons[0] in exon_type_3:
										common = True
										
									elif exon_codons[0] in exon_type_4:

										condition_code = boolian_logic(cell, self.common_proteins, exon_codons[1:])

										if condition_code != None:
											condition = False
											condition_code = condition_code.replace('self.common_proteins','common_proteins')
											loc = {'common_proteins':self.common_proteins, 'cell':cell}
											common_proteins = self.common_proteins
											try:
												exec(condition_code, globals(), loc)
											except KeyError:
												pass
											try:
												condition = loc['condition']
											except KeyError:
												pass
									elif exon_codons[0] in exon_type_5:
										random_number = randint(1, codons_to_number(exon_type_5))

							else:			
								pass
							
							
							if common == True:

								if full_protein_seq in self.common_proteins:
									if self.common_proteins[full_protein_seq] < rate:
										self.common_proteins[full_protein_seq] = rate
								else:
									self.common_proteins[full_protein_seq] = rate

							if condition == True:
								if self_rate_down <= rate:
								
									if self_rate_up >= rate:
										if random_number != None:
											cell[1][full_protein_seq] = random_number
						#--------------------------------

						







						#splicing regulatory factor
						#--------------------------------
						elif protein_type in protein_types[16]:
							#print('splicing regulatory factor')
							exon_type_1, exon_type_2, exon_type_3, exon_type_4, exon_type_5, exon_type_6, exon_type_7, exon_none = exon_types(7)
							common=False
							condition=True
							self_rate_down = 1
							self_rate_up = 5000
							seq = ''
							splice_type = None
							right_work = True

							
							if working_exons != []:
								for exon in working_exons:

									exon_codons = exon

									if exon_codons[0] in exon_type_1:
										seq += codons_to_nucl(exon_codons[1:])
									
									elif exon_codons[0] in exon_type_2:
										self_rate_down = codons_to_number(exon_codons[1:])
									
									elif exon_codons[0] in exon_type_3:
										self_rate_up = codons_to_number(exon_codons[1:])

									elif exon_codons[0] in exon_type_4:
										common = True
										
									elif exon_codons[0] in exon_type_5:

										condition_code = boolian_logic(cell, self.common_proteins, exon_codons[1:])

										if condition_code != None:
											condition = False
											condition_code = condition_code.replace('self.common_proteins','common_proteins')
											loc = {'common_proteins':self.common_proteins, 'cell':cell}
											common_proteins = self.common_proteins
											try:
												exec(condition_code, globals(), loc)
											except KeyError:
												pass
											try:
												condition = loc['condition']
											except KeyError:
												pass

									elif exon_codons[0] in exon_type_6:
										ex1, ex2, ex3, ex4, ex5, ex6, n_ex = exon_types(6)
										if exon_codons[1] in ex1:
											splice_type = 'l'
										elif exon_codons[1] in ex2:
											splice_type = 'r'
										elif exon_codons[1] in ex3:
											splice_type = 'b'
										elif exon_codons[1] in ex4:
											splice_type = 'f'
										elif exon_codons[1] in ex5:
											splice_type = 'p'
										elif exon_codons[1] in ex6:
											splice_type = 'n'
									elif exon_codons[0] in exon_type_7:
										ex1, ex2, ex3, ex4, ex5, n_ex = exon_types(5)
										if exon_codons[1] in ex1:
												splice_type = 's'
										elif exon_codons[1] in ex2:
												splice_type = 'a'
										elif exon_codons[1] in ex3:
												splice_type = 'z'
										elif exon_codons[1] in ex4:
												right_work = None
										elif exon_codons[1] in ex5:
												right_work = False



							else:			
								pass
							
							
							if common == True:

								if full_protein_seq in self.common_proteins:
									if self.common_proteins[full_protein_seq] < rate:
										self.common_proteins[full_protein_seq] = rate
								else:
									self.common_proteins[full_protein_seq] = rate

							if condition == True:
								if self_rate_down <= rate:
									if self_rate_up >= rate:
										if seq != '':
											if splice_type != None:
												if splice_type == 's' or splice_type == 'a' or splice_type == 'z':
													self.splice_regul[1][seq] = [splice_type, right_work]
												else:
													self.splice_regul[0][seq] = [splice_type, right_work]




						#--------------------------------






						#cell name
						#--------------------------------
						elif protein_type in protein_types[17]:
							#print('cell name')
							exon_type_1, exon_type_2, exon_type_3, exon_type_4, exon_type_5, exon_type_6, exon_none = exon_types(6)
							common=False
							condition=True
							self_rate_down = 1
							self_rate_up = 5000
							seq = ''
							new_name = False

							if working_exons != []:
								for exon in working_exons:

									exon_codons = exon

									if exon_codons[0] in exon_type_1:
										seq += codons_to_nucl(exon_codons[1:])

									elif exon_codons[0] in exon_type_2:
										self_rate_down = codons_to_number(exon_codons[1:])
									
									elif exon_codons[0] in exon_type_3:
										self_rate_up = codons_to_number(exon_codons[1:])

									elif exon_codons[0] in exon_type_4:
										common = True
										
									elif exon_codons[0] in exon_type_5:

										condition_code = boolian_logic(cell, self.common_proteins, exon_codons[1:])

										if condition_code != None:
											condition = False
											condition_code = condition_code.replace('self.common_proteins','common_proteins')
											loc = {'common_proteins':self.common_proteins, 'cell':cell}
											common_proteins = self.common_proteins
											try:
												exec(condition_code, globals(), loc)
											except KeyError:
												pass
											try:
												condition = loc['condition']
											except KeyError:
												pass
									elif exon_codons[0] in exon_type_6:
										
										seq = codons_to_nucl(exon_codons[1:])
										new_name = True

							else:			
								pass
							
							
							if common == True:

								if full_protein_seq in self.common_proteins:
									if self.common_proteins[full_protein_seq] < rate:
										self.common_proteins[full_protein_seq] = rate
								else:
									self.common_proteins[full_protein_seq] = rate

							if condition == True:
								if self_rate_down <= rate:
									if self_rate_up >= rate:
										if seq != '':
											
											if new_name == False:
												
												cell[2] = cell[2] + seq
											else:
												
												cell[2] = seq


						#--------------------------------

						



						#connection growth factor
						#--------------------------------
						elif protein_type in protein_types[18]:
							#print('growth factor')
							exon_type_1, exon_type_2, exon_type_3, exon_type_4, exon_type_5, exon_type_6, exon_type_7, exon_none = exon_types(7)
							common=False
							condition=True
							self_rate_down = 1
							self_rate_up = 5000
							find_connection_to = ''
							list_connections = []
							
							
							if working_exons != []:
								for exon in working_exons:

									exon_codons = exon


									if exon_codons[0] in exon_type_1:
										find_connection_to += codons_to_nucl(exon_codons[1:])
										

									elif exon_codons[0] in exon_type_2:
										self_rate_down = codons_to_number(exon_codons[1:])
									
									elif exon_codons[0] in exon_type_3:
										self_rate_up = codons_to_number(exon_codons[1:])

									elif exon_codons[0] in exon_type_4:
										common = True
										
									elif exon_codons[0] in exon_type_5:

										condition_code = boolian_logic(cell, self.common_proteins, exon_codons[1:])

										if condition_code != None:
											condition = False
											condition_code = condition_code.replace('self.common_proteins','common_proteins')
											loc = {'common_proteins':self.common_proteins, 'cell':cell}
											common_proteins = self.common_proteins
											try:
												exec(condition_code, globals(), loc)
											except KeyError:
												pass
											try:
												condition = loc['condition']
											except KeyError:
												pass
									if exon_codons[0] in exon_type_6:
										list_connections.append(codons_to_nucl(exon_codons[1:]))
										

									if exon_codons[0] in exon_type_7:
										list_connections.append(cell[2])


										

							else:			
								pass
							
							
							if common == True:

								if full_protein_seq in self.common_proteins:
									if self.common_proteins[full_protein_seq] < rate:
										self.common_proteins[full_protein_seq] = rate
								else:
									self.common_proteins[full_protein_seq] = rate
							
							if condition == True:
								if self_rate_down <= rate:
									if self_rate_up >= rate:
										if find_connection_to != '':
											for cell_n in self.nn:
												if find_connection_to in cell_n[2]:
													if cell_n[2] not in cell[3]:
														 cell[3].append(cell_n[2])

										if list_connections != []:
											for cell_n in self.nn:
												for find_con_seq in list_connections:
													if find_con_seq in cell_n[2]:
														if cell_n[2] not in cell[3]:
															 cell[3].append(cell_n[2])
							

						#--------------------------------



						#cell matrix operation type
						#--------------------------------
						elif protein_type in protein_types[19] or protein_type in protein_types[29]:#old - 19
							#print('matrix operation')
							#cell[4] = 'AAACCC'
							exon_type_1, exon_type_2, exon_type_3, exon_type_4, exon_type_5, exon_type_6, exon_type_7, exon_none = exon_types(7)
							common=False
							condition=True
							self_rate_down = 1
							self_rate_up = 5000
							add_operation_type = ''
							new_operation_type = True
							
				
							if working_exons != []:
								for exon in working_exons:

									exon_codons = exon


									if exon_codons[0] in exon_type_1:
										add_operation_type += codons_to_nucl(exon_codons[1:])
										
									elif exon_codons[0] in exon_type_2:
										self_rate_down = codons_to_number(exon_codons[1:])
									
									elif exon_codons[0] in exon_type_3:
										self_rate_up = codons_to_number(exon_codons[1:])

									elif exon_codons[0] in exon_type_4:
										common = True
										
									elif exon_codons[0] in exon_type_5:

										condition_code = boolian_logic(cell, self.common_proteins, exon_codons[1:])

										if condition_code != None:
											condition = False
											condition_code = condition_code.replace('self.common_proteins','common_proteins')
											loc = {'common_proteins':self.common_proteins, 'cell':cell}
											common_proteins = self.common_proteins
											try:
												exec(condition_code, globals(), loc)
											except KeyError:
												pass
											try:
												condition = loc['condition']
											except KeyError:
												pass
									if exon_codons[0] in exon_type_6:
										add_operation_type = codons_to_nucl(exon_codons[1:])


									if exon_codons[0] in exon_type_7:
										new_operation_type = True


										

							else:			
								pass
							
							
							if common == True:

								if full_protein_seq in self.common_proteins:
									if self.common_proteins[full_protein_seq] < rate:
										self.common_proteins[full_protein_seq] = rate
								else:
									self.common_proteins[full_protein_seq] = rate

							#print('add_operation_type:',add_operation_type, rate, self_rate_down, self_rate_up)
							if condition == True:
								#print(self_rate_down,rate)
								if self_rate_down <= rate:
									if self_rate_up >= rate:
										if add_operation_type != '':
											if new_operation_type == False:
												cell[4] += add_operation_type
											else:
												cell[4] = add_operation_type
											#print('new matrix:',cell[4])

						#--------------------------------







						#cell activation function
						#--------------------------------
						elif protein_type in protein_types[20]:
							#print('activation function')
							exon_type_1, exon_type_2, exon_type_3, exon_type_4, exon_type_5, exon_type_6, exon_type_7, exon_none = exon_types(7)
							common=False
							condition=True
							self_rate_down = 1
							self_rate_up = 5000
							add_operation_type = ''
							new_operation_type = False
							
				
							if working_exons != []:
								for exon in working_exons:

									exon_codons = exon


									if exon_codons[0] in exon_type_1:
										add_operation_type += codons_to_nucl(exon_codons[1:])
										

									elif exon_codons[0] in exon_type_2:
										self_rate_down = codons_to_number(exon_codons[1:])
									
									elif exon_codons[0] in exon_type_3:
										self_rate_up = codons_to_number(exon_codons[1:])

									elif exon_codons[0] in exon_type_4:
										common = True
										
									elif exon_codons[0] in exon_type_5:

										condition_code = boolian_logic(cell, self.common_proteins, exon_codons[1:])

										if condition_code != None:
											condition = False
											condition_code = condition_code.replace('self.common_proteins','common_proteins')
											loc = {'common_proteins':self.common_proteins, 'cell':cell}
											common_proteins = self.common_proteins
											try:
												exec(condition_code, globals(), loc)
											except KeyError:
												pass
											try:
												condition = loc['condition']
											except KeyError:
												pass
									if exon_codons[0] in exon_type_6:
										add_operation_type = codons_to_nucl(exon_codons[1:])


									if exon_codons[0] in exon_type_7:
										new_operation_type = True


										

							else:			
								pass
							
							
							if common == True:

								if full_protein_seq in self.common_proteins:
									if self.common_proteins[full_protein_seq] < rate:
										self.common_proteins[full_protein_seq] = rate
								else:
									self.common_proteins[full_protein_seq] = rate

							if condition == True:
								if self_rate_down <= rate:
									if self_rate_up >= rate:
										if add_operation_type != '':
											if new_operation_type == False:
												cell[5] += add_operation_type
											else:
												cell[5] = add_operation_type


						#--------------------------------








						#cell weights
						#--------------------------------
						elif protein_type in protein_types[21]:
							#print(' weights')
							exon_type_1, exon_type_2, exon_type_3, exon_type_4, exon_type_5, exon_type_6, exon_type_7, exon_none = exon_types(7)
							common=False
							condition=True
							self_rate_down = 1
							self_rate_up = 5000
							add_operation_type = ''
							new_operation_type = False
							rate_multiply = False
							
				
							if working_exons != []:
								for exon in working_exons:

									exon_codons = exon


									if exon_codons[0] in exon_type_1:
										add_operation_type += codons_to_nucl(exon_codons[1:])
										

									elif exon_codons[0] in exon_type_2:
										self_rate_down = codons_to_number(exon_codons[1:])
									
									elif exon_codons[0] in exon_type_3:
										self_rate_up = codons_to_number(exon_codons[1:])

									elif exon_codons[0] in exon_type_4:
										common = True
										
									elif exon_codons[0] in exon_type_5:

										condition_code = boolian_logic(cell, self.common_proteins, exon_codons[1:])

										if condition_code != None:
											condition = False
											condition_code = condition_code.replace('self.common_proteins','common_proteins')
											loc = {'common_proteins':self.common_proteins, 'cell':cell}
											common_proteins = self.common_proteins
											try:
												exec(condition_code, globals(), loc)
											except KeyError:
												pass
											try:
												condition = loc['condition']
											except KeyError:
												pass
									if exon_codons[0] in exon_type_6:
										add_operation_type = codons_to_nucl(exon_codons[1:])


									if exon_codons[0] in exon_type_7:
										ex1, ex2, ex_none = exon_types(2)
										if exon_codons[1] in ex1:
											new_operation_type = True
										elif exon_codons[1] in ex2:
											rate_multiply = True


										

							else:			
								pass
							
							
							if common == True:

								if full_protein_seq in self.common_proteins:
									if self.common_proteins[full_protein_seq] < rate:
										self.common_proteins[full_protein_seq] = rate
								else:
									self.common_proteins[full_protein_seq] = rate

							if condition == True:
								if self_rate_down <= rate:
									if self_rate_up >= rate:
										if add_operation_type != '':
											if rate_multiply == False:
												if new_operation_type == False:
													cell[6] += add_operation_type
												else:
													cell[6] = add_operation_type
											else:
												if new_operation_type == False:
													cell[6] += add_operation_type*rate
												else:
													cell[6] = add_operation_type*rate

						#--------------------------------
						#proteins do nothing
						else:
							#print('other proteins')
							exon_type_1, exon_type_2, exon_type_3, exon_type_4, exon_type_5, exon_type_6, exon_type_7, exon_none = exon_types(7)
							common=False
							condition=True
							self_rate_down = 1
							self_rate_up = 5000
							
				
							if working_exons != []:
								for exon in working_exons:

									exon_codons = exon
									
									if exon_codons[0] in exon_type_2:
										self_rate_down = codons_to_number(exon_codons[1:])
									
									elif exon_codons[0] in exon_type_3:
										self_rate_up = codons_to_number(exon_codons[1:])

									elif exon_codons[0] in exon_type_4:
										common = True
										
									elif exon_codons[0] in exon_type_5:

										condition_code = boolian_logic(cell, self.common_proteins, exon_codons[1:])

										if condition_code != None:
											condition = False
											condition_code = condition_code.replace('self.common_proteins','common_proteins')
											loc = {'common_proteins':self.common_proteins, 'cell':cell}
											common_proteins = self.common_proteins
											try:
												exec(condition_code, globals(), loc)
											except KeyError:
												pass
											try:
												condition = loc['condition']
											except KeyError:
												pass

							else:			
								pass
							
							if common == True:

								if full_protein_seq in self.common_proteins:
									if self.common_proteins[full_protein_seq] < rate:
										self.common_proteins[full_protein_seq] = rate
								else:
									self.common_proteins[full_protein_seq] = rate
					#---------------------------------------------------------------------------------------------------------------------
					

















					#start expressing new genes 
					#---------------------------------------------------------------------------------------------------------------------
					#silence transcription
					#print('do silence transcription')
					for sile in self.silence:
						del_activation_list = []
						for activ in self.transcription:
							if (sile - self.silence_strength) < activ and (sile + self.silence_strength) > activ:
								new_rate = self.transcription[activ] - self.silence[sile]
								if new_rate > 0:
									self.transcription[activ] = new_rate
								else:
									del_activation_list.append(activ)
						for del_dic_el in del_activation_list:
							del self.transcription[del_dic_el]



					#transctibe here
					#print('do transcription')
					for number, rate in self.transcription.items():
						#lookin for start codon in the next 128 nucleotides
						start_index = -1
						if number < len(cell[0]):
							start_index = cell[0][number:number+self.activation_strength].find(TATA_box)
						#if start codon wasnt found
						if start_index==-1:	
							pass
						#if start codon was found
						else:
							term = cell[0][number+start_index+4:].find(termination_site)
							if term == -1:
								pass
							#if temination site was found
							else:
								#cutting dna pice accorind to start and termination sites
								pre_mRNA = cell[0][number+start_index+4:number+start_index+4+term]
								#looking for cut sites in mrna
								cut_numbers = [m.start() for m in re.finditer('(?={})'.format(protein_cut_codon), pre_mRNA)]
								#if not cuts was found
								if 'a' in pre_mRNA or 'g' in pre_mRNA or 'c' in pre_mRNA or 't' in pre_mRNA:
									continue
								if cut_numbers == []: 
									mRNA = pre_mRNA
									if mRNA in self.splicing:
										self.splicing[mRNA] = self.splicing[mRNA]+rate 
									else:
										self.splicing[mRNA] = rate
								#if cut site was found in mRNA
								else:
									mRNSs = []
									old_cut = -3
									cut_numbers.append(len(pre_mRNA))
									#cutting mRNA into different pieces
									for cut in cut_numbers:
										mRNSs.append(pre_mRNA[old_cut+3:cut])
										old_cut = cut
									#adding each cut mrna imnto slicing list
									for mRNA_temp in mRNSs:
										if len(mRNA_temp) > 2:
											if mRNA_temp in self.splicing:
												self.splicing[mRNA_temp] = self.splicing[mRNA_temp]+rate 
											else:
												self.splicing[mRNA_temp] = rate

					#---------------------------------------------------------------------------------------------------------------------




















					#splicing and translation
					#print('do splicing and translation')
					#---------------------------------------------------------------------------------------------------------------------
					for pre_mRNA, rate in self.splicing.items():
						#searching for start codon in mRNA

						start_reading = pre_mRNA.find(start_codon)

						#if nothign was found, pass
						if start_reading == -1:
							pass
						#if start codon exist:
						elif len(pre_mRNA[start_reading:]) < 4:
							pass
						else:

							#cut mRNA after start codon
							mRNA = pre_mRNA[start_reading+3:]
				
							start_splice_list = [m.start() for m in re.finditer('(?={})'.format(exon_start_site), mRNA[:-1])]
							
							#if not cut splice:
							if start_splice_list == []:
								if mRNA in cell[1]:
									cell[1][mRNA] = cell[1][mRNA]+rate
								else:
									cell[1][mRNA] = rate
							#if cut splice exist
							else:
								
								#------------------
								end_splice_list = [m.start()+start_splice_list[0] for m in re.finditer('(?={})'.format(exon_end_site), mRNA[start_splice_list[0]:])]
				
								if end_splice_list != []:

									splice_start_last = 0
									start_splice_x = []
									end_splice_x = []
									check_exist = False


									for spice_start_site in start_splice_list:
										if spice_start_site < splice_start_last:
											pass
										else:
											for index_loop, splice_site_end in enumerate(end_splice_list):
												if spice_start_site < splice_site_end and spice_start_site > splice_start_last:
													splice_start_last = splice_site_end
													if spice_start_site+2 < splice_site_end:
														start_splice_x.append(spice_start_site)
														end_splice_x.append(splice_site_end)
														end_splice_list = end_splice_list[index_loop+1:]
														check_exist = True
														break
			
										if check_exist == False:
											break
								else:
									end_splice_x = []
									end_splice_x.append(len(mRNA[start_splice_list[0]:]))
									start_splice_x = []
									start_splice_x.append(start_splice_list[0])
								
								#------------------
								

								#check if splcie site does not cut important part of proteins
								if start_splice_x != []:
									if start_splice_x[0] > 5:
										mature_mRNA = mRNA[:6]
										
										start_intron = 6
										next_start = start_splice_x[1:]
										next_start.append(0)
										start_check = True
										intron_past = ''
										inron_next = ''
										
										for splice_start, splice_end, nxt_st in zip(start_splice_x, end_splice_x, next_start):
											
											if start_intron == 6:
												if splice_start > 6:
													intron_past = mRNA[start_intron:splice_start]
											else:
												intron_past = mRNA[start_intron+2:splice_start]
											
											if nxt_st == 0:
												if splice_end+2 < len(mRNA):
													inron_next = mRNA[splice_end+2:]
												else:
													inron_next = ''
											else:
												intron_next = mRNA[splice_end+2:nxt_st]
											
											if splice_end-splice_start+2 > 2:

												#splice_regul = [{'ABC':regul_type}][{'CBA':regul_type}]
												#regul type - ls (left silence), rs(right silence),b (bowth silence), ib(inluce fulll intron),
												#il(include left part of intron), ir (incldue right part of intron), se(silence exon)
												#ns(new splcie site), cl (new protein type left) cr(new protein type right)
												# l-ls, r-rs,b-b, f-ib, p-il, n-ir, s-se, g-ns, r-cr, l-cl
												# intron_only: l,r,b,f,p,n,
												# exon_only: s, r, l

												# c
												
												pre_exon = mRNA[splice_start+2:splice_end]
												
												
												exon_silence_check = False
												
												for seq, tmp in self.splice_regul[0].items():
													reg_type, right_work = tmp
												
													if seq in intron_past:
														if reg_type == 'r':
															exon_silence_check = True
														elif reg_type == 'b':

															exon_silence_check = True
														elif reg_type == 'f':
															if right_work == None or right_work == True:
																pre_exon = intron_past+pre_exon
															else:
																pass
														elif reg_type == 'p':
															if right_work == None or right_work == True:
																tmp_indx = intron_past.find(seq)
																left_part_intr = intron_past[:tmp_indx]
																pre_exon = left_part_intr + pre_exon
															else:
																pass
														elif reg_type == 'n':
															if right_work == None or right_work == True:
																tmp_indx = intron_past.find(seq)
																right_part_intr = intron_past[tmp_indx+len(seq):]
																pre_exon = right_part_intr + pre_exon
															else:
																pass
													
													if seq in intron_next:
														if reg_type == 'l':
															exon_silence_check = True
														elif reg_type == 'b':
															exon_silence_check = True
														elif reg_type == 'f':
															if right_work == None or right_work == False:
																pre_exon = pre_exon+intron_next
															else:
																pass
														elif reg_type == 'p':
															if right_work == None or right_work == False:
																tmp_indx = intron_next.find(seq)
																left_part_intr = intron_next[:tmp_indx]
																pre_exon = pre_exon + left_part_intr
															else:
																pass
														elif reg_type == 'n':
															if right_work == None or right_work == False:
																tmp_indx = intron_next.find(seq)
																
																right_part_intr = intron_next[tmp_indx+len(seq):]
																pre_exon = pre_exon + right_part_intr
															else:
																pass

												for seq, tmp in self.splice_regul[1].items():
													reg_type, right_work = tmp
													if seq in pre_exon:
														if reg_type == 's':
															exon_silence_check = True
														elif reg_type == 'a':
															tmp_indx = pre_exon.find(seq)
															if tmp_indx > 5:
																exon_silence_check = True
																left_part_exon = pre_exon[tmp_indx-6:tmp_indx]
																mature_mRNA = left_part_exon + mature_mRNA[6:]
														elif reg_type == 'z':
															tmp_indx = pre_exon.find(seq)
															if len(pre_exon) - tmp_indx - len(seq) > 5:
																exon_silence_check = True
																right_part_exon = pre_exon[tmp_indx+len(seq):tmp_indx+len(seq)+6]
																mature_mRNA = right_part_exon + mature_mRNA[6:]


												
												if exon_silence_check == False:
													mature_mRNA+='GC'
													mature_mRNA+=pre_exon
													mature_mRNA+='GC'

											start_intron = splice_end
											
										



										if mature_mRNA in cell[1]:
											cell[1][mature_mRNA] = cell[1][mature_mRNA]+rate
										else:
											cell[1][mature_mRNA] = rate


										

					#print('do shuffle_var')
					if shuffle_var == True:
						protein_list_tmp=list(cell[1].items())
						random.shuffle(protein_list_tmp)
						cell[1] = dict(protein_list_tmp)

					#print('do transposone')
					if self.transposone != []:

						start_site_tmp, end_site_tmp, paste_site, full_protein_seq = self.transposone
						trans_elem = cell[0][start_site_tmp:end_site_tmp]

						
						cell[0] = cell[0][:start_site_tmp] + cell[0][end_site_tmp:]
						if end_site_tmp < paste_site:
							cell[0] = cell[0][:paste_site-len(trans_elem)] + trans_elem +  cell[0][paste_site-len(trans_elem):]
						else:
							cell[0] = cell[0][:paste_site] + trans_elem + cell[0][paste_site:] 
						
						try:
							del cell[1][full_protein_seq]
						except KeyError:#unknoe happanes here!!!!!!!!!!, hae no idea why Keyerror happens
							pass

				


					
					#---------------------------------------------------------------------------------------------------------------------

					if apoptosis == True:
						apoptosis_list.append(cell)

				
					
				for del_cell in apoptosis_list:
					self.nn.remove(del_cell)


				shuffle_var = False
				proteins_full = []
				delete_list = []
				#print('prepare common proteins')
				for protein, rate in self.common_proteins.items():
					if rate > concentration_maximum:
						rate = concentration_maximum
						self.common_proteins[protein] = rate


					if rate < 1:
						delete_list.append(protein)
					elif protein == '':
						delete_list.append(protein)


					#findng AND codons in protein(one protein can do multipel functions. If tis is divided into two fucntional units,
					#AND codon is used) 
					prot_ands_first_test = [m.start() for m in re.finditer('(?={})'.format(protein_and_codon), protein)]
					prot_ands_second_test = []
					#if no AND codon found
					if prot_ands_first_test == []:
						proteins_full.append((protein, rate, protein))
					else:
						for and_codon in prot_ands_first_test:
							#each fucntion protein part is coded with 6 letters, so if after AND codon,
							#there is no 6 letter - there will be no function
							if len(protein[and_codon:]) > 9:
								if exon_cut not in protein[and_codon:and_codon+10]:
									prot_ands_second_test.append(and_codon)
						#adding each functional unit of single protein into list of proteins_full, which
						#contains each functional unit of single protein 
						start_proteins_active_center = -4
						for protein_and in prot_ands_second_test:
							proteins_full.append((protein[start_proteins_active_center+4:protein_and], rate, protein))
							start_proteins_active_center = protein_and
						

						proteins_full.append((protein[start_proteins_active_center+4:], rate, protein))


				for del_prot in delete_list:
					del self.common_proteins[del_prot]
				#for each functional unit of a proteinm isolae exons and type of protein unit
				#print('prepare common proteins 2')
				for protein, rate, full_protein_seq in proteins_full:
					protein_termination_codon = False
					#6 first latter - type of protein
					protein_type = protein[:6]
					protein_exons = protein[6:]
					#finding all exons in a protein
					exons_index = [m.start() for m in re.finditer('(?={})'.format(exon_cut), protein_exons)]
					exons = []
					exon_start = 0
					#make shure there are exons in protein
					if exons_index != []:
						#it is easier to have even numebr of exons cuts, so this pece if code deals with off umebr of exons cuts
						if len(exons_index) == 1:
							#if there is only one exon cuts and,after the exon cut there is some letter, this pice of code add
							#another exon cut sequence in the end, so that there be even number
							if len(protein_exons) > exons_index[0]:
								exons_index.append(len(protein_exons))

							#however, if the exon cut sequence is in end, make sure just delete it
							else:
								exons = []
						#adding exon cut in the end, if exosn cuts is uneven, andthere is meaningul sequence after lust exon cut
						if len(exons_index) % 2 != 0:
							if len(protein_exons) > exons_index[-1]:
								exons_index.append(len(protein_exons))
							else:
								exons = []
						
						#after applying the rules to exon cuts, now actual exon cuts are made
						exon_start = exons_index[0]
						for exon_ind in exons_index[1:]:
							exons.append(protein_exons[exon_start+2:exon_ind])
							exon_start = exon_ind

					else:
						working_exons = []
					#as information is in codonds, if there is no 3 letter in exons, remooves it from exons list
					for i in exons:
						working_exons = [codon_reader(exon) for exon in exons if len(exon) > 5]
					
					for ex_ind, ex_tmp in enumerate(working_exons):
						if stop_codon in ex_tmp:
							protein_termination_codon = True
							termination_ex_ind = ex_ind+1
							termination_cod_ind = ex_tmp.index(stop_codon)
							
					if protein_termination_codon == True:
						working_exons = working_exons[:termination_ex_ind]
						working_exons[-1] = working_exons[-1][:termination_cod_ind]
						if len(working_exons[-1]) < 2:
							working_exons = working_exons[:-1]
				
					







					cell = ['', {'':1} , '', [], '','', '']
					



					#Common protein shaperon add
					#--------------------------------

					if protein_type in protein_types[2] or protein_type in protein_types[22]:
						#print('Common protein shaperon add')
						exon_type_1, exon_type_2, exon_type_3, exon_type_4, exon_type_5, exon_none = exon_types(5)
						protein_search=[]
						seq_to_add = ''
						cell_pass=False
						cell_throuhgt_list = []
						cell_throuhgt = ''
						condition=True
						#add condition - left:1, right:2, center:3, delete:4
						add_condition = 1
						if working_exons != []:
							for exon in working_exons:
								exon_codons = exon
								if exon_codons[0] in exon_type_1:

									seq_find = codons_to_nucl(exon_codons[1:])
									found_prots_tmp = []
									#print('self.common_proteins before:',self.common_proteins)
									for prot, prot_rate in self.common_proteins.items():
										if seq_find in prot:
											if prot not in found_prots_tmp:
												protein_search.append((prot, prot_rate, seq_find))
												found_prots_tmp.append(prot)
									found_prots_tmp.clear()
								
								elif exon_codons[0] in exon_type_2:
									seq_add = codons_to_nucl(exon_codons[1:])
									seq_to_add+=seq_add

								elif exon_codons[0] in exon_type_3:
									ex, ex2, ex3, ex_none = exon_types(3)

									if len(exon_codons) > 2:
										if exon_codons[1] in ex2:
												cell_pass = True
												cell_throuhgt_list.append( codons_to_nucl(exon_codons[2:]) )

										if exon_codons[1] in ex3:
												cell_pass = True
												cell_throuhgt += codons_to_nucl(exon_codons[2:])
									
									
									
								elif exon_codons[0] in exon_type_4:

									condition_code = boolian_logic(cell, self.common_proteins, exon_codons[1:])

									if condition_code != None:
										condition = False
										condition_code = condition_code.replace('self.common_proteins','common_proteins')
										loc = {'common_proteins':self.common_proteins, 'cell':cell}
										common_proteins = self.common_proteins
										try:
											exec(condition_code, globals(), loc)
										except KeyError:
											pass
										try:
											condition = loc['condition']
										except KeyError:
											pass
								elif exon_codons[0] in exon_type_5:
				
									if exon_codons[1] in ('AAA', 'AAC', 'AAG', 'AAT', 'ACA', 'ACC', 'ACG', 'ACT', 'AGA', 'AGC', 'AGG', 'AGT', 'ATA', 'ATC', 'ATG','ATT'):
										add_condition = 1
									if exon_codons[1] in ('GAA', 'GAC', 'GAG', 'GAT', 'GCA', 'GCC', 'GCG', 'GCT', 'GGA', 'GGC', 'GGG', 'GGT', 'GTA', 'GTC', 'GTG', 'GTT'):
										add_condition = 2
									if exon_codons[1] in ('TAA', 'TAC', 'TAG', 'TAT', 'TCA', 'TCC', 'TCG', 'TCT', 'TGA', 'TGC', 'TGG', 'TGT', 'TTA', 'TTC', 'TTG', 'TTT'):
										add_condition = 3
									if exon_codons[1] in ('CAA', 'CAC', 'CAG', 'CAT', 'CCA', 'CCC', 'CCG', 'CCT', 'CGA', 'CGC', 'CGG', 'CGT', 'CTA', 'CTC', 'CTG', 'CTT'):
										add_condition = 4
						
						else:			
							pass


						if cell_pass == True:

							if cell_throuhgt != '':
								cell_throuhgt_list.append(cell_throuhgt)
							if cell_throuhgt_list != []:
								for cell_n in self.nn:
									for find_seq in cell_throuhgt_list:
										
										if find_seq in cell_n[2]:

											if full_protein_seq in cell_n[1]:
												if cell_n[1][full_protein_seq] < rate:
												
													cell_n[1][full_protein_seq] = rate
													

											else:
												cell_n[1][full_protein_seq] = rate
											

						if condition == True:
							for prot, prot_rate, seq_find in protein_search:

								if seq_to_add not in prot:
									
									
									if add_condition == 1:
										new_value = prot+seq_to_add
									elif add_condition == 2:
										new_value = seq_to_add+prot
									elif add_condition == 3:
										index_tmp = prot.find(seq_find)
										new_value = prot[:index_tmp] + seq_to_add + prot[index_tmp:]

									elif add_condition == 4:
										index_tmp = prot.find(seq_find)
										new_value = prot[:index_tmp] + seq_to_add +  prot[index_tmp+len(seq_find):]																		
									
									try:
							
										if prot_rate < rate:
											self.common_proteins[new_value] =self.common_proteins[new_value]+prot_rate
											del self.common_proteins[prot]
										else:
											self.common_proteins[new_value] = self.common_proteins[new_value] + rate
											self.common_proteins[prot] = prot_rate - rate

									except KeyError:
										try:
											#print('prot:',prot)
											#print('self.common_proteins after:',self.common_proteins)
											if prot_rate < rate:
												self.common_proteins[new_value] = self.common_proteins.pop(prot)
											else:
												self.common_proteins[new_value] = rate
												self.common_proteins[prot] = prot_rate - rate
										except KeyError:
											pass
							

								
					#--------------------------------


					#Common protein shaperon remoove
					#--------------------------------
					elif protein_type in protein_types[3] or protein_type in protein_types[23]:
						#print('Common protein shaperon remoove')
						exon_type_1, exon_type_2, exon_type_3, exon_type_4, exon_type_5, exon_type_6, exon_none = exon_types(6)
						protein_search=[]
						seq_to_add = ''
						common=False
						condition=True
						splice_range = 16
						#add condition - left:1, right:2, center:3, delete:4
						remoove_condition = True
						cell_pass=False
						cell_throuhgt_list = []
						cell_throuhgt = ''
					
						if working_exons != []:
							for exon in working_exons:
								exon_codons = exon
								if exon_codons[0] in exon_type_1:

									seq_find = codons_to_nucl(exon_codons[1:])
								
									for prot, prot_rate in self.common_proteins.items():
										if seq_find in prot:
											protein_search.append((prot, prot_rate, seq_find))
								
								elif exon_codons[0] in exon_type_2:
									seq_add = codons_to_nucl(exon_codons[1:])
									seq_to_add+=seq_add

								elif exon_codons[0] in exon_type_3:
									ex, ex2, ex3, ex_none = exon_types(3)

									if len(exon_codons) > 2:
										if exon_codons[1] in ex2:
												cell_pass = True
												cell_throuhgt_list.append( codons_to_nucl(exon_codons[2:]) )

										if exon_codons[1] in ex3:
												cell_pass = True
												cell_throuhgt += codons_to_nucl(exon_codons[2:])
									
								elif exon_codons[0] in exon_type_4:

									condition_code = boolian_logic(cell, self.common_proteins, exon_codons[1:])

									if condition_code != None:
										condition = False
										condition_code = condition_code.replace('self.common_proteins','common_proteins')
										loc = {'common_proteins':self.common_proteins, 'cell':cell}
										common_proteins = self.common_proteins
										try:
											exec(condition_code, globals(), loc)
										except KeyError:
											pass
										try:
											condition = loc['condition']
										except KeyError:
											pass
								elif exon_codons[0] in exon_type_5:
									remoove_condition = False
									
								
								elif exon_codons[0] in exon_type_6:
									if len(exon_codons) > 2:
										splice_range = codons_to_number(exon_codons[-2:])
									else:
										splice_range = codons_to_number([exon_codons[1]])
						else:		
							pass

						if cell_pass == True:

							if cell_throuhgt != '':
								cell_throuhgt_list.append(cell_throuhgt)
							if cell_throuhgt_list != []:
								for cell_n in self.nn:
									for find_seq in cell_throuhgt_list:
										
										if find_seq in cell_n[2]:

											if full_protein_seq in cell_n[1]:
												if cell_n[1][full_protein_seq] < rate:
												
													cell_n[1][full_protein_seq] = rate
													

											else:
												cell_n[1][full_protein_seq] = rate


						if condition == True:
							if seq_to_add == '':
								for prot, prot_rate, seq_find in protein_search:
									if prot not in self.common_proteins:
										pass
									else:

										if remoove_condition == True:
											index_tmp = prot.find(seq_find)
											range_right = min(len(prot[index_tmp+len(seq_find):]), splice_range)
											new_value = prot[:index_tmp] + prot[index_tmp+len(seq_find)+range_right:]

										else:
											
											index_tmp = prot.find(seq_find)

											range_left = min(len(prot[:index_tmp]), splice_range)

											new_value = prot[:index_tmp-range_left]+prot[index_tmp+len(seq_find):]


										try:
											if prot_rate <= rate:

												self.common_proteins[new_value] = self.common_proteins[new_value]+self.common_proteins[prot]

										
												del self.common_proteins[prot]



											else:
												
												self.common_proteins[new_value] = self.common_proteins[new_value] + rate
												self.common_proteins[prot] = prot_rate - rate
												

										except KeyError:
											try:
												if prot_rate <= rate:
													
													self.common_proteins[new_value] = self.common_proteins.pop(prot)
													
												else:
													
													self.common_proteins[new_value] = rate
													self.common_proteins[prot] = prot_rate - rate
											except KeyError:
												pass
												
								
							else:
								for prot, prot_rate, seq_find in protein_search:
									if prot not in self.common_proteins:
										pass
									else:
										if seq_to_add in prot:
											seq_find = seq_to_add
											if remoove_condition == True:
												index_tmp = prot.find(seq_find)
												range_right = min(len(prot[index_tmp+len(seq_find):]), splice_range)
												new_value = prot[:index_tmp] + prot[index_tmp+len(seq_find)+range_right:]

											else:
												index_tmp = prot.find(seq_find)
												range_left = min(len(prot[:index_tmp]), splice_range)
												
												new_value = prot[:index_tmp-range_left]+prot[index_tmp+len(seq_find):]

															
											
											try:
												if prot_rate <= rate:
													self.common_proteins[new_value] = self.common_proteins[new_value]+self.common_proteins[prot]
													del self.common_proteins[prot]
												else:
													self.common_proteins[new_value] = self.common_proteins[new_value] + rate
													self.common_proteins[prot] = prot_rate - rate

											except KeyError:
												if prot_rate <= rate:
													self.common_proteins[new_value] = self.common_proteins.pop(prot)
												else:
													self.common_proteins[new_value] = rate
													self.common_proteins[prot] = prot_rate - rate
					
						
							
					#--------------------------------




					#Common protein deletion
					#--------------------------------

					elif protein_type in protein_types[9] or protein_type in protein_types[23]:
						#print('Common protein deletion')
						exon_type_1, exon_type_2, exon_type_3, exon_type_4, exon_type_5, exon_none = exon_types(5)
						common=False
						condition=True
						self_rate_down = 1
						self_rate_up = 5000
						search_site = ''
						#add condition - left:1, right:2, center:3, delete:4
						cell_pass=False
						cell_throuhgt_list = []
						cell_throuhgt = ''
						if working_exons != []:
							for exon in working_exons:

								exon_codons = exon

								if exon_codons[0] in exon_type_1:
									search_site+=codons_to_nucl(exon_codons[1:])
								
								if exon_codons[0] in exon_type_2:
									self_rate_down = codons_to_number(exon_codons[1:])
								
								elif exon_codons[0] in exon_type_3:
									self_rate_up = codons_to_number(exon_codons[1:])

								elif exon_codons[0] in exon_type_4:
									ex, ex2, ex3, ex_none = exon_types(3)

									if len(exon_codons) > 2:
										if exon_codons[1] in ex2:
												cell_pass = True
												cell_throuhgt_list.append( codons_to_nucl(exon_codons[2:]) )

										if exon_codons[1] in ex3:
												cell_pass = True
												cell_throuhgt += codons_to_nucl(exon_codons[2:])
									
								elif exon_codons[0] in exon_type_5:

									condition_code = boolian_logic(cell, self.common_proteins, exon_codons[1:])

									if condition_code != None:
										condition = False
										condition_code = condition_code.replace('self.common_proteins','common_proteins')
										loc = {'common_proteins':self.common_proteins, 'cell':cell}
										common_proteins = self.common_proteins
										try:
											exec(condition_code, globals(), loc)
										except KeyError:
											pass
										try:
											condition = loc['condition']
										except KeyError:
											pass
						else:			
							pass
						

						if cell_pass == True:

							if cell_throuhgt != '':
								cell_throuhgt_list.append(cell_throuhgt)
							if cell_throuhgt_list != []:
								for cell_n in self.nn:
									for find_seq in cell_throuhgt_list:
										
										if find_seq in cell_n[2]:

											if full_protein_seq in cell_n[1]:
												if cell_n[1][full_protein_seq] < rate:
												
													cell_n[1][full_protein_seq] = rate
													

											else:
												cell_n[1][full_protein_seq] = rate

						if condition == True:
					
							if search_site != '':
							
								if self_rate_down <= rate:
									
									if self_rate_up >= rate:

										proteins_list = []
										for prot in self.common_proteins:
											if search_site in prot:
												proteins_list.append([prot, self.common_proteins[prot]])
										for prot, prot_rate in proteins_list:

											if rate >= prot_rate:
												del self.common_proteins[prot]
											else:
												self.common_proteins[prot] = self.common_proteins[prot]-rate
					

					#--------------------------------







					#common secrete
					#--------------------------------
					elif protein_type in protein_types[13] or protein_type in protein_types[24]:
						#print('common secrete')
						exon_type_1, exon_type_2, exon_type_3, exon_type_4, exon_type_5, exon_type_6, exon_type_7, exon_none = exon_types(7)
						common=False
						condition=True
						self_rate_down = 1
						self_rate_up = 5000
						produce = []
						super_prot = ''
						produce_rate = 1
						cell_pass=False
						cell_throuhgt_list = []
						cell_throuhgt = ''
						
						
						if working_exons != []:
							for exon in working_exons:

								exon_codons = exon

								if exon_codons[0] in exon_type_1:
									self_rate_down = codons_to_number(exon_codons[1:])
								
								elif exon_codons[0] in exon_type_2:
									self_rate_up = codons_to_number(exon_codons[1:])

								elif exon_codons[0] in exon_type_3:
									ex, ex2, ex3, ex_none = exon_types(3)

									if len(exon_codons) > 2:
										if exon_codons[1] in ex2:
												cell_pass = True
												cell_throuhgt_list.append( codons_to_nucl(exon_codons[2:]) )

										if exon_codons[1] in ex3:
												cell_pass = True
												cell_throuhgt += codons_to_nucl(exon_codons[2:])
									
								elif exon_codons[0] in exon_type_4:

									condition_code = boolian_logic(cell, self.common_proteins, exon_codons[1:])

									if condition_code != None:
										condition = False
										condition_code = condition_code.replace('self.common_proteins','common_proteins')
										loc = {'common_proteins':self.common_proteins, 'cell':cell}
										common_proteins = self.common_proteins
										try:
											exec(condition_code, globals(), loc)
										except KeyError:
											pass
										try:
											condition = loc['condition']
										except KeyError:
											pass
								elif exon_codons[0] in exon_type_5:
									produce.append(codons_to_nucl(exon_codons[1:]))
								elif exon_codons[0] in exon_type_6:
									super_prot += codons_to_nucl(exon_codons[1:])
								elif exon_codons[0] in exon_type_7:
									produce_rate = codons_to_number(exon_codons[1:])
									
						else:			
							pass
						

						if cell_pass == True:

							if cell_throuhgt != '':
								cell_throuhgt_list.append(cell_throuhgt)
							if cell_throuhgt_list != []:
								for cell_n in self.nn:
									for find_seq in cell_throuhgt_list:
										
										if find_seq in cell_n[2]:

											if full_protein_seq in cell_n[1]:
												if cell_n[1][full_protein_seq] < rate:
												
													cell_n[1][full_protein_seq] = rate
													

											else:
												cell_n[1][full_protein_seq] = rate

						if condition == True:
							if self_rate_down <= rate:
							
								if self_rate_up >= rate:
							
									if super_prot != '':
										if super_prot in self.common_proteins:
											self.common_proteins[super_prot] = self.common_proteins[super_prot] + (rate * produce_rate)
										else:
											self.common_proteins[super_prot] = rate * produce_rate
									
									for sec_prod in produce:
										if sec_prod in self.common_proteins:
											self.common_proteins[sec_prod] = self.common_proteins[sec_prod] + (rate * produce_rate)
										else:
											self.common_proteins[sec_prod] = rate * produce_rate

					#--------------------------------





					#Common signaling
					#--------------------------------
					elif protein_type in protein_types[15] or protein_type in protein_types[25]:
						#print('Common signaling')
						exon_type_1, exon_type_2, exon_type_3, exon_type_4, exon_type_5, exon_none = exon_types(5)
						common=False
						condition=True
						self_rate_down = 1
						self_rate_up = 5000
						random_number = None
						produce_rate = 1
						cell_pass=False
						cell_throuhgt_list = []
						cell_throuhgt = ''
						
						if working_exons != []:
							for exon in working_exons:

								exon_codons = exon

								if exon_codons[0] in exon_type_1:
									self_rate_down = codons_to_number(exon_codons[1:])
								
								elif exon_codons[0] in exon_type_2:
									self_rate_up = codons_to_number(exon_codons[1:])

								elif exon_codons[0] in exon_type_3:
									ex, ex2, ex3, ex_none = exon_types(3)

									if len(exon_codons) > 2:
										if exon_codons[1] in ex2:
												cell_pass = True
												cell_throuhgt_list.append( codons_to_nucl(exon_codons[2:]) )

										if exon_codons[1] in ex3:
												cell_pass = True
												cell_throuhgt += codons_to_nucl(exon_codons[2:])
									
								elif exon_codons[0] in exon_type_4:

									condition_code = boolian_logic(cell, self.common_proteins, exon_codons[1:])

									if condition_code != None:
										condition = False
										condition_code = condition_code.replace('self.common_proteins','common_proteins')
										loc = {'common_proteins':self.common_proteins, 'cell':cell}
										common_proteins = self.common_proteins
										try:
											exec(condition_code, globals(), loc)
										except KeyError:
											pass
										try:
											condition = loc['condition']
										except KeyError:
											pass
								elif exon_codons[0] in exon_type_5:
									random_number = randint(1, codons_to_number(exon_type_5))

						else:			
							pass
						
						if cell_pass == True:

							if cell_throuhgt != '':
								cell_throuhgt_list.append(cell_throuhgt)
							if cell_throuhgt_list != []:
								for cell_n in self.nn:
									for find_seq in cell_throuhgt_list:
										
										if find_seq in cell_n[2]:

											if full_protein_seq in cell_n[1]:
												if cell_n[1][full_protein_seq] < rate:
												
													cell_n[1][full_protein_seq] = rate
													

											else:
												cell_n[1][full_protein_seq] = rate

						if condition == True:
							if self_rate_down <= rate:
							
								if self_rate_up >= rate:
									if random_number != None:
										self.common_proteins[full_protein_seq] = random_number



					#--------------------------------



					#Common connection growth factor
					#--------------------------------
					elif protein_type in protein_types[18] or protein_type in protein_types[26]:
						#print('Common connection growth factor')
						exon_type_1, exon_type_2, exon_type_3, exon_type_4, exon_type_5, exon_type_6, exon_type_7, exon_none = exon_types(7)
						common=False
						condition=True
						self_rate_down = 1
						self_rate_up = 5000
						find_connection_to = ''
						list_connections = []
						connect_from = []
						cell_pass=False
						cell_throuhgt_list = []
						cell_throuhgt = ''
						
						if working_exons != []:
							for exon in working_exons:

								exon_codons = exon


								if exon_codons[0] in exon_type_1:
									find_connection_to += codons_to_nucl(exon_codons[1:])
									

								elif exon_codons[0] in exon_type_2:
									self_rate_down = codons_to_number(exon_codons[1:])
								
								elif exon_codons[0] in exon_type_3:
									self_rate_up = codons_to_number(exon_codons[1:])

								elif exon_codons[0] in exon_type_4:
									ex, ex2, ex3, ex_none = exon_types(3)

									if len(exon_codons) > 2:
										if exon_codons[1] in ex2:
												cell_pass = True
												cell_throuhgt_list.append( codons_to_nucl(exon_codons[2:]) )

										if exon_codons[1] in ex3:
												cell_pass = True
												cell_throuhgt += codons_to_nucl(exon_codons[2:])
									
								elif exon_codons[0] in exon_type_5:

									condition_code = boolian_logic(cell, self.common_proteins, exon_codons[1:])

									if condition_code != None:
										condition = False
										condition_code = condition_code.replace('self.common_proteins','common_proteins')
										loc = {'common_proteins':self.common_proteins, 'cell':cell}
										common_proteins = self.common_proteins
										try:
											exec(condition_code, globals(), loc)
										except KeyError:
											pass
										try:
											condition = loc['condition']
										except KeyError:
											pass
								if exon_codons[0] in exon_type_6:
									list_connections.append(codons_to_nucl(exon_codons[1:]))
									

								if exon_codons[0] in exon_type_7:
									cell_to_grow = codons_to_nucl(exon_codons[1:])
									for cell_indx, cell_n in enumerate(self.nn):
										if cell_to_grow in cell_n[2]:
											connect_from.append(cell_indx)


									

						else:			
							pass
						
						if cell_pass == True:

							if cell_throuhgt != '':
								cell_throuhgt_list.append(cell_throuhgt)
							if cell_throuhgt_list != []:
								for cell_n in self.nn:
									for find_seq in cell_throuhgt_list:
										
										if find_seq in cell_n[2]:

											if full_protein_seq in cell_n[1]:
												if cell_n[1][full_protein_seq] < rate:
												
													cell_n[1][full_protein_seq] = rate
													

											else:
												cell_n[1][full_protein_seq] = rate

						if condition == True:
							if self_rate_down <= rate:
								if self_rate_up >= rate:
									if connect_from != []:
										if find_connection_to != '':
											for cell_n in self.nn:
												if find_connection_to in cell_n[2]:
													for cell_indx in connect_from:
														if cell_n[2] not in self.nn[cell_indx][3]:
															 self.nn[cell_indx][3].append(cell_n[2])

										if list_connections != []:
											for cell_n in self.nn:
												for find_con_seq in list_connections:
													if find_con_seq in cell_n[2]:
														for cell_indx in connect_from:
															if cell_n[2] not in self.nn[cell_indx][3]:
																 self.nn[cell_indx][3].append(cell_n[2])



					#--------------------------------





					#Stop development
					#--------------------------------
					elif protein_type in protein_types[27]:
						#print('Stop development')
						exon_type_1, exon_type_2, exon_type_3, exon_type_4, exon_type_5, exon_none = exon_types(5)
						common=False
						condition=True
						self_rate_down = 1
						self_rate_up = 5000
						protein_work = False
						cell_pass=False
						cell_throuhgt_list = []
						cell_throuhgt = ''
						
						
						if working_exons != []:
							for exon in working_exons:

								exon_codons = exon

								if exon_codons[0] in exon_type_1:
									self_rate_down = codons_to_number(exon_codons[1:])
								
								elif exon_codons[0] in exon_type_2:
									self_rate_up = codons_to_number(exon_codons[1:])

								elif exon_codons[0] in exon_type_3:
									ex, ex2, ex3, ex_none = exon_types(3)

									if len(exon_codons) > 2:
										if exon_codons[1] in ex2:
												cell_pass = True
												cell_throuhgt_list.append( codons_to_nucl(exon_codons[2:]) )

										if exon_codons[1] in ex3:
												cell_pass = True
												cell_throuhgt += codons_to_nucl(exon_codons[2:])
									
								elif exon_codons[0] in exon_type_4:

									condition_code = boolian_logic(cell, self.common_proteins, exon_codons[1:])

									if condition_code != None:
										condition = False
										condition_code = condition_code.replace('self.common_proteins','common_proteins')
										loc = {'common_proteins':self.common_proteins, 'cell':cell}
										common_proteins = self.common_proteins
										try:
											exec(condition_code, globals(), loc)
										except KeyError:
											pass
										try:
											condition = loc['condition']
										except KeyError:
											pass
								elif exon_codons[0] in exon_type_5:
									protein_work = True

						else:			
							pass
						
						
						if cell_pass == True:

							if cell_throuhgt != '':
								cell_throuhgt_list.append(cell_throuhgt)
							if cell_throuhgt_list != []:
								for cell_n in self.nn:
									for find_seq in cell_throuhgt_list:
										
										if find_seq in cell_n[2]:

											if full_protein_seq in cell_n[1]:
												if cell_n[1][full_protein_seq] < rate:
												
													cell_n[1][full_protein_seq] = rate
													

											else:
												cell_n[1][full_protein_seq] = rate

						if condition == True:
							if self_rate_down <= rate:
							
								if self_rate_up >= rate:
									if protein_work == True:
					
										development = False



					#--------------------------------



					#learning_rate
					elif protein_type in protein_types[28]:
						#print('Learning_rate')
						#print('mutation!')
						exon_type_1, exon_type_2, exon_type_3, exon_type_4, exon_type_5, exon_type_6, exon_none = exon_types(6)
						common=False
						condition=True
						self_rate_down = 1
						self_rate_up = 5000
						protein_work = False
						cell_pass=False
						cell_throuhgt_list = []
						cell_throuhgt = ''
						lr = []
						mut_type = None
						#print('working_exons:',working_exons)
						#interable = iter(exon_types(6))
						#for i in range(6):
							#print(i+1, next(interable))
						if working_exons != []:
							for exon in working_exons:

								exon_codons = exon
								if exon_codons[0] in exon_type_1 and len(exon_codons) > 2:
									if rate != 0:
										lr.append( 1/(codons_to_number(exon_codons[1:])*rate) )

								elif exon_codons[0] in exon_type_2:
									self_rate_down = codons_to_number(exon_codons[1:])
								
								elif exon_codons[0] in exon_type_3:
									self_rate_up = codons_to_number(exon_codons[1:])

								elif exon_codons[0] in exon_type_4:
									common = True
									
								elif exon_codons[0] in exon_type_5:

									condition_code = boolian_logic(cell, self.common_proteins, exon_codons[1:])

									if condition_code != None:
										condition = False
										condition_code = condition_code.replace('self.common_proteins','common_proteins')
										loc = {'common_proteins':self.common_proteins, 'cell':cell}
										common_proteins = self.common_proteins
										try:
											exec(condition_code, globals(), loc)
										except KeyError:
											pass
										try:
											condition = loc['condition']
										except KeyError:
											pass

								elif exon_codons[0] in exon_type_6:
									ex1, ex2, ex_none = exon_types(2)
									if exon_codons[1] in ex1:
										mut_type = True
									elif exon_codons[1] in ex2:
										mut_type = False


						else:			
							pass
						if cell_pass == True:

							if cell_throuhgt != '':
								cell_throuhgt_list.append(cell_throuhgt)
							if cell_throuhgt_list != []:
								for cell_n in self.nn:
									for find_seq in cell_throuhgt_list:
										
										if find_seq in cell_n[2]:

											if full_protein_seq in cell_n[1]:
												if cell_n[1][full_protein_seq] < rate:
												
													cell_n[1][full_protein_seq] = rate
													

											else:
												cell_n[1][full_protein_seq] = rate
						if condition == True:
							if self_rate_down <= rate:
								if self_rate_up >= rate:
									if lr != []:
										self.learning_rate += lr
						'''
						#Mutation rate
						#--------------------------------
						elif protein_type in protein_types[28]:
							print('Mutation rate')
							#print('mutation!')
							exon_type_1, exon_type_2, exon_type_3, exon_type_4, exon_type_5, exon_type_6, exon_none = exon_types(6)
							common=False
							condition=True
							self_rate_down = 1
							self_rate_up = 5000
							protein_work = False
							cell_pass=False
							cell_throuhgt_list = []
							cell_throuhgt = ''
							mutation_rate = [None, None, None, None, None, None, None]
							mut_type = None
							#print('working_exons:',working_exons)
							#interable = iter(exon_types(6))
							#for i in range(6):
								#print(i+1, next(interable))
							if working_exons != []:
								for exon in working_exons:

									exon_codons = exon
									if exon_codons[0] in exon_type_1 and len(exon_codons) > 2:

										ex1, ex2, ex3, ex4, ex5, ex6, ex7, ex_none = exon_types(7)
								
										if exon_codons[1] in ex1:
											mutation_rate[0] = codons_to_number(exon_codons[2:])
										if exon_codons[1] in ex2:
											mutation_rate[1] = codons_to_number(exon_codons[2:])
										if exon_codons[1] in ex3:
											mutation_rate[2] = codons_to_number(exon_codons[2:])
										if exon_codons[1] in ex4:
											mutation_rate[3] = codons_to_number(exon_codons[2:])
										if exon_codons[1] in ex5:
											mutation_rate[4] = codons_to_number(exon_codons[2:])
										if exon_codons[1] in ex6:
											mutation_rate[5] = codons_to_number(exon_codons[2:])
										if exon_codons[1] in ex7:
											mutation_rate[6] = codons_to_number(exon_codons[2:])

									elif exon_codons[0] in exon_type_2:
										self_rate_down = codons_to_number(exon_codons[1:])
									
									elif exon_codons[0] in exon_type_3:
										self_rate_up = codons_to_number(exon_codons[1:])

									elif exon_codons[0] in exon_type_4:
										common = True
										
									elif exon_codons[0] in exon_type_5:

										condition_code = boolian_logic(cell, self.common_proteins, exon_codons[1:])

										if condition_code != None:
											condition = False
											condition_code = condition_code.replace('self.common_proteins','common_proteins')
											loc = {'common_proteins':self.common_proteins, 'cell':cell}
											common_proteins = self.common_proteins
											try:
												exec(condition_code, globals(), loc)
											except KeyError:
												pass
											try:
												condition = loc['condition']
											except KeyError:
												pass

									elif exon_codons[0] in exon_type_6:
										ex1, ex2, ex_none = exon_types(2)
										if exon_codons[1] in ex1:
											mut_type = True
										elif exon_codons[1] in ex2:
											mut_type = False


							else:			
								pass
							if cell_pass == True:

								if cell_throuhgt != '':
									cell_throuhgt_list.append(cell_throuhgt)
								if cell_throuhgt_list != []:
									for cell_n in self.nn:
										for find_seq in cell_throuhgt_list:
											
											if find_seq in cell_n[2]:

												if full_protein_seq in cell_n[1]:
													if cell_n[1][full_protein_seq] < rate:
													
														cell_n[1][full_protein_seq] = rate
														

												else:
													cell_n[1][full_protein_seq] = rate
							if condition == True:
								if self_rate_down <= rate:
									if self_rate_up >= rate:
										if mutation_rate != [None, None, None, None, None, None, None]:
											for mut_indx, mut_r in enumerate(mutation_rate):
												if mut_r != None:
													if mut_type == None:
														self.mutation_rate[mut_indx] = int(mut_r/rate)
													elif mut_type == True:
														if self.mutation_rate[mut_indx] == None:
															self.mutation_rate[mut_indx] = int(mut_r/rate)
														elif type(self.mutation_rate[mut_indx]) == int:
															self.mutation_rate[mut_indx] = self.mutation_rate[mut_indx] + mut_r * rate
													elif mut_type == False:
														if self.mutation_rate[mut_indx] == None:
															self.mutation_rate[mut_indx] = int(mut_r/rate)
														elif type(self.mutation_rate[mut_indx]) == int:
															add_i = self.mutation_rate[mut_indx] - mut_r * rate
															if add_i > 1:
																self.mutation_rate[mut_indx] = add_i
													print(self.mutation_rate)

						




						'''

					#all other proteins
					else:
						#print('all other common proteins')
						exon_type_1, exon_type_2, exon_type_3, exon_type_4, exon_type_5, exon_type_6, exon_type_7, exon_none = exon_types(7)
						common=False
						condition=True
						self_rate_down = 1
						self_rate_up = 5000

						connect_from = []
						cell_pass=False
						cell_throuhgt_list = []
						cell_throuhgt = ''
						
						if working_exons != []:
							for exon in working_exons:

								exon_codons = exon

								if exon_codons[0] in exon_type_2:
									self_rate_down = codons_to_number(exon_codons[1:])
								
								elif exon_codons[0] in exon_type_6:
									self_rate_up = codons_to_number(exon_codons[1:])

								elif exon_codons[0] in exon_type_4 or exon_codons[0] in exon_type_3:
									ex, ex2, ex3, ex_none = exon_types(3)

									if len(exon_codons) > 2:
										if exon_codons[1] in ex2:
												cell_pass = True
												cell_throuhgt_list.append( codons_to_nucl(exon_codons[2:]) )

										if exon_codons[1] in ex3:
												cell_pass = True
												cell_throuhgt += codons_to_nucl(exon_codons[2:])
									
								elif exon_codons[0] in exon_type_5:

									condition_code = boolian_logic(cell, self.common_proteins, exon_codons[1:])

									if condition_code != None:
										condition = False
										condition_code = condition_code.replace('self.common_proteins','common_proteins')
										loc = {'common_proteins':self.common_proteins, 'cell':cell}
										common_proteins = self.common_proteins
										try:
											exec(condition_code, globals(), loc)
										except KeyError:
											pass
										try:
											condition = loc['condition']
										except KeyError:
											pass


						else:			
							pass
						
						if cell_pass == True:

							if cell_throuhgt != '':
								cell_throuhgt_list.append(cell_throuhgt)
							if cell_throuhgt_list != []:
								for cell_n in self.nn:
									for find_seq in cell_throuhgt_list:
										
										if find_seq in cell_n[2]:

											if full_protein_seq in cell_n[1]:
												if cell_n[1][full_protein_seq] < rate:
												
													cell_n[1][full_protein_seq] = rate
													
											else:
												cell_n[1][full_protein_seq] = rate
					#--------------------------------
		except RuntimeError:
			pass
		signal.alarm(0)



		#signal.alarm(0)
	#--------------------------------------------------------------------------------------------------------------------------------
	#--------------------------------------------------------------------------------------------------------------------------------
	#--------------------------------------------------------------------------------------------------------------------------------

'''
dna_letters = 'ACTG'
genome_test = ''
for i in range(1000000):
	genome_test +=dna_letters[randint(0,3)]
'AAATCTATCTACTTACCGCGTCGTCTG ACG AAAAAA TCGCTCGCTG AT GTTTGATAG CG CG CG T AT AGGTAGTGTA CG AT'
#splicing = {genome_test .replace(' ',''):10}


neural_network = NN(genome_test)
proteins = []
for i in range(1000):
	prot = ''
	for i in range(500):
		prot += dna_letters[randint(0,3)]
	proteins.append(prot)
for prot in proteins:
	neural_network.nn[0][1][prot] = 99
'''




test_protein = 'AAAATT CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCTA GC _AAA ACG ACG ACG_ GC GGCTGCTAGCG GC _GAA ATG AT_ GC GGGCTGCTGC !GTAA! AGAAGA GGAGCTAG GC _ACA AGA AGA AGA _ GC GCGACGTGCGA GC _ CAA GGG AATAATAATAAT ACT CAT ACG AGG AAAAAGAAC _ GC GATAGAT CG _ACT CGT CGT CGT CGT _ CG' 
#LOOK FOR GGGG IN DNA
test_protein = test_protein.replace(' ','')
test_protein = test_protein.replace('_','')
test_protein = test_protein.replace('!','')
genome_test = '''GTGGTGTAGTAGT 
CTCTCT GATGATTGCGTAGC TATA GTCAGTCGT ACG | AGAAGA GAGAGA | AAAAA 
GACGTACGTGTACGTAGCAGATTATCTGCGCGCGATGATAGAATCGGCGCGCATGATATCGCGTAGCATCGAGTCGTATCGG ACG TCGTGTCAGCGTATCGAGTCGAGGTCGTAGTCGTATTGCGATTGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGGCTCGCGCGCGCGCGCGCGGCGCGCGCGCCTATGAGTCAGTCA 

GAGAGA ATTAGTGT TATA GTCG ACG | AGAAGA GC TGA AGTTGATCTA GC CG  GTCA AGAAGA GC AAC GTCA AGAAGA GC TCG | AAAAA 
GATAGATAGAGATTGATAGATATAGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG

GAGAGA ATTAGTGT TATA GTCG ACG | AGAAGG GC TGA AGTTG AT AAC ACG ACG ACG GC GGGCG   GTCA AGAAGA GC AAC GTCA AGAAGA GC TCG | AAAAA
GATAGATAGAGATTGATAGATATAGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG

GAGAGA ATTAGTGT TATA ACG | AGAAGC AT AAC CCC GC AT GAA CTGTATCTGTATCTGTATCTGTATCTGTATCTGTATCTGTATCTGTATCTGTATCTGTATCTGTATCTGTATCTGTATCTGTATCTGTATCTGTAT GC TCGAGCTGCGTCTG AT GTC GAC GTG GC| AAAAA
GATAGATAGAGATTGATAGATATAGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG

GAGAGA ATTAGTGT TATA ACG | AGAACC AT AAC CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC GC AT GGG CAT CAT GC AT ATG GGG GGG GC  | AAAAA
GATAGATAGAGATTGATAGATATAGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG

GAGAGA TATA ACG | AGATTT AT GGG GC|  AAAAA

'''.replace(' ','')
genome_test = genome_test.replace('\n','')
genome_test = genome_test.replace('|','')




genome_test = '''\
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG




CTCTCT TATA ACG | GGAGCC        AT GAA TAC TAC TAC  GC          AT GAA TTG ACCCCCCC ACT GAC TGA CTG ACT GGGGGGGA TTG TTG GC         AT GGA TGA TGA TGA GC   CCC CCC CCC   ACTG ACTG ACTG ACTG    GGG GGG GGG    AT GGA TGG TGG TGG GC        AT TAG ACT GC  |  AAAAA

GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG



CTCTCT TATA ACG | CCCCCC     AT ATG AAC ATT AAC ATT GC     AT ATG AAC ATT AAC ATT GC   AT TAT TAA AGG GC       AT TAT CCT AGG GC  |  AAAAA


GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
'''




genome_test = '''

GATAGATAGAGATTGATAGAGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG

GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG

CTCTCT TATA ACG | AGAGGG AT CAA AGT AGT AGT GC AT ACA AGG AGG ACT GC ACA  |  AAAAA

GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG

ATATAT TATA ACG | AGAGGA AT GGGGGGGGGGGGGGGGGGGGGGGG GC AT ACA AGG AGG AGA GC ACA  |  AAAAA

GGATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG

GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG

AGAGAG TATA ACG | AGAGGA AT TTTTTTTTTTTTTTTTTTTTTTTT GC AT ACA AGG AGG AGA GC ACA  |  AAAAA

GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG

CTCTCT TATA ACG | AGAAGA AT ATA AAG AAG AAG GC AT CAT ATT GC AT GAC ACA  TAC TAC TAC    ACT    AAG  GGA    GC |  AAAAA


GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG

ATATAT TATA ACG | GGGACA AT TTA AAA TCA GC  AT TAA CCC GGT GC  AT ACT ACT ACT GC  AT GGA AAT TAT AAT TAT AAT TAT AAT TAT  GC  AT GGA AAC TAT  AAC TAT  AAC TAT  AAC TAT GC|  AAAAA 

GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG

GAGAGA ATTAGTGT TATA ACG | AGAACC AT AAC AAT AAT AAT GC AT AAG CTGTATCTGTATCTGTATCTGTATCTGTATCTGTATCTGTATCTGTATCTGTATCTGTATCTGTATCTGTATCTGTATCTGTATCTGTATCTGTAT GC TCGAGCTGCGTCTG AT GTC GAC GTG GC AT CTC TCT GC | AAAAA

GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG

CTCTCT ATTAGTGT TATA ACG | CCCGGC AT ATC CTA TCA TTG GAT GC | AAAAA

GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG

GAGAGA ATTAGTGT TATA ACG | AGAACC AT AAC AAT AAT AAT GC AT AAG CTGTATCTGTATCTGTATCTGTATCTGTATCTGTATCTGTATCTGTATCTGTATCTGTATCTGTATCTGTATCTGTATCTGTATCTGTATCTGTAT GC TCGAGCTGCGTCTG AT GTC GAC GTG GC AT CTC TCT GC | AAAAA



GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG

CTCTCT ATTAGTGT TATA ACG | AGAAGC  AT |ACA AAT  AAA| GC         AT TAC TTG TTG TTG GC         AT TTC TTA TTA  GC  | AAAAA

GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG

CTCTCT ATTAGTGT TATA ACG | TCTCTA    AT | ATG TTA ATA ATC ATC TAG | GC         AT ATG AAA TTG TTG GTA TCA GTA GC         AT ATG TCG TTA  CTA TCG TCT GC     AT AAT GAC ACT GC  AT TCA ATACTA GC  | AAAAA

GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG

CTCTCT ATTAGTGT TATA ACG | TGTGTT AT | ATG TTA ATA ATC ATC TAG | GC         AT ATG AAA TTG TTG GTA TCA GTA GC         AT ATG TCG TTA  CTA TCG TCT GC     AT AAT GAC ACT GC  AT TCA ATACTA GC| AAAAA

GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG

'''.replace(' ','')


genome_test = '''GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG



GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG

CTCTCT ATTAGTGT TATA ACG | ACACTT AT CCT ACG ACA GC   CTGTCGCTGCTTGT | AAAAA

GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG
GATAGATAGAGATTGATAGATAATAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGGGATAGATAGAGATTGATAGATAGTGAGTTAGCTATCGTACGAGCTAGCGTCATGGCTAGCAGTAGCTAGCATCGTATGCATGCTAGCTAGCTAGG


'''
'''
import json
with open('/Users/danilkutny/Desktop/ENN/generations2/gen_1088.json', 'r') as file:
	data = json.load(file)
	genome = data[0][-1]
'''
#print(genome[:100])
'''
nn = NN(genome_test, max_time=1)
t1 = time()
nn.develop()
t2 = time()
print('time:', t2-t1)
for cell in nn.nn:
	print(cell[1:])
print(nn.common_proteins)
'''
#CODE
#mutation gene - TTCTA    AT | ATG TTA ATA ATC ATC TAG | GC         AT ATG AAA TTG TTG GTA TCA GTA GC         AT ATG TCG TTA  CTA TCG TCT GC     AT AAT GAC ACT GC  AT TCA ATACTA GC
#division gene  - ACACTT | AT CCT ACG ACA GC | CTGTCGCTGCTTGT 

#FORBIDDEN:
'''
termination_site = 'AAAAA'

start_codon = 'ACG'
stop_codon = 'CAG'
protein_and_codon = 'GTAA'
protein_cut_codon = 'GTCA'

CGC
TAG

GC

'''


#AGAGAG ATTAGTGT TATA ACG | CGGGGT AT GGG TTA TTA GC   | AAAAA - apoptosis 
'''
genome_test = genome_test.replace(' ','')
genome_test = genome_test.replace('\n','')
genome_test = genome_test.replace('|','')


neural_network = NN(genome_test)
#neural_network.common_proteins['CCCDCGGCGTCTTAATAATCATCTAGGCGCCTGAAATTGTTGGTATCAGTAGCGCCTGTCGTTACTATCGTCTGC'] = 1
neural_network.develop()
print(neural_network.mutation_rate)
print('nn:',neural_network.nn[0][1])
print('len:',len(neural_network.nn))
print('GenPiler ended')
'''






'''
counet = 0
while True:
	counet+=1
	genome_test = ''
	for i in range(100000):
		genome_test += DNA_letters[randint(0,3)]
	#for i in range(100):
		#index_tmp = randint(1,999999)
		#genome_test[:index_tmp] + 'CTCTCT' + genome_test[index_tmp:]



	genome_test = genome_test.replace(' ','')
	genome_test = genome_test.replace('\n','')
	genome_test = genome_test.replace('|','')



	test_protein = 'AAAATT GC _ATA ACG ACG ACG_ GC  GC CAT ACT GC AAAAAAAAAA' 
	test_protein = test_protein.replace(' ','')
	test_protein = test_protein.replace('_','')
	test_protein = test_protein.replace('!','')
	neural_network = NN(genome_test, max_time=4)
	neural_network.nn[0][1][test_protein] = 1


	try:
		neural_network.develop()
	except RuntimeError:
		print('time out')
		pass





	print('#:',counet, neural_network.counter)
	if len(neural_network.nn)!=0:
		print('len prot:',len(neural_network.nn[0][1]))
		check_conns = False
		for i in neural_network.nn:
			if i[3]!=[]:
				check_conns = True
		if check_conns == True:
			print('nn length:',len(neural_network.nn))
			for i in neural_network.nn:
				print('nn:',len(i[1]), i[2:4])


print('GenPiler ended')
'''

#evolutions steps: multicelular - multiple names - multiple connections - 1 out layer - no cycles - 1 input layer - prediction

'''
protein_done = ['dna-binding protein, whiich activate gene','pretorein bind to another protein, and then do something', 'protein combine with other protein and make new protein',
'protein silence gene, buy adding specific elemnt to a list, wheretranscriptase should stop.','protein which lowercase dna sequnces or methilate it, and a protein whcih do oposite ',
'bridge protein, which send molecule from one cell to other type of cells depending on their name','protein change other ptorein','protein chaneg other ptorein, whiel it exists','change numebr of other ptroein',
'stop gene expression when some point is met', 'shuffle dictionary protein', 'self-destructuin', 'protein produce other proteins', 'aproptossi protein',
'self activation strength regulation', 'new cell name',
'splicing protein whcih regulate, weather this splice happend or not','specific dna sequnce, which says weather intrin starts or not. Also, it binding site can depend on the situatuion in the cell',
]

not_done = ['signal other neurons types to grow', 'protein whoich induce connections, from oen sell to another, in this cell and etc','protein which says, which operations neurons does',
'mutation protein']

common_types = ['prtotein directly signaling to other cells', 'signaling protein, can be send to common or specific cells','stop development','moleculw which can pass 
membrane of cell and gett inside', channel, that transport molecule from common to the cell]

proeins_types = ['splicer','divicions activation','activate dna','disactivate dna','signal protein','new cell name','grow connection',
'gene activator','gene supressor','mutation protein','protein connection to another protein','aproptossi protein',  ]


think_about_it = ['activation protein','ligand-recepto activation', 'ligand can be detryed or not','protein, which canount and do something', 
'for example change from beeing activated to regular state after some time','when protein activted, it can express genes',
'if other protein act/disacr, do somtehing', 'prtotein directly signaling to other cells', ,'dna cahnging proteins - 180 orintational change, moove',
'count protein', change code (find seq and replce it with other seq in exon types) ]
'''

