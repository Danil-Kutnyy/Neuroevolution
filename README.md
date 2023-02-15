# Neuroevolution
Simulation of neural network evolution
(https://user-images.githubusercontent.com/121340828/219107823-70b1ffb4-ce73-4e16-9d07-33f3f5908b07.png)
My project is to create neural networks that can evolve like living organisms. This mechanism of evolution is inspired by real-world biology and 
is heavily focused on biochemistry. Much like real living organisms, my neural networks consist of cells, each with their own genome and proteins. 
Proteins can express and repress genes, manipulate their own genetic code and other proteins, regulate neural network connections, 
facilitate gene splicing, and manage the flow of proteins between cells - all of which contribute to creating a complex gene regulatory network and 
an indirect encoding mechanism for neural networks, where even a single letter mutation can cause dramatic changes to a model.

The code for this project consists of three parts: 
1. Genpiler (a genetical compiler) - the heart of the evolution code, which simulates many known biochemistry processes of living organisms, 
transforming a sequence of "ACGT" letters (the genetic code) into a mature neural network with complex interconnections, defined matrix operations, 
activation functions, training parameters and meta parameters. 
2. Tensorflow_model.py transcribes the resulting neural network into a TensorFlow model. 
3. Population.py creates a population of neural networks, evaluates them and creates a new generation by taking the best-sampled networks, 
recombining their genomes (through sexual reproduction) and mutating them.

Some results of my neural network evolution after a few hundred generations of training on MNIST can be foun in googel drive:
https://drive.google.com/drive/folders/1pOU_IcQCDtSLHNmk3QrCadB2PXCU5ryX?usp=sharing

Neural networks are composed of cells, a list of common proteins, and metaparameters. Each cell is a basic unit of the neural network, 
and it carries out matrix operations in a TensorFlow model. In Python code, cells are represented as a list. This list includes a genome, 
a protein dictionary, a cell name, connections, a matrix operation, an activation function, and weights:

1. The genome is a sequence of arbitrary A, C, T, and G letter combinations. Over time, lowercase letters (a, c, t, g) may be included, 
to indicate sequences that are not available for transcription.
2. The protein dictionary is a set of proteins, each represented by a sequence of A, C, T, and G letters, as well as a rate parameter. 
This rate parameter is a number between 1 and 4000, and it simulates the concentration rate of the protein. Some proteins can only be 
activated when the concentration reaches a certain level. 
3. The cell name is a specific sequence, in the same form as the protein and genome. It is used to identify specific cells and cell types, 
so that proteins can work with the exact cell and cell types. For example, a protein can work with all cells 
that have the sequence "ACTGACTGAC" in their name.
4. The connections list shows all the forward connections of the cell.
5. The matrix operation is defined by the type of matrix operation available in the TensorFlow documentation.
6. The activation function is also defined by the type of activation function available in the TensorFlow documentation. 
7. The weights define the weights of the parameters in the TensorFlow model.

Common Proteins
Common proteins are similar to the proteins found in a single cell, but they play an important role in cell-to-cell communication. 
These proteins are able to move between cells, allowing them to act as a signaling mechanism or to perform other functions. For example, 
a protein may exit one cell and enter another cell through the common_proteins dictionary, allowing for communication between the two cells.

Metaparematers:
1. self.time_limit - maximum time for neural network development
2. self.learning_rate = []
3. self.mutation_rate = [None, None, None, None, None, None, None](don’t work!)





Gene structure and transcription mechanism
All cells start with some genome and a protein, such as «AAAATTGCATAACGACGACGGC». What does this protein do?

This is a gene transcription protein, and it starts a gene transcription cascade. To better understand its structure, 
let’s divide the protein into pieces:
AAAATT GC |ATA ACG ACG ACG| GC
The first 6 letters - AAAATT - indicate what type of protein it is. There are 23 types of different proteins, and this is type 1 - gene transcription protein.
The sequence «GCATAACGACGACGGC» encodes how this protein works.

*(If there are GTAA or GTCA sequences in the gene, the protein contains multiple “functional centers” and the program will cut the protein 
into multiple parts (according to how many GTAA or GTCA there are) and act as if these are different proteins. In this way, one protein 
can perform multiple different functions of different protein types - it can express some genes, and repress others, for example).

If we add “GTAA” and the same “AAAATTGCATAACGACGACGGC” one more time, we will have “AAAATTGCATAACGACGACGGCGTAAAAAATTGCATAACGACGACGGC” protein. 
The program will read this as one protein with two active sites and do two of the same functions in a row.

GC part is called an exon cut, as you can see in the example. It means that the pieces of the genome between the "GC" do the actual function, 
while the "GC" site itself acts as a separator for the parameters. I will show an example later.
ATA ACG ACG ACG is the exon (parameter) of a gene transcription protein, divided into codons, which are three-letter sequences.

Each protein, though it has a specific name, in this case "gene transcription activation," can do multiple things, for example:
1. Express a gene at a specific site (shown later)
2. Express such a gene with a specific rate (how much protein to express, usually 1-4000)
3. Express such a gene at a controllable random rate (rate = randint(1, N), where N is a number that can be encoded in the exon)
4. Pass a cell barrier and diffuse into the common_protein environment

The "gene transcription activation" protein can do all of these things, so each exon (protein parameter) encodes an exact action. 
The first codon (three-letter sequence) encodes what type of exon it is, and the other codons encode other information.
In the example, the first codon "ATA" of this parameter shows the type of parameter. "ATA" means that this is an expression site parameter, 
so the next three codons: ACG ACG ACG specify the site to which the gene expression protein will bind to express a gene (shown in the example later).
A special function "codons_to_nucl" is used to transcribe codons into a sequence of "ACTG" alphabet. In our case, the "ACG ACG ACG" codons encode 
the sequence "CTCTCT". This sequence will be used as a binding site.


Now, after we understand how the protein sequence «AAAATTGCATAACGACGACGGC» will be read by our program and do its function, I will show you how gene expression happens. 

Imagine such a piece of genetic code is present in the genome:
*Spaces & «|» are used for separation and readability
«CTCTCT TATA ACG | AGAGGG AT CAA AGT AGT AGT GC AT ACA AGG AGG ACT GC ACA  |  AAAAA»

If we have a gene transcription protein in a protein_list dictionary in the cell, with a binding parameter - «CTCTCT» sequence. Then, the program will simulate as what you would expect in biology:
1. The gene transcription protein binds to the CTCTCT sequence. 
2. Then, it looks for a «TATA box». In my code - TATA is a sequence representing the start of a gene. So, after the binding sequence is found in the genome and after the TATA sequence is found next, gene expression starts.
3. AAAAA is the termination site. It indicates that the gene ends here. 
4. Rate is the number describing protein concentration. By default, the expression rate is set to 1, so in our case only 1 protein will be created (protein:1), however the expression rate can be regulated, as previously mentioned, by a special parameter in the gene expression protein.

So, in the process of expression, the protein is added to a proteins_list, simulating gene expression, and then it can do its function. However, there are a few additional steps before the protein is expressed.
1. There are repression proteins. They are used to repress gene expression and they work similarly to gene expression activation, but in the opposite direction. They can encode a special sequence and strength of silence, so that the transcription rate lowers, depending on how close the binding expression occurs and what the strength of silence is.
2. The gene splicing mechanism cuts the gene into different pieces, then deletes introns and recombines exons. Splicing can also be regulated in the cell by a special slicing regulation protein.

Here is the list of all protein types with a short description: 
Types of proteins:
1. Gene transcription - finds an exact sequence in the genome and starts to express the gene near that sequence
2. Gene repressor - represses specific gene activation 
3. Gene shaperon add - adds a specific sequence at an exact place and to a specific protein (changes a protein from «ACGT» to «ACCCGT» by adding «CC» after the «AC» sequence)
4. Gene shaperon remove - removes a specific sequence at a specific place of an existing protein
5. Cell division activator - divides a cell into multiple identical ones
6. Cell protein shuffle - shuffles all proteins inside a cell and changes them. It helps to change all indexes
7. Cell transposone - if activated, changes its own location in the genome according to some rules
8. Cell chromatin pack - makes specific genome parts unreadable for the expression
9. Cell chromatin unpack - does the opposite, makes some genome parts readable for the expression process
10. Cell protein deletion - removes specific proteins from the existing proteins 
11. Cell channels passive - allows specific proteins to passively flow from one cell to another (for example, if a cell A has 10 «G» proteins, and it has this passive channel protein, which allows «G» proteins to flow to a cell B, then the protein concentration in cell A will lower to 5, while increasing in cell B to 5. Allows for specific proteins to flow between cell environments
12. Cell channels active - unlike the passive channel, this protein forces an exact protein to flow from one cell to another, so in the previous example, this channel will decrease the concentration of «G» proteins from 10 to 0 in cell A and increase the protein rate from 0 to 10 in cell B
13. Cell apoptosis - destroys a cell
14. Cell secrete - produces proteins with a specific sequence
15. Cell activation and silence strength - changes the overall parameters of how much to silence and express proteins in a specific cell, and at which part of the genome
16. Signalling - other than doing nothing, can change its concentration in the cell using a random function, with a specific random characteristic
17. Splicing regulatory factor - changes parameters of splicing in an exact cell
18. Cell name - changes a cell name
19. Connection growth factor - regulates cell connections to other cells
20. Cell matrix operation type - this protein can encode a specific Tensorflow matrix operation. It indicates which matrix operation the cell will use as a neural network model
21. Cell activation function - this protein can encode a specific Tensorflow activation function used by the cell
22. Cell weights - this protein can encode specific Tensorflow weight parameters for the cell
23. Proteins do nothing - do nothing 

What else does a cell do?
1. Activate or silence transcription
2. Protein splicing and translation

Common_protein - intercell protein list. Some proteins can only do its function in the common_protein intercell environment:
1. Common connection growth factor - regulates connection growth between cells
2. Stop development
3. Learning_rate - sets a specific learning_rate 
4. Mutation rate - changes the mutation parameter, how actively the cell will mutate


NN object has a develop method.

In order for development to start:
1. NN should have at least one cell, with a working genetic code. First, I write a simple code myself, it is very simple. From there, it can evolve.
2. Also, for development to start, NN should contain at least one expression protein in its protein dictionary for proteins expression network to start making its thing.

Code logic:
When the neural network satisfies the previously written condition, you can use the develop method. This method starts an infinite loop. There are two ways to stop this loop:
1. Expression of a protein in a common_protein list, which stops the development process (usually it works like this - some cell expresses the "stop developing" protein, which then travels to the common_protein).
2. The development loop has a time limit, a neural network meta parameter.

How development works:
1. Loop over neural network cells.
    1. Loop over each protein in each cell and add what the protein should do to a specific "to do" list.
    2. After this cell loop ends, everything said in the "to do" list is done, one by one.
2. After each cell has done all the actions its proteins have said to do, the common proteins loop starts. This loop is very similar to the loop in each cell and it makes all the actions which the "common proteins" say to do.
3. If the development parameter is still True - the loop repeats itself.

Tensorflow_model.py:
Transforming a neural network in the form of a list to a Tensorflow model. It creates a model_tmp.py, which is a python written code of a Tensorflow model. If you remove both "'''" in the end of the file, you can see the model.summary, a visual representation of the model (random_model.png) and test it on the MNIST dataset. You can see such a file in the repository.

Population:
Creating a population of neural networks using genome samples, developing them, transforming them to a Tensorflow model, evaluating them and creating a new generation by taking the best performing neural networks, recombining their genomes (sex reproduction) and mutating. This code performs actual Evolution and saves all models in the "boost_performance_gen" directory in the form of .json in a python list, with some log information and genome of each NN in the form of a 2-d list: 
[[ "total number of cells in nn", "number of divergent cell names", "number of layer connections", "genome"], […],
…]

Main parameters in population.py:
1. number_nns - number of neural networks to take per population (10 default)
2. start_pop - file with genetic code of population. /boost_performance_gen/default_gen_284.json by default
3. save_folder - where to save the result of your population's evolution

Test.py
If you want to test a specific neural network, use test.py to see the visualization of its structure (saved as a png) and test it on the MNIST data.
