# Neuroevolution, Simulation of neural network evolution.
*Example of evolved neural network:*
![This is an image](https://preview.redd.it/dgxjwq5g9eia1.png?width=4123&format=png&auto=webp&v=enabled&s=0b1206a1764629978bc98350cf9d1915dd954a63)

My project is to create neural networks that can evolve like living organisms. This mechanism of evolution is inspired by real-world biology and 
is heavily focused on biochemistry. Much like real living organisms, my neural networks consist of cells, each with their own genome and proteins. 
Proteins can express and repress genes, manipulate their own genetic code and other proteins, regulate neural network connections, 
facilitate gene splicing, and manage the flow of proteins between cells - all of which contribute to creating a complex gene regulatory network and 
an indirect encoding mechanism for neural networks, where even a single letter mutation can cause dramatic changes to a model.

Some cool results of my neural network evolution after a few hundred generations of training on MNIST can be found in [Google drive](https://drive.google.com/drive/folders/1pOU_IcQCDtSLHNmk3QrCadB2PXCU5ryX?usp=sharing)

## How to evolve your own neural network?

If you want to try evolve your own Neural Networks, you only need python interpreter and Tenserflow installed. And the code of course!
Start with `population.py` - run the script, in my case I use zsh terminal on MacOS.

```
python3 path/to/destination/population.py
```

Default number of neural networks in a population is set to 10 and maximum development time - 12 second, so it will take about 120 second to develop all NNs. 

<img width="282" alt="Screenshot" src="https://user-images.githubusercontent.com/121340828/219373144-2700b606-f5a1-4d6a-ba15-74376229ea2b.png"> <img width="500" alt="Screenshot1" src="https://user-images.githubusercontent.com/121340828/219373597-32d733af-42a2-4e7f-9aad-6f8c2e036c07.png">

Then, each one will start to learn MNIST dataset for 3 epochs and will be evaluated. This process of leaning will be shown interactively, and you will see, how much accuracy does a model get each time(from 0 to 1).
After each model has been evaluated, best will be selected and their genes will be recombined, and population will be saved in a `boost_perfomnc_gen` folder, in the `gen_N.json` file, where `N` - number of your generation.

<img width="246" alt="Screenshot2" src="https://user-images.githubusercontent.com/121340828/219373894-72f81cf7-50e0-4741-a4b3-4c8e7c310067.png">

If you would like to see the resulted neural network architecture:
1. choose last `gen_N.json` file (represents last generation of neural network models)
2. open `test.py`
3. On the 1st line of code, there will be: `generation_file = "default_gen_284.json"`
4. change `default_gen_284.json` to `gen_N.json`
5. By default, 1st neural network in population is choosen(`neural_network_pop_number=0`). Choose, which exact network in present generation you want to visualise(by default there exist 10 NNs, index numbers: 0-9)
6. run the script
7. full model architecture will be saved as `test_model.png`
<img width="350" alt="Screenshot3" src="https://user-images.githubusercontent.com/121340828/219374366-49b94d8a-1327-42b4-bbea-55018217ded0.png">
______________test_model.png______________


### About

The code for this project consists of three parts:

1. Genpiler (a genetical compiler) - the heart of the evolution code, which simulates many known biochemistry processes of living organisms, transforming a sequence of "ACGT" letters (the genetic code) into a mature neural network with complex interconnections, defined matrix operations, activation functions, training parameters and meta parameters.
2. Tensorflow_model.py transcribes the resulting neural network into a TensorFlow model.
3. Population.py creates a population of neural networks, evaluates them with MNIST dataset and creates a new generation by taking the best-performed networks, recombining their genomes (through sexual reproduction) and mutating them.

### Other interesting results:

<img width="500" alt="Screenshot4" src="https://user-images.githubusercontent.com/121340828/219380757-25f5c0a7-241f-44d9-a3c5-c09e47681569.png">
<img width="1000" alt="Screenshot6" src="https://user-images.githubusercontent.com/121340828/219381971-6e978d77-562a-419a-9896-c38a8114e100.png">



## How the genetic compiler works
Neural networks are composed of cells, a list of common proteins, and metaparameters. Each cell is a basic unit of the neural network, and it carries out matrix operations in a TensorFlow model. In Python code, cells are represented as a list. This list includes a genome, a protein dictionary, a cell name, connections, a matrix operation, an activation function, and weights:
1. The genome is a sequence of arbitrary A, C, T, and G letter combinations. Over time, lowercase letters (a, c, t, g) may be included, to indicate sequences that are not available for transcription.
2. The protein dictionary is a set of proteins, each represented by a sequence of A, C, T, and G letters, as well as a rate parameter. This rate parameter is a number between 1 and 4000, and it simulates the concentration rate of the protein. Some proteins can only be activated when the concentration reaches a certain level.
3. The cell name is a specific sequence, in the same form as the protein and genome. It is used to identify specific cells and cell types, so that proteins can work with the exact cell and cell types. For example, a protein can work with all cells that have the sequence "ACTGACTGAC" in their name.
4. The connections list shows all the forward connections of the cell.
5. The matrix operation is defined by the type of matrix operation available in the TensorFlow documentation.
6. The activation function is also defined by the type of activation function available in the TensorFlow documentation.
7. The weights define the weights of the parameters in the TensorFlow model.
#### Common Proteins
Common proteins are similar to the proteins found in a single cell, but they play an important role in cell-to-cell communication. These proteins are able to move between cells, allowing them to act as a signaling mechanism or to perform other functions. For example, a protein may exit one cell and enter another cell through the common_proteins dictionary, allowing for communication between the two cells.
#### Metaparematers:
1. `time_limit` - maximum time for neural network development
2. `learning_rate` = []
3. `mutation_rate` = [None, None, None, None, None, None, None](don’t work!)

## Gene transcription and expression
### Gene transcription
All cells start with some genome and a protein, such as `AAAATTGCATAACGACGACGGC`. What does this protein do?

![r/artificial](https://preview.redd.it/jllu4rgn6eia1.jpg?width=318&format=pjpg&auto=webp&v=enabled&s=14021fa3d36360831e3bf8196a154de96426007e) 


This is a gene transcription protein, and it starts a gene transcription cascade. To better understand its structure, let’s divide the protein into pieces: `AAAATT GC |ATA ACG ACG ACG| GC` The first 6 letters - `AAAATT` - indicate what type of protein it is. There are 23 types of different proteins, and this is type 1 - gene transcription protein. The sequence `GCATAACGACGACGGC` encodes how this protein works.

* (If there are `GTAA` or `GTCA` sequences in the gene, the protein contains multiple “functional centers” and the program will cut the protein into multiple parts (according to how many `GTAA` or `GTCA` there are) and act as if these are different proteins. In this way, one protein can perform multiple different functions of different protein types - it can express some genes, and repress others, for example). If we add `GTAA` and the same `AAAATTGCATAACGACGACGGC` one more time, we will have `AAAATTGCATAACGACGACGGCGTAAAAAATTGCATAACGACGACGGC` protein. The program will read this as one protein with two active sites and do two of the same functions in a row.


`GC` part is called an exon cut, as you can see in the example. It means that the pieces of the genome between the "GC" do the actual function, while the `GC` site itself acts as a separator for the parameters. I will show an example later. `ATA ACG ACG ACG` is the exon (parameter) of a gene transcription protein, divided into codons, which are three-letter sequences.


Each protein, though it has a specific name, in this case "gene transcription activation," can do multiple things, for example:
1. Express a gene at a specific site (shown later)
2. Express such a gene with a specific rate (how much protein to express, usually 1-4000)
3. Express such a gene at a controllable random rate (`rate = randint(1, N)`, where `N` is a number that can be encoded in the exon)
4. Pass a cell barrier and diffuse into the `common_protein` environment


The "gene transcription activation" protein can do all of these things, so each exon (protein parameter) encodes an exact action. The first codon (three-letter sequence) encodes what type of exon it is, and the other codons encode other information. In the example, the first codon `ATA` of this parameter shows the type of parameter. `ATA` means that this is an expression site parameter, so the next three codons: `ACG ACG ACG` specify the site to which the gene expression protein will bind to express a gene (shown in the example later). A special function `codons_to_nucl` is used to transcribe codons into a sequence of `ACTG` alphabet. In our case, the `AC ACG ACG` codons encode the sequence `CTCTCT`. This sequence will be used as a binding site.
Now, after we understand how the protein sequence `AAAATTGCATAACGACGACGGC` will be read by our program and do its function, I will show you how gene expression happens.
### Gene expression
Imagine such a piece of genetic code is present in the genome(Spaces & «|» are used for separation and readability): 

`CTCTCT TATA ACG | AGAGGG AT CAA AGT AGT AGT GC AT ACA AGG AGG ACT GC ACA | AAAAA`

If we have a gene transcription protein in a protein_list dictionary in the cell, with a binding parameter - `CTCTCT` sequence. Then, the program will simulate as what you would expect in biology:
The gene transcription protein binds to the `CTCTCT` sequence.
Then, it looks for a «TATA box». In my code - `TATA` is a sequence representing the start of a gene. So, after the binding sequence is found in the genome and after the TATA sequence is found next, gene expression starts.
`AAAAA` is the termination site. It indicates that the gene ends here.
Rate is the number describing protein concentration. By default, the expression rate is set to 1, so in our case only 1 protein will be created (protein:1), however the expression rate can be regulated, as previously mentioned, by a special parameter in the gene expression protein.

So, in the process of expression, the protein is added to a proteins_list, simulating gene expression, and then it can do its function. However, there are a few additional steps before the protein is expressed.
There are repression proteins. They are used to repress gene expression and they work similarly to gene expression activation, but in the opposite direction. They can encode a special sequence and strength of silence, so that the transcription rate lowers, depending on how close the binding expression occurs and what the strength of silence is.
The gene splicing mechanism cuts the gene into different pieces, then deletes introns and recombines exons. Splicing can also be regulated in the cell by a special slicing regulation protein.

![image splicing](https://preview.redd.it/poiu8b6q6eia1.jpg?width=400&format=pjpg&auto=webp&v=enabled&s=65690c2e4b3e7fb3e90c77af4d295e9b1e12593b)

### Here is the list of all protein types with a short description:
1. Gene transcription - finds an exact sequence in the genome and starts to express the gene near that sequence
2. Gene repressor - represses specific gene activation
3. Gene shaperon add - adds a specific sequence at an exact place and to a specific protein (changes a protein from `ACGT` to `ACCCGT` by adding `CC` after the `AC` sequence)
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
### Other important points of code
What else does a cell do?
1. Activate or silence transcription
2. Protein splicing and translation


`Common_protein` is intercell protein list. Some proteins can only do its function in the common_protein intercell environment:

1. Common connection growth factor - regulates connection growth between cells
2. Stop development
3. Learning_rate - sets a specific learning_rate, in a for of list(new epoch - new learnign rate)
4. Mutation rate - changes the mutation parameter, how actively the cell will mutate


NN object has a develop method. In order for development to start:

1. NN should have at least one cell, with a working genetic code. First, I write a simple code myself, it is very simple. From there, it can evolve.
3. Also, for development to start, NN should contain at least one expression protein in its protein dictionary for proteins expression network to start making its thing.


How development works:


1. Loop over neural network cells.
    a. Loop over each protein in each cell and add what the protein should do to a specific "to do" list.
    b. After this cell loop ends, everything said in the "to do" list is done, one by one.
2. After each cell has done all the actions its proteins have said to do, the common proteins loop starts. This loop is very similar to the loop in each cell and it makes all the actions which the "common proteins" say to do.
3. If the development parameter is still True - the loop repeats itself.


## Main code files
### GenPile_2.py
Genetical compiler, indirectly encodes neural network into a string of genome.
### Tensorflow_model.py
Transforming a neural network in the form of a list to a Tensorflow model. It creates a `model_tmp.py`, which is a python written code of a Tensorflow model. If you remove both `"'''"` in the end of the file, you can see the model.summary, a visual representation of the model (`random_model.png`) and test it on the MNIST dataset. You can see such a file in the repository.
### Population.py
Creating a population of neural networks using genome samples, developing them, transforming them to a Tensorflow model, evaluating them and creating a new generation by taking the best performing neural networks, recombining their genomes (sex reproduction) and mutating. This code performs actual Evolution and saves all models in the `boost_performance_gen` directory in the form of `.json` in a python list, with some log information and genome of each NN in the form of a 2-d list: 
```
[[ "total number of cells in nn", "number of divergent cell names", "number of layer connections", "genome"],
[…], 
…]
```

#### Main parameters in `population.py`:
1. `number_nns` - number of neural networks to take per population (10 default)
2. `start_pop` - file with genetic code of population. `/boost_performance_gen/default_gen_284.json` by default
3. `save_folder` - where to save the result of your population's evolution
### Test.py
If you want to test a specific neural network, use test.py to see the visualization of its structure (saved as a png) and test it on the MNIST data.
### model_tmp.py
This is a temporary file, which is a python code representation of a tensorflow neural netwroks model. Each time a new tensorflow model created by `Tensorflow_model.py`, `model_tmp.py` changes. 

If you want to see thee model visuzlization, you can:
1. open `model_tmp.py`
2. scroll down code file till the end
3. You will see hidden code, invisible to python intrepreter. If you remoove first `'''` and second `'''`, you can test this model on MNIST data, see `model.summary()` and model visualization will be saved at `random_model_99.png`
