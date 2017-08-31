import random
import csv
import math
import copy
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as input1:
		reader =csv. reader(input1)
		for row in reader:
			dataset.append(row)
	numrow = len(dataset)
	numcol = len(dataset[0])
	return dataset, numrow, numcol
 
 
#coverting the string into int
def number_classlabel(dataset):
	for i in range(len(dataset)):
		for j in range(len(dataset[0])):
			if (dataset[i][j] == 'No'):
				dataset[i][j] = 0
			elif (dataset[i][j] == 'Yes'):
				dataset[i][j] = 1
 
#converting the string into float in the dataset
def string_to_float(dataset):
	for i in range(len(dataset)):
		for j in range(len(dataset[0])-1):
			dataset[i][j] = float(dataset[i][j])
			
#Create an initial population 
def create_population(population_size, chromosome_length):
	population = list()
	for i in range(population_size):
		chromosome = list()
		for j in range(chromosome_length):
			num = random.randint(0,1)
			chromosome.append(num)
		population.append(chromosome)
	return population

#fitness evaluation
def evaluate_fitness(population, dataset):
	for i in range(len(population)):
		chromosome = population[i]
		dataset_copy = copy.copy(dataset)
		for j in range(len(chromosome)):
			if(chromosome[j] == 1):
				
		
		
		

#roulette wheel selection method
def selection(fitness_list):
	fitness_sum = sum(fitness_list)
	fitness_prob_list = list()
	for i in range(fitness_list):
		fitness_prob_list.append(fitness_list[i]/float(fitness_sum))
	cum_value = 0 
	cum_prob_list = [] 
	for prob in prob_list: 
    	cum_prob_list.append( cum_value + prob ) 
    	cum_value += prob
    cum_prob_list[-1] = 1.0
    chromosome_selected = [] 
    for i in range(len(fitness_list)): 
		rn = random.random() 
		for j, cum_prob in enumerate(cum_prob_list): 
		    if rn<= cum_prob: 
		        chromosome_selected.append(j) 
		        break 
    return chromosome_selected	
				
#crossover
def crossover(crossover_rate,selected_list,population):
	random_num = list()
	for i in range(len(population[0])):
		random_num.append(random.random())
	for in range(len(random_num)):
		if(random_num[i] > .25 ):
			random_num[i] = -1
	random_index_list = list()
	for i in range(len(random_num)):
		if(random_num[i] !=-1):
			random_index_list.append(i)
		
	
	
		
	
				
filename = 'Glass_New.csv'
dataset= load_csv(filename)
numrow = len(dataset)
numcol = len(dataset[0])
population_size = 30
crossover_rate = .25
mutation_rate = .10
population = create_population(population_size,10)
