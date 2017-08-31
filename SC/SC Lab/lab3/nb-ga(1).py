import csv
import random
import math
import timeit
import operator
def load_data(filename):
	dataset = list()
	with open(filename,'r') as filen:
		data_reader = csv.reader(filen)
		for row in data_reader:
			if not row:
				continue
			dataset.append(row)
		dataset.pop(0)
	return dataset	
	
def loadCsv(filename):
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset

def label_to_int(dataset,rows,cols):
	for i in range(rows):
		for j in range(cols-1):
			if(dataset[i][j]=='Yes'):
				dataset[i][j]= 1
			elif (dataset[i][j]=='No'):
				dataset[i][j]=0
def string_to_float(dataset,rows,cols):
	for i in range(rows):
		for j in range(cols):
			dataset[i][j]=float(dataset[i][j]);

def splitDataset(dataset, splitRatio):
	#for r in dataset:
	#	print dataset
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]

def separateByClass(dataset):
	separated = {}
	#print dataset
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	#print separated
	return separated

def mean(numbers):
	return sum(numbers)/(len(numbers))

def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers))
	return math.sqrt(variance)

def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries

def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.iteritems():
		summaries[classValue] = summarize(instances)
		#print classValue
	return summaries

def calculateProbability(x, mean, stdev):
	if(stdev==0):
		return 0;
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.iteritems():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities
			
def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	#print bestLabel
	return bestLabel

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append((result))
	return predictions

def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		#print testSet[i],predictions[i]
		if testSet[i][-1] == predictions[i]:
			correct += 1
	if(len(testSet)==0):
		return 1	
	return (correct/float(len(testSet))) * 100.0

def nb(rows,cols,data):
	splitRatio = 0.67
	#for row in data:
		#print row
	trainingSet, testSet = splitDataset(data, splitRatio)
	#print trainingSet
	summaries = summarizeByClass(trainingSet)
	predictions = getPredictions(summaries, testSet)
	#print predictions
	#print testSet
	accuracy = getAccuracy(testSet, predictions)
	return accuracy

def exchange(i,j):
	population[i] = population[j]

def cross(u,v,l):
	t = random.randint(1,l-1)
	for y in range(l):
		if(y>=t):
			temp = population[u][y]
			population[u][y]=population[v][y]
			population[v][y] = temp
start = timeit.default_timer()
at = 9
ch = []
population = []
prob = {}
'''rows = 146
cols = 10'''
size = 20
filename = 'Glass_New.csv'
dataset = load_data(filename)
rows = len(dataset)
cols = len(dataset[0])
label_to_int(dataset,rows,cols)
string_to_float(dataset,rows,cols)
lk = []
for i in range(size):
	lst = [random.randrange(0,2) for i in range(at)]
	population.append(lst)
#print nb(rows,cols,dataset)
#def selection():
for it in range(5):
	fitness = {}
	f = 0
	for i in range(size): # for each chromosome
		new_dataset = []
		cs =0
		for count in range(at):
			if(population[i][count]==1):
				cs = cs+1
		for r in range(rows):
			n_row = []
			for k in range(at):
				if(population[i][k]==1):
					n_row.append(dataset[r][k])
			n_row.append(dataset[r][-1])
			#print n_row
			new_dataset.append(n_row)
		#for row in new_dataset:
			#print row
		#print cs
		fitness[i] = nb(rows,cs+1,new_dataset)
		#print fitness
		f =f+fitness[i]
		#fitness.append(nb(rows,cs,new_dataset))
	
	for j in range(size):
		prob[j] =(fitness[j])/f
	
	csum= prob[0]	#cumulative probability
	
	for j in range(1,size):
		prob[j] =prob[j]+csum
		csum = prob[j]


	sorted_x = sorted(prob.items(), key=operator.itemgetter(1))
	for i in range(size):
    		ra = (random.random())
    		for j in range(size):
			if(sorted_x[j]>ra):
				exchange(i,j)
				break
	
#def crossover():
 	#cross mutation 
	selected_index = []
	for i in range(size):
		ra = random.random()
		if(ra < .25):
			selected_index.append(i)
	for i in range(1,len(selected_index)):
		cross(selected_index[i],selected_index[i-1],at)
			
#def mutation():
	genes = at*size/10
	j = 0	
	for i in range(genes):
		r = random.randint(0,at-1)
		if(population[j][r]):
			population[j][r]=0
		else:
			population[j][r]=1
		j = j+1

hu=0
for i in range(size): # for each chromosome
		new_dataset = []
		cs =0
		for count in range(at):
			if(population[i][count]):
				cs = cs+1
		for r in range(1,rows):
			n_row = []
			for k in range(at):
				if(population[i][k]==1):
					n_row.append(dataset[r][k])
			n_row.append(dataset[r][-1])
			new_dataset.append(n_row)
		fitness[i] = nb(rows,cs+1,new_dataset)




#Your statements here

stop = timeit.default_timer()

print stop - start 
print max(fitness.values())
