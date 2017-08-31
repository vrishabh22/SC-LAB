import csv
import math
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
			
def separateClassWise(dataset):
	class0 = list()
	class1 = list()
	for i in range(len(dataset)):
		instance = dataset[i]
		if(instance[-1] == 0):
			class0.append(instance)
		else:
			class1.append(instance)
	return class0, class1

def mean(numbers):
	mean_n = sum(numbers)/float(len(numbers))
	return mean_n
	
def std_deviation(numbers):
	avg = mean(numbers)
	s = 0
	for x in numbers:
		s = s + pow(x-avg, 2)
	variance = s/float(len(numbers)-1)
	std_dev = math.sqrt(variance)
	return std_dev


def calculate_stddev_and_mean(dataset):
	summaries = []
	for k in range(len(dataset[0])-1):
		temp = list()
		for i in range(len(dataset)):
			temp.append(dataset[i][k])
		values = list()
		mean_vlaue = mean(temp)
		std_dev = std_deviation(temp)
		values.append(mean_vlaue)
		values.append(std_dev)
		summaries.append(values)
	return summaries


def find_stddev_and_mean_classwise(dataset):
	class0,class1 = separateClassWise(dataset)
	summaries = {}
	summaries[0] = calculate_stddev_and_mean(class0)
	summaries[1] = calculate_stddev_and_mean(class1)
	return summaries

#Calculate Gaussian Probability Density Function where x is the attribute value 
#stddev and mean are coressponding to a particular class
def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
	
def calculateClassProbabilities(class0, class1, inputVector):
	probabilities = {}
	for classSummaries in summaries.iteritems():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities
	
x = 71.5
mean = 73
stdev = 6.2
probability = calculateProbability(x, mean, stdev)
print('Probability of belonging to this class: {0}').format(probability)


		
		

'''filename="Glass_New.csv"
dataset, numrow, numcol=load_csv(filename)
number_classlabel(dataset)
string_to_float(dataset)
class0, class1 = separateClassWise(dataset)
print class0, class1'''
