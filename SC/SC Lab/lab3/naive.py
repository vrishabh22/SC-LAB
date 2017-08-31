import csv
import random
import math
 
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

def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]

def label_to_int(dataset,rows,col):
	for i in range(rows):
		for j in range(col):
			#if(dataset[i][j]=="Iris-setosa"):
			#	dataset[i][j]=0;
			#elif (dataset[i][j]=="Iris-versicolor"):
			#	dataset[i][j]=1;
			if(dataset[i][j]=="No"):
				dataset[i][j]=0;
			elif (dataset[i][j]=="Yes"):
				dataset[i][j]=1;

def string_to_float(dataset,rows,col):
	for i in range(rows):
		for j in range(1,col):
			dataset[i][j]=float(dataset[i][j]);

def cross_validation_split(dataset,no_folds):
	split = list()
	copy = list(dataset)
	fold_size = int(len(dataset) / no_folds)
	for i in range(no_folds):
		fold = list()
		while(len(fold) < fold_size):
			index = randrange(len(copy))
			fold.append(copy.pop(index))
		split.append(fold)
	return split

 
def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated
 
def mean(numbers):
	return sum(numbers)/float(len(numbers))
 
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
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
	return summaries
 
def calculateProbability(x, mean, stdev):
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
	return bestLabel
 
def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions
 
def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i][-1] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
 
def main():
	filename = 'Glass_New.csv'
	splitRatio = 0.67
	dataset= load_data(filename)
	trainingSet, testSet = splitDataset(dataset, splitRatio)
	print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingSet), len(testSet))
	# prepare model
	summaries = summarizeByClass(trainingSet)
	# test model
	predictions = getPredictions(summaries, testSet)
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: {0}%').format(accuracy)
 
main()
