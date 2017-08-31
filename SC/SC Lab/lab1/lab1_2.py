from random import seed
from random import randrange
import csv 
 
# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as input1:
		reader = csv.reader(input1)
		for row in reader:
			if not row:
				continue
			dataset.append(row)
		dataset.pop(0)
	numrow = len(dataset)
	numcol = len(dataset[0])
	return dataset, numrow, numcol
 
#coverting the string into int
def number_classlabel(dataset,numrow,numcol):
	for i in range(numrow):
		if (dataset[i][0] == 'No'):
			dataset[i][0] = 0
		elif (dataset[i][0] == 'Yes'):
			dataset[i][0] = 1
 
#converting the string into float in the dataset
def string_to_float(dataset,numrow,numcol):
	for i in range(numrow):
		for j in range(1,numcol):
			dataset[i][j] = float(dataset[i][j])


# Make a prediction with weights
def predict(row, weights, threshold):
	sum_weight = weights[0] #this is the weight of the bias
	for i in range(len(row)-1):
		sum_weight += weights[i + 1] * row[i+1]
	return 1 if sum_weight >= threshold else 0
	

# Estimate weights
def train_weights(trainset, l_rate, max_iter,threshold):
	weights = [0.022 for i in range(len(trainset[0]))]
	total_error = 0
	for itern in range(max_iter):
		for row in trainset:
			prediction = predict(row, weights, threshold)
			error = row[0] - prediction               #row[-1] returns the class label for that particular row
			total_error += error * error
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + l_rate * error * row[i+1] #this is for features weight change
		if(total_error == 0):                     #if total error becomes zero then break from that fold
			break
	return weights
	
 
#function for cross validation
def cross_validation(dataset, l_rate, max_iter, threshold,numrow,numcol):
	accuracy = []
	size = numrow/10
	
	for i in range(0,numrow,size):
		
		train_start = i + size
		trainset = dataset[train_start: ]
		if(i-1 > 0):
			for r in range(i):
				trainset.append(dataset[r])
		testset = dataset[i:i+size]
		
		weights = train_weights(trainset, l_rate, max_iter, threshold)
		
		correct_count = 0
		for r in testset:
			prediction = predict(r, weights, threshold)
			actual = r[0]
			if(prediction == actual):
				correct_count+=1
		acc = (correct_count/size) * 100.0
		accuracy.append(acc)
		print len(trainset)
		print len(testset)
		print acc
	acc_sum = sum(i for i in accuracy)
	avg_acc = acc_sum/len(accuracy)
	print avg_acc
		
filename = 'SPECTF.csv'
dataset,numrow,numcol = load_csv(filename)
number_classlabel(dataset,numrow,numcol)
string_to_float(dataset,numrow,numcol)
l_rate = .01
threshold = 75
max_iter= 100
cross_validation(dataset, l_rate, max_iter, threshold,numrow,numcol)
