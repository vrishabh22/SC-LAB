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
def number_classlabel(dataset):
	for i in range(100):
		for j in range(5):
			if (dataset[i][j] == 'Iris-setosa'):
				dataset[i][j] = 0
			elif (dataset[i][j] == 'Iris-versicolor'):
				dataset[i][j] = 1
 
#converting the string into float in the dataset
def string_to_float(dataset):
	for i in range(100):
		for j in range(4):
			dataset[i][j] = float(dataset[i][j])


# Make a prediction with weights
def predict(row, weights, threshold):
	sum_weight = weights[0] #this is the weight of the bias
	for i in range(1,len(row)):
		sum_weight += weights[i] * row[i-1]
	return 1 if sum_weight >= threshold else 0
	

# Estimate weights
def train_weights(trainset, l_rate, max_iter,threshold, weights):
	total_error = 0
	for itern in range(max_iter):
		for row in trainset:
			prediction = predict(row, weights, threshold)
			error = row[-1] - prediction               #row[-1] returns the class label for that particular row
			total_error =total_error +  error * error
			weights[0] = weights[0] + l_rate * error   #this is for bias weight change
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + l_rate * error * row[i] #this is for features weight change
		if(total_error == 0):                     #if total error becomes zero then break from that fold
			break
	return weights
 
#function for cross validation
def cross_validation(dataset, l_rate, max_iter, threshold):
	accuracy = []
	fold_size = len(dataset)/10
	for i in range(0,100,fold_size):
		weights=[]
		for j in range(5):
			weights.append(1.0/5)
		train_start = i + 10
		trainset = dataset[train_start: ] [0:]
		if(i-1 > 0):
			for r in range(i):
				trainset.append(dataset[r])
		testset = dataset[i:i+10][0:]
		#print testset
		#print trainset
		weights = train_weights(trainset, l_rate, max_iter, threshold, weights)
		#testing of data and finding the accuracy
		correct_count = 0
		for r in testset:
			prediction = predict(r, weights, threshold)
			actual = r[-1]
			if(prediction == actual):
				correct_count+=1
			print "actual",r[-1],"predicted",prediction
		acc = (correct_count/10.0) * 100
		accuracy.append(acc)
		print acc
	acc_sum = sum(i for i in accuracy)
	avg_acc = acc_sum/len(accuracy)
	print "Accuracy = ", avg_acc
		
filename = 'IRIS.csv'
dataset,numrow,numcol = load_csv(filename)
number_classlabel(dataset)
string_to_float(dataset)
l_rate = 1
threshold = 2.5
max_iter= 100
cross_validation(dataset, l_rate, max_iter, threshold)
