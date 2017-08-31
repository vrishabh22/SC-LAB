from __future__ import division
from math import exp
from csv import reader

#Loading a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename,"r") as file:
		csv_reader=reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
		dataset.pop(0)
	return dataset
#Converting string to float
def str_to_float(dataset,column):
	for row in dataset:
		row[column]=float(row[column].strip())
#Converting class labels
def str_to_int(dataset,column):
	class_label= [row[column] for row in dataset]
	unique=set(class_label)
	lookup=dict()
	for i,value in enumerate(unique):
		lookup[value]=i
	for row in dataset:
		row[column]=lookup[row[column]]
	return lookup
def dataset_minmax(dataset):
	minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats
# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
#Dividing the data into 10 folds
def divide_folds(dataset,n_folds):
	dataset_split=list()
	dataset_copy=list(dataset)
	fold_sz=int(len(dataset)/n_folds)
	for i in range(n_folds-1):
		fold=list()
		index=0
		for n in range(fold_sz):
			fold.append(dataset_copy.pop(index))
			index=index+1
		dataset_split.append(fold)
	dataset_split.append(dataset_copy)
	return dataset_split
def accuracy_metric(actual, predicted):
	correct = 0
	TP = 0
	TN = 0
	FN = 0
	FP = 0
	#print len(actual)
	for i in range(len(actual)):
		#print actual[i],predicted[i]
		if actual[i] == predicted[i]:
			correct += 1
			if actual[i]==1:
				TP += 1
			else:
				TN +=1
		else:
			if actual[i] == 1:
				FN += 1
			if actual[i] == 0:
				FP +=1
	#print TP,FP,TN,FN
	if ((TP+FP) != 0):
		precY = TP / (TP +FP)
	else:
		precY = 0
	if((TP+FN) != 0):
		recY = TP / (TP + FN)
	else:
		recY =0
	if((TN + FN) != 0):
		precN= TN / (TN + FN)
	else:
		precN = 0
	if((TN + FP) != 0):
		recN = TN /(TN + FP)
	else:
		recN = 0
	return correct / float(len(actual)) * 100.0,precY,recY,precN,recN
	
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = divide_folds(dataset, n_folds)
	scores = list()
	preY= list()
	preN = list()
	reY= list()
	reN = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy,pY,rY,pN,rN = accuracy_metric(actual, predicted)
		scores.append(accuracy)
		preY.append(pY)
		preN.append(pN)
		reY.append(rY)
		reN.append(rN)
	return scores,preY,preN,reY,reN
	
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[(1/(n_inputs*5)) for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[(1/(n_hidden*1)) for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	for layer in network:
		for l in layer:
			for key in l:
				l[key][-1]=1/6
	return network
	
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation
	
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))
	
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs
	
def transfer_derivative(output):
	return output * (1.0 - output)
	
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[0] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
			
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']
			
def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row)
			expected=[row[-1]]
			#print "expected:%s" %expected
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))]) 
			if(sum_error ==0):
				break
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		#print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
		
def predict(network, row):
	outputs = forward_propagate(network, row)
	#print "output : %s"%outputs[0]
	if((outputs[0]- 0.85) >0):
		return 1
	else:
		return 0
		
def back_propagation(train, test, l_rate, n_epoch, n_hidden):
	n_inputs = len(train[0]) - 1
	n_outputs = 1
	#print "initialize"
	network = initialize_network(n_inputs, n_hidden, n_outputs)
	#print network
	train_network(network, train, l_rate, n_epoch, n_outputs)
	predictions = list()
	for row in test:
		prediction = predict(network, row)
		predictions.append(prediction)
	return(predictions)
	
filename = 'SPECTF.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
	str_to_float(dataset, i)
# convert class column to integers
str_to_int(dataset, len(dataset[0])-1)
# normalize input variables
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)
# evaluate algorithm
n_folds = 10
l_rate=[20,0.2,0.3,0.4,0.5,0.6,0.7,0.8,-10]
n_epoch = 20
n_hidden = 5
print "l_rate\tMeanAccuracy\tPrecisionY\t\tPrecisionN\t\tRecallY\t\t     RecallN"
for l in range(len(l_rate)):
	scores,preY,preN,reY,reN = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate[l], n_epoch, n_hidden)
	MeanAccuracy= (sum(scores)/float(len(scores)))
	precisionY = (sum(preY)/float(len(preY)))
	precisionN = (sum(preN)/float(len(preN)))
	recallY = (sum(reY)/float(len(reY)))
	recallN = (sum(reN)/float(len(reN)))
	print "%s\t%s\t%s\t%s\t%s\t%s"%(l_rate[l],MeanAccuracy,precisionY,precisionN,recallY,recallN)
