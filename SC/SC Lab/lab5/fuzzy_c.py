import random
import csv
from random import randint
import math
from math import sqrt
import copy
import operator
import timeit

global Theta
Theta = 0.0001

def loadCsv(filename):
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset

def loadCsv1(filename):
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	for i in range(len(dataset)):
		for j in range(len(dataset[0])-1):
			dataset[i][j] = float(dataset[i][j])
	return dataset



def select_initial_centers(dataset, c):
	maximum_value = map(max, zip(*dataset))
	minimum_value = map(min, zip(*dataset))

	init_centers = list()
	for i in range(c):
		current = []
		for j in range(len(maximum_value)):
			
			value = random.randint(minimum_value[j], maximum_value[j])
			current.append(value)
		init_centers.append(current)
	print init_centers
	return init_centers
	
#This function randomly initializes U such that the rows add up to 1.
def initialise_U(dataset, c):
	
	U = []
	for i in range(0,len(dataset)):
		current = []
		rand_sum = 0.0
		for j in range(0,c):
			value = random.randint(1,100)
			current.append(value)
			rand_sum += value
		for j in range(0,c):
			current[j] = current[j] / rand_sum
		U.append(current)
	return U

def distance(point, center):
	dimensions = len(point)
	sum_sq = 0
	for i in range(dimensions):
		square_diff = (point[i] - center[i]) ** 2
		sum_sq = sum_sq + square_diff
	return sqrt(sum_sq)
	
	
def update_membership_value(datatet, centers, c,m, U):
	for i in range(len(dataset)):		
		for j in range(len(centers)):
			dist = []
			for k in range(len(centers)):
				dist_value = distance(dataset[i], centers[k])
				dist.append(dist_value)
			
			_sum = 0.0
			for d in range(len(dist)):
				value = math.pow(dist[j]/dist[d], 2/(m-1))
				_sum = _sum + value
			U_value = 1 / float(_sum)
			U[i][j] = U_value
	return U
	
	
def calculate_cluster_center(U, dataset, m, c):
	zip_dataset =zip(*dataset)
	new_centers = []
	for i in range(c):
		denominator_value = 0.0
		for k in range(len(U)):
			p_sum = math.pow(U[k][i], m)
			denominator_value += p_sum
		current = [] #stores the value of the current cluster
		for m in range(len(zip_dataset)):
			_sum = 0.0
			for n in range(len(zip_dataset[0])):
				value = zip_dataset[m][n] * (math.pow(U[n][i], m)) 
				_sum = _sum+ value
			current.append(_sum)
		new_centers.append(current)
	return new_centers


"""
	This is the stopping condition, it happens when the U matrix stops changing too much with successive iterations.
	"""
def stopping_condition(U,U_old):	
	global Theta
	for i in range(0,len(U)):
		for j in range(0,len(U[0])):
			if abs(U[i][j] - U_old[i][j]) > Theta :
				return False
	return True

def fuzzy_means(dataset, c , m):
	U = initialise_U(dataset, c)
	centers = select_initial_centers(dataset, c)
	while True:
		old_U = copy.copy(U)
		U = update_membership_value(dataset, centers, c, m, old_U)
		stop = stopping_condition(U,old_U)
		if (stop == True):
			break
		centers = calculate_cluster_center(U, dataset, m, c)
	return U
	
	
filename="SPECTF.csv"
dataset = loadCsv(filename)
c = 2
m = 0.99
final_U = fuzzy_means(dataset, c, m)

assignments = []
for i in range(len(final_U)):
	index, value = max(enumerate(final_U[i]), key=operator.itemgetter(1))
	assignments.append(index)


cluster_count = list()
for i in range(1,c+1):
	cluster_count.append(0)
for p in range(len(assignments)):
	cluster_count[assignments[p]] = cluster_count[assignments[p]] + 1
print 'cluster_count',cluster_count
		
filename1 = 'SPECTF_New.csv'
dataset_new = loadCsv1(filename1)

count = {}
for i in range(c):
	yes_count = 0
	no_count = 0
	for j in range(len(assignments)):
		if(assignments[j] == i):
			if(dataset_new[j][-1] == 'Yes'):
				yes_count +=1
			else:
				no_count +=1
	count[i] = [yes_count, no_count]

print count 

for i in range(len(count)):
	index, value = max(enumerate(count[i]), key=operator.itemgetter(1))
	print index, value
	if index == 1:
		count[i].append('Yes')
	else:
		count[i].append('No')

print count

for i in range(len(count)):
	if(count[i][-1] == 'Yes'):
		TP = count[i][1]
	if(count[i][-1] == 'No'):
		TN = count[i][0]
	if(count[i][-1] == 'Yes'):
		FP = count[i][0]
	if(count[i][-1] == 'No' ):
		FN = count[i][1]
print TP,TN , FP, FN

#now to find TPR and TNR
TPR = float(TP)/(TP+FN)
print 'TPR', TPR
TNR = float(TN)/(TN+FP)
print 'TNR' ,TNR

accuracy = (TP+TN)/float(FP+FN+TP+TN) *100
print 'Accuracy', accuracy
