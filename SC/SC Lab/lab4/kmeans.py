import csv
import random
from random import randint
from math import sqrt
from collections import defaultdict
import operator

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


#randomly selecting k initial centers
def select_initial_k_centers(dataset,k):
	no_of_ele = len(dataset)
	init_centers = list()
	index = list()
	for i in range(k):
		index.append(randint(0,no_of_ele))
	print 'initial index' ,index
	for i in range(len(index)):
		init_centers.append(dataset[index[i]])
	print 'initial_centers', init_centers
	return init_centers
	
#assign points to the cluster
def assign_points(dataset, centers):
	cluster_assignments = []
	for point in dataset:
		shortest_dist = 999999999  # positive infinity
		shortest_index = 0
		for i in range(len(centers)):
			dist = distance(point, centers[i])
			if dist < shortest_dist:
				shortest_dist = dist
				shortest_index = i
		cluster_assignments.append(shortest_index)
	print 'cluster_assignments',cluster_assignments
	return cluster_assignments

#finds the centroid of the set of points
def average_of_points(points):
    
    dimensions = len(points[0])

    new_center = []

    for dimension in xrange(dimensions):
        dim_sum = 0  # dimension sum
        for p in points:
            dim_sum += p[dimension]

        # average of each dimension
        new_center.append(dim_sum / float(len(points)))

    return new_center
	
def update_centers(data_set, assignments):
 
    new_means = defaultdict(list)
    centers = []
    for assignment, point in zip(assignments, data_set):
        new_means[assignment].append(point)
        
    for points in new_means.itervalues():
        centers.append(average_of_points(points))

    return centers

def distance(x, y):
	dimensions = len(x)
	sum_sq = 0
	for i in range(dimensions):
		square_diff = (x[i] - y[i]) ** 2
		sum_sq = sum_sq + square_diff
	return sqrt(sum_sq)


def kmeans(dataset, k):
	k_points = select_initial_k_centers(dataset, k)
	assignments = assign_points(dataset, k_points)
	old_assignments = None
	#continue until assignments do not change
	while assignments != old_assignments:
		new_centers = update_centers(dataset, assignments)
		old_assignments = assignments
		assignments = assign_points(dataset, new_centers)
	return assignments
	
filename="SPECTF.csv"
dataset = loadCsv(filename)
k = 2
assignments = kmeans(dataset, k)
print assignments

cluster_count = list()
for i in range(1,k+1):
	cluster_count.append(0)
for p in range(len(assignments)):
	cluster_count[assignments[p]] = cluster_count[assignments[p]] + 1
print 'cluster_count',cluster_count
		
filename1 = 'SPECTF_New.csv'
dataset_new = loadCsv1(filename1)

count = {}
for i in range(k):
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
