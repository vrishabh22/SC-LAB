#!/usr/bin/env python
import numpy as np
import csv
import itertools
weight=[0.2,]
for i in range(1,5):
    weight.append(0.2)
th=400
def kfold(initial,final):
    correct=0
    wrong=0
    with open('IRIS.csv') as csvfile:
        dataset = csv.reader(csvfile,delimiter=',')
        for row in dataset:
            col=len(row)
            break
        wt=(1/(float)(col))
        
        lr=0.1
        for row in itertools.islice(dataset,0,initial-1):
            res=0.0
            for i in range(0,col-1):
                d=float(row[i])
                #print weight[i],d
                res=res+weight[i]*d
            if res<=th:
                y="Iris-setosa"
            else:
                y="Iris-versicolor"
            if y==row[col-1]:
                err=0
            else:
                err=1
            for i in range(0,col-1):
                weight[i]=weight[i]+(lr*err*float(row[i]))
        for row in itertools.islice(dataset,final+1,101):
            res=0.0
            for i in range(0,col-1):
                d=float(row[i])
                res=res+weight[i]*d
            if res<=th:
                y="Iris-setosa"
            else:
                y="Iris-versicolor"
            if y==row[col-1]:
                err=0
            else:
                err=1
            for i in range(0,col-1):
                weight[i]=weight[i]+(lr*err*float(row[i]))
    with open('IRIS.csv') as csvfile:
        dataset = csv.reader(csvfile,delimiter=',')    
        for row in itertools.islice(dataset,initial,final):
            res=0
            for i in range(0,col-1):
                res=res+weight[i]*float(row[i])
            if res<=th:
                y="Iris-setosa"
            else:
                y="Iris-versicolor"
            if y==row[col-1]:
                correct+=1
                print y +"   Predict"
                print row[col-1] +"   Actual"
            else:
                wrong+=1
                print res
                print y +"   Predict"
                print row[col-1] +"   Actual"
            #print initial 
            #print final
            
        #print "right ",correct
        #print wrong , "wrong"
    return correct
c=0
for i in range(1,11):
    c+=kfold(10*(i-1)+1,10*i+1)
print c/100.0
