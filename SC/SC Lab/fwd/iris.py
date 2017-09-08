#!/usr/bin/env python
import numpy as np
import csv

with open('IRIS.csv') as csvfile:
    dataset = csv.reader(csvfile,delimiter=',')
    c=0
    for row in dataset:
        col=len(row)
        break
    wt=(1/(float)(col))
    weight=[wt,]
    for i in range(1,col-1):
        weight.append(wt)
    lr=0.5
    count=0
    for row in dataset:
        count=count+1
        if count==90:
            break
        res=0.0
        for i in range(0,col-1):
            d=float(row[i])
            #print type(row[i])
            res=res+weight[i]*d
        print res
        if res<=500:
            y="Iris-setosa"
        else:
            y="Iris-versicolor"
        if y==row[col-1]:
            err=0
        else:
            err=1
        for i in range(0,col-1):
            weight[i]=weight[i]+(lr*err*float(row[i]))
    for row in dataset:
        res=0
        for i in range(0,col-1):
            res=res+weight[i]*float(row[i])
        print res
        if res<=500:
            y="Iris-setosa"
        else:
            y="Iris-versicolor"
        print y +"   predict"
        print row[col-1] +"   Actual"
        #c=0
        if y==row[col-1]:
        	c=c+1
    print c*10   
     

        
        
