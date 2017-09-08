import pandas as pd
import numpy as np
import math



def makeMatrix(I,J,fill=0.0):
    m=[]
    for i in range(I):
        m.append([fill]*J)
    return m

def randomizeMatrix(matrix,wei):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix[i][j]=wei

def sigmoid(x):
    return (float)(1.0/(float)(1.0+math.exp(-1*x)))

class NN:

    def __init__(self,NI,NH,NO,BH,BO):
        self.ni=NI+1
        self.nh=NH
        self.no=NO
        print self.ni
        print self.nh
        print self.no
        self.bh=BH
        self.bo=BO

        self.ai,self.ah,self.ao=[],[],[]
        self.ai=[1.0]*self.ni
        self.ah=[1.0]*self.nh
        self.ao=[1.0]*self.no

        self.TotAccuracy=0

        self.wi=makeMatrix(self.ni,self.nh)
        self.wo=makeMatrix(self.nh,self.no)

        wii=(float)(1.0/(float)(self.ni*self.nh+self.bh))
        randomizeMatrix(self.wi,wii)
        woo=(float)(1.0/(float)(self.nh+self.bo))
        randomizeMatrix(self.wo,woo)

    def ruNN(self,inputs):
        if len(inputs)!=self.ni-1:
            print 'Incorrect number of inputs'
        for i in range(self.ni-1):
            self.ai[i]=inputs[i]
        for j in range(self.nh):
            sum=0.0
            for i in range(self.ni):
                sum+=(self.ai[i]*self.wi[i][j])
            self.ah[j]=sigmoid(sum+self.bh)
        for k in range(self.no):
            sum=0.0
            for j in range(self.nh):
                sum+=(self.ah[j]*self.wo[j][k])
            self.ao[k]=sigmoid(sum+self.bo)
        return self.ao

    def backPropagate(self,targets,N):
        #print "targets ",targets
        #targets=np.array(targets)
        #print targets
        #print targets[0]
        output_deltas=[0.0]*self.no
        for k in range(self.no):
            #print 'target[',k,'] ',targets,' self.ao[',k,'] ',self.ao[k]
            error=targets-self.ao[k]
            output_deltas[k]=error*self.ao[k]*(1-self.ao[k])
        hidden_deltas=[0.0]*self.nh
        for j in range(self.nh):
            error=0.0
            for k in range(self.no):
                error+=output_deltas[k]*self.wo[j][k]
            hidden_deltas[j]=error*self.ah[j]*(1-self.ah[j])
        #update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change=output_deltas[k]*self.ah[j]
                self.wo[j][k]+=N*change
        #update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change=hidden_deltas[j]*self.ai[i]
                self.wi[i][j]+=N*change
        #update bias
        self.bo+=N*output_deltas[0]
        self.bh+=N*hidden_deltas[0]

        #calc conbined error
        #1/2 for differential convience and & **2 for modulus
        error=0.0
        #for k in range(len(targets)):
        error=0.5*(targets-self.ao[k])**2
        return error

    def test(self,X_test,y_test):
        t=0
        for p in range(0,len(X_test)):
            x = round(self.ruNN(X_test[p])[0],0)
            y = y_test[p]
            print x,' ',y
            if x == y:
                t+=1
          #  print 'Inputs ',X_test[p],'--> ',round(self.ruNN(X_test[p])[0],0),'\tTarget',y_test[p]    
        FoldAccuracy = t/11.0
        print 'foldAccuracy: ' , FoldAccuracy
        return FoldAccuracy

    def train(self,X_train,y_train,X_test,y_test):
        max_iterations=1000
        N=0.5
        for i in range(max_iterations):
            for p in range(0,len(X_train)):
                self.ruNN(X_train[p])
                error=self.backPropagate(y_train[p],N)
        #    if i%50 ==0:
         #       print "Combined error TRAINING",error
        FoldAccuracy = self.test(X_test,y_test)
        self.TotAccuracy+=FoldAccuracy
        x = self.TotAccuracy
        return x

def kfold(k,X,y):
    x_test,x_train,y_test,y_train=[],[],[],[]
    for i in range(0,len(X)):
        if i%10==k:
            x_test.append(X[i])
            y_test.append(y[i])
        else:
            x_train.append(X[i])
            y_train.append(y[i])
    return x_train,x_test,y_train,y_test

def main():
    df=pd.read_csv("SPECTF_New.csv")
    df=df.sample(frac=1)
    X=df.as_matrix()
    nr=df.shape[1]
    nor=df.shape[0]
    #print X
    y=X[:,nr-1]
    X=X[:,0:nr-1]
    #y=X[:,4]
    #X=X[:,0:4]
    #print X
    #print y
    y=[0.0 if x=='No' else 1.0 for x in y]
    #print y
    n=X.shape[1]
    #print n
    myNN=NN(n,5,1,5,1)
    for k in range(0,10):
        X_train, X_test, y_train, y_test=kfold(k,X,y)
        x=myNN.train(X_train,y_train,X_test,y_test)
    print 'tot Accu' , x/10.0
if __name__=="__main__":
    main()
