import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Reading and visualizing data using scatter plot
CSV_Data = pd.read_csv("D:\canada_per_capita_income.csv", header=None)
CSV_Data = CSV_Data.replace(np.NaN,0)
X = CSV_Data.iloc[:,0]
Y = CSV_Data.iloc[:,1]

X = np.array(X)
Y = np.array(Y)
X = X[:,np.newaxis]
Y = Y[:,np.newaxis]

mnX = np.min(X)
mxX = np.max(X)
temp = mxX-mnX

CopyX = (X - mnX)/temp

mnY = np.min(Y)
mxY = np.max(Y)
temp1 = mxY-mnY

CopyY = (Y - mnY)/temp1

m,col = CopyX.shape
ones = np.ones((m,1))
CopyX = np.hstack((ones,CopyX))

x2 = np.power(CopyX[:,1],2)
x3 = np.power(CopyX[:,1],3)
x4 = np.power(CopyX[:,1],4)
x5 = np.power(CopyX[:,1],5)
x6 = np.power(CopyX[:,1],6)
x7 = np.power(CopyX[:,1],7)
x8 = np.power(CopyX[:,1],8)
x9 = np.power(CopyX[:,1],9)
x10 = np.power(CopyX[:,1],10)
x11 = np.power(CopyX[:,1],11)
x12 = np.power(CopyX[:,1],12)
x13 = np.power(CopyX[:,1],13)
x14 = np.power(CopyX[:,1],14)
x15 = np.power(CopyX[:,1],15)
x16 = np.power(CopyX[:,1],16)
x17 = np.power(CopyX[:,1],17)
x18 = np.power(CopyX[:,1],18)
x19 = np.power(CopyX[:,1],19)
x20 = np.power(CopyX[:,1],20)
x21 = np.power(CopyX[:,1],21)
x22 = np.power(CopyX[:,1],22)
x23 = np.power(CopyX[:,1],23)
x24 = np.power(CopyX[:,1],24)
x25 = np.power(CopyX[:,1],25)
x26 = np.power(CopyX[:,1],26)
x27 = np.power(CopyX[:,1],27)
x28 = np.power(CopyX[:,1],28)
x29 = np.power(CopyX[:,1],29)
x30 = np.power(CopyX[:,1],30)

x2 = x2[:,np.newaxis]

x3 = x3[:,np.newaxis]

x4 = x4[:,np.newaxis]

x5 = x5[:,np.newaxis]

x6 = x6[:,np.newaxis]

x7 = x7[:,np.newaxis]

x8 = x8[:,np.newaxis]

x9 = x9[:,np.newaxis]

x10 = x10[:,np.newaxis]

x11 = x11[:,np.newaxis]

x12 = x12[:,np.newaxis]

x13 = x13[:,np.newaxis]

x14 = x14[:,np.newaxis]

x15 = x15[:,np.newaxis]

x16 = x16[:,np.newaxis]

x17 = x17[:,np.newaxis]

x18 = x18[:,np.newaxis]

x19 = x19[:,np.newaxis]

x20 = x20[:,np.newaxis]

x21 = x21[:,np.newaxis]

x22 = x22[:,np.newaxis]

x23 = x23[:,np.newaxis]

x24 = x24[:,np.newaxis]

x25 = x25[:,np.newaxis]

x26 = x26[:,np.newaxis]

x27 = x27[:,np.newaxis]

x28 = x28[:,np.newaxis]

x29 = x29[:,np.newaxis]

x30 = x30[:,np.newaxis]

CopyX = np.hstack((CopyX,x2))
CopyX = np.hstack((CopyX,x3))
CopyX = np.hstack((CopyX,x4))
CopyX = np.hstack((CopyX,x5))
CopyX = np.hstack((CopyX,x6))
CopyX = np.hstack((CopyX,x7))
CopyX = np.hstack((CopyX,x8))
CopyX = np.hstack((CopyX,x9))
CopyX = np.hstack((CopyX,x10))
CopyX = np.hstack((CopyX,x11))
CopyX = np.hstack((CopyX,x12))
CopyX = np.hstack((CopyX,x13))
CopyX = np.hstack((CopyX,x14))
CopyX = np.hstack((CopyX,x15))
CopyX = np.hstack((CopyX,x16))
CopyX = np.hstack((CopyX,x17))
CopyX = np.hstack((CopyX,x18))
CopyX = np.hstack((CopyX,x19))
CopyX = np.hstack((CopyX,x20))
CopyX = np.hstack((CopyX,x21))
CopyX = np.hstack((CopyX,x22))
CopyX = np.hstack((CopyX,x23))
CopyX = np.hstack((CopyX,x24))
CopyX = np.hstack((CopyX,x25))
CopyX = np.hstack((CopyX,x26))
CopyX = np.hstack((CopyX,x27))
CopyX = np.hstack((CopyX,x28))
CopyX = np.hstack((CopyX,x29))
CopyX = np.hstack((CopyX,x30))


theta = np.zeros((31,1))

iterations = 8000
alpha = 0.01

# Cost Fucction 

def Get_cost_J(X,Y,Theta,lamda):
    
    Pridictions = np.dot(X,Theta)
    Error = Pridictions-Y
    SqrError = np.power(Error,2)
    SumSqrError = np.sum(SqrError)
    J  = (1/2*m)*SumSqrError
    reg = (lamda/2*m)*np.sum(np.power(theta,2))
    J = J+reg
    return J
# Grdient Decent

def gradient_decent(X,Y,theta,alpha,iterations,m, lamda):
    history = np.zeros((iterations,1))
    for i in range(iterations):
        Predictions =np.dot(X,theta)         
        Error = Predictions - Y
        theta = theta - alpha*(((np.dot(X.T,Error))/m) + (lamda/m*theta))        
        history[i] = Get_cost_J(X, Y, theta,lamda)   
    return (theta,history)


theta,hist = gradient_decent(CopyX, CopyY, theta, alpha, iterations, m,1) 

pridictions = np.dot(CopyX,theta)

pridictions = pridictions*temp1+mnY

plt.scatter(X,Y)
plt.plot(X,pridictions, color='red')
plt.show()