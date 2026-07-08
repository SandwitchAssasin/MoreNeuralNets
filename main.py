import numpy as np
import math
import matplotlib.pyplot as plt
import NeurNetsFromBeginNUMPY as NeurNFN
'''
m = NeurNFN.Model(2, (4,6,1))
print(m.Forward((4,2)))
databs = [[(4,2),1],[(1,1),1],[(1.5,1.6),1],[(1.2,1.4),1],
          [(3.4,4),0],[(0.3,0.5),0],[(1.9,0.1),0],[(0.2,4.4),0],[(1.5,3.4),0],
          [(4,1.4),0],[(4,3),0]]
for dt in databs:
    dt[1] = [dt[1]]

m.Train(databs, 4000, 0.03)
plt.subplot(1,2,1)
for data in databs:
    colr = 'blue'
    if(data[1][0] == 1):
        colr = 'red'
    plt.plot(data[0][0],data[0][1],marker='o',color=colr)
plt.subplot(1,2,2)
X = np.linspace(-1,5,1000)
Y = np.random.uniform(0,4.5,1000)
c = []
for cint in range(0,1000):
    c.append(m.Forward((X[cint],Y[cint])))

print(m.Forward((4,2)))
plt.scatter(X,Y,c=c,cmap='coolwarm')
plt.show()
'''

#Real project - Irises

def BladPredykcji(pred, real):
    suma = 0
    for i in range(0,len(pred)):
        suma = suma + (pred[i] - real[i])**2
    return suma/len(pred)

with open('iris.txt') as f:
    irises = {'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}
    irises_res = {0:'Iris-setosa',1:'Iris-versicolor',2:'Iris-virginica'}
    data =[]

    lines = f.readlines()
    for line in lines:
        boxes = line.rstrip('\n').split(',')
        floatline = []
        dataline = []

        floatline.append((float)(boxes[0])/10)
        floatline.append((float)(boxes[1])/10)
        floatline.append((float)(boxes[2])/10)
        floatline.append((float)(boxes[3])/10)
        dataline.append(tuple(floatline))
        irisL = [0,0,0]
        mI = irises[boxes[4]]
        irisL[mI] = 1
        dataline.append(irisL)
        data.append(dataline)
print(data)
dataTrain = []
dataTest = []
for d in data:
    r = np.random.uniform(0,1)
    if r < 0.8:
        dataTrain.append(d)
    else:
        dataTest.append(d)

l1_2 = NeurNFN.DenseLayer(12,'linear')
ld1 = NeurNFN.LayerNormalization()
ld21 = NeurNFN.ReLU_L()
l2_2 = NeurNFN.DenseLayer(40,'linear')
ld2 = NeurNFN.LayerNormalization()
ld22 = NeurNFN.ReLU_L()
l3_2 = NeurNFN.DenseLayer(40,'linear')
ld3 = NeurNFN.LayerNormalization()
ld23 = NeurNFN.ReLU_L()
l4_2 = NeurNFN.DenseLayer(3,'softmax')
lay = [l1_2,ld1,ld21,l2_2,ld2,ld22,l3_2,ld3,ld23,l4_2]
m2 = NeurNFN.Model(4, lay)
m2.Compile()

m2.Train(dataTrain,epochs = 1200, learning_rate = 0.003)



for d in dataTest:
    predictions = m2.Forward(d[0])
    real = d[1]
    print(irises_res[np.argmax(real)], ":", irises_res[np.argmax(predictions)], ":" , round(BladPredykcji(predictions,real)[0],2))
