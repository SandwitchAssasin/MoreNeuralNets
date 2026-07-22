import numpy as np
import math
import matplotlib.pyplot as plt
import NeurNetsFromBeginNUMPY as NeurNFN



'''
databs = [[(4,2),1],[(1,1),1],[(1.5,1.6),1],[(1.2,1.4),1],
          [(3.4,4),1],[(0.3,0.5),0],[(1.9,0.1),0],[(0.2,4.4),0],[(1.5,3.4),0],
          [(4,1.4),0],[(3,3),0]]
for dt in databs:
    dt[1] = [dt[1]]

l1_2 = NeurNFN.Dense(10,'sigmoid')
l2_2 = NeurNFN.Dense(15,'linear')
lds = NeurNFN.BatchNormalization()
ld2s = NeurNFN.LeakyReLU_L(0.1)
l3_2 = NeurNFN.Dense(15,'linear')
ld = NeurNFN.BatchNormalization()
ld2 = NeurNFN.LeakyReLU_L(0.1)
l4_2 = NeurNFN.Dense(1,'sigmoid')
lay = [l1_2,l2_2,lds,ld2s,l3_2,ld,ld2, l4_2]

m = NeurNFN.Model(2, lay)
m.Train(databs,epochs = 3000, learning_rate = 0.15, batch_size=8)
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

plt.scatter(X,Y,c=c,cmap='coolwarm')
plt.show()
'''


#Real project - Irises

def BladPredykcji(pred, real):
    suma = 0
    for i in range(0,len(pred[0])):
        suma = suma + (pred[0][i] - real[i])**2
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
#print(data)
dataTrain = []
dataTest = []
for d in data:
    r = np.random.uniform(0,1)
    if r < 0.8:
        dataTrain.append(d)
    else:
        dataTest.append(d)

#Mozliwe ze dziala, tylko po prostu nie potrafi poprawnie predictowac

l1_2 = NeurNFN.Dense(30,'linear')
ld1 = NeurNFN.BatchNormalization()
ld1p = NeurNFN.LeakyReLU_L(0.1)
l2_2 = NeurNFN.Dense(70,'linear')
ld2 = NeurNFN.BatchNormalization()
ld2p = NeurNFN.LeakyReLU_L(0.1)
l3_2 = NeurNFN.Dense(70,'linear')
ld3 = NeurNFN.BatchNormalization()
ld3p = NeurNFN.LeakyReLU_L(0.1)
l4_2 = NeurNFN.Dense(3,'softmax')

lay = [l1_2,ld1,ld1p,l2_2,ld2,ld2p,l3_2,ld3,ld3p,l4_2]

'''
l1_2 = NeurNFN.Dense(15,'linear',0.1)
ld21 = NeurNFN.LeakyReLU_L(0.1)
l2_2 = NeurNFN.Dense(17,'linear',0.1)
ld22 = NeurNFN.LeakyReLU_L(0.1)
l3_2 = NeurNFN.Dense(15,'linear',0.1)
ld23 = NeurNFN.LeakyReLU_L(0.1)
l4_2 = NeurNFN.Dense(3,'softmax')
lay = [l1_2,ld21,l2_2,ld22,l3_2,ld23,l4_2]
'''
print('KETE', len(dataTrain))
np.random.shuffle(dataTrain)
m2 = NeurNFN.Model(4, lay)
m2.Train(dataTrain,epochs = 250, learning_rate = 0.002, batch_size=50)


for d in dataTest:
    predictions = m2.Forward(d[0])
    real = d[1]
    print(irises_res[np.argmax(real)], ":", irises_res[np.argmax(predictions)], ":" , np.round(BladPredykcji(predictions,real),2))
    