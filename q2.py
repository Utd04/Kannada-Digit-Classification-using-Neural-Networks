import sys
import pandas as pd
import numpy as np
import math
import time
import random
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

def sigmoid_activation(a):
    a1 = np.multiply(a >= 0, a)
    a2 = np.multiply(a < 0, a)
    return np.add(1/(1+np.exp(-a1)), np.divide(np.exp(a2), (1+np.exp(a2)))) - 0.5
def sigmoid_derivative(a):
    return np.multiply(sigmoid_activation(a), 1-sigmoid_activation(a))
def relu_activation(a):
    return np.multiply(a > 0, a)
def relu_derivative(a):
    return np.multiply(a > 0, np.ones(a.shape, dtype=float))

def RandomInit(layerinfo):
    np.random.seed(1)
    modelParameters = {}
    for l in range(1, len(layerinfo)):
        # can use np.random.rand() too 
        sizeOfMatrix = (layerinfo[l],layerinfo[l-1])
        modelParameters["Weight"+str(l)] = np.random.normal(0, 1, sizeOfMatrix)*np.sqrt(2.0/layerinfo[l-1])
        sizeOfMatrix = (layerinfo[l], 1)
        modelParameters["Bias"+str(l)] = np.zeros(sizeOfMatrix, dtype=float)
    return modelParameters

def prediction(modelParameters, data_x, activationFunction):
    forward_pass = {}
    x = np.transpose(data_x)
    for i in range((int)(len(modelParameters)/2)):
        x = np.dot(modelParameters["Weight"+str(i+1)], x) + modelParameters["Bias"+str(i+1)]
        forward_pass["z"+str(i+1)] = x
        x = sigmoid_activation(x)
        forward_pass["a"+str(i+1)] = x
    output = np.exp(forward_pass["a"+str((int)(len(modelParameters)/2))])
    summer = np.sum(output, axis=0)
    output = np.divide(output, summer)
    return np.argmax(output, axis=0)


def onehot(y):
    ret = []
    for i in range(10):
        if(y==i):
            ret.append(1)
        else:
            ret.append(0)
    return np.array(ret)

def gereralNeuralNetwork(trXpath,trYpath,teXpath,outputfile,batchSize,LayerStringInputArgument,activationFunction,adaptive):
    hiddenLayer = (LayerStringInputArgument.split())
    temporarylist = []
    for i in hiddenLayer:
        temporarylist.append((int(i)))

    hiddenLayer = temporarylist

    inputBoxes = 784
    ouputBoxes = 10
    learning_rate = 0.1
    currerror = 10
    Eps = 0.1
    tolerance = 0.0001
    CostStorage = []

    if(adaptive):
        learning_rate= 1
        print("adaptive")
    
    
    xTraining = np.load(trXpath)
    yTraining = np.load(trYpath)

    spvar = xTraining.shape
    newx = np.zeros((spvar[0], spvar[1]*spvar[2]), dtype=int)
    for i in range(spvar[0]):
        newlist = []
        for j in range(spvar[1]):
            for k in range(spvar[2]):
                newlist.append(xTraining[i][j][k])
        newx[i] = np.array(newlist)
    xTraining = newx
    newy = np.zeros((yTraining.shape[0],10))
    for i in range(yTraining.shape[0]):
        newy[i] = onehot(yTraining[i])
    yTraining = newy


    # CONVERTING THE TEST X
    xTest = np.load(teXpath)
    spvar = xTest.shape
    newx = np.zeros((spvar[0], spvar[1]*spvar[2]), dtype=int)
    for i in range(spvar[0]):
        newlist = []
        for j in range(spvar[1]):
            for k in range(spvar[2]):
                newlist.append(xTest[i][j][k])
        newx[i] = np.array(newlist)
    xTest = newx


    #  TEST PURPOSE
    # yTest = np.load("y_test.npy")
    # newy = np.zeros((yTest.shape[0],10))
    # for i in range(yTest.shape[0]):
    #     newy[i] = onehot(yTest[i])
    # yTest = newy



    xMatrix = np.asmatrix(xTraining)
    yMatrix = np.asmatrix(yTraining)

    xtestMatrix = np.asmatrix(xTest)
    # ytestMatrix = np.asmatrix(yTest)

    maxEpochs = 150
    epochs = 0

    m = len(xMatrix)

    totalBatches = (int)(m/batchSize)
    Boxes = [inputBoxes]
    Boxes.extend(hiddenLayer)
    Boxes.append(ouputBoxes)
    modelParameters = RandomInit(Boxes)

    while(True):
        if(Eps > currerror):
            break
        if(epochs> maxEpochs):
            break
        currerror = 0
        for batchIndex in range(totalBatches):
            begin = batchIndex*batchSize
            end = 0
            if batchIndex == totalBatches-1:
                end = m
            else:
                end = begin+batchSize
            currentX = xMatrix[begin:end,:]
            currentY = yMatrix[begin:end,:]

            forwardValues = {}
            layerCount =(len(modelParameters)//2)
            xt = np.transpose(currentX)
            forwardValues["a0"] = xt
            for i in range(layerCount-1):
                tempMat = modelParameters["Weight"+str(i+1)]
                temp2= np.dot(tempMat, xt)
                xt = temp2 + modelParameters["Bias"+str(i+1)]
                forwardValues["z"+str(i+1)] = xt
                
                if(activationFunction =="rlu"):
                    xt = relu_activation(xt)
                else:
                    xt = sigmoid_activation(xt)
                forwardValues["a"+str(i+1)] = xt
            tempMat2 = modelParameters["Weight"+str(layerCount)]
            temp3 = np.dot(tempMat2, xt) 
            xt = temp3 + modelParameters["Bias"+str(layerCount)]
            forwardValues["z"+str(layerCount)] = xt
            xt = sigmoid_activation(xt)
            forwardValues["a"+str(layerCount)] = xt

            trueOutput = np.transpose(currentY)
            value0 = forwardValues["a"+str((len(modelParameters)//2))]
            helpval = (forwardValues["a"+str((len(modelParameters)//2))] == 0)
            value1 = np.multiply(1,helpval)
            val = np.add(value0, value1)
            loss0 = np.multiply(trueOutput, np.log(val))
            value0  = 1 - forwardValues["a"+str((int)(len(modelParameters)/2))]
            helpval = (forwardValues["a"+str((len(modelParameters)//2))] == 1)
            value1 = np.multiply(1, helpval)
            val = np.add(value0, value1)
            loss1 = np.multiply(1-trueOutput, np.log(val))
            loss = np.add(loss0, loss1)
            loss = -1*loss
            averageLoss = np.mean(loss, axis=1)
            transposemat = np.transpose(averageLoss)
            magnitude = np.dot(transposemat, averageLoss)[0, 0]
            magnitude = np.sqrt(magnitude)
            
            
            derivativeStorage= {}
            finalParameters = {}
            shapeVal = trueOutput.shape[1]

            # # derivative for last layer of network
            help0 = forwardValues["a"+str((len(modelParameters)//2))]
            lastLayerDet =  help0- trueOutput
            derivativeStorage["der"+str((int)(len(modelParameters)/2))] = lastLayerDet
            for i in range((len(modelParameters)//2) - 1, 0, -1):
                
                if(activationFunction=="rlu"):
                    t0 = relu_derivative(forwardValues["z"+str(i)])
                else:
                    t0 = sigmoid_derivative(forwardValues["z"+str(i)])
                t2= derivativeStorage["der"+str(i+1)]
                t1 = np.transpose(modelParameters["Weight"+str(i+1)])
                f0 = np.dot(t1,t2)
                lastLayerDet = np.multiply(f0, t0)
                derivativeStorage["der"+str(i)] = lastLayerDet

            for i in range(1, (len(modelParameters)//2) + 1):
                t0 = forwardValues["a"+str(i-1)]
                t1 = np.transpose(t0)
                t2 = derivativeStorage["der"+str(i)]
                f0 = np.dot(t2, t1)
                val1 = (learning_rate/shapeVal)*f0
                par1 = modelParameters["Weight"+str(i)]
                finalParameters["Weight"+str(i)] = par1 - val1
                p0 = np.sum(t2, axis=1)
                val2 = (learning_rate/shapeVal)*p0
                par2 = modelParameters["Bias"+str(i)]
                finalParameters["Bias"+str(i)] = par2 - val2
            modelParameters = finalParameters
            currerror += ((float(magnitude))/(end-begin+1))
        CostStorage.append(currerror)
        epochs = epochs + 1
        if(adaptive):
            learning_rate = learning_rate/(math.sqrt(epochs))


    # print("TRAINING DATA")
    # trainingDataPerdiction =(prediction(modelParameters, xMatrix, activationFunction))
    # trainingDataPerdiction = np.transpose(trainingDataPerdiction)
    # a = (accuracy_score(np.argmax(yMatrix, axis=1), trainingDataPerdiction))

    # print("TEST DATA")
    testDataPrediction =(prediction(modelParameters, xtestMatrix, activationFunction))
    testDataPrediction = np.transpose(testDataPrediction)

    np.savetxt(outputfile, testDataPrediction, fmt="%d", delimiter="\n")



def partd(trXpath,trYpath,teXpath,outputfile):
    xTraining = np.load(trXpath)
    yTraining = np.load(trYpath)
    xTest = np.load(teXpath)
    spvar = xTraining.shape
    newx = np.zeros((spvar[0], spvar[1]*spvar[2]), dtype=int)
    for i in range(spvar[0]):
        newlist = []
        for j in range(spvar[1]):
            for k in range(spvar[2]):
                newlist.append(xTraining[i][j][k])
        newx[i] = np.array(newlist)
    xTraining = newx

    clf = MLPClassifier(hidden_layer_sizes=(100,100),solver='sgd')
    clf.fit(xTraining,yTraining)
    # trainingDataPerdiction = clf.predict(xTraining)
    # print(accuracy_score(yTraining, trainingDataPerdiction))
    # predict over the test set...
    spvar = xTest.shape
    newx = np.zeros((spvar[0], spvar[1]*spvar[2]), dtype=int)
    for i in range(spvar[0]):
        newlist = []
        for j in range(spvar[1]):
            for k in range(spvar[2]):
                newlist.append(xTraining[i][j][k])
        newx[i] = np.array(newlist)
    xTest = newx
    DataPerdiction = clf.predict(xTraining)




def main():
    trXpath = sys.argv[1]
    trYpath = sys.argv[2]
    teXpath = sys.argv[3]
    outputfile = sys.argv[4]
    batchSize = int(sys.argv[5])
    LayerStringInputArgument = sys.argv[6]
    activationFunction = sys.argv[7]

    adaptive = False

    
    
    gereralNeuralNetwork(trXpath,trYpath,teXpath,outputfile,batchSize,LayerStringInputArgument,activationFunction,adaptive)
    

    # partb(trXpath,trYpath,teXpath,outputfile,batchSize,LayerStringInputArgument,activationFunction)

    # partd(trXpath,trYpath,teXpath,outputfile)


    # code to plot the graphs
    # layermat = ["1", "10", "50", "100", "500"]
    # trainArr = [0.0]*5
    # testArr = [0.0]*5
    # for i in range(5):
    #     print(i)
    #     (a,b)= gereralNeuralNetwork(trXpath,trYpath,teXpath,outputfile,batchSize,layermat[i],activationFunction,adaptive)
    #     trainArr[i] = a
    #     testArr[i] = b
    # print(trainArr)
    # print(testArr)

    # plt.title("Accuracy vs Hidden Layer Units")
    # plt.plot(layermat, trainArr, label = 'Training')
    # plt.plot(layermat, testArr, label = 'Testing')

    # plt.xlabel("Hidden Layer Units")
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    main()
