import torchfile as trf 
import matplotlib.pyplot as plt 
import torch
import sys
sys.path.append('src')
from Model import *
from Criterion import *
from Linear import *
from ReLU import *
from Conv2D import *
import argparse
import os

import src.ReLU as rl 



import importlib

importlib.reload(rl)

batchSize = 128
plotIndex = 0
losses = []
plotIndices = []
lossClass = Criterion()
learningRate = 1e-6
data = None
labels = None
reg = 1e-3 
batchSize = 64
dataSize = 0
model = None

def disp(x,shape=False):
    print("")
    print('##: ', str(x))
    print("#type :", type(x))
    if shape:
        print("#shape: ", x.shape)



def init_model():
        model = Model()
        model.addLayer(Linear(108*108, 1000))
        model.addLayer(ReLU())
        model.addLayer(Linear(1000, 6))
        # model.addLayer(ReLU())
        return model

def process_data(path_data, path_labels):
        global data, labels, sz
        path_labels = "./data/labels.bin"
        path_data = "./data/data.bin"


        # loading as numpy array
        l = trf.load(path_labels)
        d = trf.load(path_data)


        # converting numpy arrays into tensors
        _l = torch.from_numpy(l)
        _d = torch.from_numpy(d)
        _data = _d.contiguous().view(_d.size()[0], -1).type(torch.DoubleTensor)
        _labels = _l.type(torch.DoubleTensor)
	
        data = _data[:]
        labels = _labels[:]

        sz = data.size()[0]

        # normalizing the data
        mean = data.mean(dim=0)
        std = data.std(dim=0, keepdim=True)
        data = (data - mean)/std

        return mean, std


def train(model,lossClass,iterations, whenToPrint, batchSize, learningRate, par_regularization):
	global dataSize, plotIndex, losses, plotIndices, labels, data
	dataSize = data.size()[0]
	for i in range(iterations):
		indices = (torch.randperm(dataSize)[:batchSize]).numpy()
		currentData = data[indices, :]
		currentLabels = labels.view(dataSize, 1)[indices, :]
		yPred = model.forward(currentData)
		lossGrad, loss = lossClass.backward(yPred, currentLabels)
		if i%whenToPrint == 0:
			reg_loss = model.regularization_loss(par_regularization)
			print("Iter - %d : Training-Loss = %.4f Regularization-Loss = %.4f and Total-loss = %.4f"%(i, loss,reg_loss,loss+reg_loss))
			#losses.append(loss)
			#plotIndices.append(plotIndex)

		model.clearGradParam()
		model.backward(currentData, lossGrad)
		for layer in model.Layers:
			if layer.isTrainable:
				layer.weight -= learningRate*((1-momentum)*layer.gradWeight + momentum*layer.momentumWeight) + par_regularization*layer.weight
				layer.bias -= learningRate*((1-momentum)*layer.gradBias + momentum*layer.momentumBias) + par_regularization*layer.bias
				#layer.weight -= (learningRate*layer.gradWeight + par_regularization*layer.weight)
				#layer.bias -= (learningRate*layer.gradBias + par_regularization*layer.bias)
		if i%(whenToPrint*10) == 0:
			print(trainAcc())	
		plotIndex += 1

def trainAcc():
        global model, data, label
        yPred = model.forward(data)
        N = data.size()[0]
        acc = (yPred.max(dim=1)[1].type(torch.LongTensor) == labels.type(torch.LongTensor)).sum()/N
        print(acc)
        return acc

def trainModel():
	global model, batchSize, reg, learningRate, lossClass
	iterations_count = 128*500//batchSize
	lr_decay_iter = iterations_count//8
	reg_zero = 2*iterations_count//10

	for i in range(5):
		train(model,lossClass,lr_decay_iter,10, batchSize ,learningRate, reg)
		learningRate /= 10
		reg/=10
		print(trainAcc())
	return 

def saveModel(fileToSave):
	global model
	file = open(fileToSave, 'wb')
	torch.save(model, file)
	file.close()
	return 



if __name__ == "__main__":
        global model, batchSize, reg, learningRate, lossClass
        parser = argparse.ArgumentParser()
        parser.add_argument("-modelName", help="input model name")
        parser.add_argument("-data",help="path to data.bin")
        parser.add_argument("-target",help="path to labels.bin")

        args = parser.parse_args()
        
        print("==================input args===================")
        print(args.modelName)
        print(args.data)
        print(args.target)

        print("==========INIT Model========================")
        model = init_model()

        print("Mode Initialised")

        mean, std = process_data(args.data, args.target)

        print("#### Training")
        # trainModel()
        
        iterations_count = 128*500//batchSize
        lr_decay_iter = iterations_count//8
        reg_zero = 2*iterations_count//10

        for i in range(5):
                train(model,lossClass,lr_decay_iter,10, batchSize ,learningRate, reg)
                learningRate /= 10
                reg/=10
                print(trainAcc())
 


        print("#### Saving the model")
        cmd = "mkdir -p " + args.modelName
        os.system(cmd)

        fname = args.modelName + "/model.bin"
        saveModel(fname)




    
