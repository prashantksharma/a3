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
import numpy as np

import src.ReLU as rl 



import importlib

importlib.reload(rl)


learningRate = 1e-3
data = None
labels = None
reg = 1e-3 
batchSize = 64
dataSize = 0
# momemtum params
beta1 = 0.9
beta2 = 0.999
eps = 1e-8

configW = None
configB = None


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
        # path_labels = "./data/labels.bin"
        # path_data = "./data/data.bin"


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
        # global dataSize, plotIndex, losses, plotIndices, labels, data, configW, configB, beta1, beta2, eps
        global labels
        dataSize = data.size()[0]

        for i in range(1,iterations):
                indices = (torch.randperm(dataSize)[:batchSize]).numpy()
                currentData = data[indices, :]
                currentLabels = labels.view(dataSize, 1)[indices, :]
                yPred = model.forward(currentData)
                lossGrad= lossClass.backward(yPred, currentLabels)
                loss = lossClass.forward(yPred, currentLabels)
                if i%whenToPrint == 0:
                        reg_loss = model.regularization_loss(par_regularization)
                        print("Iteration (%d / %d) : loss = %.5f reg loss = %.5f , tot loss = %.5f" % (i, iterations,loss,reg_loss,loss+reg_loss))
                        #losses.append(loss)
                        #plotIndices.append(plotIndex)

                model.clearGradParam()
                model.backward(currentData, lossGrad)
                for layer in model.Layers:
                        status = layer.canTrain
                        if status:
                                w = layer.weight
                                dw = layer.gradWeight
                                b = layer.bias
                                db = layer.gradBias

                                #momentum update
                                np_w = w.numpy()
                                np_dw = dw.numpy()

                                np_b = b.numpy()
                                np_db = db.numpy()

                                # print("========== Layer info=============")
                                
                                # print("### before w ", np_w.shape)
                                # print("### before dw ", np_dw.shape)
                                
                                layer.weight, layer.configW = layer.momentum_update(np_w, np_dw, layer.configW)
                                # print("### before  b ", np_b.shape)
                                # print("### before db ", np_db.shape)
                                layer.bias, layer.configB = layer.momentum_update(np_b, np_db, layer.configB)

                if i%(whenToPrint*10) == 0:
                        print(accuracy())	
                # plotIndex += 1

def accuracy():
        global model, data, label
        yPred = model.forward(data)
        N = data.size()[0]
        acc = (yPred.max(dim=1)[1].type(torch.LongTensor) == labels.type(torch.LongTensor)).sum()/N
        print(acc)
        return acc

def saveModel(fileToSave):
	global model
	file = open(fileToSave, 'wb')
	torch.save(model, file)
	file.close()
	return 



if __name__ == "__main__":
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

        print("Model Initialised")

        mean, std = process_data(args.data, args.target)

        print("#### Training")
        

        tot_itr = int(128*500/batchSize)
        iter_epoch = int(tot_itr/8)
        # reg_zero = int(2*tot_itr/10)
        step_size = 10
        print("# iter per epoch ", iter_epoch)

        lossClass = Criterion()
        for i in range(5):
                train(model,lossClass,iter_epoch,step_size, batchSize ,learningRate, reg)
                learningRate /= 10
                reg/=10
                print(accuracy())
 


        print("#### Saving the model")
        cmd = "mkdir -p " + args.modelName
        os.system(cmd)

        fname = args.modelName + "/model.bin"
        saveModel(fname)




    
