import numpy as np 
import torch 
import math

class Model():
    def __init__(self):
        self.Layers = [] # list of layer specs
        self.isTrain = True

    def addLayer(self, Layer):
        self.Layers.append(Layer)

    def forward(self, input):
        tot_layers = len(self.Layers)
        self.inputs = [None]*(tot_layers + 1)
        self.inputs[0] = input

        for i in range(tot_layers):
            self.inputs[i+1] = self.Layers[i].forward(self.inputs[i])
        return self.inputs[tot_layers]

    def backward(self, input, gradOutput):
        current_grad = gradOutput.clone()
        tot_layers = len(self.Layers)

        ub = tot_layers -1 
        lb = -1 
        decr = -1 

        for i in range(ub, lb, decr):
            current_grad = (self.Layers[i]).backward(self.inputs[i], current_grad)
        
        return 

    def dispGradParam(self):
        tot_layers = len(self.Layers)

        ub = tot_layers -1 
        lb = -1 
        decr = -1

        for i in range(ub, lb, decr):
            print("##Layer %d"%(i))
            print(self.Layers[i])

            if(self.Layers[i].isTrainale):
                self.params(self.Layers[i])

    def clearGradParam(self):
        for l in self.Layers:
            # l.clear_grad()
            l.gradInput = 0
            l.gradWeight = 0	
            l.gradBias = 0
    
    def params(self, Layer):
        print("## Weight ", Layer.weight)
        print("## bias ", Layer.bias)
    
    # def saveMeanVariance(self, mean, variance):
    #     self.dataMean = mean
    #     self.dataVariance = variance
    def regularization_loss(self,regularization):
        reg = 0
        for i in self.Layers:
            if (i.isTrainable):
                reg += regularization * i.weights_norm()
        return reg			
