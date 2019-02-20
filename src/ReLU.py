import numpy as np 
import torch
import math


class ReLU():
    def __init__(self):
        # super().__init__()
        self.isTrainable = False
        

    def forward(self, input):
        input[input < 0] = 0
        self.output = input
        return self.output

    def backward(self, input, gradOutput):
        # mask = input > 0
        # dx = gradOutput* (mask.type(torch.DoubleTensor))
        # self.gradInput = dx
        # self.gradInput
        self.gradInput = gradOutput.clone()
        self.gradInput[self.output==0] = 0
        return self.gradInput

    def __str__(self):
        return "### ReLU"

    def clear_grad(self):
        self.gradInput = 0
        return