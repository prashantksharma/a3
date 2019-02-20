import numpy as np 
import torch
import math


class ReLU():
    def __init__(self):
        self.canTrain = False
        self.ip = None
        self.op = None
        self.ip_grad = None


    def forward(self, ip):

        self.op = ip.clone()
        self.op[self.op < 0] = 0
        return self.op

    def backward(self, ip, op_grad):

        self.ip_grad = op_grad.clone()
        self.ip_grad[self.op==0] = 0

        return self.ip_grad