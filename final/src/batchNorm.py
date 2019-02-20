import numpy
import torch
from math import sqrt

momentum = 0.8

class BatchNorm(object):
	"""docstring for BatchNorm"""
	def __init__(self, input_dim,isTrainable=True,momentum=0.9,weight=1,bias=0):
		super(BatchNorm, self).__init__()
		self.isTrainable = isTrainable
		self.running_mean = torch.zeros(input_dim).type(torch.DoubleTensor)
		self.running_var = torch.ones(input_dim).type(torch.DoubleTensor)
		self.momentum = momentum
		if ((type(weight)==type(1)) and weight==1):
			weight = self.running_var
		if ((type(bias)==type(0)) and bias==0):
			bias = self.running_mean	
		self.weight = weight
		self.bias=bias
		self.momentumWeight = torch.zeros(self.weight.size()).type(torch.DoubleTensor)
		self.momentumBias = torch.zeros(self.bias.size()).type(torch.DoubleTensor)
	def forward(self,input,mode='train'):
		#if (mode=='train'):
		#elif (mode=='test'):
		#return (self.weight*self.input_hat + self.bias)
		return (self.weight * (input - self.running_mean) / torch.sqrt(self.running_var + 1e-100) + self.bias)
		#else :
		#return "No such mode :/ "	
	def backward(self,input,gradOutput):
		self.sample_mean = input.mean(dim=0)
		self.sample_var = input.var(dim=0,unbiased=False)
		input_hat = (input - self.sample_mean) / torch.sqrt(self.sample_var + 1e-100)		
		self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.sample_mean
		self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.sample_var
		self.input_hat = input_hat
		
		N = input.shape[0]
		self.gradBias = gradOutput.sum(dim=0)
		self.gradWeight = torch.sum(gradOutput * self.input_hat,dim=0)
		gradVar = (gradOutput*self.weight*(input-self.sample_mean)*-0.5*torch.pow(self.sample_var + 1e-100,-1.5)).sum(dim=0)
		gradMean1 = -1*(gradOutput*self.weight/torch.sqrt(self.sample_var + 1e-100)).sum(dim=0)
		gradMean2 = gradVar * (-2 * (input-self.sample_mean)).sum(dim=0)
		gradMean = gradMean1 + gradMean2
		#print(gradMean.type())
		gradInput1 = (gradOutput*self.weight/torch.sqrt(self.sample_var + 1e-100))
		gradInput2 = (torch.ones(input.size())).type(torch.DoubleTensor) * gradMean 
		gradInput2 = gradInput2/N  
		gradInput3 = (input-self.sample_mean) * gradVar * 2./N
		gradInput = gradInput1 + gradInput2 + gradInput3
		self.momentumWeight = momentum*self.momentumWeight + (1- momentum)*self.gradWeight
		self.momentumBias = momentum*self.momentumBias + (1- momentum)*self.gradBias
		return gradInput
	def clear_grad(self):
		self.gradbias = 0
		self.gradweight = 0
		return	
	def reset_running_pars(self):
		self.running_mean = torch.zeros(input_dim).mean(dim=0).type(torch.DoubleTensor)
		self.running_var = torch.ones(input_dim).mean(dim=0).type(torch.DoubleTensor)
	def __str__(self):
		string = "This is a batch normalisation layer with input dimension = " + str(self.bias.size())
		return string
	def print_param(self):
		print("The gamma matrix (that counteracts the variance) is: ")
		print(self.weight)
		print("Gamma has a mean value of " + str(self.weight.mean()))
		print("")
		print("The beta matrix (that counteracts the mean) is: ")
		print(self.bias)
		print("Beta has a mean value of " + str(self.bias.mean()))
		return 
	def weights_norm(self):
		return torch.norm(self.weight) + torch.norm(self.bias)		