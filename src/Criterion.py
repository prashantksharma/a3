import torch
import numpy as np
import math

class Criterion():
	def __init__(self):
		# self.out_vect=None
		# self.pred_out=None
		pass

	def forward(self, ip, target):
		# Target contains the desired ouput
		# ip is the output labels predicted by last Layer
		pred_out = target.view(ip.size()[0]).numpy()
		# Finding the location of desired label in the predicted output labels
		out_vect = (torch.zeros(ip.size())).type(torch.DoubleTensor)
		
		out_vect[np.arange(ip.size()[0]), pred_out] = 1
		# Removing the maximum from individual output vector so the exponentials do not blow-up
		numerator = (ip - torch.max(ip, dim=1, keepdim=True)[0]).exp()
		# finding cross entropy loss using softtmax 
		logsfmax = -((numerator/numerator.sum(dim = 1, keepdim = True)).log())*out_vect
		loss = (logsfmax.sum())/float(ip.size()[0])

		return loss

	def backward(self, ip, target):
		# Target contains the desired ouput
		# ip is the output labels predicted by last Layer
		pred_out = target.view(ip.size()[0]).numpy()
		# Finding the location of desired label in the predicted output labels
		out_vect = (torch.zeros(ip.size())).type(torch.DoubleTensor)
		out_vect[np.arange(ip.size()[0]), pred_out] = 1
		numerator_grad = (ip - torch.max(ip, dim=1, keepdim=True)[0]).exp()
		fin_num = numerator_grad/numerator_grad.sum(dim=1,keepdim=True)
		grad = fin_num - out_vect
		grad = grad/float(ip.size()[0])

		return grad