import numpy as np
import torch
from math import sqrt
from torch.distributions import normal



class Linear():
	def __init__(self,no_input_neuron,no_output_neuron):

		m = normal.Normal(0,sqrt(2/(no_input_neuron+no_output_neuron)))
		self.canTrain = True
		self.weight = m.sample((no_input_neuron,no_output_neuron)).type(torch.DoubleTensor)
		self.bias = m.sample((1,no_output_neuron)).type(torch.DoubleTensor)
		self.dim = no_output_neuron
		self.configW = None
		self.configB = None
		self.gradInput = None
		self.gradWeight = None
		self.gradBias  = None
		return
	def forward(self,ip):
		# multiplying inputs with weight matrix
		ip_w = ip.mm(self.weight)
		# adding bias
		self.op = ip_w + self.bias
		return self.op

	def backward(self, input, gradOutput):
		# calculating transpose
		w_T = self.weight.t()
		ip_T = input.t()

		# calculating gradient of inputs 
		self.gradInput = gradOutput.mm(w_T)
		# calculating grad of weight
		self.gradWeight = ip_T.mm(gradOutput)
		# calculating grad of bias
		sm = gradOutput.sum(dim=0)
		# bias_dim = self.bias.shape[0]
		self.gradBias = sm.view(1, self.dim)

		return self.gradInput

	def momentum_update(self, w, dw, config=None):
		# Reference: 
		# http://cs231n.github.io/neural-networks-3/#ada
		# http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture7.pdf
		if config is None: config = {}
		config.setdefault('learning_rate', 1e-3)
		config.setdefault('beta1', 0.9)
		config.setdefault('beta2', 0.999)
		config.setdefault('epsilon', 1e-8)
		config.setdefault('m', np.zeros_like(w))
		config.setdefault('v', np.zeros_like(w))
		config.setdefault('t', 0)
        
		config['t'] += 1

		# print("### inside momentum ")

		# print("### after w ", w.shape)
		# print("### m ", config['m'].shape)
		# print("### after dw ", dw.shape)


		# print("### dw ", dw.shape)
		# print("### b ", np_b.shape)
		# print("### db ", np_db.shape)

		
		config['m'] = config['beta1'] * config['m'] + (1 - config['beta1']) * dw
		mt = config['m'] / (1 - config['beta1']**config['t'])
		config['v'] = config['beta2'] * config['v'] + (1 - config['beta2']) * dw**2
		vt = config['v'] / (1 - config['beta2']**config['t'])
		next_w = w - config['learning_rate'] * mt / (np.sqrt(vt) + config['epsilon'])

		return torch.from_numpy(next_w), config

	def norm_weights(self):
		# creating local copy
		w = self.weight
		b = self.bias

		# normalizing weights and bias
		norm_w = torch.norm(w)
		norm_b = torch.norm(b)
		
		return  norm_w + norm_b