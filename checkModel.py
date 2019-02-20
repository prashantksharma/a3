import sys
sys.path.append('src')
from imports import *
import torch
import torchfile as trf
import argparse

def create_model(make_model_file_path):
	_f = open(make_model_file_path, 'r')
	_fData = ''.join(line for line in _f).split('\n')
	while(_fData[-1] == ''):
		_fData = _fData[:-1]
	_f.close()

	weight_p = _fData[-2]
	bias_p = _fData[-1]
	_fData = _fData[1:-2]
	model = Model()

	for data in _fData:
		test_data = data.split()
		no_ips = int(test_data[1])
		no_ops = int(test_data[2])
		{
		"linear": model.addLayer(Linear( no_ips, no_ops )),
		"relu": model.addLayer(ReLU()) }[test_data[0]]

	i = 0
	for layer in model.Layers:
		if layer.isTrainable:
			layer.weight = torch.from_numpy(tf.load(weight_p)[i]).t()
			layer.bias = torch.from_numpy(tf.load(bias_p)[i])
			i += 1

	return model

def readInput(input_path):
	input = tf.load(input_path)
	input = torch.from_numpy(input)
	return input.view(input.size()[0], -1)

def readInputGrad(gradInput_path):
	gradInput = tf.load(gradInput_path)
	return torch.from_numpy(gradInput)

def read_ip_ip_grad(ip_path, ip_grad_path):
	ip = trf.load(ip_path)
	ip_grad = trf.load(ip_grad_path)

	ip = torch.from_numpy(ip)
	ip_grad = torch.from_numpy(ip_grad)

	return ip, ip_grad

def model_pass(model):
	w_grads = []
	b_grads = []
	layers = model.layers
	for l in layers:
		if l.isTrainable:
			w_grads.append(l.gradWeight.t())
			b_grads.append(l.gradBias)

	return w_grads, b_grads


def save(file, file_path):
	_f = open(file_path, 'wb')
	torch.save(file, _f)
	_f.close()
	
#########################################################################################


if __name__ == "__main__":
		parser = argparse.ArgumentParser()
		parser.add_argument("-config", help="model config")
		parser.add_argument("-i",help="input")
		parser.add_argument("-ig",help="inputGrad")
		parser.add_argument("-o",help="output")
		parser.add_argument("-ow",help="grad weight")
		parser.add_argument("-ob",help="grad bias")
		parser.add_argument("-og",help="grad output")


		args = parser.parse_args()

		print("==================input args===================")
		print(args.config)
		print(args.i)
		print(args.ig)
		print("etc....")

		print("## make model")
		model = create_model(args.config)

		ip, ip_grad = read_ip_ip_grad(args.i, args.ig)

		model.clearGradParam()
		model.backward(ip, ip_grad)


		# weightGrads = []
		# biasGrads = []
		# for layer in model.Layers:
		# 	if layer.isTrainable:
		# 		weightGrads.append(layer.gradWeight.t())
		# 		biasGrads.append(layer.gradBias)
		w_grads, b_grads = model_pass(model)
		save(w_grads,args.ow)
		save(b_grads, args.ob)


		gradOutput = model.Layers[0].gradInput
		save(gradOutput, args.ig)








