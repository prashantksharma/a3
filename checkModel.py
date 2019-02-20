import sys
sys.path.append('src')
from imports import *
import torch
import torchfile as tf

def makeModel(make_model_file_path):
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
		{
		"linear": model.addLayer(Linear(int(test_data[1]), int(test_data[2]))),
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

def saveToFile(file, file_path):
	_f = open(file_path, 'wb')
	torch.save(file, _f)
	_f.close()
	
#########################################################################################


argumentList = sys.argv[1:]
arguments = {}
for i in range(int(len(argumentList)/2)):
	arguments[argumentList[2*i]] = argumentList[2*i + 1]

#print('')
model = makeModel(arguments["-config"])
#print('')
input = readInput(arguments["-i"])
gradInput = readInputGrad(arguments["-ig"])
output = model.forward(input)
saveToFile(output, arguments["-o"])
model.clearGradParam()
model.backward(input, gradInput)
weightGrads = []
biasGrads = []
for layer in model.Layers:
	if layer.isTrainable:
		weightGrads.append(layer.gradWeight.t())
		biasGrads.append(layer.gradBias)
saveToFile(weightGrads, arguments["-ow"])
saveToFile(biasGrads, arguments["-ob"])
gradOutput = model.Layers[0].gradInput
saveToFile(gradOutput, arguments["-og"])
#print('')








