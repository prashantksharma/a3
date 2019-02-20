import torch
import sys
sys.path.append('src')
from imports import *
import torchfile as trf
import numpy as np
import os
import argparse

def process_test_data(path):
    d = trf.load(path)
    _d = torch.from_numpy(d)
    _data = _d.contiguous().view(_d.size()[0], -1).type(torch.DoubleTensor)

    data = _data[:]

    # normalizing the data
    mean = data.mean(dim=0)
    std = data.std(dim=0, keepdim=True)
    data = (data - mean)/std

    return data

def predict(model, test_data):
    yhat = model.forward(test_data)
    yhat = yhat.max(dim = 1)[1]

    # saving the predictions
    file = open("predictions_yhat.bin", 'wb')
    torch.save(yhat, file)
    file.close()

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("-modelName", help="input model name")
        parser.add_argument("-data",help="path to data.bin")
        # parser.add_argument("-target",help="path to labels.bin")

        args = parser.parse_args()
        
        print("==================input args===================")
        print(args.modelName)
        print(args.data)
        # print(args.target)

        print("==========Loading the model ========================")


        model_path = args.modelName + "/model.bin"
        model = torch.load(model_path)

        print("## model loaded")

        print("########## Making predictions")

        test_data = process_test_data(args.data)
        predict(model, test_data)

        print("### Prections: Saved")



