# for i in range(9, 2, -1):
#     print(i)

import torchfile as trf
import torch 
import numpy as np

pred = torch.load('./predictions_yhat_2.bin')
# print(type(pred))
print(pred.numpy())
# print(pred.n)

np_pred = pred.numpy()

tot = len(np_pred)
print("# tot", tot)

# for i in range(tot):
# np.savetxt("out1.csv",np_pred, header="predictions",fmt="%d")
np.savetxt("out4.csv", np.dstack((np.arange(0, np_pred.size),np_pred))[0],"%d,%d",header="id,label")

labels = trf.load('./data/labels.bin')
print(labels)