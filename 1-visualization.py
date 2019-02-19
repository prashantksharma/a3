
# coding: utf-8

# In[46]:


import torchfile as trf
import matplotlib.pyplot as plt
import torch


# In[2]:


# get_ipython().system('pwd')


# In[5]:


path_labels = "./data/labels.bin"
path_data = "./data/data.bin"


# In[9]:



l = trf.load(path_labels)


# In[18]:


d = trf.load(path_data)


# In[14]:


print(type(l), l.shape, l[:5])


# In[19]:


print(type(d), d.shape, d[0])


# In[20]:


d[0], 


# In[32]:


l[0],plt.imshow(d[0])


# In[43]:


l[:15]


# In[44]:


plt.imshow(d[1])


# In[45]:


plt.imshow(d[7])


# In[35]:


import pandas as pd


# In[36]:


df = pd.DataFrame(l)
df.head()


# In[38]:


df.plot()


# In[49]:


def load_data(path_data, path_labels):
    global data, labels, data_size
    
    _labels = torchfile.load(path_labels)
    _data = torchfile.load(path_data)
    
    tot_labels = torch.from_numpy(_labels)
    tot_data = torch.from_numpy(_data)
    
    tot_labels = tot_labels.type(torch.DoubleTensor)
    tot_data = tot_data.contiguous().view(tot_data.size()[0], -1).type(torch.DoubleTensor)
    
    data = tot_data[:]
    labels = tot_labels[:]
    
    data_size = data.size()[0]
    
    data_mean = data.mean(dim=0)
    data = data - data_mean
    
    data_std = data.std(dim=0, keepdim=True)
    data = data/data_std
    
    return data_mean, data_std


# In[50]:


load_data(path_data, path_labels)


# In[63]:


type(data)


# In[52]:


data.size()


# In[55]:


type(labels), labels[0], labels[:10]


# In[59]:


x = labels[:10]


# In[60]:


x


# In[61]:


x + 2


# In[62]:


data[0].con


# In[65]:


plt.imshow(d[0])


# In[73]:


im = data[0].numpy().reshape(108,108)
im


# In[74]:


plt.imshow(im)


# In[75]:


plt.imshow(d[0])


# In[76]:


d[0]


# In[77]:


im


# In[79]:


type(data), data.size()


# In[80]:


108**2


# In[81]:


torch.save(data,"data.pt")


# In[82]:


type(labels)


# In[83]:


torch.save(labels,"labels.pt")


# ## Checking saved data after normalization

# In[85]:


_d = torch.load('data.pt')


# In[ ]:


_l = torch.load('labels.pt')

