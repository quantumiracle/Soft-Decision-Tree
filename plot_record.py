# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from SDT import SDT
from torch.utils import data
from utils.dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

directory ='record/compare_lambda/'
name='run-sdt_'
label='Training_Weight_Difference'  # Testing_Accuracy, Testing_Alpha
compared_val = [0.1, 0.01, 0.001, -0.1, -0.01, -0.001]

def smooth(y, radius=100, mode='two_sided'):
    if len(y) < 2*radius+1:
        return np.ones_like(y) * y.mean()
    elif mode == 'two_sided':
        convkernel = np.ones(2 * radius+1)
        return np.convolve(y, convkernel, mode='same') / \
               np.convolve(np.ones_like(y), convkernel, mode='same')
    elif mode == 'causal':
        convkernel = np.ones(radius)
        out = np.convolve(y, convkernel,mode='full') / \
              np.convolve(np.ones_like(y), convkernel, mode='full')
        return out[:-radius+1]


def plot_with_fill(x, data, label, color=None):
    y_m=np.mean(data, axis=0)
    y_std=np.std(data, axis=0)
    y_upper=y_m+y_std
    y_lower=y_m-y_std
    if color is not None:
        plt.fill_between(
        x, list(y_lower), list(y_upper), interpolate=True, facecolor=color, linewidth=0.0, alpha=0.3
    )   
    else:
        plt.fill_between(
        x, list(y_lower), list(y_upper), interpolate=True, linewidth=0.0, alpha=0.3
    )     
    plt.plot(x, list(y_m), color=color, label=label)


plt.figure(figsize=(8,6))

for val in compared_val:
    all_data=[]
    for filename in os.listdir(directory):
        if filename.startswith(name) and filename.endswith(label+".csv"): 
            file_path = os.path.join(directory, filename)
            print(os.path.join(directory, filename))
            if '_'+str(val) in filename:  # _ for distinguishing from minus -
                with open(file_path, newline='') as csvfile:
                    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                    data=[]
                    for i, row in enumerate(spamreader):
                        if i != 0:  # skip the title
                            walltime, step, value = row[0].split(",")
                            data.append([int(step), float(value)])
                # print(data.shape)
                all_data.append(np.array(data))

    print(np.array(all_data).shape)
    all_data = np.array(all_data)
    x = np.arange(all_data.shape[1])
    y = all_data[:, :, 1]
    plot_with_fill(x, y, label = str(val))

plt.xlabel('Epoch')
plt.ylabel(label)
leg= plt.legend(loc=4)
plt.grid()
plt.savefig('./img/compare_lambda/'+label+'.png')
plt.show()


# if __name__ == '__main__':
