import json
import csv
import pandas as pd
import numpy as np
import torch
import os
from tqdm import tqdm

path = "CatA_Simple"

folder = os.fsencode(path)

data_set = {'input': [],
            'output': [],
            'operation': [],
            'kernel': []}
for file in tqdm(os.listdir(folder)):
    filename = os.fsdecode(file)
    if filename.endswith(('.json')):
        with open('{}/{}'.format(path, filename), encoding='utf-8') as inputfile:
            df = pd.read_json(inputfile)
            input = torch.zeros([4,15,15], dtype = torch.int32)
            output = torch.zeros([4,15,15], dtype = torch.int32)
            operation = torch.zeros([8] , dtype = torch.int32)
            kernel = torch.zeros([8,8], dtype = torch.int32)
            with open('{}/{}'.format(path, filename)) as data_file:
                data = json.load(data_file)
                for i,v in zip(range(4), data):
                    input[i] = torch.FloatTensor(v["input"].copy())
                    output[i] = torch.FloatTensor(v["output"].copy())
            with open('{}/{}_soln.txt'.format(path, filename[:len(filename) - 5]), 'r') as file_name:
                for line,i in zip(file_name, range(8)):
                    for word in line.split():
                        if(word == "Dilation"):
                            operation[i] = 0
                        elif(word == "Erosion"):
                            operation[i] = 1
                        else:
                            kernel[i][ord(word[-1]) - ord('0') - 1] = 1
            data_set['input'].append(input.tolist())
            data_set['output'].append(output.tolist())
            data_set['operation'].append(operation.tolist())
            data_set['kernel'].append(kernel.tolist())
        # writer.writerow(df)
df = pd.DataFrame(data_set)
# df.to_csv('dataset.csv', encoding='utf-8', header=True, index=True)
df.to_json('dataset.json')
