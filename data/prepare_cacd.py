# -*- coding:utf-8 -*-

"""
@author: Jun
@file: .py
@time: 6/25/20193:09 PM
"""
from tqdm import tqdm
import os
import pickle
import random
import numpy as np



root='C:/Projects/CACD'
folders=['14-30','31-40','41-50','51-62']
namelist_train=[]
namelist_test=[]


for folder in tqdm(folders):
    img_dir=os.path.join(root,folder)
    namelist=os.listdir(img_dir)
    random.shuffle(namelist)
    length=len(namelist)
    k=int(0.9*length)
    namelist_train+=namelist[:k]
    namelist_test+=namelist[k:]

random.shuffle(namelist_train)
random.shuffle(namelist_test)

# np.savetxt(root+'/train_ids.csv',namelist_train,fmt='%s')
# np.savetxt(root+'/test_ids.csv',namelist_test,fmt='%s')

#
# for folder in tqdm(folders):
#     img_dir=os.path.join(root,folder)
#     namelist=os.listdir(img_dir)
#     random.shuffle(namelist)
#     length=len(namelist)
#     k=int(0.9*length)
#     namelist_train.append(namelist[:k])
#     namelist_test.append(namelist[k:])
#
#
with open(root+'/train_ids.pkl','wb') as f:
    pickle.dump(namelist_train,f)
    f.close()

with open(root + '/test_ids.pkl', 'wb') as f:
    pickle.dump(namelist_test, f)
    f.close()

