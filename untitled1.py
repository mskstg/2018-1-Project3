# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 15:48:53 2018

@author: HTG
"""

import surprise
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import KNNWithZScore
from surprise import KNNBaseline
from surprise.model_selection import GridSearchCV
from collections import defaultdict
import numpy as np

np.random.seed(0)
file_path = 'data/user_artists_log.dat'
reader = Reader(line_format='user item rating', sep='\t')
data = Dataset.load_from_file(file_path, reader=reader)

bsl_options_ub = {'n_epochs': 30, 'method': 'als', 'reg_i':10, 'reg_u':0}
sim_options_ub = {'name': 'msd', 'min_support': 1, 'user_based': True}
              
algo = surprise.KNNBaseline(sim_options=sim_options_ub, bsl_options=bsl_options_ub, k=70)
surprise.model_selection.cross_validate(algo, data, measures=['RMSE'], cv=3, verbose=True)


#algo = surprise.SVD(n_epochs= 25, lr_all= 0.01, reg_all= 0.05, n_factors= 5)
#surprise.model_selection.cross_validate(algo, data, measures=['RMSE'], cv=3, verbose=True)