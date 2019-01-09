# -*- coding: utf-8 -*-
"""
Experiments with deep learning to predict social exclusion. It delegates in models.py with different machine learning models.

@author: Emilio Serrano and Javier Bajo
"""
from tools  import *
from models import *
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
import numpy







if __name__ == '__main__':


    # fix random seed for reproducibility
    seed = 123
    numpy.random.seed(seed)


    #X_train, Y_train, X_val, Y_val = preprocessData("./data/oversampledTraning.csv", "./data/oversampledValidation.csv")
    #X_train, Y_train, X_val, Y_val = preprocessData("./data/unbalancedTraining.csv", "./data/unbalancedValidation.csv")   
    #resultsBaseline=trainRandomForest(X_train,Y_train,X_val,Y_val)
    #resultsBasicNN= trainBasicNN(X_train,Y_train,X_val,Y_val)
    
    #optimization, read data with getDataHyperas where csv files are specified
    resultsHyperas=hyperasOptimization(1000,seed)
   