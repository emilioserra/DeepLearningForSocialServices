 
"""
Different machine learning models: train random forest, a basi neural network with tensorflow, and hyperas code for testing different hyperparameters.

@author: Emilio Serrano and Javier Bajo
"""

from tools  import *
import numpy
import pandas
import pandas as pd   
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from keras import initializers
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from keras.optimizers import SGD

def trainRandomForest(X_train,Y_train,X_val,Y_val):
    """
     train random forest and report results returned by getMetrics   
    """    
    model=RandomForestClassifier(n_estimators=100)
    model.fit(X_train,Y_train)
    results=(getMetrics(model, X_train,Y_train,X_val,Y_val))
    return results


def trainBasicNN(X_train,Y_train,X_val,Y_val):
    """
     basic desnes NN for binary classification with dropout and weight for classes to deal with unblanaced classification
     xavier is as default in dense, kernel_initializer='glorot_uniform'. The loss function could be redifined to weight precision.
 
    """
    # create model
    model = Sequential()
    model.add(Dense(175, input_dim=347, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(175, input_dim=347, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid')) #changes for multi-class
    #class weight, to control unbalanced data
    class_weight = {0: 1., 1: 1.} #somehow, class 0 is map to 1, it control precision, i.e. {0: 10., 1: 1.} increases precision in class 1
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    model.fit(X_train, Y_train, validation_data=(X_val,Y_val), epochs=50, batch_size=50,class_weight=class_weight)
    results = getMetrics(model, X_train,Y_train,X_val,Y_val)
    return results





def trainModelHyperas(x_train, y_train, x_test, y_test):
    """
    
    aux fucntion for hyperasOptimization, it is called several times to get the best architecture
    look up "choice" to add to the search space
    sometimes, comments are taken as values for the optimizer, so do not add values as comments
    the hidden layers loop works if you print model description, but the choices of dense and dropout are the same for all hidden layers       
    that is why, to have the same selection in the first hidden layers, neurons and dropout rate are extenral choices/varaibles

    
    
    
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    
   
    
    model = Sequential()
    
    #equal neurons and dropout are selected
    neuronsPerLayers= {{choice([50,175,256, 512, 1024])}}
    dropoutRate={{uniform(0, 1)}}
    activationFunction={{choice(['relu', 'tanh'])}}
    hiddenLayers={{choice([1,2,5,10,20,30])}}  

    
    #first hidden layer, individual choice could be defined
    model.add(Dense(neuronsPerLayers, input_shape=(347,)))
    model.add(Activation(activationFunction))
    model.add(Dropout(dropoutRate))
    
    
    #second and next hidden layers, choince for each one cannot be selected inside the loop
    for x in range(1, hiddenLayers):
        model.add(Dense(neuronsPerLayers))
        model.add(Activation(activationFunction))
        model.add(Dropout(dropoutRate))

    
 
    #output layer, choice for optimization
    model.add(Dense(1, activation='sigmoid')) #changes for multi-class
    model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                optimizer={{choice(['sgd','adam','nadam'])}})


    
    #fit, choice for batch  and ephopcs.
    model.fit(x_train, y_train,
              batch_size={{choice([32,64,512,1024])}},
              epochs={{choice([10,50,100,150])}},  
              verbose=2,
              validation_data=(x_test, y_test))
    
    
   
      
    

    score, acc = model.evaluate(x_test, y_test, verbose=0)
    
    #report information about the specific model., space parameters, and interesting metrics. This makes things slower
    #print("MODEL DESCRIPTION: \t" + model.to_json() ) #print keras final model, useful to check hyperas and hyperopt bugs
    #print('Test accuracy:', acc)

    print("Acc,AccV,Prec,Recall: \t" + str((getMetrics(model, x_train, y_train, x_test, y_test,True))))
    print("Model parameters: \t" + str(space)) 
    
    
        
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}



#aux funciton for trainModelHyperas, just returned previously load data
def getDataHyperas():
   x_train,y_train,x_test,y_test = preprocessData("./data/oversampledTraning.csv", "./data/oversampledValidation.csv")
   return x_train,y_train,x_test,y_test




#optimization, read data with getDataHyperas where csv files are specified
def hyperasOptimization(maxEvals,seed):
    
    #check options https://github.com/maxpumperla/hyperas/blob/master/hyperas/optim.py
    best_run, best_model = optim.minimize(model=trainModelHyperas,
                                          data=getDataHyperas,
                                          algo=tpe.suggest,
                                          max_evals=maxEvals,
                                          trials=Trials(),
                                          rseed=seed,)

    X_train, Y_train, X_val, Y_val = getDataHyperas()
    
    print("---------------------------------------------")
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_val, Y_val))
    metrics = getMetrics(best_model, X_train,Y_train,X_val,Y_val,True)
    print("Acc,AccV,Prec,Recall: \t", metrics)
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    return metrics

    
    
   