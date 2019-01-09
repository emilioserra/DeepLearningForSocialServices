 
"""

 
@author: Emilio Serrano and Javier Bajo

"""



import pandas 
from sklearn.metrics import *
from sklearn.preprocessing import MinMaxScaler
import time;



#get nested list, first position with numeric values in dataframe, second with categorical
def getNumericAndCategorical(df):
    cols = df.columns
    num_cols = df._get_numeric_data().columns 
    return [list(set(num_cols)),list(set(cols) - set(num_cols))]
    


#get nested list, with x_train. y_train, x_validation. 7_validation.  Datasets categorical values transformed as one hot vector and numerical values are normalized: 347 variables + class in the end
def preprocessData(trainingFile, validationFile):

    #load data training and validation
    dfT = pandas.read_csv(trainingFile)
    dfT = dfT.drop('Unnamed: 0', axis=1) #remove first column with number of row
    dfV = pandas.read_csv(validationFile)
    dfV = dfV.drop('Unnamed: 0', axis=1) #remove first column with number of row
 
    #concatenating training and validaiton to make preprocessing consistent and take class as number
    numberOfTrainingSamples= dfT.shape[0]
    df = pandas.concat([dfT,dfV], axis=0) #concatenate adding rows
    dfY=df[['class']]
    dfY=dfY.replace({'class': {'N': 0, 'S': 1} }) #change class per 0 and one
    df = df.drop('class', axis=1) #remove class
    


    #one hot vector for just cagtegories, add the numerical, and the class as number
    numeric= getNumericAndCategorical(df)[0] #numeric columns
    categorical= getNumericAndCategorical(df)[1] #categorical columns
    one_hot = pandas.get_dummies(df[categorical]) #one_hot vector for categorical
    scaler = MinMaxScaler()
    df[numeric]= scaler.fit_transform(df[numeric])#scale numeric values between 0 and 1
    df= pandas.concat([df[numeric],one_hot], axis=1)  #concatenate numerical and one_hot, adding columns, axis=1
    #df= pandas.concat([df,dfY], axis=1)  #concatenate with class, adding columns, axis=1


    #splitting again training and validation
    X_train=df[0:numberOfTrainingSamples] #get again training and test after preprocessing
    Y_train=dfY[0:numberOfTrainingSamples]
    X_val=df[numberOfTrainingSamples+1:df.shape[0]] # get again validation and test after preprocessing
    Y_val=dfY[numberOfTrainingSamples+1:df.shape[0]]
    #note: the first two cases in training oversampledTraning.csv and validation oversampledValidation.csv have  been manually checked to make sure the preprocessing is correct when comparing with the original datasets

    return X_train,Y_train,X_val,Y_val 
    
    
#return accT,acc,precision,recall, verbose to show confusion ,atrix, report, and time.
def getMetrics(model, X_train,Y_train,X_val,Y_val,verbose=True):

    y_pred = model.predict(X_train) #predict with the model the training samples
    y_pred = (y_pred > 0.5) #positive if >0.5
    accT= accuracy_score(Y_train, y_pred)

    y_pred = model.predict(X_val)  #predict with the model the validation samples
    y_pred = (y_pred > 0.5)
    acc= accuracy_score(Y_val, y_pred)
    precision= precision_score(Y_val, y_pred)
    recall= recall_score(Y_val, y_pred)
    
    if(verbose==True):
        print("Classification report:")
        print(classification_report(Y_val, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(Y_val, y_pred))
        localtime = time.asctime( time.localtime(time.time()) )
        print("Local current time :", localtime)

    
    return[round(accT,4),round(acc,4),round(precision,4),round(recall,4)]


