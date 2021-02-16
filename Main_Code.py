#IMPORTING PACKAGES REQUIRED FOR THE PROJECT 
import os
import sys

import numpy as np
import pandas as pd

import matplotlib.path as mpath
import matplotlib.pyplot as plt

from numpy.linalg import matrix_rank

from sklearn.model_selection import train_test_split

class DataPreprocessing(object):

    print('DataPreprocessing is started...')
    print()
    print('-------------------------------')
    print()
    
    # Uploading Movie lens dataset
    header = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv("u_100k.data",sep='\t')
    df.columns=header


    # Representing dataset as  User-Item matrix
    n_users = df.user_id.unique().shape[0]
   
    n_items = df.item_id.unique().shape[0]
    
    ratings = np.zeros((n_users, n_items))
    for row in df.itertuples():
        ratings[row[1]-1, row[2]-1] = row[3]

    
    #Dividing data into tarin and test set
    train_data, test_data = train_test_split(df, test_size=0.25)

    #Create two user-item matrices, one for training and another for testing
    train_data_matrix = np.zeros((n_users, n_items))

    
    #finding Rank for Train_data_matrix
    train_rank = matrix_rank(train_data_matrix)

    for line in train_data.itertuples():
        train_data_matrix[line[1]-1, line[2]-1] = line[3]

    
    test_data_matrix = np.zeros((n_users, n_items))

    #finding Rank for Train_data_matrix
    test_rank = matrix_rank(test_data_matrix)
    
    for line in test_data.itertuples():
        test_data_matrix[line[1]-1, line[2]-1] = line[3]

    print('DataPreprocessing Completed Sucessfully...')

if __name__=='__main__':

    #create an Empty DataFrame
    COLUMN_NAMES=['Epoches','Learning_Rate','Train_RMSE','Test_RMSE'] 
    analysis = pd.DataFrame(columns=COLUMN_NAMES)
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)

    #creating empty Lists for storing Epochs, Learning_rate 
    epochs=int(15)
    learning_rate=0.01

        
    obj = DataPreprocessing()


    import MatrixFactorization as m
    print('-------------------------------------')
    print('Training the data has been Started...')
    print('-------------------------------------')
    mf = m.MF(obj.train_data_matrix, K=obj.train_rank, alpha=learning_rate, beta=0.01, iterations=epochs)
    training_process = mf.train(0)
    full_matrix=mf.full_matrix()
    print('------------------------------------------')
    print('Training the Data Completed Sucessfully...')
    print('------------------------------------------')


    print()
    print()
    print('-----------------------------------------')
    print('Applying the Test DataSet on the Model...')
    print('-----------------------------------------')
    #Applying the same parameters for the test dataset to reduce the error rate.
    import MatrixFactorization as m
    mf = m.MF(obj.test_data_matrix, K=obj.test_rank, alpha=learning_rate, beta=0.01, iterations=epochs)
    test_process = mf.train(0)
    test_matrix=mf.full_matrix()


    #Calculating the Mean Square Error
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    def rmse(prediction, ground_truth):
        prediction = prediction[ground_truth.nonzero()].flatten() 
        ground_truth = ground_truth[ground_truth.nonzero()].flatten()
        return sqrt(mean_squared_error(prediction, ground_truth))


    #Storing the Rmse in the Error variable to append the data into the DataFrame called Analysis
    error=rmse(test_matrix,obj.test_data_matrix)
    train_rmse=rmse(full_matrix,obj.train_data_matrix)


    #storing all the outputs into the dataframe...    
    analysis.loc[str(0)] = [int(epochs),learning_rate,train_rmse,error,]
    
    print()
    print()
    print('------------------------------------------------')
    print('                     RESULT                     ')
    print('------------------------------------------------')
    print(analysis)


    star = mpath.Path.unit_regular_star(6)
    circle = mpath.Path.unit_circle()
    # concatenate the circle with an internal cutout of the star
    verts = np.concatenate([circle.vertices, star.vertices[::-1, ...]])
    codes = np.concatenate([circle.codes, star.codes])
    cut_star = mpath.Path(verts, codes)



    plt.plot(training_process, '--r', marker=cut_star, markersize=15)
    plt.plot(test_process, '--g', marker=cut_star, markersize=15)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Train','Test'])
    plt.show()


    
