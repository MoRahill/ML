import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')

    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    #print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result

df = pd.read_csv('breast-cancer-wisconsin.data', names=['id','clump_thickness','uniform_cell_size','uniform_cell_shape',
                'marginal_adhesion','single_epithelial_size','bare_nuclei',
                'bland_chromatin','normal_nucleoli','mitoses','class'])
df.replace('?', -99999, inplace=True) #replace missing data with outlier
df.drop(['id'], axis=1, inplace=True) #drop id column as it is not useful for prediction
full_data = df.astype(float).values.tolist() #convert dataframe to list of lists
random.shuffle(full_data) #shuffle the data to ensure randomness

test_size = 0.2
train_set = {2:[], 4:[]} #create a dictionary to hold the training data
test_set = {2:[], 4:[]} #create a dictionary to hold the testing data
train_data = full_data[:-int(test_size*len(full_data))] #split the data into training and testing sets
test_data = full_data[-int(test_size*len(full_data)):] #split the data into training and testing sets

for i in train_data:
    train_set[i[-1]].append(i[:-1]) #append the features to the corresponding class in the training set

for i in test_data:
    test_set[i[-1]].append(i[:-1]) #append the features to the corresponding class in the testing set

correct = 0
total = 0
for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neighbors(train_set, data, k=5) #make prediction on the test data
        if group == vote:
            correct += 1
        total += 1

print('Accuracy:', correct/total) #print the accuracy of the model