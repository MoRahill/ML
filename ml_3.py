import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd

column_names = ['id','clump_thickness','uniform_cell_size','uniform_cell_shape',
                'marginal_adhesion','single_epithelial_size','bare_nuclei',
                'bland_chromatin','normal_nucleoli','mitoses','class']

df = pd.read_csv('breast-cancer-wisconsin.data', names=column_names)


df.replace('?', -99999, inplace=True) #replace missing data with outlier
df.drop(['id'], axis=1, inplace=True) #drop id column as it is not useful for prediction

X = np.array(df.drop(['class'], axis=1)) #features
y = np.array(df['class']) #labels

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2) #split data into training and testing sets

clf = neighbors.KNeighborsClassifier() #create KNN classifier
clf.fit(X_train, y_train) #train the classifier on the training data

accuracy = clf.score(X_test, y_test) #evaluate the classifier on the testing data
print(accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,1]]) #example data to predict 
example_measures = example_measures.reshape(1, -1) #reshape the data to be 2D as required by sklearn

example_measures = np.array([[4,2,1,1,1,2,3,2,1], [4,2,1,1,1,2,3,2,1]])  
example_measures = example_measures.reshape(2, -1) 

example_measures = np.array([[4,2,1,1,1,2,3,2,1], [4,2,1,1,1,2,3,2,1], [4,2,1,3,3,2,3,2,1]]) #example data to predict 
example_measures = example_measures.reshape(len(example_measures), -1) #len(example_measures) will give the number of rows in the array, which is 3 in this case. The -1 will automatically calculate the number of columns based on the number of rows and the total number of elements in the array.

prediction = clf.predict(example_measures) #make prediction on example data
print(prediction)