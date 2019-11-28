# Import packages required
import pandas as pd
import pickle
from sklearn import neural_network as nn
from sklearn import preprocessing
import datetime

# Reading data from .csv file with pandas
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# Converting sets from a DataFrame object -> a ndarray obj
# So that to compute in the classifier
X_test = test.to_numpy()
tmp = train.to_numpy()

# For the training set, column1 is label, others are input data.
# So split them into 2 parts.
X_train = tmp[:, 1:]
y_train = tmp[:, 0]

# Pre-processing
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)

# Get the time now!
now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Create a classifier (say I have a MLPC aka neural network here)
classifier = nn.MLPClassifier(activation='logistic', hidden_layer_sizes=(200, 100))

print("Classifier fitting...")
classifier.fit(X_train, y_train)
print("Classifier is predicting...")
y_test = classifier.predict(X_test)
print("Writing prediction result to ../output/pred_" + now_time + ".csv ...")
# Convert y_test type Array->DataFrame, and add the column label 'Label'
y_test = pd.DataFrame(data=y_test, columns={'Label'})
# Change the indices from 0~27999 -> 1~28000
y_test.index = y_test.index + 1

pass
y_test.to_csv("../output/pred_" + now_time + ".csv", index_label=['ImageId'])

# Pickle the classifier into a .joblib file
print("Pickling the model in ../model/classifier_"+now_time+'.joblib')
with open('../model/classifier_'+now_time+'.joblib', 'wb') as clf_file:
    pickle.dump(classifier, clf_file)
