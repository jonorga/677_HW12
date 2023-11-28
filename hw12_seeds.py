###
### CS667 Data Science with Python, Homework 12, Jon Organ
###

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import svm

# Last digit of BUID: 6
# R = 0, class L = 1 (negative) and L = 2 (positive)

df = pd.read_excel("seeds_dataset.xlsx", header=None, names=['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'L'], nrows=140)

# Question 1 ========================================================================================
print("Question 1:")

Y = df[["L"]].values
X = df[['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7']].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5, random_state = 0)
print("Data subset split into training and testing data...")

def SVM_clf(svm, svm_type):
	svm.fit(X_train, Y_train.ravel())
	svm_acc = svm.score(X_test, Y_test)
	print("SVM " + svm_type + " accuracy: " + str(round(svm_acc * 100, 2)) + "%")

	predictions = svm.predict(X_test)
	y_actu = pd.Series(Y_test.ravel(), name="Actual")
	y_pred = pd.Series(predictions, name="Predicted")
	cm = pd.crosstab(y_actu, y_pred)
	print("SVM " + svm_type + " confusion matrix:")
	print(cm)

SVM_clf(svm.SVC(kernel='linear'), "Linear")
print()
SVM_clf(svm.SVC(kernel='rbf'), "Gaussian")
print()
SVM_clf(svm.SVC(kernel='poly', degree=3), "Polynomial")


print("\n")
# Question 2 ========================================================================================
print("Question 2:")






