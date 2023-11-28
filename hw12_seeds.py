###
### CS667 Data Science with Python, Homework 12, Jon Organ
###

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

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
	print(svm_type + " accuracy: " + str(round(svm_acc * 100, 2)) + "%")

	predictions = svm.predict(X_test)
	y_actu = pd.Series(Y_test.ravel(), name="Actual")
	y_pred = pd.Series(predictions, name="Predicted")
	cm = pd.crosstab(y_actu, y_pred)
	print(svm_type + " confusion matrix:")
	print(cm)
	return [cm, svm_acc * 100]

cm_svmlin = SVM_clf(svm.SVC(kernel='linear'), "SVM Linear")
print()
cm_svmgauss = SVM_clf(svm.SVC(kernel='rbf'), "SVM Gaussian")
print()
cm_svmpoly = SVM_clf(svm.SVC(kernel='poly', degree=3), "SVM Polynomial")


print("\n")
# Question 2 ========================================================================================
print("Question 2:")

cm_nb = SVM_clf(GaussianNB(), "Naive Bayesian")
print()

print("{:<16} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}".format('Model' ,'TP', 'FP', 'TN', 'FN', 
	'accuracy', 'TPR', 'TNR'))

def PrintTableLine(cm, method, acc):
	TP = cm[1].iloc[0]
	FP = cm[1].iloc[1]
	TN = cm[2].iloc[1]
	FN = cm[2].iloc[0]
	TPR = round(TP / (TP + FN), 2)
	TNR = round(TN / (TN + FP), 2)
	print("{:<16} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}".format(method ,TP, FP, TN, FN, 
		round(acc, 2), TPR, TNR))


PrintTableLine(cm_svmlin[0], "SVM Linear", cm_svmlin[1])
PrintTableLine(cm_svmgauss[0], "SVM Gaussian", cm_svmgauss[1])
PrintTableLine(cm_svmpoly[0], "SVM Polynomial", cm_svmpoly[1])
PrintTableLine(cm_nb[0], "Naive Bayesian", cm_nb[1])

print("\n")
# Question 3 ========================================================================================
print("Question 3:")


