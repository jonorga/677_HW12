###
### CS667 Data Science with Python, Homework 12, Jon Organ
###

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import random

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
# Question 3.1 ========================================================================================
print("Question 3.1:")

k = 1
k_and_distortion = []
while k <= 8:
	kmeans_clf = KMeans(n_clusters=k)
	kmeans_clf.fit(X)
	k_and_distortion.append([k, kmeans_clf.inertia_])
	k += 1

fig, ax = plt.subplots()
temp_df = pd.DataFrame(k_and_distortion, columns=['k', 'Distortion'])
ax.plot(temp_df['k'], temp_df['Distortion'])
ax.set(xlabel='k Value', ylabel='Distortion', title='k Distortion by Value')
ax.grid()
print("Saving Q3 graph...")
fig.savefig("Q3_kDistortion_Graph.png")
print("Using the knee method, the point of diminishing returns seems to be around k = 5")


print("\n")
# Question 3.2 ========================================================================================
print("Question 3.2:")
kmeans_clf = KMeans(n_clusters=5)

rand_feat_1 = random.randrange(7)
rand_feat_2 = random.randrange(7)
if rand_feat_1 == rand_feat_2:
	if rand_feat_2 == 6:
		rand_feat_2 = 0
	else:
		rand_feat_2 += 1

rand_feat_1 += 1
rand_feat_2 += 1
rand_feat_1 = "f" + str(rand_feat_1)
rand_feat_2 = "f" + str(rand_feat_2)

X32 = df[[rand_feat_1, rand_feat_2]]
kmeans_clf.fit(X32)


scatter_plot = plt.figure()
ax = scatter_plot.add_subplot(1, 1, 1)
ax.scatter(X32[rand_feat_1], X32[rand_feat_2], s=30, c=df['L'])

cluster_df = pd.DataFrame(kmeans_clf.cluster_centers_)
ax.scatter(cluster_df[0], cluster_df[1], s=820, alpha=0.5)
for i in range(5):
	mark = "$" + str(i + 1) + "$"
	ax.scatter(cluster_df[0].iloc[i], cluster_df[1].iloc[i], s=220, marker=mark, color="Red")

ax.set_title("Scatter plot for " + rand_feat_1 + " and " + rand_feat_2)
ax.set_xlabel(rand_feat_1)
ax.set_ylabel(rand_feat_2)
print("Saving Q3.2 scatter plot...")
scatter_plot.savefig("Q3.2_scatterplot.png")
print("Looking at multiple iterations of the Q3.2 scatter plot it immediately jumps out at me that"
	" most of the feature combbinations seem to have a pattern that could be predicted by linear"
	" or logistic regression")





