###
### CS667 Data Science with Python, Homework 12, Jon Organ
###

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df_cmg = pd.read_csv("cmg_weeks.csv")
df_spy = pd.read_csv("spy_weeks.csv")

print("Question 1:")

def Q1(df):
	kmeans_clf = KMeans(n_clusters=3)
	X = df[['Avg_Return', 'Volatility']][df['Week'] <= 50]
	y_means = kmeans_clf.fit_predict(X)
	print("k = 3 k-means classifier implemented...")

	print("Distortion vs k:")
	print("{:<8} {:<8}".format('k Val', 'Distortion'))
	k = 1
	fig, ax = plt.subplots()
	k_and_distortion = []
	while k <= 8:
		kmeans_clf = KMeans(n_clusters=k)
		kmeans_clf.fit(X)
		print("{:<8} {:<8}".format(k, round(kmeans_clf.inertia_, 6)))
		k_and_distortion.append([k, kmeans_clf.inertia_])
		k += 1
	temp_df = pd.DataFrame(k_and_distortion, columns=['k', 'Distortion'])
	ax.plot(temp_df['k'], temp_df['Distortion'])
	ax.set(xlabel='k Value', ylabel='Distortion', title='k Distortion by Value')
	ax.grid()
	print("Saving Q1 graph...")
	fig.savefig("Q1_kDistortion_Graph.png")
	print("Using the knee method, the point of diminishing returns seems to be around k = 6")

Q1(df_cmg)