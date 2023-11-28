###
### CS667 Data Science with Python, Homework 12, Jon Organ
###

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df_cmg = pd.read_csv("cmg_weeks.csv")

# Question 1 ========================================================================================
print("Question 1:")

X = df_cmg[['Avg_Return', 'Volatility']][df_cmg['Week'] <= 50]
Y = df_cmg[['Color']][df_cmg['Week'] <= 50]

def Q1(df):
	kmeans_clf = KMeans(n_clusters=3)
	
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


print("\n")
# Question 2 ========================================================================================
print("Question 2:")
kmeans_clf = KMeans(n_clusters=6)
y_means = kmeans_clf.fit_predict(X)

i = 0
# cluster num, green count, red count
cluster_analysis = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
while i < 50:
	if Y['Color'].iloc[i] == "Green":
		cluster_analysis[y_means[i]][0] += 1
	else:
		cluster_analysis[y_means[i]][1] += 1
	i += 1

print("{:<8} {:<8} {:<8}".format('Cluster', 'Green %', 'Red %'))

count = 1
cluster_percs = []
for cluster in cluster_analysis:
	total = cluster[0] + cluster[1]
	green_perc = round((cluster[0] / total) * 100, 2)
	red_perc = round((cluster[1] / total) * 100, 2)
	cluster_percs.append([green_perc, red_perc])
	print("{:<8} {:<8} {:<8}".format(str(count), str(green_perc) + "%", str(red_perc) + "%"))
	count += 1


print("\n")
# Question 3 ========================================================================================
print("Question 3:")

count = 1
for cluster in cluster_percs:
	if cluster[0] > 90:
		print("Cluster " + str(count) + " is a pure green cluster (" + str(cluster[0]) + "%)")
	elif cluster[1] > 90:
		print("Cluster " + str(count) + " is a pure red cluster (" + str(cluster[1]) + "%)")
	count += 1




