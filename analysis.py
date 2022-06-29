import json
import gzip
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Circle
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from scipy import cluster
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from scipy.stats import chi2_contingency, pearsonr
from itertools import combinations
import sys
sys.path.append("lib")
# import myorngCA


def circleOfCorrelations(pc_infos, ebouli):
	plt.Circle((0,0),radius=10, color='g', fill=False)
	circle1=plt.Circle((0,0),radius=1, color='g', fill=False)
	fig = plt.gcf()
	fig.gca().add_artist(circle1)
	for idx in range(len(pc_infos["PC-0"])):
		x = pc_infos["PC-0"][idx]
		y = pc_infos["PC-1"][idx]
		plt.plot([0.0,x],[0.0,y],'k-')
		plt.plot(x, y, 'rx')
		plt.annotate(pc_infos.index[idx], xy=(x,y))
	plt.xlabel("PC-0 (%s%%)" % str(ebouli[0])[:4].lstrip("0."))
	plt.ylabel("PC-1 (%s%%)" % str(ebouli[1])[:4].lstrip("0."))
	plt.xlim((-1,1))
	plt.ylim((-1,1))
	plt.title("Circle of Correlations")
 
def myScatter(df):
	"""
	散点图
	"""
	# http://stackoverflow.com/a/23010837/1565438
	axs = pd.plotting.scatter_matrix(df, diagonal='kde')
	for ax in axs[:,0]: # the left boundary
		ax.grid('off', axis='both')
		ax.set_ylabel(ax.get_ylabel(), rotation=0, labelpad=len(ax.get_ylabel())+40)
		ax.set_yticks([])

	for ax in axs[-1,:]: # the lower boundary
		ax.grid('off', axis='both')
		ax.set_xlabel(ax.get_xlabel(), rotation=90)
		ax.set_xticks([])
	plt.show()
 
def myPCA(df, clusters=None):
	"""
	画成分和属性相关圈和pca图
	"""
	# Normalize data
	df_norm = (df - df.mean()) / df.std()
	# PCA
	pca = PCA(n_components='mle')
	pca_res = pca.fit_transform(df_norm.values)
	# Ebouli
	ebouli = pd.Series(pca.explained_variance_ratio_)
	ebouli.plot(kind='bar', title="Ebouli des valeurs propres")
	plt.show()
	# Circle of correlations
	# http://stackoverflow.com/a/22996786/1565438
	coef = np.transpose(pca.components_)
	cols = ['PC-'+str(x) for x in range(len(ebouli))]
	pc_infos = pd.DataFrame(coef, columns=cols, index=df_norm.columns)
	circleOfCorrelations(pc_infos, ebouli)
	plt.show()
	# Plot PCA
	dat = pd.DataFrame(pca_res, columns=cols)
	if isinstance(clusters, np.ndarray):
		for clust in set(clusters):
			colors = list("bgrcmyk")
			plt.scatter(dat["PC-0"][clusters==clust],dat["PC-1"][clusters==clust],c=colors[clust])
	else:
		plt.scatter(dat["PC-0"],dat["PC-1"])
	plt.xlabel("PC-0 (%s%%)" % str(ebouli[0])[:4].lstrip("0."))
	plt.ylabel("PC-1 (%s%%)" % str(ebouli[1])[:4].lstrip("0."))
	plt.title("PCA")
	plt.show()
	return pc_infos, ebouli

def myKmeans(df, nb_clusters):
	centroids, _ = cluster.vq.kmeans(df.values, nb_clusters, iter=100)
	idx, _ = cluster.vq.vq(df.values, centroids)
	return idx

def loadJSON(path):
	data = json.loads(gzip.open(path).read())
	df = pd.DataFrame(data)
	df = df.T
	df = df.fillna(0)
	return df

def myHClust(df):
	X = df.values
	Y = pdist(X)
	Z = linkage(Y, method="average")
	res = dendrogram(Z, labels=df.index)
	plt.title("Hierarchical Clustering (dendrogram)")
	plt.show()
	return res

def myCorrPlot(df):
	"""
	相关圈， 是把皮尔森相关系数用圈的不同颜色表示
	Correlation plot ( ~ corrplot with R)
	Forked from https://github.com/louridas/corrplot
	"""
	plt.figure(1)
	ax = plt.subplot(1, 1, 1, aspect='equal')
	poscm = cm.get_cmap('Blues')
	negcm = cm.get_cmap('Reds')
	labels = df.columns

	for pair in combinations(labels, 2):
		corr = pearsonr(df[pair[0]].values, df[pair[1]].values)[0]
		clrmap = poscm if corr >= 0 else negcm
		circle = Circle((labels.get_loc(pair[0]),labels.get_loc(pair[1])), radius = 0.4)
		circle.set_edgecolor('black')
		circle.set_facecolor(clrmap(np.abs(corr)))
		mirrorCircle = Circle((labels.get_loc(pair[1]),labels.get_loc(pair[0])), radius = 0.4)
		mirrorCircle.set_edgecolor('black')
		mirrorCircle.set_facecolor(clrmap(np.abs(corr)))
		ax.add_artist(circle)
		ax.add_artist(mirrorCircle)

	ax.set_xlim(-1, len(labels))
	ax.set_ylim(-1, len(labels))
		
	ax.xaxis.tick_top()
	xtickslocs = np.arange(len(labels))
	ax.set_xticks(xtickslocs)
	ax.set_xticklabels(labels, rotation=30, fontsize='small', ha='left')

	ax.invert_yaxis()
	ytickslocs = np.arange(len(labels))
	ax.set_yticks(ytickslocs)
	ax.set_yticklabels(labels, fontsize='small')
	plt.show()
	return plt

def myChiSquaredTest(df):
	chi2, p, _, _ = chi2_contingency(df.values)
	print("chi2 score:",chi2)
	print("pvalue:", p)
	if p < 0.05:
		return True
	else:
		return False



if __name__ == '__main__':

	# # An example with IRIS dataset
	from sklearn import datasets
	iris = datasets.load_iris()
	df = pd.DataFrame(iris.data, columns=iris.feature_names)
	# # Chi-Square test
	# myChiSquaredTest(df)
	# # Scatter Matrix of features
	# myScatter(df)
	# # Correlations plot
	# myCorrPlot(df)
	# # PCA
	myPCA(df)
	# # PCA和Kmeans的结合，就是PCA上的点是Kmeans已经进行了聚类的，然后好分颜色
	# myPCA(df, clusters=myKmeans(df, 3))
	# # Hierarchical Clustering, 层次聚类
	# myHClust(df)
	# # Correlation Plot, 相关圈
	# plot = myCorrPlot(df)


