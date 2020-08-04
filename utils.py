import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score, accuracy_score

from lightgbm import LGBMClassifier

class Utils(object):
	def plot_map(lat1, lat2, lon1, lon2):
		from mpl_toolkits.basemap import Basemap
		plt.figure(figsize = (15, 25))
		map = Basemap(projection='cyl', resolution='h',
		      llcrnrlat=lat1, urcrnrlat=lat2,
		      llcrnrlon=lon1, urcrnrlon=lon2)
		map.drawcoastlines()
		map.fillcontinents(color='palegoldenrod', lake_color='lightskyblue')
		map.drawmapboundary(fill_color='lightskyblue')
		map.drawparallels(np.arange(lat1, lat2 + 0.5, 0.1), labels=[1, 0, 0, 0])
		map.drawmeridians(np.arange(lon1, lon2 + 0.5, 0.1), labels=[0, 0, 0, 1])
		return map
	    
	    
	#######


	def plot_hist_shape(df, column, value):
		plt.figure(figsize=(10,5))
		plt.hist(df[column][df[column] <= value], bins = 10)
		print(df[column][df[column] <= value].shape[0])
	    
	    
	#######

	  
	def multiclass_roc_auc_score(y_test, y_pred, average="macro",multi_class="ovr"):
		lb = LabelBinarizer()
		lb.fit(y_test)
		y_test = lb.transform(y_test)
		y_pred = lb.transform(y_pred)
		return roc_auc_score(y_test, y_pred, average=average, multi_class=multi_class)
	    
	    
	#######


	def plot_confusion_matrix(cm, classes,
		normalize=False,
		title='Matriz de Confusão',
		cmap=plt.cm.Blues):
		plt.figure(figsize=(7, 7))
		"""
		This function prints and plots the confusion matrix.
		Normalization can be applied by setting `normalize=True`.
		"""
		plt.imshow(cm, interpolation='nearest', cmap=cmap)
		plt.title(title)
		plt.colorbar()
		tick_marks = np.arange(len(classes))
		plt.xticks(tick_marks, classes, rotation=90)
		plt.yticks(tick_marks, classes)

		thresh = cm.max() / 2.
		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
			plt.text(j, i, cm[i, j],
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

		plt.tight_layout()
		plt.ylabel('Label Verdadeiras')
		plt.xlabel('Labels da Previsão')
