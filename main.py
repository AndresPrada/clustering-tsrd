# 
#
#
#
#
# This is the script for the technical task for the process in Motor AI.
# Author: Andrés Prada

import os, glob, cv2, sys, time
from pathlib import Path
import numpy as np
from keras import Model
from cifar100vgg import cifar100vgg
from keras.applications.resnet import ResNet152
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.nasnet import NASNetLarge
from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
from sklearn.cluster import AgglomerativeClustering
from sklearn.utils.multiclass import unique_labels
from sklearn import manifold
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score
import matplotlib.pyplot as plt

def visualize_data(Z, labels, num_clusters):
	'''
		This function helps to visualize the data performing a dimensionality reduction with TSNE.
	'''
	tsne = manifold.TSNE(n_components=2, init='pca', random_state=0).fit_transform(Z)
	fig = plt.figure()
	plt.scatter(tsne[:, 0], tsne[:, 1], s=2, c=labels, cmap=plt.cm.get_cmap("jet", num_clusters))
	plt.colorbar(ticks=range(num_clusters))
	plt.show()

def plot_matrix(y_true, y_pred, classes, title=None, cmap=plt.cm.Blues):
	"""
	    This function plots the matrix true label vs cluster label.
	    It is edited from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
	    """

	# Compute confusion matrix
	cm = confusion_matrix(y_true, y_pred)

	# Only use the labels that appear in the data
	classes = classes[unique_labels(y_true, y_pred)]

	fig, ax = plt.subplots()
	im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
	ax.figure.colorbar(im, ax=ax)
	# We want to show all ticks...
	ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes, title=title, ylabel='True label', xlabel='Cluster label')
	    
	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), fontsize=6, rotation=45, ha="right", rotation_mode="anchor")
	plt.setp(ax.get_yticklabels(), fontsize=6)

	# Loop over data dimensions and create text annotations.
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			if cm[i,j] > 0:
				ax.text(j, i, cm[i, j], fontsize=6, ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
	fig.tight_layout()
	plt.show()
	return ax

if __name__ == '__main__':

	start_time = time.time()
	# Open images folder
	image_path = "tsrd-train/"
	image_paths = [str(p)for p in Path(image_path).glob('*.png')]

	# Extracct labels from image name
	labels = [int(str(p).split("/")[1].split("_")[0][-2:]) for p in image_paths]


	# Commented models. For testing, please check the README to adjust image dimensions and feature vector size

	#model = Xception(include_top=False, weights='imagenet', pooling='avg')
	#model = NASNetLarge(include_top=False, weights='imagenet', pooling='avg')
	#model = InceptionResNetV2(include_top=False, weights='imagenet', pooling='avg')
	#model = VGG16(include_top=False, weights='imagenet', pooling='avg')
	#model = cifar100vgg(train=False)
	#my_layer = model.model.layers[56]

	#model = Model(model.model.input, outputs=my_layer.output)
	# Define the model
	model = ResNet152(include_top=False, weights='imagenet', pooling='avg')
	model.layers[0].trainable = False
	dims = [224,224]
	vect_len = 2048

	# Define list to store vector values
	feature_vects = np.zeros((len(image_paths), vect_len), dtype=float)

	# Extract vectors
	for idx, img in enumerate(image_paths):
		# Print
		print("Extracting vector features for image: "+str(idx))

		# Load and reshape the image to input to the network
		img = cv2.resize(cv2.imread(img), (dims[0], dims[1]))

		# Add the 4th dim (1, 224, 224, 3)
		img = np.expand_dims(img.copy(), axis=0)

		# Predict and store value
		feature_vects[idx, :] = model.predict(img).flatten()

	# Cluster the vectors
	clusters = AgglomerativeClustering(n_clusters=58).fit(feature_vects)

	# Check running time
	print("--- %s seconds ---" % (time.time() - start_time))

	#Evaluate similarity normalized_mutual_info_score
	nmi = normalized_mutual_info_score(labels, clusters.labels_, average_method='warn')
	print('Evaluation of similarity with normalized mutual score: ' + str(nmi))

	# Print true label vs cluster
	plot_matrix(labels, clusters.labels_, np.unique(labels), title='Labeled images')

	# Finally, visualize data
	visualize_data(feature_vects, labels, 58)









