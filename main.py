# This is the script for the technical task for the process in Motor AI.
#
# Andrés Prada

import os, glob, cv2, sys, pdb, time
from pathlib import Path
import numpy as np
from keras import Model
from cifar100vgg import cifar100vgg
from keras.applications.resnet import ResNet152
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.nasnet import NASNetLarge
from keras.applications.xception import Xception
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, adjusted_rand_score, normalized_mutual_info_score
from munkres import Munkres
import matplotlib.pyplot as plt

def ass_matrix(true_labels, pred_labels):
	n_clusters = len(np.unique(pred_labels))
	ass_matrix = np.zeros((n_clusters, n_clusters), dtype=np.int32)
	for idx_true, idx_pred in enumerate(pred_labels):
		ass_matrix[true_labels[idx_true], idx_pred] += 1
	return ass_matrix

def clusters_mapping(matrix):
	'''	
	http://software.clapper.org/munkres/

	'''

	# Convert matrix to cost matrix
	cost_matrix = []
	for row in matrix:
		cost_row = []
		for col in row:
			cost_row += [sys.maxsize - col]
		cost_matrix += [cost_row]
	
	# Load Munkres() Object
	m = Munkres()

	# Return mapping to maximize the cost
	return m.compute(cost_matrix)

def plot_confusion_matrix(cm, classes):
	"""
	This function plots the confusion matrix.
	This function was edited from the original found in: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
	"""
	cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 0.00001)
	fig, ax = plt.subplots()
	im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	ax.figure.colorbar(im, ax=ax)
	# We want to show all ticks...
	ax.set(xticks=np.arange(cm.shape[1]),
			yticks=np.arange(cm.shape[0]),
			# ... and label them with the respective list entries
			xticklabels=classes, yticklabels=classes,
			title='Confusion Matrix',
			ylabel='True label',
			xlabel='Predicted label')

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			if cm[i,j] > 0.1:
				ax.text(j, i, format(cm[i, j]*100, '.0f'), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
	fig.tight_layout()
	plt.show()
	return ax

if __name__ == '__main__':

	start_time = time.time()
	# Open images folder
	image_path = "TSRD/"
	image_paths = [str(p)for p in Path(image_path).glob('*.png')]

	# Extracct labels from image name
	labels = [int(str(p).split("/")[1].split("_")[0][-2:]) for p in image_paths]

	# Define the model
	#model = ResNet152(include_top=False, weights='imagenet', pooling='avg')
	#model = Xception(include_top=True, weights='imagenet', pooling='avg')
	model = NASNetLarge(include_top=False, weights='imagenet', pooling='avg')
	#model = InceptionResNetV2(include_top=False, weights='imagenet', pooling='avg')
	#model = cifar100vgg(train=False)
	#my_layer = model.model.layers[56]
	#model = Model(model.model.input, outputs=my_layer.output)
	model.layers[0].trainable = False
	dims = [331,331]

	# Define list to store vector values
	feature_vects = []

	# Extract vectors
	for idx, img in enumerate(image_paths):
		# Print
		print("Extracting vector features for image: "+str(idx))

		# Load and reshape the image to input to the network
		img = cv2.resize(cv2.imread(img), (dims[0], dims[1]))

		# Add the 4th dim (1, 224, 224, 3)
		img = np.expand_dims(img.copy(), axis=0)

		# Predict and store value
		feature_vects.append(model.predict(img).flatten())

	# Cluster the vectors
	clusters = KMeans(n_clusters=58).fit(feature_vects)

	#Evaluate similarity normalized_mutual_info_score
	nmi = normalized_mutual_info_score(labels, clusters.labels_, average_method='warn')
	print('Evaluation of similarity with normalized mutual score: ' + str(nmi))
	
	print("--- %s seconds ---" % (time.time() - start_time))
	# Extract the assignment matrix
	assignment_matrix = ass_matrix(labels, clusters.labels_)

	# Extract mapping real label -> cluster
	mapping = clusters_mapping(assignment_matrix)

	# Map labels to real labels
	corrected_labels = np.zeros(clusters.labels_.shape, dtype=np.int32)
	for idx, label in enumerate(clusters.labels_):
		for map_ in mapping:
			if map_[0] == label:
				corrected_labels[idx] = map_[1]

	# Sort them
	#labels, corrected_labels = zip(*sorted(zip(labels, corrected_labels), key=lambda x: x[0]))
	
	# Get confussion matrix
	confusion_matrix = confusion_matrix(labels, corrected_labels)
	
	# Extract label names
	labels_sort = labels.copy()
	labels_sort.sort()
	labels_names = np.unique(labels_sort)
	
	# Plot confussion matrix 
	plot_confusion_matrix(confusion_matrix, labels_names)

	# # Find accuracy of the matrix
	# acc = accuracy_score(labels, corrected_labels)
	# print("Final accuracy of the system: "+str(acc))

	# #Evaluate similarity adjusted rand score
	# ars = adjusted_rand_score(labels, corrected_labels)
	# print('Evaluation of similarity with adjusted rand score: ' + str(ars))








