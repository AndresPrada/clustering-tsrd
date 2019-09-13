from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import manifold
from sklearn.metrics import confusion_matrix, accuracy_score, adjusted_rand_score, normalized_mutual_info_score
from munkres import Munkres
import matplotlib.pyplot as plt
import os, glob, cv2, sys, pdb, time
from pathlib import Path
import numpy as np
from sklearn.utils.multiclass import unique_labels
from munkres import Munkres

def visualize_data(Z, labels, num_clusters):
    '''
        TSNE visualization of the points in latent space Z
        :param Z: Numpy array containing points in latent space in which clustering was performed
        :param labels: True labels - used for coloring points
        :param num_clusters: Total number of clusters
        :param title: filename where the plot should be saved
        :return: None - (side effect) saves clustering visualization plot in specified location
        '''
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    Z_tsne = tsne.fit_transform(Z)
    fig = plt.figure()
    plt.scatter(Z_tsne[:, 0], Z_tsne[:, 1], s=2, c=labels, cmap=plt.cm.get_cmap("jet", num_clusters))
    plt.colorbar(ticks=range(num_clusters))
    plt.show()

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

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
	"""
	    This function prints and plots the confusion matrix.
	    Normalization can be applied by setting `normalize=True`.
	    """
	if not title:
	    if normalize:
	        title = 'Normalized confusion matrix'
	    else:
	        title = 'Confusion matrix, without normalization'
	# Compute confusion matrix
	cm = confusion_matrix(y_true, y_pred)

	# Only use the labels that appear in the data
	classes = classes[unique_labels(y_true, y_pred)]
	if normalize:
	    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	    print("Normalized confusion matrix")
	else:
	    print('Confusion matrix, without normalization')
	    
	print(cm)

	fig, ax = plt.subplots()
	im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
	ax.figure.colorbar(im, ax=ax)
	# We want to show all ticks...
	ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes, title=title, ylabel='True label', xlabel='Cluster label')
	    
	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	fmt = '.0f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			if cm[i,j] > 0.01:
				ax.text(j, i, format(cm[i, j]*100, fmt), fontsize=8, ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
	fig.tight_layout()
	plt.rc('font', size=8)
	plt.show()
	return ax

if __name__ == '__main__':
	image_path = "tsrd-train/"
	image_paths = [str(p)for p in Path(image_path).glob('*.png')]

	# Extracct labels from image name
	labels = [int(str(p).split("/")[1].split("_")[0][-2:]) for p in image_paths]
	feature_vects = np.load('feats.npy')

	# Cluster the vectors
	#clusters_km = KMeans(n_clusters=58).fit(feature_vects)
	clusters_ag = AgglomerativeClustering(n_clusters=58).fit(feature_vects)

	#Evaluate similarity normalized_mutual_info_score
	#nmi_km = normalized_mutual_info_score(labels, clusters_km.labels_, average_method='warn')
	#print('Evaluation of similarity with normalized mutual score km: ' + str(nmi_km))

	#Evaluate similarity normalized_mutual_info_score
	nmi_ag = normalized_mutual_info_score(labels, clusters_ag.labels_, average_method='warn')
	print('Evaluation of similarity with normalized mutual score ag: ' + str(nmi_ag))

	# Visualize data
	# visualize_data(feature_vects, labels, 58)

	# Extract the assignment matrix
	assignment_matrix = ass_matrix(labels, clusters_ag.labels_)

	# Extract mapping real label -> cluster
	mapping = clusters_mapping(assignment_matrix)

	# Map labels to real labels
	corrected_labels = np.zeros(clusters_ag.labels_.shape, dtype=np.int32)
	for idx, label in enumerate(clusters_ag.labels_):
		for map_ in mapping:
			if map_[0] == label:
				corrected_labels[idx] = map_[1]
	# labels_sort = labels.copy()
	# labels_sort.sort()
	# labels_names = np.unique(labels_sort)
	# final_labels = []
	# for i in clusters_ag.labels_:
	# 	final_labels.append(corrected_labels[i])
    
	# corresp_labels = corrected_labels.tolist()
	# corresp_labels.sort()
    
    #conf_mat = confusion_matrix2(final_labels, labels)
    # Plot normalized confusion matrix
	#plot_confusion_matrix(labels, final_labels, classes=np.asarray(corresp_labels), normalize=True, title='Normalized confusion matrix')
	plot_confusion_matrix(labels, clusters_ag.labels_, np.unique(labels), normalize=True, title='Labeled images')
