# Technical Task Motor AI

#### Author: Andrés Prada

## Description: 
This notebook presents the solution for the technical task proposed for the selection process in Motor AI. This task requires to compress the images in the dataset TRDS through an existing pre-trained model and extract the feature vector at a certain layer. These feature vectors need to be clustered afterward using a clustering algorithm. Finally, the performance of the model needs to be evaluated using the proper metrics.

## Methodology
This task can be divided into two smaller tasks, that each of them is independent of the other one. The first task would consist of choosing a pre-trained model that could perform fine with the dataset given. Also, it is important to wisely choose the layer on which the feature vectors are extracted would be. The second task is related to the clustering technique that could be used to cluster those feature vectors, and the metrics that could show how well the model performs. For the first task, the selection of an efficient model has its limitations:

* Since one of the constraints is to get rid of the training step, the number of possible solutions drastically. The first idea was to find a pre-trained model that was used for a similar problem. Ideally, any model which was trained on some dataset containing Traffic Signs of any type would perform better than other models trained on other types of data. The first approach was to research this problem. In 'Deep neural network for traffic sign recognition systems: An analysis of spatial transformers and stochastic optimisation methods', a novel CNN is presented. This architecture relies on spatial transformers and three different convolution layers. The implemented code could be found in GitHub, however, it is implemented in Lua, and the weights attached use some CUDNN layers, making it impossible to parse the weights file without a CUDA supported machine.

* The second option was to research any pre-trained model used for classification tasks and compare the performance of those networks on different datasets. All of the pre-trained models in Keras were pre-trained on the ImageNet dataset. Checking the benchmark (`https://keras.io/applications/#documentation-for-individual-models`) of the different available pre-trained models on this dataset, it is possible to observe the models that perform the best for the problem. The best results on the pre-trained dataset were obtained by NASNetLarge, followed by InceptionResNetV2 and Xception. Despite these models could perform reasonably good for the TSRD dataset, it does not mean that any of them could be the best model for the task. They also have a main drawback: they take as input images of very big sizes compared to TSRD dataset: 331x331 for the NASNet and 299x299 for the InceptionResNetV2 and Xception. This expansion in each of the images could not exploit all of the possibilities of such deep models.

*  A third alternative to this first task was found when the pre-trained model in `https://github.com/geifmany/cifar-vgg` with the weights in the CIFAR100 dataset was found. This was trained on the VGG16 model. This pre-trained model had the advantage that the images taken as input were resized to 32x32, which are closer to the 96x96 size that the images in TSRD have. 

To this point, four different architectures were selected to model this problem. The goal would be to build the whole pipeline and stick with the model that could perform the best. The layer on which the feature vectors would depend on the architecture. For the VGG16, they would be extracted after the last batch normalization, before the last Dense layer that performs the classification itself.  In the rest of the networks, those vectors would be extracted after the last convolutional layer, where a global average is applied. The size of these vectors will be of 512 for the VGG16 model and 2048 for the rest of them. The feature vectors are extracted in this position due to its compression size of the features. This is where the feature vectors are better represented, just before the classification step.

The second part of this task consists of the clustering of all of these vectors and evaluate, with the appropriate metric, the chosen model. The most famous and used technique that could be used straight forward for this problem is the KMeans algorithm. This algorithm assigns each of the features vectors to a cluster in which each feature vector belongs to the cluster with the nearest mean, serving as a model of the cluster. Also, Agglomerative Clustering could provide efficient performance for the task.

Finally, a metric is needed to validate the clustering method. According to the research, the most common metric to validate unsupervised learning is the evaluation of similarity with Normalized Mutual Information (NMI). This metric is computed as is shown in the image below. It takes as input the true labels and the predicted labels obtained for each cluster.

![](https://github.com/AndresPrada/clustering-tsrd/blob/master/nmi.png){:height="50%" width="50%"} This was found in https://course.ccs.neu.edu/cs6140sp15/7_locality_cluster/Assignment-6/NMI.pdf

One final step is that once all the feature vectors are clustered into different sets, each of those sets would have a random label from 0 to 58 (which is the number of classes in the TSRD dataset). The problem now was to assign the correct label - or at least, an approximation - of the category label. To do this, the Hungarian algorithm (or Munkres algorithm) comes handy. This algorithm aims to maximize the cost of a given matrix. In this case, a cost matrix was created by defining a 58x58 matrix, where each entry for the y-axis corresponded to the true label of each image, and the x-axis to the predicted cluster's label. The Munkres algorithm creates a map that assigns the cluster with the maximum number of instances to the true label. This process can be leaky as it is not granted that the maximum number of instances for a cluster maps to the correct label. However, it is an approximate way to map each of the classes to a category. Furthermore, it allows us to explore which samples were clustered correctly together or not, and to extract the confusion matrix.

##Results
All of the networks explained above were tested using all of the images in the dataset, containing both training and testing sets, a total number of images of 6164. Each of the architectures was tested using the following pipeline:

* Images were resized to the corresponding input size of each network and added a fourth dimension. For example, the tensor input for the NASNetLarge is [1, 331, 331, 3].
* The images are fed into the network and the feature vectors for each of the images is extracted at each point.
* These vectors were clustered with KMeans clustering since the Agglomerative Clustering proved to perform worse than KMeans.
* The NMI was computed.

Finally, for the network that achieved the highest NMI, the last step was to compute the confusion matrix through the Menkres algorithm. The results for the different networks were the following:

|  Model | Weights | Time(s) | NMI |
|---|---|---|---|
| VGG16 | CIFAR100 | 92.35 | 0.37782 |
| InceptionResNetV2 |  ImageNet | 2233.29 |0.2382 |
|  NASNetLarge | ImageNet |   |
|  Xception | ImageNet |   |


Since the model that performed the best was the *****, it will be used to analyze the performance of the clustering. To do this, a confusion matrix will be created as explained above. This confusion matrix can be seen in the figure below. For visualization purposes, the normalized float numbers have been multiplied by 100 and remove the decimals.

![]()

Although given the number of classes it is difficult to properly observe each of the classes, some of them show very clear results. For example, this algorithm have clustered together images from categories ** and **. More examples of this can be found below:


On the other hand, for classes 



