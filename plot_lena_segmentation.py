'''
plot_lena_segmentation.py
Segmenting the picture of Lena in regions 
'''

import time 
import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt 

from sklearn.feature_extraction import image 
from sklearn.cluster import spectral_clustering 

lena = sp.misc.lena()

# downsample the image by a factor of 4 
lena = lena[::2, ::2] + lena[1::2, ::2] + \
        lena[::2, 1::2] + lena[1::2, 1::2]
lena = lena[::2, ::2] + lena[1::2, ::2] + \
        lena[::2, 1::2] + lena[1::2, 1::2]

# convert the image into a graph with the value of the gradient 
# on the edges

graph = image.img_to_graph(lena)

'''
take a decreasing function of the gradient: an exponential 
the smaller beta is, the more independent the segmentation is 
of the actual image. For beta=1, the segmentation is close to 
a voronoi 
'''

beta = 5 
eps = 1e-6 
graph.data = np.exp(-beta * graph.data / lena.std()) + eps 

'''
apply spectral clustering (this step goes much faster if you 
have pyamg installed)
'''

N_REGIONS = 11 

# visualize the resulting regions 

for assign_labels in ('kmeans', 'discretize'):
    t0 = time.time()
    labels = spectral_clustering(
        graph, n_clusters=N_REGIONS,
        assign_labels=assign_labels,
        random_state=1
    )

    t1 = time.time()
    labels = labels.reshape(lena.shape)
    plt.figure(figsize=(5, 5))
    plt.imshow(lena, cmap=plt.cm.gray)
    for l in range(N_REGIONS):
        plt.contour(
            labels==l, contours=1,
            colors=[plt.cm.spectral(l / float(N_REGIONS)), ])
    plt.xticks(())
    plt.yticks(())
    plt.title(
        'Spectral clustering: %s, %.2fs' % (assign_labels, (t1-t0)))

plt.show()
