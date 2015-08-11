'''
plot_dict_face_patches.py
Online learning of a dictionary of parts of faces
'''

import time 
import matplotlib.pyplot as plt 
import numpy as np 

from sklearn import datasets 
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.image import extract_patches_2d 

faces = datasets.fetch_olivetti_faces()

# learn the dictionary of images 

print('Learning the dictionary ...')
rng = np.random.RandomState(0)
kmeans = MiniBatchKMeans(
    n_clusters=81,
    random_state=rng,
    verbose=True,
)
patch_size = (20, 20)

buffer = []
index = 1 
t0 = time.time()

# the online learning part: cycle over the whole dataset 6 times 
index = 0 
for _ in range(6):
    for img in faces.images:
        data = extract_patches_2d(
            img, patch_size, max_patches=50, random_state=rng
        )
        data = np.reshape(data, (len(data), -1))
        buffer.append(data)
        index += 1
        if index % 10 == 0:
            data = np.concatenate(buffer, axis=0)
            data -= np.mean(data, axis=0)
            data /= np.std(data, axis=0)
            kmeans.partial_fit(data)
            buffer = []
        if index % 100 == 0:
            print('Partial fit of %4i out of %i'
                % (index, 6 * len(faces.images)))
dt = time.time() - t0 
print('done in %.2fs.' % dt)

# plot the results 
plt.figure(figsize=(4.2, 4))
for i, patch in enumerate(kmeans.cluster_centers_):
    plt.subplot(9, 9, i+1)
    plt.imshow(
        patch.reshape(patch_size),
        cmap=plt.cm.gray,
        interpolation='nearest'
    )
    plt.xticks(())
    plt.yticks(())

plt.suptitle('Patches of faces\nTrain time %.2fs on %d patches' 
    % (dt, 8 * len(faces.images)), fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

plt.show()
