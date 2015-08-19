'''
plot_sparse_cov.py 

Sparse inverse covariance estimation 

'''

import numpy as np 
from scipy import linalg 
from sklearn.datasets import make_sparse_spd_matrix 
from sklearn.covariance import GraphLassoCV, ledoit_wolf 
import matplotlib.pyplot as plt 

# generate the data 
n_samples = 60 
n_features = 20 

prng = np.random.RandomState(1)
prec = make_sparse_spd_matrix(
    n_features,
    alpha = 0.98, 
    smallest_coef = 0.4,
    largest_coef = 0.7,
    random_state = prng
)

cov = linalg.inv(prec)
d = np.sqrt(np.diag(cov))
cov /= d 
cov /= d[:, np.newaxis] 
prec *= d 
prec *= d[:, np.newaxis]
X = prng.multivariate_normal(
    np.zeros(n_features),
    cov, 
    size=n_samples
)
X -= X.mean(axis=0)
X /= X.std(axis=0) 

# estimate the covariance 
emp_cov = np.dot(X.T, X) / n_samples 

model = GraphLassoCV()
model.fit(X)
cov_ = model.covariance_ 
prec_ = model.precision_ 

lw_cov_, _ = ledoit_wolf(X)
lw_prec_ = linalg.inv(lw_cov_)

# plot the results 
plt.figure(figsize=(10, 6)) 
plt.subplots_adjust(left=0.02, right=0.98) 

# plot the covariances 
covs = [
    ('Empirical', emp_cov),
    ('Ledoit-Wolf', lw_cov_),
    ('GraphLasso', cov_),
    ('True', cov)
]

vmax = cov_.max()
for i, (name, this_cov) in enumerate(covs):
    plt.subplot(2, 4, i+1)
    plt.imshow(
        this_cov, 
        interpolation='nearest',
        vmin = -vmax, 
        vmax = vmax,
        cmap = plt.cm.RdBu_r
    )

    plt.xticks(())
    plt.yticks(())
    plt.title('%s covariance' % name)

# plot the precisions 
precs = [
    ('Empirical', linalg.inv(emp_cov)),
    ('Ledoit-Wolf', lw_prec_),
    ('GraphLasso', prec_),
    ('True', prec)
]

vmax = 0.9 * prec_.max()

for i, (name, this_prec) in enumerate(precs):
    ax = plt.subplot(2, 4, i+5)
    plt.imshow(
        np.ma.masked_equal(this_prec, 0),
        interpolation = 'nearest',
        vmin = -vmax,
        vmax = vmax,
        cmap = plt.cm.RdBu_r
    )
    plt.xticks(())
    plt.yticks(())
    plt.title('%s precision' % name)
    ax.set_axis_bgcolor('0.7') 

# plot the model selection metric 
plt.figure(figsize=(4, 3)) 
plt.axes([0.2, 0.15, 0.75, 0.7])
plt.plot(model.cv_alphas_, np.mean(model.grid_scores, axis=1), 'o-')
plt.axvline(model.alpha_, color='0.5') 
plt.title('Model seleciton')
plt.ylabel('Cross-validation score') 
plt.xlabel('alpha') 

plt.show()
