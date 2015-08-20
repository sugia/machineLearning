'''
plot_covariance_estimation.py 

Shrinkage covariance estimation: LedoitWolf vs OAS and max-likelihood 

'''

import numpy as np 
import matplotlib.pyplot as plt 
from scipy import linalg 
from sklearn.covariance import LedoitWolf, OAS, ShrunkCovariance, \
    log_likelihood, empirical_covariance 
from sklearn.grid_search import GridSearchCV 

# generate sample data 
n_features, n_samples = 40, 20 
np.random.seed(42) 
base_X_train = np.random.normal(size=(n_samples, n_features))
base_X_test = np.random.normal(size=(n_samples, n_features))

# color samples 
coloring_matrix = np.random.normal(size=(n_features, n_features))
X_train = np.dot(base_X_train, coloring_matrix)
X_test = np.dot(base_X_test, coloring_matrix)

# compute the likelihood on test data 
# spanning a range of possible shrinkage coefficient values 
shrinkages = np.logspace(-2, 0, 30) 
negative_logliks = [
    -ShrunkCovariance(shrinkage=s).fit(X_train).score(X_test)
    for s in shrinkages 
]

# under the ground-truth model, which we whould not have 
# access to in real settings 

real_cov = np.dot(coloring_matrix.T, coloring_matrix)
emp_cov = empirical_covariance(X_train)
loglik_real = -log_likelihood(emp_cov, linalg.inv(real_cov))

# compute different approaches to setting the parameter 
# gridsearch for an optimal shrinkage coefficient 

tuned_parameters = [
    {'shrinkage': shrinkages}
]
cv = GridSearchCV(ShrunkCovariance(), tuned_parameters)

cv.fit(X_train)

# Ledoit-Wolf optimal shrinkage coefficient estimate 
lw = LedoitWolf()
loglik_lw = lw.fit(X_train).score(X_test) 

# OAS coefficient estimate 
oa = OAS()
loglik_oa = oa.fit(X_train).score(X_test) 

# plot results 
fig = plt.figure()
plt.title('Regularized covariance: likelihood and shrinkage coefficient')
plt.xlabel('Regularization parameter: shrinkage coefficient') 
plt.ylabel('Error: negative log-likelihood on test data')

# range shirnkage curve 
plt.loglog(shrinkages, negative_logliks, label='Negative log-likelihood')

plt.plot(plt.xlim(), 2 * [loglik_real], '--r',
    label='Real covariance likelihood') 

# adjust view 
lik_max = np.amax(negative_logliks)
lik_min = np.amin(negative_logliks)
ymin = lik_min - 6.0 * np.log((plt.ylim()[1] - plt.ylim()[0])) 
ymax = lik_max + 10.0 * np.log(lik_max - lik_min) 
xmin = shrinkages[0] 
xmax = shrinkages[-1] 

# lw likelihood 
plt.vlines(lw.shrinkage_, ymin, -loglik_lw, color='magenta',
    linewidth=3, label='Ledoit-Wolf estimate')

# oas likelihood 
plt.vlines(
    oa.shrinkage_, ymin, -loglik_oa, color='purple',
    linewidth=3, label='OAS estimate'
)

# best CV estimator likelihood 
plt.vlines(
    cv.best_estimator_.shrinkage, ymin,
    -cv.best_estimator_.score(X_test), color='cyan',
    linewidth=3, label='Cross-validataion best estimate'
)

plt.ylim(ymin, ymax)
plt.xlim(xmin, xmax)
plt.legend()

plt.show()
