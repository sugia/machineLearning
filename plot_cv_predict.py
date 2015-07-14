'''
plot_cv_predict.py
Plotting Cross-Validated Predictions
This example shows how to use cross_val_predict 
to visualize prediction errors.
http://scikit-learn.org/stable/auto_examples/plot_cv_predict.html
'''

from sklearn import datasets
from sklearn.cross_validation import cross_val_predict 
from sklearn import linear_model 
import matplotlib.pyplot as plt 

if __name__ == '__main__':
    lr = linear_model.LinearRegression()
    boston = datasets.load_boston()
    y = boston.target 

    '''
    cross_val_predict returns an array of the same size as `y` 
    where each entry is a prediction obtained by cross validated.
    '''

    predicted = cross_val_predict(lr, boston.data, y, cv=10)

    fig, ax = plt.subplots()
    ax.scatter(y, predicted)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    fig.show()

    flag = raw_input()
