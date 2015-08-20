'''
plot_digits_last_image.py 

The Digit Dataset 
'''

from sklearn import datasets 
import matplotlib.pyplot as plt 

# load the digits dataset 
digits = datasets.load_digits()

# display the first digit 
plt.figure(1, figsize=(3, 3)) 
plt.imshow(
    digits.images[-1], 
    cmap = plt.cm.gray_r, 
    interpolation = 'nearest',
)
plt.show()
