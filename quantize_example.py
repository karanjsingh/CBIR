# import the necessary packages
from __future__ import print_function
from pyimagesearch.ir import bagsofvisualwords
from sklearn.metrics import pairwise
import numpy as np

# randomly generate the vocabulary/cluster centers along with the feature
# vectors -- we'll generate 10 feature vectors containing 6 real-valued
# entries, along with a codebook containing 3 'visual words'
np.random.seed(42)
vocab = np.random.uniform(size=(4,6))# general vocab words we get from the dataset
features = np.random.uniform(size=(12,6)) # all these features belong to single image, we get single histogram at the end
print("[INFO] vocab :\n {} \n".format(vocab)) 
print("[INFO] features : \n {} \n".format(features))
# we have 4 words(vocabulary) to quantize our vector and 12 features to play with
# initialize our bag of visual words histogram -- it will contain 4 entries,
# one for each of the possible visual words
hist=np.zeros((4,),dtype="int32")

# loop over the feature vectors
for (i,f) in enumerate(features):
	# compute the Euclidean distance between the current feature vector
	# and the 4 visual words; then, find the index of the visual word
	# with the smallest distance
	D= pairwise.euclidean_distances(f.reshape(1,-1),Y=vocab)
	#print(D)
	j = np.argmin(D)

	print("[INFO] Closest visual word to {} feature is {} ".format(i,j))
	hist[j] += 1
	print("[INFO] Update histogram: {}".format(hist))

# apply our `BagOfVisualWords` class and make this process super
# speedy
bovw = bagsofvisualwords.BagsOfVisualWords(vocab, sparse=False)
hist = bovw.describe(features)
print("[INFO] BOVW histogram: {}".format(hist))







