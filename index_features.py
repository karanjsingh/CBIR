# import the necessary packages
from __future__ import print_function
from pyimagesearch.discriptors import detectanddescribe
from pyimagesearch.indexer import featureindexer
from imutils.feature import factories
from imutils import paths
import argparse
import imutils
import cv2

# construct the argument parser and pass the arguments
ap=argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,
	help="Path to the directory that contains images to be indexed")
ap.add_argument("-f","--featuresdb",required=True,
	help="Path to where the features dataset will be stored")
ap.add_argument("-a","--approximages",type=int,default=500,
	help="Approximate # of images in the dataset")
ap.add_argument("-b","--maxbuffersize",type=int,default=50000,
	help="Maximum buffer size for number of features of images stored in the memory")
args=vars(ap.parse_args())

# initialise the keypoint detector, local invariant descriptor, 
# and the descriptor pipeline
detector = factories.FeatureDetector_create("SURF")
descriptor= factories.DescriptorExtractor_create("RootSIFT")
dad = detectanddescribe.DetectAndDescribe(detector, descriptor)

# initialize the feature indexer
fi = featureindexer.FeatureIndexer(args["featuresdb"],estNumImages=args["approximages"],
	maxBufferSize=args["maxbuffersize"],verbose=True)

#loop over theimages in the dataset
for (i,imagePath) in enumerate(list(paths.list_images(args["dataset"]))):
	# chech to see if progress should be displayed
	if i > 0 and i % 10 == 0:
		fi._debug("processed {} images".format(i), msgType="[PROGRESS]")

	## extract the filename (i.e the unique image ID) from the
	# image path , then load the image itself
	filename = imagePath[imagePath.rfind("/")+1:]
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width = 320)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# describe the image
	(kps, descs) = dad.describe(image)

	# if either the key points or descriptors are none, thenignore the image
	if kps is None or descs is None:
		continue
	
	# index the features
	fi.add(filename,kps,descs)

# finish the indexing
fi.finish()









































