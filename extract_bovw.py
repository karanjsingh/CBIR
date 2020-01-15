# import the necessary packages
from pyimagesearch.ir import bagsofvisualwords
from pyimagesearch.indexer import bovwindexer
import argparse
import pickle
import h5py
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--featuresdb", required=True,
	help="Path the features database")
ap.add_argument("-c", "--codebook", required=True,
	help="Path to the codebook")
ap.add_argument("-b", "--bovwdb", required=True,
	help="Path to where the bag-of-visual-words database will be stored")
ap.add_argument("-d", "--idf", required=True,
	help="Path to inverse document frequency counts will be stored")
ap.add_argument("-s", "--maxbuffersize", type=int, default=500,
	help="Maximum buffer size for # of features to be stored in memory")
args = vars(ap.parse_args())

# load the codebook vocabulary and initialize BOVWs transformer
vocab = pickle.loads(open(args["codebook"],"rb").read())

bovw = bagsofvisualwords.BagsOfVisualWords(vocab)

# open the features database and initialize the bovw indexer
featuresDB = h5py.File(args["featuresdb"],mode="r")

bi = bovwindexer.BOVWIndexer(bovw.codebook.shape[0], args["bovwdb"],
	estNumImages = featuresDB["image_ids"].shape[0],
	maxBufferSize=args["maxbuffersize"])

# loop over the image IDs and index
for (i,(imageID,offset)) in enumerate(zip(featuresDB["image_ids"], featuresDB["index"])):
	# check to see if progress should be displayed
	if i > 0 and i % 10 ==0:
		bi._debug("processed {} images".format(i),msgType="[PROGRESS]")

	# extract the feature vectors for the current image using the starting and
	# ending offsets (while ignoring the keypoints) and then quantize the
	# features to construct the bag-of-visual-words histogram
	features = featuresDB["features"][offset[0]:offset[1]][:, 2:]
	hist = bovw.describe(features)
	# add the bovw to the index
	# add the bag-of-visual-words to the index
	bi.add(hist)
 
# close the features database and finish the indexing process
featuresDB.close()
bi.finish()
 
# dump the inverse document frequency counts to file
f = open(args["idf"], "wb")
f.write(pickle.dumps(bi.df(method="idf")))
f.close()
