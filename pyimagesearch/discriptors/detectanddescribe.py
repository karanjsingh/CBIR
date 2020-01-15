# import necessary pachkages
import numpy as np

class DetectAndDescribe:
	def __init__(self,detector,descriptor):
		#store the key point detectors and local invariant descriptors
		self.detector=detector
		self.descriptor=descriptor

	def describe(self, image,useKpList = True):
		# detect the keypoints in the image and extract local invariant descriptors
		kps = self.detector.detect(image)
		(kps,descs)=self.descriptor.compute(image,kps)

		# if there are nokey points or descriptors, return None
		if len(kps) == 0:
			return (None,None)

		# check to see if the keypoints should be converted to the numpy arrays
		if useKpList:
			kps = np.int0([kp.pt for kp in kps])

		# return the tuple of keypoints and descriptors
		return (kps,descs)








