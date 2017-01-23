import tensorflow as tf
import numpy as np
import os
import math as math
from scipy import misc

def ip_layer(inTens,outWidth,iLayer):

	#Slope for leaky ReLU
	leakSlope = 0.2

	inWidth = inTens.get_shape()[1]

	W = tf.get_variable('W_ip' + str(iLayer) + '_C',(inWidth,outWidth),initializer=tf.random_normal_initializer(0,0.02))
	b = tf.get_variable('b_ip' + str(iLayer) + '_C',outWidth,initializer = tf.constant_initializer(0.1))

	ip = tf.matmul(inTens,W) + b

	#Leaky ReLU 
	ipH = tf.maximum(ip,ip*leakSlope,name='ipH_' + str(iLayer))

	return ipH

def linear_layer(inTens,outWidth,iLayer):
	

	inWidth = inTens.get_shape()[1]

	W = tf.get_variable('W_lin' + str(iLayer) + '_C',(inWidth,outWidth),initializer=tf.random_normal_initializer(0,0.02))
	b = tf.get_variable('b_lin' + str(iLayer) + '_C',outWidth,initializer = tf.constant_initializer(0.1))

	h = tf.matmul(inTens,W) + b	

	return h


def pre_proc_im(im):

		im = misc.imresize(im,(128,128)) #We downsample by 2x since we have kludgy all-in-RAM data ingestion
		im = im.astype(float)/255
		im = (im - .5) * 2; #zero centered [-1 1] range
		if np.ndim(im) == 2: #Handle graysale images
			im = np.expand_dims(im,2)

		return im			


def load_class_data(parentDir,maxImPerClass):

	#Find class sub-directories.
	classNames = sorted(os.listdir(parentDir))

	nClasses = len(classNames)

	print('Found ' + str(nClasses) + ' class folders:')
	print(classNames)

	nImsPerClass = np.zeros(nClasses,dtype=np.int)	
	imFiles = [ [] for _ in range(nClasses)]
	classIms =  [ [] for _ in range(nClasses)]
	classLabels = [ [] for _ in range(nClasses)]


	for iClass in range(0, nClasses):

		#Get all file names and count images
		currTrainDir = parentDir + os.path.sep + classNames[iClass]
		imFiles[iClass] = os.listdir(currTrainDir)
		nFilesCurr = int(len(imFiles[iClass]))
		nImsPerClass[iClass] = min(nFilesCurr,maxImPerClass)					

		tmp = pre_proc_im(misc.imread(currTrainDir + os.path.sep + imFiles[iClass][0]))

		imSize = tmp.shape[0:2]
		if tmp.ndim > 2:
			nImChan = tmp.shape[2]
		else:
			nImChan = 1;


		print("Image size:" + str(imSize))

		classIms[iClass] = np.zeros(imSize + (nImsPerClass[iClass], nImChan)) #Use a list of np arrays to allow for varying numbers of iamges per class 		
		iLoadIms = np.random.randint(0,nFilesCurr-1,nImsPerClass[iClass]) #Randomize image order in case a subset is loaded
		print("Loading " + str(nImsPerClass[iClass]) + " images out of " + str(nFilesCurr) + " total for class " + str(iClass))

		for iIm in range(0,nImsPerClass[iClass]):
			classIms[iClass][:,:,iIm,:] = pre_proc_im(misc.imread(currTrainDir + os.path.sep + imFiles[iClass][iLoadIms[iIm]]))
			if iIm%500 == 0:
				print("Loaded image " + str(iIm) + " of " + str(nImsPerClass[iClass]))

		#Setup labels as one-hot vectors
		classLabels[iClass] = np.zeros((nImsPerClass[iClass],nClasses),dtype=np.bool)
		classLabels[iClass][:,iClass] = True		

	print('Number of examples per class:')
	print(nImsPerClass)	

	return classIms,classLabels,classNames,nImsPerClass

def get_class_batch(imsIn,labelsIn,batchSize):

	#Returns a balanced batch from the input dataset, permuted to the correct dimension order
	#NOTE: currently assuming batchsize is multiple of number of classes
	nClasses = len(imsIn)
	nPer = batchSize / nClasses
	
	
	imBatch = [[] for _ in range(nClasses)]
	labelBatch = [[] for _ in range(nClasses)]
	for iClass in range(0,nClasses):
		
		iBatch =  np.random.randint(0,imsIn[iClass].shape[2],nPer)
		imBatch[iClass] = imsIn[iClass][:,:,iBatch,:]
		labelBatch[iClass] = labelsIn[iClass][iBatch,:]
	
	imBatch = np.concatenate(imBatch,2)
	imBatch = np.transpose(imBatch,(2,0,1,3))

	labelBatch = np.concatenate(labelBatch,0)

	return imBatch,labelBatch



