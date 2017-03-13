import tensorflow as tf
import numpy as np
import os
import math as math
from matplotlib import pyplot as plt
import time

import cam_gan_ops as cgo

#----------------------------
#--------- Parameters -------
#----------------------------


# ---------- Specify unsupervised model -------#
#This model, trained in a fully unsupervised manner, is used for initialization

initModel = '/home/he19/files/CellBiology/IDAC/Projects/CAMELYON/GAN/Snapshots/CAMELYON_Balanced_EarlyStart60kPer3_128_resume4_iter5000.ckpt'


# ---- Specify data ----- 

#trainParentDir = '/ssd/CAMELYON/Train'
#trainParentDir = '/ssd/CAMELYON/SmallDevSet/Train'
#testParentDir = '/ssd/CAMELYON/SmallDevSet/Test'
testParentDir = '/ssd/CAMELYON/Test'
trainParentDir = '/ssd/CAMELYON/Train'
#testParentDir = '/ssd/CAMELYON/Test'
maxImPerClass = int(50) #Since we are doing kludigy all-data-in-RAM for speed we sub-sample the training datasets
maxImPerClassTest = int(2.5e4)

# ---- Classifier module architecture  --- #

#Width of fully-connected (inner product) layers that will be connected to the last ip layer of the discriminator.
ipLayerWidth = (1024, 512, 256)
nIPLayers = len(ipLayerWidth)

# --- Optimization --- #

baseLR = 1e-4 #Base step size for ADAM optimizer
nIters = int(1e4) #Number of iterations

# --- Loggin / saving ---#

logInterval = int(10) #How fequently to calculate val accuracy/log loss
#outputDir = '/home/he19/files/CellBiology/IDAC/Projects/CAMELYON/GAN_Xfer/FinetuneFitC_75kTrain25kTest_Res4i5k'
outputDir = '/home/he19/files/CellBiology/IDAC/Projects/CAMELYON/GAN_Xfer/FinetuneFitC_CombLoss_75kTrain25kTest_Res4i5k'

#outputDir = '/home/he19/files/CellBiology/IDAC/Projects/CAMELYON/GAN_Xfer/FitAll_25kTrain25kTest_Res4i5k'
#outputDir = '/home/he19/files/CellBiology/IDAC/Projects/CAMELYON/GAN_Xfer/FineTuneFitC_CombLoss_50Train25kTest_Res4i5k'
#outputDir = '/home/he19/files/CellBiology/IDAC/Projects/CAMELYON/GAN_Xfer/FitCOnly_50Train25kTest_Res4i5k'


testOutDir = outputDir + os.path.sep + 'Test'
trainOutDir = outputDir + os.path.sep + 'Train'
if not(os.path.exists(testOutDir)):
	os.makedirs(testOutDir)
if not(os.path.exists(trainOutDir)):
	os.makedirs(trainOutDir)

saveInterval = int(1e3) #How frequently to save snapshots. If zero no saving.
snapOutDir = outputDir + os.path.sep + 'Snapshots'
baseSnapName = 'CAMELYON_xfer_'
if not(os.path.exists(snapOutDir)):
	os.makedirs(snapOutDir)


# --- GPUs -----

gpuID = 3 #GPU to use. -1 is CPU
if gpuID >= 0:
	#Make only this GPU visible to TF
	os.environ["CUDA_VISIBLE_DEVICES"]=str(gpuID)
	gpuString = '/gpu:0' #TF will now think the visible GPU is 0...
else:
	gpuString = '/cpu:0' #Will still use multiple cores if present


#----------------------------
#----- Intialization --------
#----------------------------


# --- Load data ---- 

print("Loading training data from " + trainParentDir)
trainIms,trainLabels,classNames,nImsPerClass = cgo.load_class_data(trainParentDir,maxImPerClass)
nClasses = len(classNames)

print("Loading test data from " + testParentDir)
testIms,testLabels,_,nImsPerClassTest = cgo.load_class_data(testParentDir,maxImPerClassTest)

sessConfig = tf.ConfigProto(allow_soft_placement=True)

with tf.variable_scope("discriminators_shared") as scope, tf.device(gpuString):

	
	
	sess = tf.InteractiveSession(config=sessConfig)

	#Restore the graph and variable values from the unsupervised training phase
	print("Restoring model from " + initModel )
	saver = tf.train.import_meta_graph(initModel + '.meta')
	saver.restore(sess,initModel)
	
	#Get the tensors we'll need for our classifier

	#The last inner-product layer - will be the input features for our classifier
	ipLast_D = tf.get_default_graph().get_tensor_by_name("discriminators_shared/Maximum_5:0")
	#Data placeholder
	data = tf.get_default_graph().get_tensor_by_name("discriminators_shared/Placeholder:0")
	batchSize = data.get_shape()[0]
	Z = tf.get_default_graph().get_tensor_by_name("discriminators_shared/Placeholder_2:0")

	#The Discriminator loss on generator images (for combined loss)
	loss_DofGofZ = tf.get_default_graph().get_tensor_by_name("discriminators_shared/Mean_1:0")

	print("Done")

	
	#---- Model construction --------#	


	#Add a classifier module (MLP) that uses the discriminators learned representation
	print("Constructing classifier module...")
	currInput = ipLast_D
	for iLayer in range(0,nIPLayers):

		currInput = cgo.ip_layer(currInput,ipLayerWidth[iLayer],iLayer)

	#Add linear logit output layer 
	forward_C = cgo.linear_layer(currInput,nClasses,nIPLayers)
	#And softmax for evaluation (loss uses logits)
	forward_C_SoftMax = tf.nn.softmax(forward_C,-1,name='forward_SoftMax_C')


	print("Done")


	# ---- Loss, accuracy layers ---- #

	labels = tf.placeholder("float",shape=(batchSize,nClasses))
	
	#Loss on classifier only 
	#loss_C = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(forward_C,labels),name='loss_C')
	#Combined loss on discriminator and classifier
	loss_C = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(forward_C,labels),name='loss_C') + loss_DofGofZ
	
	
	
	evaluation = tf.equal(tf.argmax(forward_C_SoftMax,1),tf.argmax(labels,1))	
	accuracy = tf.reduce_mean(tf.cast(evaluation,"float"))

	#Use summary to log loss over time
	tf.summary.scalar("Loss",loss_C)
	tf.summary.scalar("Accuracy",accuracy)

	summary_op = tf.summary.merge_all()
	
	#----------------------------
	#----- Optimization  --------
	#----------------------------



	#Find only variables associated with classification module
	vars_all_t = tf.trainable_variables()	
	vars_C = [var for var in vars_all_t if '_C' in var.name]

	#Only optimize the classifier weights
	optimizer = tf.train.AdamOptimizer(baseLR).minimize(loss_C,var_list=vars_C)
	#Optimize all weights (fine-tuning on Discriminator)
	#optimizer = tf.train.AdamOptimizer(baseLR).minimize(loss_C)
	
	#TEMP - for testing variable preservation
	Wtest = tf.get_default_graph().get_tensor_by_name("discriminators_shared/W_ip_0_G:0")
	Wtest = Wtest.eval()

	print("Initializing new variables")	
	#Initialize only uninitialized variables to preserve our starting model
	#There HAS to be a better way to do this...????
	initVarNames = sess.run(tf.report_uninitialized_variables())
	nVarsInit = len(initVarNames)
	initVars = [[] for _ in range(nVarsInit)]
	vars_all = tf.global_variables()
	nVarTot = len(vars_all)
	nInit = 0
	for iVar in range(0,nVarTot):
		if not(sess.run(tf.is_variable_initialized(vars_all[iVar]))):
			initVars[nInit] = vars_all[iVar]
			nInit += 1			

	sess.run(tf.variables_initializer(initVars))

	# print("Initializing ALL variables")	
	# sess.run(tf.global_variables_initializer())

	test_summary_writer = tf.summary.FileWriter(testOutDir,graph=tf.get_default_graph())
	train_summary_writer = tf.summary.FileWriter(trainOutDir,graph=tf.get_default_graph())

	print("Starting optimization...")
	startTime = time.time();	

	iVal = -1
	for iIter in range(0,nIters):

		imBatch,labelBatch = cgo.get_class_batch(trainIms,trainLabels,batchSize)
		imBatchTest,labelBatchTest = cgo.get_class_batch(testIms,testLabels,batchSize)
		ZBatch = np.random.uniform(-1,1,(batchSize,Z.get_shape()[1]))	#For combined loss experiments

		optimizer.run(feed_dict={data:imBatch, labels:labelBatch, Z:ZBatch})

		if iIter%logInterval == 0:
			iVal += 1

			summary_train,train_loss,train_acc= sess.run([summary_op,loss_C,accuracy],feed_dict={data:imBatch, labels:labelBatch, Z:ZBatch})
			summary_test,test_loss,test_acc= sess.run([summary_op,loss_C,accuracy],feed_dict={data:imBatchTest, labels:labelBatch, Z:ZBatch})
			
			train_summary_writer.add_summary(summary_train,iIter)
			test_summary_writer.add_summary(summary_test,iIter)
			
			print("Training loss " + str(train_loss) + ", accuracy " + str(train_acc) + " Test loss " + str(test_loss) + ", accuracy " + str(test_acc))			
			print("Iteration " + str(iIter) + " of " + str(nIters))

		if saveInterval > 0 and (iIter%saveInterval == 0 or iIter == (nIters - 1)):
			outFile = snapOutDir + os.path.sep + baseSnapName + '_iter' + str(iIter) + '.ckpt'
			save_path = saver.save(sess,outFile)
  			print("Model saved in file: " + save_path)



