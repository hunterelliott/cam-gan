import numpy as np
import tensorflow as tf
import random
import math
import os
from scipy import misc
import time
from skimage.transform import resize
import imageio

#----------------------------------
# --- parameters -------------

# --- GPUs --

gpuID = 1 #GPU to use. -1 is CPU
if gpuID >= 0:
    #Make only this GPU visible to TF
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpuID)
    gpuString = '/gpu:0' #TF will now think the visible GPU is 0...
else:
    gpuString = '/cpu:0' #Will still use multiple cores if present



# --- Architecture -- 

batchSize = 128;#minibatch size
#layerWidth = (128, 256, 512, 1024) #Num feature maps per conv layer. (Matches figure in Radford & Metz)
#batchNormLayer = (False,True,True, True) #Whether to apply batchnorm to a layer. Radford & Metz do not norm image layers (discrim input and gen output), but this is handled below
layerWidth = (128, 256, 512, 1024, 2056) #Num feature maps per conv layer. 
batchNormLayer = (False,True,True, True, True) #Whether to apply batchnorm to a layer. Radford & Metz do not norm image layers (discrim input and gen output), but this is handled below
#layerWidth = (32,64,128)
#batchNormLayer = (False,True,True)
nLayers = len(layerWidth)
ipLayerWidth = (1024,) #width of first fully connected layers. Last will have nClasses output
batchNormIPLayer = (False,)
#ipLayerWidth = (128,)
#batchNormIPLayer = (False,)
kernelSize = (5,5) #filter kernel support
maxImLoadPerClass = int(2e3) #For datasets too large to fit in memory load a random subset

zDims = 100 #Dimensionality of noise vector which is input of generator


# --- Optimization -- 

baseLR_D = 2e-4 #Base learning rate for discriminator
#baseLR_D = 1e-3
#baseLR_D = 1e-2
#baseLR_G = baseLR_D*1.9 #LR for generator. Works OK with 2 iters per D before batchNorm
#baseLR_G = baseLR_D * 1.8
#baseLR_G = baseLR_D * .7
baseLR_G = baseLR_D
beta1 = .5 #Beta 1 1st momement momentum in ADAM optimizer. Radford and Metz recommend lower than default of .5
nIters = int(2e4) #Number of iterations)
logInterval = 20 #Display loss every N iters

k_D = 1 #Number of discrimator updates per iteration (this was usually 1 in goodfellow 2014)
k_G = 2 #Number of generator updates per iteration (this was always 1 in goodfellow 2014, but 2 in github example tensorflow DCGAN implementation

# --- dataset

#testParentDir = '/home/hunter/Desktop/TEMP_LOCAL/Data/MNIST/Test'
#testParentDir = '/ssd/MNIST/Test'
#trainParentDir = '/home/hunter/Desktop/TEMP_LOCAL/Data/MNIST/Train'
#trainParentDir = '/ssd/MNIST/Train'
#trainParentDir = '/ssd/DCGAN/data/celebs' #celebA dataset
#trainParentDir = '/ssd/DCGAN/data/CIAN'
#trainParentDir = '/ssd/DeadNetLocal/trainSet_Combined_Feb12_And_8_25_2016_allSickDie/train'
#trainParentDir = '/ssd/CAMELYON/Train'
trainParentDir = '/media/hunter/storage/Google_Drive/Google Photos/'

# --- restore

restoreModel = False
#restoreFile = '/home/he19/files/CellBiology/IDAC/Projects/CAMELYON/GAN/Snapshots/CAMELYON_Subs1and2_128_iter16000.ckpt'
#restoreFile = '/home/he19/files/CellBiology/IDAC/Projects/CAMELYON/GAN/Snapshots/CAMELYON_Balanced_EarlyStart50kPer3_128_iter6000.ckpt'
#restoreFile = '/home/he19/files/CellBiology/IDAC/Projects/CAMELYON/GAN/Snapshots/CAMELYON_Balanced_EarlyStart60kPer3_128_resume_iter5000.ckpt'
#restoreFile = '/home/he19/files/CellBiology/IDAC/Projects/CAMELYON/GAN/Snapshots/CAMELYON_Balanced_EarlyStart60kPer3_128_resume2_iter5000.ckpt'
#restoreFile = '/home/he19/files/CellBiology/IDAC/Projects/CAMELYON/GAN/Snapshots/CAMELYON_Balanced_EarlyStart60kPer3_128_resume3_iter3000.ckpt'
restoreFile = None

#restoreFile = ''

# --- Output --

showPlots = True; #Show various figures at the end
saveInterval = int(1e3) #How frequently to save snapshots. If zero no saving.
#saveInterval = int(0)

if saveInterval  > 0:
    #snapshotDir = '/home/he19/files/CellBiology/IDAC/Hunter/GANs/DCGAN/MNIST_Testing/Snapshots'
    #baseName = 'CelebA_128'
    #baseName = 'CIAN_128'
    #baseName = 'CIAN'
    #baseName ='DeadNet'
    #baseName = 'CAMELYON_Subs1and2_128_resume'
    #baseName = 'CAMELYON_Balanced_EarlyStart50kPer3_128'
    #baseName = 'CAMELYON_Balanced_EarlyStart60kPer3_128_resume'
    #baseName = 'CAMELYON_Balanced_EarlyStart60kPer3_128_resume4'
    baseName = 'myPhotos_v0'
    snapshotDir = '/media/hunter/1E52113152110F61/shared/Training_Experiments/DCGAN/myphotos/v0'





if showPlots:
    from matplotlib import pyplot as plt


with tf.variable_scope("discriminators_shared") as scope, tf.device(gpuString):

    #Image pre-processing function
    def preProcIm(im):

        im = resize(im,(128,128))
        #im = misc.imresize(im,(64,64))#Resize to match architecture in Radford & Metz
        #im = misc.imresize(im,(32,32))#Resize to power of 2
        #im = misc.imresize(im,(24,24))
        im = im.astype(float)/255
        im = (im - .5) * 2;
        if np.ndim(im) == 2:
            im = np.expand_dims(im,2)

        return im

    #----------------------------------
    # ---- load data ----------------


    #Find class sub-directories. We assume same structure in test and train
    classNames = sorted(os.listdir(trainParentDir))
    classNames = [name for name in classNames if os.path.isdir(os.path.join(trainParentDir,name))]

    nClasses = len(classNames)

    print('Found ' + str(nClasses) + ' class folders:')
    print(classNames)

    nTrainPerClass = np.zeros(nClasses,dtype=np.int)
    #nTestPerClass = np.zeros(nClasses,dtype=np.int)

    trainFiles = [ [] for _ in range(nClasses)]
    #testFiles = [ [] for _ in range(nClasses)]

    trainIms =  [ [] for _ in range(nClasses)]
    #testIms =  [ [] for _ in range(nClasses)]

    trainLabels = [ [] for _ in range(nClasses)]
    #testLabels = [ [] for _ in range(nClasses)]

    for iClass in range(0, nClasses):

        #Get all file names and count images
        currTrainDir = trainParentDir + os.path.sep + classNames[iClass]
        for root, directory, files in os.walk(currTrainDir):
            for file in files:
                if any([ext in file for ext in ['.png', '.jpg','.JPG','.PNG','.BMP']]):
                    trainFiles[iClass].append(os.path.join(root,file))

        nFilesCurr = int(len(trainFiles[iClass]))
        nTrainPerClass[iClass] = min(nFilesCurr,maxImLoadPerClass)

        #currTestDir = testParentDir + os.path.sep + classNames[iClass]
        #testFiles[iClass] = os.listdir(currTestDir)
        #nTestPerClass[iClass] = len(testFiles[iClass])

        #Read all images into memory since it's MNIST...

        #tmp = preProcIm(misc.imread(currTrainDir + os.path.sep + trainFiles[iClass][0]))
        tmp = preProcIm(imageio.imread(trainFiles[iClass][0]))

        imSize = tmp.shape[0:2]
        if tmp.ndim > 2:
            nImChan = tmp.shape[2]
        else:
            nImChan = 1;


        print("Image size:" + str(imSize))

        trainIms[iClass] = np.zeros(imSize + (nTrainPerClass[iClass], nImChan)) #Use a list of np arrays to allow for varying numbers of iamges per class
        #testIms[iClass] = np.zeros(imSize + (nTestPerClass[iClass],))

        # Randomize image order in case a subset is loaded
        random.shuffle(trainFiles[iClass])
        print("Loading " + str(nTrainPerClass[iClass]) + " images out of " + str(nFilesCurr) + " total for class " + str(iClass))

        iIm = 0
        while iIm < nTrainPerClass[iClass]:

            try:
                newIm = preProcIm(imageio.imread(trainFiles[iClass][iIm]))
                trainIms[iClass][:, :, iIm, :] = newIm
            except:
                print("error reading image " + str(iIm) + ", skipping")

            iIm += 1

            if iIm%500 == 0:

                print("Loaded image " + str(iIm) + " of " + str(nTrainPerClass[iClass]))
        #for iIm in range(0,nTestPerClass[iClass]):
        #	testIms[iClass][:,:,iIm] = preProcIm(misc.imread(currTestDir + os.path.sep + testFiles[iClass][iIm]))


        #Setup labels
        trainLabels[iClass] = np.zeros((nTrainPerClass[iClass],nClasses),dtype=np.bool)
        trainLabels[iClass][:,iClass] = True
        #testLabels[iClass] = np.zeros((nTestPerClass[iClass],nClasses),dtype=np.bool)
        #testLabels[iClass][:,iClass] = True

    #Define final IP layer output size - scalar with probability datum is from data distribution
    ipLayerWidth = ipLayerWidth + (1,)
    batchNormIPLayer = batchNormIPLayer + (False,)
    nIPLayers = len(ipLayerWidth)

    print('Number of training examples per class:')
    print(nTrainPerClass)
    #print('Number of test examples per class:')
    #print(nTestPerClass)

    #Un-list the data & labels for ease of use below
    trainIms = np.concatenate(trainIms,2)
    nTrainTot = sum(nTrainPerClass)
    #testIms = np.concatenate(testIms,2)
    #nTestTot = sum(nTestPerClass)

    trainLabels = np.concatenate(trainLabels[:])
    #testLabels = np.concatenate(testLabels[:])

    data = tf.placeholder("float",shape=(batchSize, imSize[0], imSize[1], nImChan))
    labels = tf.placeholder("float",shape=(batchSize,nClasses))

    def construct_discriminator(inTens,layerWidth,ipLayerWidth,kernelSize,batchNormLayer,batchNormIPLayer):

        #----------------------------------
        # --- define discriminator ---------

        leakSlope = 0.2 #Slope for leaky ReLU

        nLayers = len(layerWidth)
        nIPLayers = len(ipLayerWidth)

        W_convD = [ [] for _ in range(nLayers)]
        b_convD = [ [] for _ in range(nLayers)]
        conv_D = [ [] for _ in range(nLayers)] #feature maps
        convNorm_D  = [ [] for _ in range(nLayers)] #batch normed layers
        batchMean = [ [] for _ in range(nLayers)]
        batchVar = [ [] for _ in range(nLayers)]
        beta = [ [] for _ in range(nLayers)] #Learned scale
        gamma = [ [] for _ in range(nLayers)] #Learned offset
        convH_D = [ [] for _ in range(nLayers)] #activations (after nonlinearity)

        W_ipD = [ [] for _ in range(nIPLayers)]
        b_ipD = [ [] for _ in range(nIPLayers)]
        ip_D = [ [] for _ in range(nIPLayers)]
        ipNorm_D = [ [] for _ in range(nIPLayers)]
        ipBatchMean = [ [] for _ in range(nIPLayers)]
        ipBatchVar = [ [] for _ in range(nIPLayers)]
        ipBeta = [ [] for _ in range(nLayers)] #Learned scale
        ipGamma = [ [] for _ in range(nLayers)] #Learned offset
        ipH_D = [ [] for _ in range(nIPLayers)] #activations


        #Create the conv/relu/pool layers
        for iLayer in range(0,nLayers):

            print("Creating discrimator conv layer " + str(iLayer))

            #initialize weights and biases
            if iLayer == 0:
                kernelDepth = nImChan
            else:
                kernelDepth = layerWidth[iLayer-1]



            #Use get_variable so we can share these variables accross the two discriminators
            W_convD[iLayer] = tf.get_variable('W_conv' + str(iLayer) + '_D',(kernelSize[0],kernelSize[1],kernelDepth,layerWidth[iLayer]),initializer=tf.random_normal_initializer(0,0.02))
            b_convD[iLayer] = tf.get_variable('b_conv' + str(iLayer) + '_D',layerWidth[iLayer],initializer=tf.constant_initializer(0.0))
            #W_convD[iLayer] = tf.Variable(tf.truncated_normal((kernelSize[0],kernelSize[1],kernelDepth,layerWidth[iLayer]),stddev=0.02 ))
            #b_convD[iLayer] = tf.Variable(tf.constant(0.1,shape=(layerWidth[iLayer],)))

            #create conv layer
            if iLayer == 0:
                inLayer = inTens
            else:
                inLayer = convH_D[iLayer-1]

            conv_D[iLayer] = tf.nn.conv2d(inLayer,W_convD[iLayer],strides=(1,2,2,1),padding='SAME')

            if batchNormLayer[iLayer]:
                batchMean[iLayer], batchVar[iLayer] = tf.nn.moments(conv_D[iLayer],[0])
                #beta[iLayer] = tf.Variable(tf.zeros(layerWidth[iLayer]))
                #gamma[iLayer] = tf.Variable(tf.ones(layerWidth[iLayer]))
                beta[iLayer] = tf.get_variable('beta_conv' + str(iLayer) + '_D',layerWidth[iLayer],initializer=tf.constant_initializer(0.1))
                gamma[iLayer] = tf.get_variable('gamma_conv' + str(iLayer) + '_D',layerWidth[iLayer],initializer=tf.constant_initializer(1.0))
                #beta[iLayer] = tf.zeros(layerWidth[iLayer]) #In DCGAN paper they didn't learn these on MNIST
                #gamma[iLayer] = tf.ones(layerWidth[iLayer])
                convNorm_D[iLayer] = tf.nn.batch_normalization(conv_D[iLayer],batchMean[iLayer],batchVar[iLayer],beta[iLayer],gamma[iLayer],1e-5)
            else:
                convNorm_D[iLayer] = conv_D[iLayer] + b_convD[iLayer]

            #activations after nonlinearity
            #convH_D[iLayer] = tf.nn.relu(convNorm_D[iLayer])
            convH_D[iLayer] = tf.maximum(convNorm_D[iLayer],convNorm_D[iLayer] * leakSlope) #leaky ReLU


        conv_end_flat_D = tf.reshape(convH_D[nLayers-1],(-1,convH_D[nLayers-1].get_shape()[1::].num_elements()))

        #Add the fully connected (inner product) layers
        for iLayer in range(0,nIPLayers):

            print("Creating discriminator fully-connected layer " + str(iLayer))

            if iLayer == 0:
                inSize = convH_D[nLayers-1].get_shape()[1::].num_elements() #First dimension is batch size
            else:
                inSize = ipLayerWidth[iLayer-1]

            W_ipD[iLayer] = tf.get_variable('W_ip' + str(iLayer) + '_D',(inSize,ipLayerWidth[iLayer]),initializer=tf.random_normal_initializer(0,0.02))
            b_ipD[iLayer] = tf.get_variable('b_ip' + str(iLayer) + '_D',ipLayerWidth[iLayer],initializer=tf.constant_initializer(0.1))
            #W_ipD[iLayer] = tf.Variable(tf.truncated_normal((inSize,ipLayerWidth[iLayer]),stddev=0.02))
            #b_ipD[iLayer] = tf.Variable(tf.constant(0.1,shape=(ipLayerWidth[iLayer],)))

            if iLayer == 0:
                inLayer = conv_end_flat_D
            else:
                inLayer = ipH_D[iLayer-1]

            ip_D[iLayer] = tf.matmul(inLayer,W_ipD[iLayer])

            if batchNormIPLayer[iLayer]:
                ipBatchMean[iLayer], ipBatchVar[iLayer] = tf.nn.moments(ip_D[iLayer],[0])
                ipBeta[iLayer] = tf.get_variable('beta_ip' + str(iLayer) + '_D',ipLayerWidth[iLayer],initializer=tf.constant_initializer(0.1))
                ipGamma[iLayer] = tf.get_variable('gamma_ip' + str(iLayer) + '_D',ipLayerWidth[iLayer],initializer=tf.constant_initializer(1.0))
                #ipBeta[iLayer] = tf.Variable(tf.zeros(ipLayerWidth[iLayer]))
                #ipGamma[iLayer] = tf.Variable(tf.ones(ipLayerWidth[iLayer]))
                #ipBeta[iLayer] = tf.zeros(ipLayerWidth[iLayer]) #In DCGAN paper they didn't learn these on MNIST
                #ipGamma[iLayer] = tf.ones(ipLayerWidth[iLayer])
                ipNorm_D[iLayer] = tf.nn.batch_normalization(ip_D[iLayer],ipBatchMean[iLayer],ipBatchVar[iLayer],ipBeta[iLayer],ipGamma[iLayer],1e-5)
            else:
                ipNorm_D[iLayer] = ip_D[iLayer] + b_ipD[iLayer]

            if iLayer == (nIPLayers-1):
                #ipH_D[iLayer] = tf.nn.sigmoid(ipNorm_D[iLayer])#Scalar probability output
                ipH_D[iLayer] = ipNorm_D[iLayer] #No activation as we use logit loss
            else:
                #ipH_D[iLayer] = tf.nn.relu(ipNorm_D[iLayer])
                ipH_D[iLayer] = tf.maximum(ipNorm_D[iLayer],ipNorm_D[iLayer] * leakSlope) #leaky ReLU

        #for cleanliness - pointer to full forward pass
        forward = ipH_D[nIPLayers-1]

        return forward



    #----------------------------------
    # --- define generator ---------


    W_convG = [ [] for _ in range(nLayers)]
    b_convG = [ [] for _ in range(nLayers)]
    conv_G = [ [] for _ in range(nLayers)] #feature maps
    convNorm_G = [ [] for _ in range(nLayers)] #feature maps
    convH_G = [ [] for _ in range(nLayers)] #activations (after nonlinearity)

    batchMean_G = [ [] for _ in range(nLayers)]
    batchVar_G = [ [] for _ in range(nLayers)]
    beta_G = [ [] for _ in range(nLayers)] #Learned scale
    gamma_G = [ [] for _ in range(nLayers)] #Learned offset

    W_ipG = [ [] for _ in range(nIPLayers)]
    b_ipG = [ [] for _ in range(nIPLayers)]
    ip_G = [ [] for _ in range(nIPLayers)]
    ipNorm_G = [ [] for _ in range(nIPLayers)]
    ipH_G = [ [] for _ in range(nIPLayers)] #activations

    ipBatchMean_G = [ [] for _ in range(nLayers)]
    ipBatchVar_G = [ [] for _ in range(nLayers)]
    ipBeta_G = [ [] for _ in range(nLayers)] #Learned scale
    ipGamma_G = [ [] for _ in range(nLayers)] #Learned offset


    #Reverse structure and Last up-conv layer must map to image depth
    layerWidth_G = layerWidth[::-1] + (nImChan,)
    batchNormLayer_G = batchNormLayer[::-1] + (False,) #Radford & Metz say output layer norm causes instability
    nLayers_G = len(layerWidth_G)-1 #First feature map is produced by inner product
    firstFeatDim = int(np.prod(imSize) / np.power(4,nLayers) * layerWidth_G[0])
    if ipLayerWidth[0] != firstFeatDim: #Last IP layer will be reshaped to make first feature map
        ipLayerWidth_G = ipLayerWidth[-2::-1] + (firstFeatDim,)
        batchNormIPLayer_G = batchNormIPLayer[::-1] + (False,)
    else:
        ipLayerWidth_G = ipLayerWidth[-2::-1]
        batchNormIPLayer_G = batchNormIPLayer[::-1]

    nIPLayers_G = len(ipLayerWidth_G)
    #create the noise vector layer
    Z = tf.placeholder("float",shape=(batchSize, zDims))

    #create the fully connected layers
    for iLayer in range(0,nIPLayers_G):

        print("Creating generator fully-connected layer " + str(iLayer))

        if iLayer == 0:
            inSize = zDims
        else:
            inSize = ipLayerWidth_G[iLayer-1]

        W_ipG[iLayer] = tf.Variable(tf.truncated_normal((inSize,ipLayerWidth_G[iLayer]),stddev=0.02),name='W_ip_' +str(iLayer) + '_G')
        b_ipG[iLayer] = tf.Variable(tf.constant(0.1,shape=(ipLayerWidth_G[iLayer],)),name='b_ip' + str(iLayer) + '_G')

        if iLayer == 0:
            inLayer = Z
        else:
            inLayer = ipH_G[iLayer-1]

        ip_G[iLayer] = tf.matmul(inLayer,W_ipG[iLayer])

        if batchNormIPLayer_G[iLayer]:
            ipBatchMean_G[iLayer], ipBatchVar_G[iLayer] = tf.nn.moments(ip_G[iLayer],[0])
            ipBeta_G[iLayer] = tf.Variable(tf.zeros(ipLayerWidth_G[iLayer]),name='beta_ip' + str(iLayer) + '_G')
            ipGamma_G[iLayer] = tf.Variable(tf.ones(ipLayerWidth_G[iLayer]),name='gamma_ip' + str(iLayer) + '_G')
            #ipBeta_G[iLayer] = tf.zeros(ipLayerWidth_G[iLayer])
            #ipGamma_G[iLayer] = tf.ones(ipLayerWidth_G[iLayer])
            ipNorm_G[iLayer] = tf.nn.batch_normalization(ip_G[iLayer],ipBatchMean_G[iLayer],ipBatchVar_G[iLayer],ipBeta_G[iLayer],ipGamma_G[iLayer],1e-5)
        else:
            ipNorm_G[iLayer] = ip_G[iLayer] + b_ipG[iLayer]


        ipH_G[iLayer] = tf.nn.relu(ipNorm_G[iLayer])

    #reshape last IP layer into first feature creation layer
    ip_end_square = tf.reshape(ipH_G[nIPLayers_G-1],np.concatenate(((batchSize,),np.divide(imSize,np.power(2,nLayers_G)),(layerWidth[nLayers_G-1],))))

    #create up-convolution layers
    for iLayer in range(0,nLayers_G):

        print("Creating generator up-convolution layer " + str(iLayer))

        W_convG[iLayer] = tf.Variable(tf.truncated_normal((kernelSize[0],kernelSize[1],layerWidth_G[iLayer+1],layerWidth_G[iLayer]),stddev=0.02 ),name='W_conv' + str(iLayer) + '_G')
        b_convG[iLayer] = tf.Variable(tf.constant(0.1,shape=(layerWidth_G[iLayer+1],)),name='b_conv' + str(iLayer) + '_G')

        if iLayer == 0:
            inLayer = ip_end_square
        else:
            inLayer = convH_G[iLayer-1]

        outSize = (np.concatenate(((batchSize,),np.divide(imSize,np.power(2,nLayers_G-iLayer-1)),(layerWidth_G[iLayer+1],))))
        conv_G[iLayer] = tf.nn.conv2d_transpose(inLayer,W_convG[iLayer],outSize.astype(np.int32),strides=(1,2,2,1),padding='SAME',data_format='NHWC')

        if batchNormLayer_G[iLayer]:
            batchMean_G[iLayer], batchVar_G[iLayer] = tf.nn.moments(conv_G[iLayer],[0])
            beta_G[iLayer] = tf.Variable(tf.zeros(layerWidth_G[iLayer+1]),name='beta_conv' + str(iLayer) + '_G')
            gamma_G[iLayer] = tf.Variable(tf.ones(layerWidth_G[iLayer+1]),name='gamma_conv' + str(iLayer) + '_G')
            #beta_G[iLayer] = tf.zeros(layerWidth_G[iLayer+1])
            #gamma_G[iLayer] = tf.ones(layerWidth_G[iLayer+1])
            convNorm_G[iLayer] = tf.nn.batch_normalization(conv_G[iLayer],batchMean_G[iLayer],batchVar_G[iLayer],beta_G[iLayer],gamma_G[iLayer],1e-5)
        else:
            convNorm_G[iLayer] = conv_G[iLayer] + b_convG[iLayer]

        if iLayer == (nLayers_G-1):
            convH_G[iLayer] = tf.nn.tanh(convNorm_G[iLayer])#tanh to map to image range
        else:
            convH_G[iLayer] = tf.nn.relu(convNorm_G[iLayer])
            #convH_G[iLayer] = tf.nn.tanh(convNorm_G[iLayer])

    forward_G = convH_G[nLayers_G-1] #generator forward pass

    def plotResults(loss_log_D,loss_log_G,accuracy_log_D,accuracy_log_DofG,batchSize,genIms):

        cf = plt.figure()
        cf.add_subplot(2,2,1)
        plt.plot(loss_log_D)
        cf.add_subplot(2,2,2)
        plt.plot(loss_log_G)
        cf.add_subplot(2,2,3)
        plt.plot(accuracy_log_D)
        cf.add_subplot(2,2,4)
        plt.plot(accuracy_log_DofG)

        imPanelShape = (4,4)
        nImShow = np.prod(imPanelShape)
        iSampShow = np.random.randint(0,batchSize-1,nImShow)
        plt.rcParams['image.interpolation'] = 'nearest'
        plt.rcParams['image.cmap'] = 'gray'

        cf = plt.figure()
        for iShow in range(0,nImShow):
            cf.add_subplot(imPanelShape[0],imPanelShape[1],iShow+1)
            plt.imshow(np.squeeze((genIms[iSampShow[iShow],:,:,:] / 2) + .5))


        plt.show()

    # -- Create discriminators
    forward_D = construct_discriminator(data,layerWidth,ipLayerWidth,kernelSize,batchNormLayer,batchNormIPLayer)
    scope.reuse_variables()
    forward_DofG = construct_discriminator(forward_G,layerWidth,ipLayerWidth,kernelSize,batchNormLayer,batchNormIPLayer) #Discrimator output on generated images


    #----------------------------------
    # ------ loss layers ----------


    #use loss from original Goodfellow 2014 GAN paper
    #loss_D = -tf.reduce_mean(tf.log(forward_D) + tf.log(1-forward_DofG))#Invert because discriminator maximizes (old loss using sigmoid output)
    loss_D_X = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=forward_D,labels=tf.ones_like(forward_D))) #Discriminator loss on data
    loss_D_GofZ = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=forward_DofG,labels=tf.zeros_like(forward_DofG))) #Discriminator loss on G output
    loss_D = loss_D_X + loss_D_GofZ
    #loss_G = tf.reduce_mean(tf.log(1-forward_DofG))#Generator minimizes
    loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=forward_DofG,labels=tf.ones_like(forward_DofG))) #Generator loss
    #loss_G = -tf.reduce_mean(tf.log(forward_DofG))#Generator maximizes in alternative loss from Goodfellow 2014

    #Discriminator accuracy
    evaluation_D = tf.greater(tf.nn.sigmoid(forward_D),.5) #Discriminator prediction is correct on data if probability of being data > .5
    accuracy_D = tf.reduce_mean(tf.cast(evaluation_D,"float"))
    evaluation_DofG = tf.less(tf.nn.sigmoid(forward_DofG),.5) #Discriminator prediction is correct on G if probability of being data < .5
    accuracy_DofG = tf.reduce_mean(tf.cast(evaluation_DofG,"float"))



    #----------------------------------
    # ----- optimization -----

print("Starting optimization...")
startTime = time.time();

loss_log_D = np.zeros(int(math.floor(nIters/logInterval)))
loss_log_G = np.zeros(int(math.floor(nIters/logInterval)))
accuracy_log_D = np.zeros(int(math.floor(nIters/logInterval)))
accuracy_log_DofG = np.zeros(int(math.floor(nIters/logInterval)))

sess = tf.InteractiveSession() #Need interactive session?? switch to regular...
#sess = tf.Session() #Need interactive session?? switch to regular...

#Get variable lists so we can independently update the generator and discriminator
vars_All = tf.trainable_variables()
vars_G = [var for var in vars_All if '_G' in var.name]
vars_D = [var for var in vars_All if '_D' in var.name]

sess.run(tf.global_variables_initializer())

optimizer_D = tf.train.AdamOptimizer(baseLR_D,beta1).minimize(loss_D,var_list=vars_D)
optimizer_G = tf.train.AdamOptimizer(baseLR_G,beta1).minimize(loss_G,var_list=vars_G)

sess.run(tf.global_variables_initializer())

if saveInterval  > 0:
    with tf.device('/cpu:0'):
        saver = tf.train.Saver()

if restoreModel:
    print("restoring model from file: " + restoreFile)
    saver.restore(sess,restoreFile)

iLog = -1

for i in range(nIters):


    # -- Update the discriminator

    for k in range(0,k_D):

        iTrainBatch = np.random.randint(0,nTrainTot-1,batchSize)
        #trainImBatch = np.expand_dims(np.transpose(trainIms[:,:,iTrainBatch],(2,0,1,3)),3)
        trainImBatch = np.transpose(trainIms[:,:,iTrainBatch],(2,0,1,3))

        trainZBatch = np.random.uniform(-1,1,(batchSize,zDims))

        optimizer_D.run(feed_dict={data:trainImBatch, Z:trainZBatch})


    # -- Update the generator

    for k in range(k_G):

        trainZBatch = np.random.uniform(-1,1,(batchSize,zDims))

        optimizer_G.run(feed_dict={Z:trainZBatch})



    #Log loss intermittently
    if i%logInterval == 0:

        iLog += 1
        loss_log_D[iLog] = loss_D.eval(feed_dict={data:trainImBatch, Z:trainZBatch})
        loss_log_G[iLog] = loss_G.eval(feed_dict={Z:trainZBatch})

        accuracy_log_D[iLog] =accuracy_D.eval(feed_dict={data:trainImBatch})
        accuracy_log_DofG[iLog] =accuracy_DofG.eval(feed_dict={Z:trainZBatch})

        print("Discrimator loss: " + str(loss_log_D[iLog]) + ", generator loss: " + str(loss_log_G[iLog]))
        print("Discrimator accuracy on data : " + str(accuracy_log_D[iLog]) + ", on generator output: " + str(accuracy_log_DofG[iLog]))
        print("Iteration " + str(i) + " of " + str(nIters))

    if saveInterval > 0 and (i%saveInterval == 0 or i == (nIters - 1)):
        outFile = snapshotDir + os.path.sep + baseName + '_iter' + str(i) + '.ckpt'
        save_path = saver.save(sess,outFile)
        print("Model saved in file: " + save_path)

#Generate some examples
genIms = forward_G.eval(feed_dict={Z: np.random.uniform(-1,1,(batchSize,zDims))})

endTime = time.time();
print("Finished optimization. Elapsed time: ")
print(endTime-startTime)

#sess.close()


if showPlots:

    plotResults(loss_log_D,loss_log_G,accuracy_log_D,accuracy_log_DofG,batchSize,genIms)

