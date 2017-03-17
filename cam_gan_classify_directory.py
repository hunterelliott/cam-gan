#Performs inference using a classifier trained from the GAN discriminator

import tensorflow as tf
import time
import os
import cam_gan_ops as cgo

# --- Device ----- #

devID = -1  #GPU to use. -1 is CPU
if devID >= 0:
	#Make only this GPU visible to TF
	os.environ["CUDA_VISIBLE_DEVICES"]=str(devID)
	devString = '/gpu:0' #TF will now think the visible GPU is 0...
else:
	devString = '/cpu:0' #Will still use multiple cores if present
	os.environ["CUDA_VISIBLE_DEVICES"]=''




# ---- input ----- #

#modelFile = '/media/hunter/New Volume/camgan_xfertraining/Snapshots/CAMELYON_xfer_FitMLPOnly_iter2000.ckpt'
#modelFile = '/media/extra/hunter_temp/Snapshots/CAMELYON_xfer_FitMLPOnly_iter2000.ckpt'
modelFile = '/home/hunter/Desktop/TEMP_LOCAL/camgan_xfertraining/Snapshots/CAMELYON_xfer_FitMLPOnly_iter2000.ckpt'

dataDir = '/home/hunter/Desktop/TEMP_LOCAL/Data/CAMELYON/MixedForInferenceTest'
dataFiles = os.listdir(dataDir)


nFiles = len(dataFiles)

for i in range(nFiles):
	dataFiles[i] = os.path.join(dataDir, dataFiles[i])


print("Found " + str(nFiles) + " files")

# ----- init ------ #


#Reads one image from the current file queue
def get_image(file_queue):

	
	reader = tf.WholeFileReader()
	key,image_data = reader.read(file_queue)

	image = tf.image.decode_image(image_data)	


	return image

def get_image_batch(file_queue,batch_size):


	images = [[] for _ in range(batch_size)]
	for i in range(batch_size):
		images[i] = get_image(file_queue)		


	images = tf.stack(images,0)

	return images

start_time = time.time();
print("Initializing...")
batch_size = 128

#Initialize the input queue ops on the CPU
with tf.device('/cpu:0'):

	file_queue = tf.train.string_input_producer(dataFiles,capacity=1e4,shared_name=
		    'chief_queue',num_epochs=1,shuffle=False)

	image = get_image(file_queue)

	images = get_image_batch(file_queue,batch_size)



# ---- inference ------ #

with tf.variable_scope("discriminators_shared") as scope:#, tf.device(devString): #, tf.Session() as sess





	# -- session, model init

	#Some ops in our GAN don't support GPU so allow TF to do the device placement when necessary
	sessConfig = tf.ConfigProto(allow_soft_placement=True)
#	sessConfig.gpu_options.allow_growth=True
	sess = tf.Session(config=sessConfig)
	sess.run(tf.local_variables_initializer())


	print("Restoring model from " + modelFile )
	saver = tf.train.import_meta_graph(modelFile + '.meta')
	saver.restore(sess,modelFile)
	#Get the operations and tensors we'll need
	predict = tf.get_default_graph().get_tensor_by_name("discriminators_shared/forward_SoftMax_C:0")
	data = tf.get_default_graph().get_tensor_by_name("discriminators_shared/Placeholder:0")
	batchSize = data.get_shape()[0]


	sess.run(tf.global_variables_initializer()) #TEMP FOR DEBUG!!!!


 	# -- queue init
 	#Do this AFTER the model restoration to avoid weird errors
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord,sess=sess)


	end_time = time.time();
	print("Finished initialization. Elapsed time: " + str(end_time-start_time) + " seconds.")
	print("Starting inference...")
	start_time = time.time()

	#im_shape = sess.run(im_shape)

	# -- process the images!

	try:
	
		nIm_proc = 0
		while not(coord.should_stop()):

			#Get a batch of images
			ims = sess.run(images)
			ims = sess.run(tf.image.resize_images(ims,[128, 128]))

			#Run inference on them
			pred = sess.run(predict,feed_dict={data:ims})


			nIm_proc += ims.shape[0]

			if nIm_proc%(10*batch_size) == 0:
				print("Finished " + str(nIm_proc) + " images")

	except tf.errors.OutOfRangeError:
		print("Done! Completed " + str(nIm_proc) + " images total")

	finally:
		coord.request_stop()


	coord.join(threads)
	sess.close()

end_time = time.time();
print("Finished inference. Elapsed time: " + str(end_time-start_time) + " seconds.")
