#Performs inference using a classifier trained from the GAN discriminator

import tensorflow as tf
import time
import os

# --- Device ----- #

devID = -1 #GPU to use. -1 is CPU
if devID >= 0:
	#Make only this GPU visible to TF
	os.environ["CUDA_VISIBLE_DEVICES"]=str(devID)
	devString = '/gpu:0' #TF will now think the visible GPU is 0...
else:
	devString = '/cpu:0' #Will still use multiple cores if present
	os.environ["CUDA_VISIBLE_DEVICES"]=''




# ---- input ----- #

modelFile = '/media/hunter/New Volume/camgan_xfertraining/Snapshots/CAMELYON_xfer_FitMLPOnly_iter2000.ckpt.index'
# dataFiles = ['/home/hunter/Desktop/TEMP_LOCAL/CEMELYON_MixedMedTest_1.tfrecords',
# 	'/home/hunter/Desktop/TEMP_LOCAL/CEMELYON_MixedMedTest_1.tfrecords']
dataFiles = ['/home/hunter/Desktop/TEMP_LOCAL/data/MNIST/train.tfrecords',
			 '/home/hunter/Desktop/TEMP_LOCAL/data/MNIST/test.tfrecords']

nFiles = len(dataFiles)


# ----- init ------ #


#Reads one image from the current tfrecord file
def get_image(file_queue):
	
	
	reader = tf.TFRecordReader()
	key, serial_record = reader.read(file_queue)
 	record = tf.parse_single_example(serial_record,features={'image_raw': tf.FixedLenFeature([], tf.string),
 		'height':tf.FixedLenFeature([],tf.int64),
 		'width':tf.FixedLenFeature([],tf.int64),
 		'depth':tf.FixedLenFeature([],tf.int64)})
	image = tf.decode_raw(record['image_raw'], tf.uint8)

	image.set_shape([784])
	#im_sz = [record['height'], record['width'], record['depth']]

	return image

start_time = time.time();


# ---- inference ------ #

with tf.variable_scope("discriminators_shared") as scope, tf.device(devString): #, tf.Session() as sess


	sess = tf.Session()

	file_queue = tf.train.string_input_producer(dataFiles,capacity=1e4,shared_name=
	    'chief_queue',num_epochs=1,shuffle=False)



	image = get_image(file_queue)

	batch_size = 64
	# images = tf.train.shuffle_batch(
 #        [image], batch_size=batch_size, num_threads=2,
 #        capacity=5000 + 3 * batch_size,
 #        # Ensures a minimum amount of shuffling of examples.
 #        min_after_dequeue=1000)
	images = tf.train.batch(
        [image], batch_size=batch_size, num_threads=2,
        capacity=5000 + 3 * batch_size)
 	
 # 	im_list = [[image] for _ in range(8)]
	# images = tf.train.batch_join(
 #        im_list, batch_size=batch_size,
 #        capacity=5000 + 3 * batch_size)


	sess.run(tf.local_variables_initializer())

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord,sess=sess)

	try:
	
		nIm_proc = 0
		while not(coord.should_stop()):

			#Get a batch of images
			ims = sess.run(images)

			nIm_proc += ims.shape[0]

			if nIm_proc%100 == 0:
				print("Finished " + str(nIm_proc) + " images")

	except tf.errors.OutOfRangeError:
		print("Done! Completed " + str(nIm_proc) + " images total")

	finally:
		coord.request_stop()


	coord.join(threads)
	sess.close()

end_time = time.time();
print("Finished conversion. Elapsed time: " + str(end_time-start_time) + " seconds.")

# 	reader = tf.TFRecordReader()
# #	key, serial_record = sess.run(reader.read(file_queue))
# 	key, serial_record = reader.read(file_queue)
#  	record = tf.parse_single_example(serial_record,features={'image_raw': tf.FixedLenFeature([], tf.string),
#  		'height':tf.FixedLenFeature([],tf.int64),
#  		'width':tf.FixedLenFeature([],tf.int64),
#  		'depth':tf.FixedLenFeature([],tf.int64)})
# 	image = tf.decode_raw(record['image_raw'], tf.uint8)

#	im = sess.run(get_image(file_queue))
	

	#a = bas


	#Initialize the variables the file queue needs
 #  	sess.run(tf.local_variables_initializer())

	# coord = tf.train.Coordinator()
	# threads = tf.train.start_queue_runners(coord=coord,sess=sess)

	# #Restore the graph and variable values from the unsupervised training phase
	# print("Restoring model from " + modelFile )
	# saver = tf.train.import_meta_graph(modelFile + '.meta')
	# saver.restore(sess,modelFile)

	# #Get the operations and tensors we'll need

	# predict = tf.get_default_graph().get_tensor_by_name("discriminators_shared/forward_SoftMax_C")
	# data = tf.get_default_graph().get_tensor_by_name("discriminators_shared/Placeholder:0")
	# batchSize = data.get_shape()[0]


	# print("Done")




