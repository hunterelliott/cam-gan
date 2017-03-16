#Performs distributed inference using a classifier trained from the GAN discriminator

import tensorflow as tf
import time
import os
import cam_gan_ops as cgo
import argparse

# ---- input ----- #

parser = argparse.ArgumentParser()
parser.add_argument('task_id', metavar='task_id', type=int,
                    help='index of which task this worker should be assigned to')

parser.add_argument('gpu', metavar='gpu', type=int,
                    help='GPU device to use')


args = parser.parse_args()
devID = args.gpu
task_ind = args.task_id

#modelFile = '/media/hunter/New Volume/camgan_xfertraining/Snapshots/CAMELYON_xfer_FitMLPOnly_iter2000.ckpt'
modelFile = '/media/extra/hunter_temp/Snapshots/CAMELYON_xfer_FitMLPOnly_iter2000.ckpt'
# dataFiles = ['/home/hunter/Desktop/TEMP_LOCAL/CEMELYON_MixedMedTest_1.tfrecords',
# 	'/home/hunter/Desktop/TEMP_LOCAL/CEMELYON_MixedMedTest_1.tfrecords']
# dataFiles = ['/home/hunter/Desktop/TEMP_LOCAL/data/MNIST/train.tfrecords',
# 			 '/home/hunter/Desktop/TEMP_LOCAL/data/MNIST/test.tfrecords']
#dataFiles = ['/home/hunter/Desktop/TEMP_LOCAL/data/camelyon_test_10k.tfrecords']
#dataFiles = ['/home/hunter/Desktop/TEMP_LOCAL/data/camelyon_test_5k.tfrecords',
#			 '/home/hunter/Desktop/TEMP_LOCAL/data/camelyon_test_5k_2.tfrecords']
dataFiles = ['/media/extra/hunter_temp/camelyon_test_10k.tfrecords',
			 '/media/extra/hunter_temp/camelyon_test_10k_2.tfrecords']

#dataFiles = ['/media/extra/hunter_temp/camelyon_test_10k.tfrecords']

nFiles = len(dataFiles)

print("Selected GPU" + str(devID))
print("Task index " + str(task_ind))



# --- Device ----- #

#devID = 0 #GPU to use. -1 is CPU
if devID >= 0:
	#Make only this GPU visible to TF
	os.environ["CUDA_VISIBLE_DEVICES"]=str(devID)
#	os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
	devString = '/gpu:0' #TF will now think the visible GPU is 0...
else:
	devString = '/cpu:0' #Will still use multiple cores if present
	os.environ["CUDA_VISIBLE_DEVICES"]=''

#task_ind = 1
worker_hosts = ['pathai1.local:2222','pathai3.local:2223']



# ----- init ------ #


#Reads one image from the current tfrecord file
def get_image(file_queue):
	
	reader_opts = tf.python_io.TFRecordOptions(2)#Set compression to zlib
	reader = tf.TFRecordReader(options=reader_opts)
	key, serial_record = reader.read(file_queue)
 	record = tf.parse_single_example(serial_record,features={'image_raw': tf.FixedLenFeature([], tf.string),
 		'height':tf.FixedLenFeature([],tf.int64),
 		'width':tf.FixedLenFeature([],tf.int64),
 		'depth':tf.FixedLenFeature([],tf.int64)})
	image = tf.decode_raw(record['image_raw'], tf.uint8)
	
	#im_shape = tf.stack([record['height'], record['width'], record['depth']],0)
	
	#image.set_shape([784])	
	image = tf.reshape(image,[256,256,3])
	image = tf.image.resize_images(image,[128, 128])
	#image = tf.reshape(image,im_shape)

	#im_sz = [record['height'], record['width'], record['depth']]
	return image

start_time = time.time();
print("Initializing...")
batch_size = 128


cluster = tf.train.ClusterSpec({"local": worker_hosts})
server = tf.train.Server(cluster, job_name="local", task_index=task_ind)


#Initialize the input queue ops on the CPU
with tf.device('/job:local/task:0'):

	file_queue = tf.train.string_input_producer(dataFiles,capacity=1e4,shared_name=
		    'shared_file_queue',num_epochs=1,shuffle=False)

	image = get_image(file_queue)

	try:
		images = tf.train.batch(
	        [image], batch_size=batch_size, num_threads=16,
	        capacity=5000 + 3 * batch_size,shared_name='shared_image_queue')
	except:
		print("Queue exhausted...")

 # 	im_list = [[image] for _ in range(8)]
	# images = tf.train.batch_join(
 #        im_list, batch_size=batch_size,
 #        capacity=5000 + 3 * batch_size)


# ---- inference ------ #

with tf.variable_scope("discriminators_shared") as scope, tf.device('/job:local/task:' + str(task_ind) + devString):
	# -- session, model init

	#Some ops in our GAN don't support GPU so allow TF to do the device placement when necessary
	sessConfig = tf.ConfigProto(allow_soft_placement=True)
#	sessConfig.gpu_options.allow_growth=True
	sess = tf.Session(server.target,config=sessConfig)
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

			#Run inference on them
			pred = sess.run(predict,feed_dict={data:ims})


			nIm_proc += ims.shape[0]

			if nIm_proc%100 == 0:
				print("Finished " + str(nIm_proc) + " images")

	except:# tf.errors.OutOfRangeError:
		print("Done! Completed " + str(nIm_proc) + " images total")

	#finally:
	#coord.request_stop()


	#coord.join(threads)
	#sess.close()

end_time = time.time();
print("Finished inference. Elapsed time: " + str(end_time-start_time) + " seconds.")

