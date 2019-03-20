# -*- coding: utf-8 -*-
import sys
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
from tensorflow.python.client import device_lib
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization, Dropout, SpatialDropout2D, Activation
from keras.models import Model
from keras import backend as K
from keras.models import load_model
from keras.optimizers import Adam
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import time
from keras.utils import multi_gpu_model
from keras.models import model_from_json
from keras.layers.advanced_activations import LeakyReLU

def get_available_gpus():
	local_devices_protos = device_lib.list_local_devices()
	return [x.name for x in local_devices_protos if x.device_type == 'GPU']
print("liste des GPUs a utiliser",get_available_gpus())


def myloss(y_true, y_pred):
	# wasserstein_loss
	return K.mean(y_true*y_pred)

def tilted_loss(q,y,f):
	e = (y-f)
	return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)


def CNN_autoencoder_double_entry(workdir,instant,subresolution,n_training_set,n_test_set,nepoch,nbatch):

	source_train_set=np.load(workdir+"source_train_set_2D.npy")
	dens_train_set=np.load(workdir+"dens_train_set_2D.npy")
	x_train_set=np.load(workdir+"x_train_set_2D.npy")
	print (x_train_set.shape)
	

	source_test_set=np.load(workdir+"source_test_set_2D.npy")
	dens_test_set=np.load(workdir+"dens_test_set_2D.npy")
	x_test_set=np.load(workdir+"x_test_set_2D.npy")
	print (x_test_set.shape)
	



	input_source = Input(shape=(subresolution, subresolution, 1))  
	input_dens = Input(shape=(subresolution, subresolution, 1))  

	dp=0.5
	# create filter for the source field
	x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_source)
	x = BatchNormalization()(x)
	x = SpatialDropout2D(dp)(x)

	x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
	x = BatchNormalization()(x)
	x = SpatialDropout2D(dp)(x)
	x = MaxPooling2D((2, 2), padding='same')(x)

	x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
	x = BatchNormalization()(x)
	x = SpatialDropout2D(dp)(x)

	x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
	x = BatchNormalization()(x)
	x = SpatialDropout2D(dp)(x)
	x = MaxPooling2D((2, 2), padding='same')(x)


	x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
	x = BatchNormalization()(x)
	x = SpatialDropout2D(dp)(x)

	x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
	x = BatchNormalization()(x)
	x = SpatialDropout2D(dp)(x)

	encoded_source = MaxPooling2D((2, 2), padding='same')(x)


	# create filter for the density field
	x2 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_dens)
	x2 = BatchNormalization()(x2)
	x2 = SpatialDropout2D(dp)(x2)

	x2 = Conv2D(32, (3, 3), activation='relu', padding='same')(x2)
	x2 = BatchNormalization()(x2)
	x2 = SpatialDropout2D(dp)(x2)
	x2 = MaxPooling2D((2, 2), padding='same')(x2)


	x2 = Conv2D(64, (3, 3), activation='relu', padding='same')(x2)
	x2 = BatchNormalization()(x2)
	x2 = SpatialDropout2D(dp)(x2)

	x2 = Conv2D(64, (3, 3), activation='relu', padding='same')(x2)
	x2 = BatchNormalization()(x2)
	x2 = SpatialDropout2D(dp)(x2)
	x2 = MaxPooling2D((2, 2), padding='same')(x2)


	x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x2)
	x2 = BatchNormalization()(x2)
	x2 = SpatialDropout2D(dp)(x2)

	x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x2)
	x2 = BatchNormalization()(x2)
	x2 = SpatialDropout2D(dp)(x2)

	encoded_dens = MaxPooling2D((2, 2), padding='same')(x2)


	# concatenate source and density
	x3 = concatenate([encoded_source, encoded_dens])


	# deconvolution
	x4 = Conv2D(128, (3, 3), activation='relu', padding='same')(x3)
	x4 = BatchNormalization()(x4)
	x4 = SpatialDropout2D(dp)(x4)
	x4 = UpSampling2D((2, 2))(x4)


	x4 = Conv2D(64, (3, 3), activation='relu', padding='same')(x4)
	x4 = BatchNormalization()(x4)
	x4 = SpatialDropout2D(dp)(x4)

	x4 = Conv2D(64, (3, 3), activation='relu', padding='same')(x4)
	x4 = BatchNormalization()(x4)
	x4 = SpatialDropout2D(dp)(x4)
	x4 = UpSampling2D((2, 2))(x4)


	x4 = Conv2D(32, (3, 3), activation='relu', padding='same')(x4)
	x4 = BatchNormalization()(x4)
	x4 = SpatialDropout2D(dp)(x4)

	x4 = Conv2D(32, (3, 3), activation='relu', padding='same')(x4)
	x4 = BatchNormalization()(x4)
	x4 = SpatialDropout2D(dp)(x4)
	x4 = UpSampling2D((2, 2))(x4)

	decoded = Conv2D(1, (3, 3), activation='linear', padding='same')(x4)
	


	autoencoder = Model(inputs=[input_source, input_dens], outputs=decoded)

	parallel_autoencoder = multi_gpu_model(autoencoder, gpus=2)

	parallel_autoencoder.summary()
	adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	parallel_autoencoder.compile(optimizer=adam, loss='mse', metrics=['mse'])
	#autoencoder.compile(optimizer=adam, loss=mylossfunction, metrics=['accuracy'])
	#parallel_autoencoder.compile(optimizer=adam, loss=myloss, metrics=['mse'])
	#quantile = 10
	#parallel_autoencoder.compile(optimizer=adam, loss=lambda y,f: tilted_loss(quantile,y,f), metrics=['mse'])


	history = parallel_autoencoder.fit([source_train_set,dens_train_set], x_train_set,
		        epochs=1,
			verbose=1,
		        batch_size=nbatch,
		        shuffle=True,
		        validation_data=([source_test_set,dens_test_set], x_test_set))

	# serialize model to JSON
	save_dir = workdir
	autoencoder_json = autoencoder.to_json()
	with open(save_dir+"autoencoder.json", "w") as json_file:
		json_file.write(autoencoder_json)
	# serialize weights to HDF5
	autoencoder.save_weights(save_dir+"autoencoder_0.h5")
	print("Saved model to disk")

	accuracy_train=np.array(history.history['mean_squared_error'])
	accuracy_test=np.array(history.history['val_mean_squared_error'])
	loss_train=np.array(history.history['loss'])
	loss_test=np.array(history.history['val_loss'])


	
	np.save(workdir+"accuracy_2D_real_data_train_nepoch_"+str(0)+".npy",accuracy_train)
	np.save(workdir+"accuracy_2D_real_data_test_nepoch_"+str(0)+".npy",accuracy_test)
	np.save(workdir+"loss_2D_real_data_train_nepoch_"+str(0)+".npy",loss_train)
	np.save(workdir+"loss_2D_real_data_test_nepoch_"+str(0)+".npy",loss_test)
	






def continue_training(workdir,instant,subresolution,n_training_set,n_test_set,nepoch,nbatch,nstart):

	save_dir = workdir
	
	source_train_set=np.load(workdir+"source_train_set_2D.npy")
	dens_train_set=np.load(workdir+"dens_train_set_2D.npy")
	x_train_set=np.load(workdir+"x_train_set_2D.npy")
	
	source_test_set=np.load(workdir+"source_test_set_2D.npy")
	dens_test_set=np.load(workdir+"dens_test_set_2D.npy")
	x_test_set=np.load(workdir+"x_test_set_2D.npy")
	

	for i in range(nstart,nepoch):
		print(i)
		if i==0:
			CNN_autoencoder_double_entry(workdir,instant,subresolution,n_training_set,n_test_set,nepoch,nbatch)
		if i>0:
			
			if (i-1)%100==0:
				save_dir = workdir
				json_file = open(save_dir+'autoencoder.json', 'r')
				loaded_model_json = json_file.read()
				json_file.close()
				autoencoder = model_from_json(loaded_model_json)
				# load weights into new model
				autoencoder.load_weights(save_dir+"autoencoder_"+str(i-1)+".h5")

	
				parallel_autoencoder = multi_gpu_model(autoencoder, gpus=2)
				adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
				parallel_autoencoder.compile(optimizer=adam, loss='mse', metrics=['mse'])
				#parallel_autoencoder.compile(optimizer=adam, loss=myloss, metrics=['mse'])
			
			# for double entry
			history = parallel_autoencoder.fit([source_train_set,dens_train_set], x_train_set,
		        epochs=1,
			verbose=1,
		        batch_size=nbatch,
		        shuffle=True,
		        validation_data=([source_test_set,dens_test_set], x_test_set))
			

			if (i)%100==0:
				# serialize model to JSON
				save_dir = workdir
				autoencoder_json = autoencoder.to_json()
				with open(save_dir+"autoencoder.json", "w") as json_file:
					json_file.write(autoencoder_json)
				# serialize weights to HDF5
				autoencoder.save_weights(save_dir+"autoencoder_"+str(i)+".h5")
				print("Saved model "+str(i)+" to disk")


			accuracy_train=np.array(history.history['mean_squared_error'])
			accuracy_test=np.array(history.history['val_mean_squared_error'])
			loss_train=np.array(history.history['loss'])
			loss_test=np.array(history.history['val_loss'])


			np.save(workdir+"accuracy_2D_real_data_train_nepoch_"+str(i)+".npy",accuracy_train)
			np.save(workdir+"accuracy_2D_real_data_test_nepoch_"+str(i)+".npy",accuracy_test)
			np.save(workdir+"loss_2D_real_data_train_nepoch_"+str(i)+".npy",loss_train)
			np.save(workdir+"loss_2D_real_data_test_nepoch_"+str(i)+".npy",loss_test)














def recover_map(workdir,instant,subresolution,number_training):

	source_test_set=np.load(workdir+"source_test_set_2D.npy")
	dens_test_set=np.load(workdir+"dens_test_set_2D.npy")
	x_test_set=np.load(workdir+"x_test_set_2D.npy")

	#source_test_set=np.load(workdir+"source_train_set_2D.npy")
	#dens_test_set=np.load(workdir+"dens_train_set_2D.npy")
	#x_test_set=np.load(workdir+"x_train_set_2D.npy")
	

	meandenstrain=np.load(workdir+"meandenstrain.npy")
	stddenstrain=np.load(workdir+"stddenstrainset.npy")
	
	# load json and create model
	save_dir = workdir
	json_file = open(save_dir+'autoencoder.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	autoencoder = model_from_json(loaded_model_json)
	# load weights into new model
	autoencoder.load_weights(save_dir+"autoencoder_"+str(number_training)+".h5")
	print("Loaded model from disk")




	decoded_imgs = autoencoder.predict([source_test_set,dens_test_set])
	
	print (decoded_imgs.shape)
	

	# plot multiple cube one slice
	n=10
	idep=0
	for k in range(5):
		idep=k*n
		plt.figure(k,figsize=(16.5, 5))
		for i in range(n):
			ax = plt.subplot(4, n, i+1 )
			plt.imshow(np.log10(dens_test_set[i+idep,:,:,0]*stddenstrain+meandenstrain))
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			ax = plt.subplot(4, n, i+1+n )
			plt.imshow(source_test_set[i+idep,:,:,0])
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			ax = plt.subplot(4, n, i+1+2*n )
			plt.imshow(x_test_set[i+idep,:,:,0])
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			ax = plt.subplot(4, n, i+1+3*n )
			plt.imshow(decoded_imgs[i+idep].reshape(subresolution,subresolution))
			#print(decoded_imgs[i+idep].reshape(resolution,resolution))
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
		plt.subplots_adjust(wspace=0, hspace=0)
		plt.tight_layout()
		if k==0:
			plt.savefig("/data/chardin/figure/test_set_recovered.pdf")

	




############################################################################################################
#
#
#              
#                     				Main
#
#
#
############################################################################################################



workdir="/data/chardin/2D/200_80_64_xHI_dp_05/"



instant=8
nbatch=1000


resolution=64
n_training_set=200
n_test_set=80
nepoch=1001
nstart=0




start = time.clock()
continue_training(workdir,instant,resolution,n_training_set,n_test_set,nepoch,nbatch,nstart)
elapsed = time.clock()
elapsed = elapsed - start
print("Time spent in (function name) is: ", elapsed)




epoch_number=0
recover_map(workdir,instant,resolution,epoch_number)



plt.show()






