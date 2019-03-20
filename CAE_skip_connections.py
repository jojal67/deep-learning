# -*- coding: utf-8 -*-
import sys
import os
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

	dp=0.2
	# create filter for the source field
	S1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_source)
	S1 = BatchNormalization()(S1)
	S1 = SpatialDropout2D(dp)(S1)

	S2 = Conv2D(32, (3, 3), activation='relu', padding='same')(S1)
	S2 = BatchNormalization()(S2)
	S2 = SpatialDropout2D(dp)(S2)
	S2 = MaxPooling2D((2, 2), padding='same')(S2)

	S3 = Conv2D(64, (3, 3), activation='relu', padding='same')(S2)
	S3 = BatchNormalization()(S3)
	S3 = SpatialDropout2D(dp)(S3)

	S4 = Conv2D(64, (3, 3), activation='relu', padding='same')(S3)
	S4 = BatchNormalization()(S4)
	S4 = SpatialDropout2D(dp)(S4)
	S4 = MaxPooling2D((2, 2), padding='same')(S4)


	S5 = Conv2D(128, (3, 3), activation='relu', padding='same')(S4)
	S5 = BatchNormalization()(S5)
	S5 = SpatialDropout2D(dp)(S5)

	S6 = Conv2D(128, (3, 3), activation='relu', padding='same')(S5)
	S6 = BatchNormalization()(S6)
	S6 = SpatialDropout2D(dp)(S6)

	encoded_source = MaxPooling2D((2, 2), padding='same')(S6)


	# create filter for the density field
	D1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_dens)
	D1 = BatchNormalization()(D1)
	D1 = SpatialDropout2D(dp)(D1)

	D2 = Conv2D(32, (3, 3), activation='relu', padding='same')(D1)
	D2 = BatchNormalization()(D2)
	D2 = SpatialDropout2D(dp)(D2)
	D2 = MaxPooling2D((2, 2), padding='same')(D2)


	D3 = Conv2D(64, (3, 3), activation='relu', padding='same')(D2)
	D3 = BatchNormalization()(D3)
	D3 = SpatialDropout2D(dp)(D3)

	D4 = Conv2D(64, (3, 3), activation='relu', padding='same')(D3)
	D4 = BatchNormalization()(D4)
	D4 = SpatialDropout2D(dp)(D4)
	D4 = MaxPooling2D((2, 2), padding='same')(D4)


	D5 = Conv2D(128, (3, 3), activation='relu', padding='same')(D4)
	D5 = BatchNormalization()(D5)
	D5 = SpatialDropout2D(dp)(D5)

	D6 = Conv2D(128, (3, 3), activation='relu', padding='same')(D5)
	D6 = BatchNormalization()(D6)
	D6 = SpatialDropout2D(dp)(D6)

	encoded_dens = MaxPooling2D((2, 2), padding='same')(D6)


	# concatenate source and density
	X1 = concatenate([encoded_source, encoded_dens])


	# deconvolution
	X2 = Conv2D(128, (3, 3), activation='relu', padding='same')(X1)
	X2 = BatchNormalization()(X2)
	X2 = SpatialDropout2D(dp)(X2)
	X2 = UpSampling2D((2, 2))(X2)

	X2 = concatenate([X2, D5, S5])


	X3 = Conv2D(64, (3, 3), activation='relu', padding='same')(X2)
	X3 = BatchNormalization()(X3)
	X3 = SpatialDropout2D(dp)(X3)

	X3 = concatenate([X3, D4, S4])

	X4 = Conv2D(64, (3, 3), activation='relu', padding='same')(X3)
	X4 = BatchNormalization()(X4)
	X4 = SpatialDropout2D(dp)(X4)
	X4 = UpSampling2D((2, 2))(X4)

	X4 = concatenate([X4, D3, S3])

	X5 = Conv2D(32, (3, 3), activation='relu', padding='same')(X4)
	X5 = BatchNormalization()(X5)
	X5 = SpatialDropout2D(dp)(X5)

	X5 = concatenate([X5, D2, S2])

	X6 = Conv2D(32, (3, 3), activation='relu', padding='same')(X5)
	X6 = BatchNormalization()(X6)
	X6 = SpatialDropout2D(dp)(X6)
	X6 = UpSampling2D((2, 2))(X6)

	X6 = concatenate([X6, D1, S1])

	decoded = Conv2D(1, (3, 3), activation='linear', padding='same')(X6)
	


	autoencoder = Model(inputs=[input_source, input_dens], outputs=decoded)

	parallel_autoencoder = multi_gpu_model(autoencoder, gpus=2)

	parallel_autoencoder.summary()
	adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	parallel_autoencoder.compile(optimizer=adam, loss='mse', metrics=['mse'])
	

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






workdir="/data/chardin/2D/2000_250_256/"



instant=8
nbatch=8


resolution=256
n_training_set=2000
n_test_set=250
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






