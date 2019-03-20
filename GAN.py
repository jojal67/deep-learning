# -*- coding: utf-8 -*-
import sys
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
from tensorflow.python.client import device_lib
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization, Dropout, SpatialDropout2D
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




def GAN_autoencoder(workdir,instant,subresolution,n_training_set,n_test_set,nepoch,nbatch):

	source_train_set=np.load(workdir+"source_train_set_2D.npy")
	dens_train_set=np.load(workdir+"dens_train_set_2D.npy")
	x_train_set=np.load(workdir+"x_train_set_2D.npy")


	source_test_set=np.load(workdir+"source_test_set_2D.npy")
	dens_test_set=np.load(workdir+"dens_test_set_2D.npy")
	x_test_set=np.load(workdir+"x_test_set_2D.npy")

	input_source = Input(shape=(subresolution, subresolution, 1))  
	input_dens = Input(shape=(subresolution, subresolution, 1)) 
	
	dropout = 0.4
	# create filter for the source field
	x = Conv2D(32, (3, 3), activation=LeakyReLU(alpha=0.2)), padding='same')(input_source)
	x = BatchNormalization()(x)
	x = SpatialDropout2D(dropout)(x)
	x = Conv2D(32, (3, 3), activation=LeakyReLU(alpha=0.2)), padding='same')(x)
	x = BatchNormalization()(x)
	x = SpatialDropout2D(dropout)(x)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(64, (3, 3), activation=LeakyReLU(alpha=0.2)), padding='same')(x)
	x = BatchNormalization()(x)
	x = SpatialDropout2D(dropout)(x)
	x = Conv2D(64, (3, 3), activation=LeakyReLU(alpha=0.2)), padding='same')(x)
	x = BatchNormalization()(x)
	x = SpatialDropout2D(dropout)(x)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(128, (3, 3), activation=LeakyReLU(alpha=0.2)), padding='same')(x)
	x = BatchNormalization()(x)
	x = SpatialDropout2D(dropout)(x)
	x = Conv2D(128, (3, 3), activation=LeakyReLU(alpha=0.2)), padding='same')(x)
	x = BatchNormalization()(x)
	x = SpatialDropout2D(dropout)(x)
	encoded_source = MaxPooling2D((2, 2), padding='same')(x)




	self.D = Sequential()
	depth = 64
	dropout = 0.4
	# In: 28 x 28 x 1, depth = 1
	# Out: 14 x 14 x 1, depth=64
	self.D.add(Conv2D(depth*1, 5, strides=2, input_shape=input_source,padding='same', activation=LeakyReLU(alpha=0.2)))
	self.D.add(Dropout(dropout))
	self.D.add(Conv2D(depth*2, 5, strides=2, padding='same',activation=LeakyReLU(alpha=0.2)))
	self.D.add(Dropout(dropout))
	self.D.add(Conv2D(depth*4, 5, strides=2, padding='same',activation=LeakyReLU(alpha=0.2)))
	self.D.add(Dropout(dropout))
	self.D.add(Conv2D(depth*8, 5, strides=1, padding='same',activation=LeakyReLU(alpha=0.2)))
	self.D.add(Dropout(dropout))
	# Out: 1-dim probability
	self.D.add(Flatten())
	self.D.add(Dense(1))
	self.D.add(Activation('sigmoid'))
	self.D.summary()


	self.G = Sequential()
	dropout = 0.4
	depth = 64+64+64+64
	dim = 7
	# In: 100
	# Out: dim x dim x depth
	self.G.add(Dense(dim*dim*depth, input_dim=100))
	self.G.add(BatchNormalization(momentum=0.9))
	self.G.add(Activation('relu'))
	self.G.add(Reshape((dim, dim, depth)))
	self.G.add(Dropout(dropout))
	# In: dim x dim x depth
	# Out: 2*dim x 2*dim x depth/2
	self.G.add(UpSampling2D())
	self.G.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
	self.G.add(BatchNormalization(momentum=0.9))
	self.G.add(Activation('relu'))
	self.G.add(UpSampling2D())
	self.G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
	self.G.add(BatchNormalization(momentum=0.9))
	self.G.add(Activation('relu'))
	self.G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
	self.G.add(BatchNormalization(momentum=0.9))
	self.G.add(Activation('relu'))
	# Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
	self.G.add(Conv2DTranspose(1, 5, padding='same'))
	self.G.add(Activation('sigmoid'))
	self.G.summary()
	return self.G




	optimizer = RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8)
	self.DM = Sequential()
	self.DM.add(self.discriminator())
	self.DM.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])





	# saving the model
	save_dir = workdir
	model_name = 'CNN_autoencoder2D_real_data_ntraining_number_0.h5'
	model_path = os.path.join(save_dir, model_name)
	parallel_autoencoder.save(model_path)
	print('Saved trained model at %s ' % model_path)

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
			GAN_autoencoder(workdir,instant,subresolution,n_training_set,n_test_set,nepoch,nbatch)
		if i>0:
			
			if (i-1)%100==0:
				print("toto")
				model_name = 'CNN_autoencoder2D_real_data_ntraining_number_'+str(i-1)+'.h5'
				model_path = os.path.join(save_dir, model_name)
				autoencoder = load_model(model_path)
				
			# for double entry
			history = autoencoder.fit([source_train_set,dens_train_set], x_train_set,
		        epochs=1,
			verbose=1,
		        batch_size=nbatch,
		        shuffle=True,
		        validation_data=([source_test_set,dens_test_set], x_test_set))
			
			if (i)%100==0:
				# saving the model
				model_name = 'CNN_autoencoder2D_real_data_ntraining_number_'+str(i)+'.h5'
				model_path = os.path.join(save_dir, model_name)
				autoencoder.save(model_path)
				print('Saved trained model at %s ' % model_path)

			accuracy_train=np.array(history.history['mean_squared_error'])
			accuracy_test=np.array(history.history['val_mean_squared_error'])
			loss_train=np.array(history.history['loss'])
			loss_test=np.array(history.history['val_loss'])

			np.save(workdir+"accuracy_2D_real_data_train_nepoch_"+str(i)+".npy",accuracy_train)
			np.save(workdir+"accuracy_2D_real_data_test_nepoch_"+str(i)+".npy",accuracy_test)
			np.save(workdir+"loss_2D_real_data_train_nepoch_"+str(i)+".npy",loss_train)
			np.save(workdir+"loss_2D_real_data_test_nepoch_"+str(i)+".npy",loss_test)















############################################################################################################
#
#
#              
#                     				Main
#
#
#
############################################################################################################



workdir="/data/chardin/2D/test_GAN/"

instant=8
nbatch=100
resolution=64
n_training_set=200
n_test_set=80
nepoch=101
nstart=0



continue_training(workdir,instant,resolution,n_training_set,n_test_set,nepoch,nbatch,nstart)







plt.show()






