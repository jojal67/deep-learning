# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy as sc
import scipy.ndimage as ndimage



def build_sample(workdir,instant,resolution,subresolution,n_training_set,n_test_set,shuffle):
	workcube="/data/chardin/"
	print("there will be ",n_training_set," entries for the training set")
	print("there will be ",n_test_set," entries for the test set")
	print("loading ")
	with np.load(workcube+'field.npz') as data:
		dens = data['mapd']
		x = data['mapx']
		S = data['mapstar']
	print("loaded ")
	
	#####################################################################################
	# create cube of ionization fraction with 0 and 1 only (0 if xion<0.5 or 1 if x>=0.5)
	#####################################################################################
	print("calculating xion ")
	"""
	x_neutral_ion=np.zeros((resolution,resolution,resolution))
	indices=np.where(x>=0.5)
	x_neutral_ion[indices]=1
	"""
	x_neutral_ion=np.log10(1.-x)
	del x
	print("xion calculated ")
	#######################################
	# Creating the whole source cube
	#######################################
	print("calculating source position ")
	indices=np.where(S!=0)
	xhalo,yhalo,zhalo=indices[0],indices[1],indices[2]
	nbsources=len(xhalo)
	print("NUMBER OF SOURCES = ",nbsources)
	indicessort=np.argsort(zhalo)
	xhalo=xhalo[indicessort]
	yhalo=yhalo[indicessort]
	zhalo=zhalo[indicessort]
	#xhalo=xhalo[0:30000]
	#yhalo=yhalo[0:30000]
	#zhalo=zhalo[0:30000]
	print("last z slice used for the training set : ",zhalo[len(zhalo)-1])
	randomize = np.arange(len(xhalo))
	np.random.shuffle(randomize)
	xhalo = xhalo[randomize]
	yhalo = yhalo[randomize]
	zhalo = zhalo[randomize]
	print("source position calculated")
	print("source filtering")
	S = ndimage.gaussian_filter(S, sigma=[0,0,10], order=0)
	#dens = ndimage.gaussian_filter(dens, sigma=[0,0,10], order=0)
	print("source filtered")
	####################################################################################################################################################
	# Creating the whole training and testing set :
	#
	# The training set will consist of three arrays (one for the source field and one 
	# for the density field which compose the input of the neural network and one that is the xion value expected (0 or 1) )
	# The source and density array have shape ((n_training_set,subresolution,subresolution,subresolution,1)) where "n_training_set" is the number of input over which the neural network will be trained
	# Each input consist of an image of shape subresolutionXsubresolutionXsubresolution pixels, hence the subresolution,subresolution,subresolution of the numpy array
	####################################################################################################################################################
	source_train_set=np.zeros((n_training_set,subresolution,subresolution,1),dtype=np.float32) 
	dens_train_set=np.zeros((n_training_set,subresolution,subresolution,1),dtype=np.float32) 
	x_train_set=np.zeros((n_training_set,subresolution,subresolution,1),dtype=np.float32)


	source_test_set=np.zeros((n_test_set,subresolution,subresolution,1),dtype=np.float32)
	dens_test_set=np.zeros((n_test_set,subresolution,subresolution,1),dtype=np.float32)
	x_test_set=np.zeros((n_test_set,subresolution,subresolution,1),dtype=np.float32)

	


	

	########################################################
	#                  Training set
	########################################################
	cpt=0
	i=0
	while cpt<n_training_set:
		
		#rand=np.random.randint(0,nbsources-1)
		if int(xhalo[i])>subresolution/2 and int(xhalo[i])<(resolution-subresolution/2) and int(yhalo[i])>subresolution/2 and int(yhalo[i])<(resolution-subresolution/2) and int(zhalo[i])>subresolution/2 and int(zhalo[i])<(resolution-subresolution/2): 
			xmin=int(xhalo[i]-subresolution/2) #np.random.randint(subresolution,resolution-subresolution-1)#
			xmax=xmin+subresolution
			ymin=int(yhalo[i]-subresolution/2) #np.random.randint(subresolution,resolution-subresolution-1)#
			ymax=ymin+subresolution
			z=int(zhalo[i]) #np.random.randint(subresolution,resolution-subresolution-1)#

			xfield=x_neutral_ion[xmin:xmax,ymin:ymax,z]
			ncells_ionized=len(np.where(xfield==1)[0])
			if ncells_ionized/(subresolution*subresolution)<=0.5:			
				print ("train cpt, x , y , z = ",cpt, int(xhalo[i]),int(yhalo[i]),int(zhalo[i]))
				source_train_set[cpt,:,:,0]=S[xmin:xmax,ymin:ymax,z]
				dens_train_set[cpt,:,:,0]=dens[xmin:xmax,ymin:ymax,z]
				x_train_set[cpt,:,:,0]=xfield
				cpt=cpt+1
		i=i+1

		
	########################################################
	#                  Testing set
	########################################################
	cpt=0
	while cpt<n_test_set:
		
		#rand=np.random.randint(0,nbsources-1)
		if int(xhalo[i])>subresolution/2 and int(xhalo[i])<(resolution-subresolution/2) and int(yhalo[i])>subresolution/2 and int(yhalo[i])<(resolution-subresolution/2) and int(zhalo[i])>subresolution/2 and int(zhalo[i])<(resolution-subresolution/2):
			xmin=int(xhalo[i]-subresolution/2) #np.random.randint(subresolution,resolution-subresolution-1)#
			xmax=xmin+subresolution
			ymin=int(yhalo[i]-subresolution/2) #np.random.randint(subresolution,resolution-subresolution-1)#
			ymax=ymin+subresolution
			z=int(zhalo[i]) #np.random.randint(subresolution,resolution-subresolution-1)#

			xfield=x_neutral_ion[xmin:xmax,ymin:ymax,z]
			ncells_ionized=len(np.where(xfield==1)[0])
			if ncells_ionized/(subresolution*subresolution)<=0.5:
				print ("test cpt, x , y , z = ",cpt, int(xhalo[i]),int(yhalo[i]),int(zhalo[i]))			
				source_test_set[cpt,:,:,0]=S[xmin:xmax,ymin:ymax,z]
				dens_test_set[cpt,:,:,0]=dens[xmin:xmax,ymin:ymax,z]
				x_test_set[cpt,:,:,0]=xfield
				cpt=cpt+1
		i=i+1
		


	#######################################################
	#     zero centering of training and testing data 
	#######################################################
	meansourcetrain = np.mean(source_train_set)
	meandenstrain = np.mean(dens_train_set)
	meanxtrain = np.mean(x_train_set)

	source_train_set=source_train_set-meansourcetrain
	dens_train_set=dens_train_set-meandenstrain
	#x_train_set=x_train_set-meanxtrain

	source_test_set=source_test_set-meansourcetrain
	dens_test_set=dens_test_set-meandenstrain
	#x_test_set=x_test_set-meanxtrain
	#######################################################
	#    normalization of training and testing data 
	#######################################################
	stdsourcetrain=np.std(source_train_set)
	stddenstrainset=np.std(dens_train_set)
	stdxtrainset=np.std(x_train_set)

	source_train_set=source_train_set/stdsourcetrain
	dens_train_set=dens_train_set/stddenstrainset
	#x_train_set=x_train_set/stdxtrainset

	source_test_set=source_test_set/stdsourcetrain
	dens_test_set=dens_test_set/stddenstrainset
	#x_test_set=x_test_set/stdxtrainset
	#######################################################
	# Saving the training and testing set on the disk
	#######################################################
	np.save(workdir+"source_train_set_2D.npy",source_train_set)
	np.save(workdir+"dens_train_set_2D.npy",dens_train_set)
	np.save(workdir+"x_train_set_2D.npy",x_train_set)

	np.save(workdir+"source_test_set_2D.npy",source_test_set)
	np.save(workdir+"dens_test_set_2D.npy",dens_test_set)
	np.save(workdir+"x_test_set_2D.npy",x_test_set)

	np.save(workdir+"meansourcetrain.npy",meansourcetrain)
	np.save(workdir+"meandenstrain.npy",meandenstrain)
	np.save(workdir+"meanxtrain.npy",meanxtrain)
	
	np.save(workdir+"stdsourcetrain.npy",stdsourcetrain)
	np.save(workdir+"stddenstrainset.npy",stddenstrainset)
	np.save(workdir+"stdxtrainset.npy",stdxtrainset)




def plot_sample(workdir,instant,resolution):
	
	source_train_set_mean=np.load(workdir+"source_train_set_2D.npy")
	dens_train_set_mean=np.load(workdir+"dens_train_set_2D.npy")
	x_train_set_mean=np.load(workdir+"x_train_set_2D.npy")

	"""
	meanxtrain=np.load(workdir+"meanxtrain.npy")
	stdxtrainset=np.load(workdir+"stdxtrainset.npy")
	print("xtrain mean = ",meanxtrain)
	print("mean xtrain after zero centering = ",np.average(x_train_set_mean*stdxtrainset))
	"""

	# plot one cube multiple slice
	n=10
	nlice=32
	idep=0
	nbsubplot=5
	for j in range(nbsubplot):
		plt.figure(j+200,figsize=(20, 3))
		for i in range(n):
			ax = plt.subplot(3, n, i+1 )
			plt.imshow(source_train_set_mean[idep+i,:,:,0].reshape((int(resolution),int(resolution))))
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			plt.colorbar()
			ax = plt.subplot(3, n, i+1+n )
			plt.imshow(dens_train_set_mean[idep+i,:,:,0].reshape((int(resolution),int(resolution))))
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			plt.colorbar()
			ax = plt.subplot(3, n, i+1+n+n)
			plt.imshow(x_train_set_mean[idep+i,:,:,0].reshape((int(resolution),int(resolution))))
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			plt.colorbar()
		idep =idep + n
		print(idep)
		
	plt.subplots_adjust(wspace=0, hspace=0)


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
resolution=1024
boxsize=128 # cMpc/h


n_training_set=2000
n_test_set=250
subresolution=256
shuffle=0 # one for shuffling the maps


print("The subcube for the training are ",(boxsize/resolution)*subresolution," cMpc/h large")


build_sample(workdir,instant,resolution,subresolution,n_training_set,n_test_set,shuffle)

plot_sample(workdir,instant,subresolution)





plt.show()
