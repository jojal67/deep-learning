# -*- coding: utf-8 -*-
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from scipy import fftpack
import pylab as py
from keras.models import model_from_json



def azimuthalAverage(image,freqmap):
	"""
	Calculate the azimuthally averaged radial profile.
	image - The 2D image
	center - The [x,y,z] pixel coordinates used as the center. The default is 
	     None, which then uses the center of the image (including 
	     fracitonal pixels).
	"""
	# Calculate the indices from the image
	x,y = np.indices(image.shape)
	center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])
	#print (" center = ", center)
	r = np.hypot(x - center[0], y - center[1])
	#print (" shape r = ",np.shape(r))
	# Get sorted radii
	ind = np.argsort(r.flat)
	r_sorted = r.flat[ind]
	i_sorted = image.flat[ind]
	f_sorted = freqmap.flat[ind]
	#print (" shape r sorted = ",np.shape(r_sorted))
	#print (" shape i sorted = ",np.shape(i_sorted))
	# Get the integer part of the radii (bin size = 1)
	r_int = r_sorted.astype(int)
	#print (" shape r int = ",np.shape(r_int))
	# Find all pixels that fall within each radial bin.
	deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
	rind = np.where(deltar)[0]       # location of changed radius
	nr = rind[1:] - rind[:-1]        # number of radius bin
	#print ("nr = ",nr)
	# Cumulative sum to figure out sums for each radius bin
	csim = np.cumsum(i_sorted, dtype=float)
	fsim = np.cumsum(f_sorted, dtype=float)
	tbin = csim[rind[1:]] - csim[rind[:-1]]
	fbin = fsim[rind[1:]] - fsim[rind[:-1]]
	radial_prof = tbin / nr
	f_prof = fbin / nr
	return f_prof, radial_prof



def pspectrum(resolution,box_size,image):
	#print (np.shape(image))
	resolution=len(image)
	kf = 1./float(box_size) # h/Mpc 
	nhalf = resolution / 2
	w = np.zeros(resolution)
	for i in range(resolution):
		if i > nhalf:
			iw = i-resolution 
		else: 
			iw = i 
		w[i] = kf*iw	
	freq=fftpack.fftfreq(resolution, d=float(box_size)/resolution)
	#print (w,len(w))
	#print (freq,len(freq))
	Freq=fftpack.fftshift( freq )
	#print("shape freq = ",np.shape(Freq))
	#print (Freq)
	freq_x,freq_y=np.meshgrid(Freq,Freq)
	freq=np.sqrt(freq_x**2+freq_y**2)

	# Take the fourier transform of the field.
	F1 = fftpack.fftn(image) 
	# Now shift the quadrants around so that low spatial frequencies are in
	# the center of the 2D fourier transformed field.
	F2 = fftpack.fftshift( F1 ) 
	#print("shape F2 = ",np.shape(F2))
	# Calculate a 2D power spectrum
	psd2D = np.abs( F2 )**2
	# Calculate the azimuthally averaged 1D power spectrum
	knorm, psd1D = azimuthalAverage(psd2D,freq)	
	# Now plot up both
	return knorm,psd1D













def recover_all_map_density_plus_source(workdir,resolution,subresolution,nslice,number_training):

	worksave="/data/chardin/2D/map_to_recover/"
	source_real_2D=np.load(worksave+"source_map_2D_real_slice_"+str(nslice)+"_2.npy")
	dens_real_2D=np.load(worksave+"dens_map_2D_real_slice_"+str(nslice)+"_2.npy")
	#x_real_2D=np.load(worksave+"x_real_value_map_2D_real_slice_"+str(nslice)+"_2.npy")
	x_real_2D=np.load(worksave+"x_vrai_slice_"+str(nslice)+"_2.npy")


	meansourcetrain=np.load(workdir+"meansourcetrain.npy")
	meandenstrain=np.load(workdir+"meandenstrain.npy")
	meanxtrain=np.load(workdir+"meanxtrain.npy")

	stdsourcetrain=np.load(workdir+"stdsourcetrain.npy")
	stddenstrain=np.load(workdir+"stddenstrainset.npy")
	stdxtrainset=np.load(workdir+"stdxtrainset.npy")
	
	source_real_2D=source_real_2D-meansourcetrain
	dens_real_2D=dens_real_2D-meandenstrain
	source_real_2D=source_real_2D/stdsourcetrain
	dens_real_2D=dens_real_2D/stddenstrain
	
	print("mean log10(1-x) train = ",meanxtrain)
	print("std log10(1-x) train = ",stdxtrainset)
	

	# load json and create model # for multi GPU training
	save_dir = workdir
	json_file = open(save_dir+'autoencoder.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	autoencoder = model_from_json(loaded_model_json)
	# load weights into new model
	autoencoder.load_weights(save_dir+"autoencoder_"+str(number_training)+".h5")
	print("Loaded model from disk")



	Nx=Ny=resolution/subresolution
	N_cube_to_recover=int((resolution*resolution)/(subresolution*subresolution))
	all_submap_to_recover_source=np.zeros((N_cube_to_recover,subresolution,subresolution,1),dtype=np.float32) 
	all_submap_to_recover_dens=np.zeros((N_cube_to_recover,subresolution,subresolution,1),dtype=np.float32) 

	cpty=0
	for i in range(N_cube_to_recover):
		if i%Nx==0 and i>0:
			cpty=cpty+1
		all_submap_to_recover_source[i,:,:,0]=source_real_2D[int((i%Nx)*subresolution):int((i%Nx)*subresolution+subresolution),int(cpty*subresolution):int(cpty*subresolution+subresolution)]
		all_submap_to_recover_dens[i,:,:,0]=dens_real_2D[int((i%Nx)*subresolution):int((i%Nx)*subresolution+subresolution),int(cpty*subresolution):int(cpty*subresolution+subresolution)]
	decoded_imgs = autoencoder.predict([all_submap_to_recover_source,all_submap_to_recover_dens])


	densrecovered=np.zeros((resolution,resolution))
	xrecovered=np.zeros((resolution,resolution))
	cpty=0
	for i in range(N_cube_to_recover):
		if i%Nx==0 and i>0:
			cpty=cpty+1
		densrecovered[int((i%Nx)*subresolution):int((i%Nx)*subresolution+subresolution),int(cpty*subresolution):int(cpty*subresolution+subresolution)]=all_submap_to_recover_dens[i,:,:,0].reshape((subresolution,subresolution))*stddenstrain+meandenstrain
		xrecovered[int((i%Nx)*subresolution):int((i%Nx)*subresolution+subresolution),int(cpty*subresolution):int(cpty*subresolution+subresolution)]=decoded_imgs[i].reshape((subresolution,subresolution))



	x_real_2D=np.log10(1.-x_real_2D)
	xrecovered=xrecovered


		
	plt.figure(1,figsize=(10,10))
	ax = plt.subplot(2,2,1)
	plt.imshow(np.log10(densrecovered))
	#plt.imshow(densrecovered)
	for i in range(int(Nx)):
		plt.axhline(i*subresolution,c="w",ls=":")
		plt.axvline(i*subresolution,c="w",ls=":")
	plt.axhline(resolution-1,c="r", linewidth=5)
	plt.axhline(1,c="r", linewidth=5)
	plt.axvline(resolution-1,c="r", linewidth=5)
	plt.axvline(1,c="r", linewidth=5)
	#plt.text(100,100,'Density',color="w",fontsize=20)
	ax.get_xaxis().set_visible(False)
	ax = plt.subplot(2,2,2)
	plt.imshow(x_real_2D)
	for i in range(int(Nx)):
		plt.axhline(i*subresolution,c="w",ls=":")
		plt.axvline(i*subresolution,c="w",ls=":")
	plt.axhline(resolution-1,c="r", linewidth=5)
	plt.axhline(1,c="r", linewidth=5)
	plt.axvline(resolution-1,c="r", linewidth=5)
	plt.axvline(1,c="r", linewidth=5)
	#plt.text(100,100,'Xion original',color="w",fontsize=20)
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	ax = plt.subplot(2,2,3)
	plt.imshow(np.log10(source_real_2D))
	for i in range(int(Nx)):
		plt.axhline(i*subresolution,c="w",ls=":")
		plt.axvline(i*subresolution,c="w",ls=":")
	plt.axhline(resolution-1,c="r", linewidth=5)
	plt.axhline(1,c="r", linewidth=5)
	plt.axvline(resolution-1,c="r", linewidth=5)
	plt.axvline(1,c="r", linewidth=5)
	#plt.text(100,100,'Source',color="w",fontsize=20)
	ax = plt.subplot(2,2,4)
	plt.imshow(xrecovered)
	for i in range(int(Nx)):
		plt.axhline(i*subresolution,c="w",ls=":")
		plt.axvline(i*subresolution,c="w",ls=":")
	plt.axhline(resolution-1,c="r", linewidth=5)
	plt.axhline(1,c="r", linewidth=5)
	plt.axvline(resolution-1,c="r", linewidth=5)
	plt.axvline(1,c="r", linewidth=5)
	#plt.text(100,100,'Xion model',color="w",fontsize=20)
	ax.get_yaxis().set_visible(False)
	plt.subplots_adjust(wspace=0, hspace=0)
	plt.savefig("/data/chardin/figure/model_visualisation.pdf")




	knorm,ps_real=pspectrum(resolution,128,x_real_2D)
	knorm,ps_model=pspectrum(resolution,128,xrecovered)
	print("x real average = ",np.average(x_real_2D))
	print("x cnn average = ",np.average(xrecovered))
	plt.figure(2)
	plt.loglog(knorm,ps_real,c="b",label="Simulation")
	plt.loglog(knorm,ps_model,c="r",label="CNN")
	plt.loglog(knorm,ps_real/ps_model,c="g",label="Simulation/CNN")
	plt.loglog(knorm,np.zeros(len(knorm))+1,c="k",ls=":")
	plt.loglog(knorm,np.zeros(len(knorm))+10,c="k",ls=":")
	plt.loglog(knorm,np.zeros(len(knorm))+0.1,c="k",ls=":")
	plt.xlabel('Spatial Frequency ')
	plt.ylabel('Power Spectrum')
	plt.legend()
	plt.savefig("/data/chardin/figure/pspectrum.pdf")

############################################################################################################
#
#
#              
#                     				Main
#
#
#
############################################################################################################


resolution=1024
subresolution=128
nslice=512
nepoch=300
 
workdir="/data/chardin/2D/2000_250_128/"

recover_all_map_density_plus_source(workdir,resolution,subresolution,nslice,nepoch)


plt.show()






