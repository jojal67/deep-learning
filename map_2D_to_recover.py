# -*- coding: utf-8 -*-
import sys
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

def save_map_to_recover():

	resolution=1024
	workcube="/data/chardin/"
	worksave="/data/chardin/2D/map_to_recover/"

		
	print("loading ")
	with np.load(workcube+'field.npz') as data:
		dens = data['mapd']
		x = data['mapx']
		S = data['mapstar']
	print("loaded ")
	print("calculating xion ")
	x_neutral_ion=x#np.zeros((resolution,resolution,resolution))
	#indices=np.where(x>=0.5)
	del x
	#x_neutral_ion[indices]=1
	#S = ndimage.gaussian_filter(S, sigma=[0,0,10], order=0)
	
	for i in range(resolution):
		#source_real_2D=S[:,:,i]
		#dens_real_2D=dens[:,:,i]
		x_real_2D=x_neutral_ion[:,:,i]
		#np.save(worksave+"source_map_2D_real_slice_"+str(i)+".npy",source_real_2D)
		#np.save(worksave+"dens_map_2D_real_slice_"+str(i)+".npy",dens_real_2D)
		#np.save(worksave+"x_real_value_map_2D_real_slice_"+str(i)+".npy",x_real_2D)
		np.save(worksave+"x_vrai_slice_"+str(i)+".npy",x_real_2D)
	

	"""
	print("loading ")
	with np.load(workcube+'field2.npz') as data:
		dens = data['mapd']
		x = data['mapx']
		S = data['mapstar']
	print("loaded ")
	print("calculating xion ")
	x_neutral_ion=x#np.zeros((resolution,resolution,resolution))
	#indices=np.where(x>=0.5)
	del x
	#x_neutral_ion[indices]=1
	#S = ndimage.gaussian_filter(S, sigma=[0,0,10], order=0)
	
	for i in range(resolution):
		#source_real_2D=S[:,:,i]
		#dens_real_2D=dens[:,:,i]
		x_real_2D=x_neutral_ion[:,:,i]
		#np.save(worksave+"source_map_2D_real_slice_"+str(i)+"_2.npy",source_real_2D)
		#np.save(worksave+"dens_map_2D_real_slice_"+str(i)+"_2.npy",dens_real_2D)
		#np.save(worksave+"x_real_value_map_2D_real_slice_"+str(i)+"_2.npy",x_real_2D)
		np.save(worksave+"x_vrai_slice_"+str(i)+"_2.npy",x_real_2D)
	"""


############################################################################################################
#
#
#              
#                     				Main
#
#
#
############################################################################################################



save_map_to_recover()









