# -*- coding: utf-8 -*-
import sys
import os
import subprocess
import numpy as np
import matplotlib
import matplotlib.pyplot as plt










def plot_convergence_emma():

	
	color_model_train=["r","r","r"]
	color_model_test=["b","b","b"]
	ls_model=["-",":","--"]
	model_label=["1","2","3"]

	
	workdir0="/data/chardin/2D/200_80_64_diff_train_test/"
	workdir1="/data/chardin/2D/2000_250_128/"
	workdir2="/data/chardin/2D/200_80_64_smooth_source_all_direction/"
	workdir3="/data/chardin/2D/200_80_64_smoothS_5/"
	workdir4="/data/chardin/2D/200_80_64_smoothS_15/"
	workdir5="/data/chardin/2D/200_80_64_smoothS_20/"
	workdi6r="/data/chardin/2D/200_80_64_x_normalized/"
	workdir7="/data/chardin/2D/200_80_64_x_normalized_mse/"
	workdir8="/data/chardin/2D/200_80_64_x_HI_mse/"



	workdir9="/data/chardin/2D/200_80_64_x_HI_mse_nosigmoid/"
	workdir10="/data/chardin/2D/200_80_64_xHI_dp_05/"
	workdir11="/data/chardin/2D/200_80_64_xHI_skip/"
	workdir12="/data/chardin/2D/1000_150_64/"
	workdir13="/data/chardin/2D/2000_250_128/"

	workdir14="/data/chardin/2D/1000_150_256/"
	workdir="/data/chardin/2D/2000_250_256/"

	workdir=[workdir14,workdir]
	for j in range(len(workdir)):

		cmd="ls "+workdir[j]+"accuracy_2D_real_data_train_nepoch_* | wc -l"
		toto=subprocess.run(cmd,shell=True, stdout=subprocess.PIPE)
		nepoch=int(toto.stdout)

		epoch=np.zeros(nepoch)
		accuracy_train=np.zeros(nepoch)
		accuracy_test=np.zeros(nepoch)
		loss_train=np.zeros(nepoch)
		loss_test=np.zeros(nepoch)
		cpt=0
		for i in range(nepoch):
			epoch[i]=cpt
		
			accuracy_train[i]=np.load(workdir[j]+"accuracy_2D_real_data_train_nepoch_"+str(i)+".npy")
			accuracy_test[i]=np.load(workdir[j]+"accuracy_2D_real_data_test_nepoch_"+str(i)+".npy")
			loss_train[i]=np.load(workdir[j]+"loss_2D_real_data_train_nepoch_"+str(i)+".npy")
			loss_test[i]=np.load(workdir[j]+"loss_2D_real_data_test_nepoch_"+str(i)+".npy")
			cpt=cpt+1
		
		print(epoch)

		plt.figure(11)
		plt.subplot(2,1,1)
		plt.plot(epoch,accuracy_train,color=color_model_train[j],ls=ls_model[j],label=model_label[j])
		plt.plot(epoch,accuracy_test,color=color_model_test[j],ls=ls_model[j])
		#plt.semilogy(epoch,accuracy_train,color=color_model_train[j],ls=ls_model[j],label=model_label[j])
		#plt.semilogy(epoch,accuracy_test,color=color_model_test[j],ls=ls_model[j])
		plt.axhline(0.9)
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='lower right')
	

		plt.subplot(2,1,2)
		#plt.plot(epoch,loss_train,color=color_model_train[j],ls=ls_model[j])
		#plt.plot(epoch,loss_test,color=color_model_test[j],ls=ls_model[j])
		plt.semilogy(epoch,loss_train,color=color_model_train[j],ls=ls_model[j])
		plt.semilogy(epoch,loss_test,color=color_model_test[j],ls=ls_model[j])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper right')

		plt.tight_layout()

	plt.show()



############################################################################################################
#
#
#              
#                     				Main
#
#
#
############################################################################################################






plot_convergence_emma()










