# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def copy(workdirin,workdirout):

	
	os.system("cp "+workdirin+"source_train_set_2D.npy "+workdirout)
	os.system("cp "+workdirin+"dens_train_set_2D.npy "+workdirout)
	os.system("cp "+workdirin+"x_train_set_2D.npy "+workdirout)

	os.system("cp "+workdirin+"source_test_set_2D.npy "+workdirout)
	os.system("cp "+workdirin+"dens_test_set_2D.npy "+workdirout)
	os.system("cp "+workdirin+"x_test_set_2D.npy "+workdirout)


	os.system("cp "+workdirin+"meansourcetrain.npy "+workdirout)
	os.system("cp "+workdirin+"meandenstrain.npy "+workdirout)
	os.system("cp "+workdirin+"meanxtrain.npy "+workdirout)
	os.system("cp "+workdirin+"stdsourcetrain.npy "+workdirout)
	os.system("cp "+workdirin+"stddenstrainset.npy "+workdirout)
	os.system("cp "+workdirin+"stdxtrainset.npy "+workdirout)
	




workdirin="/data/chardin/2D/200_80_64_xHI_dp_05/"
workdirout="/data/chardin/2D/200_80_64_xHI_skip/"

copy(workdirin,workdirout)
