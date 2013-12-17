""" Script for running the OOP classifiers from """
from numpy import *
import numpy as np
from pylab import *
import scipy.io
import scipy.ndimage
#from OOP_max_likelihood import *

class classifier:
	dataLink = 0
	whiskerData = 0
	
	def loading(self):
		self.whiskerData = scipy.io.loadmat(self.dataLink)['data']
		# self.data = data_roomba['data'] # texdata = 4 textures * 16 trials


	def likelihood(self):	# First find the max + min of the data, to set the bins
		i = 0
		mx = zeros((size(self.whiskerData)/2,1)) 			#size(texdata)/2)
		for x in xrange(0,shape(self.whiskerData)[0]): 		# number of classes
		    for y in xrange(0,shape(self.whiskerData)[1]/2): 	# number of files
		        tex = self.whiskerData[x,y]
		        mx[shape(self.whiskerData)[1]/2*x+y] = max(max(tex[:,0]),max(tex[:,1]))

		i = 0
		mn = zeros((size(self.whiskerData)/2,1)) 			#size(texdata)/2)
		for x in xrange(0,shape(self.whiskerData)[0]/2): 		# number of classes
		    for y in xrange(0,shape(self.whiskerData)[1]): 	# number of files
		        tex = self.whiskerData[x,y]
		        mn[shape(self.whiskerData)[1]/2*x+y] = min(min(tex[:,0]),min(tex[:,1]))
		        
		self.binsy = linspace(min(mn), max(mx), 501) # 501 numbers from min to max








ML = classifier()

ML.dataLink = '../../../jonathan/paperfigs_RSIresub_1011/data_roomba.mat'

ML.loading() # Load the data into ML.whiskerData using the link above

plot(ML.whiskerData[1,1])

show()

# TRAINING
ML.likelihood() # generate likelihoods from the data
