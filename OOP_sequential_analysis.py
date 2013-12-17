""" OOP Sequential analysis code. Hopefully inheriting from the Max_likelihood classifier() class """
from numpy import *
import numpy as np
from pylab import *
import scipy.io
import scipy.ndimage
from OOP_max_likelihood import *

ML = classifier()
smoo = True
norm = True
bins = 301


print "%s" % ML.status
ML.dataLink = '../../../Nathans/paperfigs_RSIresub_1011/data_roomba.mat'

ML.loading() # Load the data into ML.whiskerData using the link above
print "%s" % ML.status

ML.dataSorting()
print "%s" % ML.status

# TRAINING
ML.concatenate() # concatentate data into classes
print "%s" % ML.status

# ML.setDistro(bins)   # define max/min/bins of the distribution
ML.binsy = linspace(1,3,bins)
print "%s" % ML.status

ML.likelihood(norm,smoo,bins)  # generate likelihoods distributions (histograms) from the data
print "%s" % ML.status

# PLOT THE HISTOGRAMS TO SEE IF THEY'RE RIGHT
for x in xrange(0,4):
	# ML.hists[x,:,0] = np.histogram(ML.training[x,:,0],ML.binsy)[0]
	# ML.hists[x,:,1] = np.histogram(ML.training[x,:,1],ML.binsy)[0] # generate a histogram of the y input
	figure(1) # Plot the histograms for a cheeky look
	histy = subplot(2,2,x)
	histy.plot(ML.hists[x])
	# xlim((100,300))

# TESTING
# """
# Run for one chunk size
chunk = 20 #400 # Length of each sample to classify
Y = 10*ones(100)
thresh = 1- np.power(Y,linspace(0,-10,100))
ML.sortTest(chunk)
print "%s" % ML.status

ML.findDistro(bins,chunk)
print "%s" % ML.status

# Compute likelihoods in short chunks. Streaming one in, one out, then computed sum log posteriors
ML.findPosterior(chunk)
print "%s" % ML.status

ML.score = ML.findScore()
print "%s" % ML.status

print ML.score
show()




