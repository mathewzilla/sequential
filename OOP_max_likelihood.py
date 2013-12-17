""" OOP max likelihood classifier """
""" To Do: change likelihood code to process whole array of training data (not half). i.e. 
Training/Test data needs to be set before hand by user. 
Also, change code so that it isn't just training/test. Maybe add 'update' likelihoods, 
to take previous posterior as new likelihood?

""" 
from numpy import *
import numpy as np
from pylab import *
import scipy.io
import scipy.ndimage
#from OOP_max_likelihood import *

class classifier:
	dataLink = None
	whiskerData = None
	train = None
	test = None
	training = None
	binsy = None
	hists = None
	testing = None
	trueClass = None
	dataDist = None
	sumlik = None
	posterior = None
	score = None
	sumPos = None
	decTime = None
	classed = None
	status = 'Ready to start'
	
	def loading(self):
		self.whiskerData = scipy.io.loadmat(self.dataLink)['data']
		# self.data = data_roomba['data'] # texdata = 4 textures * 16 trials
	 	self.status = 'Loaded the data'

	def dataSorting(self):
	 	w = shape(self.whiskerData)[0] 		# Width 					4
		l = shape(self.whiskerData)[1/4] 		# Length 					16
		s = shape(self.whiskerData[0,0])[0]	# Size (of each trial) 		32000

	 	self.train = np.empty([w,l], dtype=np.object)
	 	self.test  = np.empty([w,l], dtype=np.object)
	 	for x in xrange(0,w):
	 		for y in xrange(0,l):
	 			self.train[x,y] = (self.whiskerData[x,y])[0:s/2]
	 			self.test[x,y]  = (self.whiskerData[x,y])[s/2:s]

	  	self.status = 'Sorted data into train/test sets'

	def concatenate(self):	# Concatentate training data into classes
		w = shape(self.train)[0] 			# Width 					4
		l = shape(self.train)[1] 			# Length 					16
		s = shape(self.train[0,0])[0]		# Size (of each trial) 		16000
		c = shape(self.train[0,0])[1]		# Number of input channels 	2

		Training = empty((w,(s*l),c))						
		for x in xrange(0,w): 			# number of classes
		    for y in xrange(0,l): 		# half number of files
		    	Training[x,s*y:s*(y+1)] = self.train[x,y]
		    	self.training = Training
		    	
	 	self.status = 'Concatenated training data'

	def setDistro(self,bins):	# First find the max + min of the data, to set the bins
		w = shape(self.train)[0] 			# Width 					4
		l = shape(self.train)[1] 			# Length 					16
		T = size(self.train)				# Number of trials in set	64
		i = 0
		mx = empty((T,1)) 				
		for x in xrange(0,w): 			# number of classes
		    for y in xrange(0,l): 		# number of files
			    tex = self.train[x,y]
			    mx[l*x+y] = max(max(tex[:,0]),max(tex[:,1]))

		i = 0
		mn = empty((T,1)) 				
		for x in xrange(0,w): 			# number of classes
		    for y in xrange(0,l): 		# number of files
			    tex = self.train[x,y]
			    mn[l*x+y] = min(min(tex[:,0]),min(tex[:,1]))

		self.binsy = linspace(min(mn), max(mx), bins) # 501 numbers from min to max
		self.status = 'Set the distribution for the likelihoods (max,min,bins)'

	def likelihood(self,norm,smoo,bins):   # Generate the likelihood distributions (histgrams)
		w = shape(self.train)[0] 			# Width 					4
		c = shape(self.train[0,0])[1]		# Number of input channels 	2
		
		Hists = zeros((w,bins-1,2)) # 4 Textures, 500 bins, 2 whisker channels 
		for x in xrange(0,w):
			Hists[x,:,0] = np.histogram(self.training[x,:,0],self.binsy)[0] # generate a histogram of the x input
			Hists[x,:,1] = np.histogram(self.training[x,:,1],self.binsy)[0] # generate a histogram of the y input
			self.status = 'Generated likelihood histograms'

		Hists = Hists + np.finfo(np.float).eps # get rid of that pesky integer maths

		if norm: # normalise each histogram (so integral of each likelihood = 1)
			for x in xrange(0,w):
				Hists[x,:,0] = Hists[x,:,0]/sum(Hists[x,:,0])
				Hists[x,:,1] = Hists[x,:,1]/sum(Hists[x,:,1])
				self.status = 'Normalised the likelihoods'

		if smoo: # Optional smoothing
			for x in xrange(0,w):
				for y in xrange(0,c):
					Hists[x,:,y] = scipy.ndimage.filters.gaussian_filter1d(Hists[x,:,y],1.5) # Smooth with gaussian sd 1.5
					Hists[x,:,y] = Hists[x,:,y]/sum(Hists[x,:,y])
					self.status = 'Smoothed the likelihoods'

		self.hists = Hists
		# self.hists = hists

	def sortTest(self,chunk):
		w = shape(self.test)[0] 			# Width 					4
		l = shape(self.test)[1] 			# Length 					16
		s = shape(self.test[0,0])[0]		# Size (of each trial) 		16000
		c = shape(self.test[0,0])[1]		# Number of input channels 	2

		samps = s/chunk 					# Nu. samples per trial 	40
		TSamps = samps*w*l 					# Total number of samples   2560

		# Chunk in to 200ms (400 sample) bins
		Testing = empty((TSamps,chunk,c))
		TrueClass = empty((TSamps))
		for x in xrange(0,w):
		    for y in xrange(0,l):
		        for z in xrange(0,samps):
		            Testing[l*samps*x+samps*y+z,:] = self.test[x,y][z*chunk:z*chunk+chunk]
		            TrueClass[l*samps*x+samps*y+z] = x

		self.status = 'Separated test data into samples'
		self.testing = Testing
		self.trueClass = TrueClass

	def findDistro(self,bins,chunk):
		w = shape(self.test)[0] 			# Width 					4
		l = shape(self.test)[1] 			# Length 					16
		s = shape(self.test[0,0])[0]		# Size (of each trial) 		16000
		c = shape(self.test[0,0])[1]		# Number of input channels 	2
		
		samps = s/chunk 					# Nu. samples per trial 	40
		TSamps = samps*w*l 					# Total number of samples   2560
		
		## When new data comes in, determine which bin each sample will go in (like matlab [h,binID] = histc())
		DataDist = zeros((TSamps,chunk,c))     			# Distribution of test data. I.E. an array of bin IDs
		# For x
		for x in xrange(0,TSamps):             				 # For each sample
			for y in xrange(0,chunk):         				 # For each data point in that sample
				for z in xrange(0,bins-1):        				 # For each bin in the likelihood bins
					if self.binsy[z] > self.testing[x,y,0]:  # Work out which bin that x value should go in
						DataDist[x,y,0] = z
						break

		# For y
		for x in xrange(0,TSamps):             				 # For each sample
			for y in xrange(0,chunk):          				 # For each data point in that sample
				for z in xrange(0,bins-1):        				 # For each bin in the likelihood bins
					if self.binsy[z] > self.testing[x,y,1]:  # Work out which bin that y value should go in
						DataDist[x,y,1] = z
						break

		self.status = 'Determined distribution of test data'
		self.dataDist = DataDist.astype(np.integer) # Change type to int for use as index


	def findPosterior(self,chunk):
		w = shape(self.test)[0] 			# Width 					4
		l = shape(self.test)[1] 			# Length 					16
		s = shape(self.test[0,0])[0]		# Size (of each trial) 		16000

		samps = s/chunk 					# Nu. samples per trial 	40
		TSamps = samps*w*l 					# Total number of samples   2560

		## Summing log likelihoods for posterior
		Posterior = zeros((TSamps))
		sumlik = zeros((TSamps,w))
		for x in xrange(0,TSamps):             # For each 'trial'
			for y in xrange(0,w):              # For each class
				sumlik[x,y] = sum(log(self.hists[y,self.dataDist[x,:,0],0]+np.finfo(np.float).eps) + log(self.hists[y,self.dataDist[x,:,1],1]+np.finfo(np.float).eps)) # sum log likelihoods + eps
			Posterior[x] = argmax(sumlik[x])

		self.status = 'Computed posterior distribution'
		self.sumlik = sumlik
		self.posterior = Posterior

	def findMarginal(self,thresh,chunk):
		w = shape(self.test)[0] 			# Width 					4
		l = shape(self.test)[1] 			# Length 					16
		s = shape(self.test[0,0])[0]		# Size (of each trial) 		16000

		samps = s/chunk 					# Nu. samples per trial 	40
		TSamps = samps*w*l 					# Total number of samples   2560

		# Need code for adding up self.sumlik to a threshold using marginalisation 
		# First instance, make it just like NL's code. Later, rework it to an 'online' version
		logpr = log(1/4.0)*ones((4))
		sumPos = logpr
		logm = zeros((4))
		decTime = []
		classed = []
		for x in xrange(0,TSamps):
			
			i = i+1 # Put in sanity checks?
			
			sumPos[i] = sumPos[i-1] + self.sumlik[x]/chunk
			logm = log(sum(exp(sumPos[i]))) # log marginal
			sumPos[i] = sumPos[i] - logm 	# recomputed posterior

			# Check threshold crossing
			if any(exp(sumlik[i]) > thresh):
				# store decision time
				decTime = append(decTime,i)
				classed = append(classed,argmax(sumPos[i]))
				self.TrueClass
				
				# reset summed posteriors
				sumPos[i] = logpr

		self.decTime = decTime
		self.classed = classed
		self.sumPos = sumPos



	def findScore(self):
		score = zeros((4));
		for x in xrange(0,size(self.trueClass)):
			if self.trueClass[x] == 0 and self.posterior[x] == 0:
				score[0] = score[0] + 1
			if self.trueClass[x] == 1 and self.posterior[x] == 1:
				score[1] = score[1] + 1
			if self.trueClass[x] == 2 and self.posterior[x] == 2:
				score[2] = score[2] + 1
			if self.trueClass[x] == 3 and self.posterior[x] == 3:
				score[3] = score[3] + 1
		
		self.status = 'Computed normalised classification score'
		return score*100.0/(size(self.trueClass)/4.0)

"""
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
# Run for one chunk size
chunk = 20 #400 # Length of each sample to classify
ML.sortTest(chunk)
print "%s" % ML.status

ML.findDistro(bins,chunk)
print "%s" % ML.status

ML.findPosterior(chunk)
print "%s" % ML.status

ML.score = ML.findScore()
print "%s" % ML.status

print ML.score
show()


# Run for different chunk sizes
steps = range(10,410,10)
ML.score = zeros((4,size(steps)))
for z in xrange(0,size(steps)):
	ch = steps[z]
	print ch
	chunk = ch #400 # Length of each sample to classify
	ML.sortTest(chunk)
	# print "%s" % ML.status

	ML.findDistro(bins,chunk)
	# print "%s" % ML.status

	ML.findPosterior(chunk)
	# print "%s" % ML.status
	
	ML.score[:,z] = ML.findScore()
	# print "%s" % ML.status

# normTerm = [256000/x for x in steps] # Need to normalise score for number of trials
# score = score/newList

np.save('1-16-score.npy',ML.score)

figure(2)
plot(ML.score.T)
# plot(ML.posterior)
# plot(ML.trueClass)
show()
"""

