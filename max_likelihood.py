""" Coding up maximum likelihood in python """

from numpy import *
import numpy as np
from pylab import *
import scipy.io
import scipy.ndimage

# Load up the texture data
data_roomba = scipy.io.loadmat('../../../Nathans/paperfigs_RSIresub_1011/data_roomba.mat')
texdata = data_roomba['data'] # texdata = 4 textures * 16 trials

# plot some raw data, to see how it's looking

for x in xrange(0,4):
    texA = texdata[x,5]
    sigs = subplot(2,2,x)
    figure(1)
    sigs.plot(texA)
    title('Whisker data from Roomba', size="small")
    ylabel('Magnitude of deflection (V)', size="small")
    xlabel('Time (msec)', size="small")
   # legend(size="small")
   # axes(size="small")
    ylim((0,5))
#show()

########## TRAINING ###########

## Generate histograms from the training data (First 8 of 16 whole files for now)

# First find the max + min of the data, to set the bins
i = 0
mx = zeros((32,1)) #size(texdata))
for x in xrange(0,4):
    for y in xrange(0,8):
        tex = texdata[x,y]
        mx[i] = max(max(tex[:,0]),max(tex[:,1]))
        i = i+1
#print max(mx)

i = 0
mn = zeros((32,1)) #size(texdata))
for x in xrange(0,4):
    for y in xrange(0,8):
        tex = texdata[x,y]
        mn[i] = min(min(tex[:,0]),min(tex[:,1]))
        i = i+1
        
#print min(mn)

binsy = linspace(min(mn), max(mx), 501) # 501 numbers from min to max

#print(binsy)

# Concatenate training data
Training = zeros((4,256000,2))
for x in xrange(0,4):
    Training[x] = vstack((texdata[x,0],texdata[x,1],texdata[x,2],texdata[x,3],texdata[x,4],texdata[x,5],texdata[x,6],texdata[x,7]))
    
# Generate the histograms. Changed to 4 classes instead of 32
#i = 0
hists = zeros((4,500,2)) # 4 Textures, 500 bins, 2 whisker channels 
for x in xrange(0,4):
#    for y in xrange(0,8):
#        tex = texdata[x,y]
    hists[x,:,0] = np.histogram(Training[x,:,0],binsy)[0] # generate a histogram of the x input
    hists[x,:,1] = np.histogram(Training[x,:,1],binsy)[0] # generate a histogram of the y input

    # normalise each histogram (so integral of each likelihood = 1)
    hists[x,:,0] = hists[x,:,0]/sum(hists[x,:,0])
    hists[x,:,1] = hists[x,:,1]/sum(hists[x,:,1])

    figure(2) # Plot the histograms for a cheeky look
    histy = subplot(2,2,x)
    histy.plot(hists[x])
    xlim((100,300))
#        i = i+1
#show()
# print hist
# print bin_edges


# Optional smoothing
for x in xrange(0,4):
    for y in xrange(0,2):
         hists[x,:,y] = scipy.ndimage.filters.gaussian_filter1d(hists[x,:,y],2) # Smooth with gaussian sd 2
         hists[x,:,y] = hists[x,:,y]/sum(hists[x,:,y])

############# TESTING ##############

# Chunk in to 200ms (400 sample) bins
Testing = zeros((2560,400,2))
TrueClass = zeros((2560))
for x in xrange(0,4):
    for y in xrange(0,8):
        for z in xrange(0,80):
            Testing[640*x+80*y+z,:] = texdata[x,y+8][z*400:z*400+400]
            TrueClass[640*x+80*y+z] = x
## When new data comes in, determine which bin each sample will go in (like matlab [h,binID] = histc())
dataDist = zeros((2560,400,2))    # Distribution of test data. I.E. an array of bin IDs

# For x
for x in xrange(0,2560):             # For each 'trial'
    for y in xrange(0,400):          # For each sample in that trial
        for z in xrange(0,500):      # For each bin in the likelihood bins
            if binsy[z] > Testing[x,y,0]:  # Work out which bin that x value should go in
                dataDist[x,y,0] = z
                break
              
# For y
for x in xrange(0,2560):             # For each 'trial'
    for y in xrange(0,400):          # For each sample in that trial
        for z in xrange(0,500):      # For each bin in the likelihood bins
            if binsy[z] > Testing[x,y,1]:  # Work out which bin that y value should go in
                dataDist[x,y,1] = z
                break 
            
## Summing log likelihoods for posterior
dataDist = dataDist.astype(np.integer) # Change type to int for use as index
posterior = zeros((2560))
sumlik = zeros((4))
for x in xrange(0,2560):               # For each 'trial'
    for y in xrange(0,4):              # For each class
        sumlik[y] = sum(log(hists[y,dataDist[x]]+np.finfo(np.float).eps)) # sum log likelihoods + eps
    posterior[x] = argmax(sumlik)

figure(3)
plot(posterior)
plot(TrueClass)
show()


