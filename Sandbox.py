""" Sandbox for probing the data quickly in iPython """

for x in xrange(0,4):
	figure(1) # Plot the histograms for a cheeky look
	histy = subplot(2,2,x)
	histy.plot(ML.training[x])
	# xlim((100,300))

show()