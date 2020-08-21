import numpy             as np
import pandas            as pd 
import matplotlib.pyplot as plt
from   spike_probability import compute_voltage_dependent_spike_probability
from   fit_phi           import exponential_fit

# Reading data
time_series = np.squeeze( pd.read_csv('data/time_series.dat', header=None).values.T )

##############################################################################
# Plotting the voltage time series
##############################################################################
plt.figure(figsize = (10, 4))
plt.subplot(4,1,1)
plt.plot(time_series[5000:5000+100000])
plt.axis('off')
plt.subplot(4,1,2)
plt.plot(time_series[50000:50000+100000])
plt.axis('off')
plt.subplot(4,1,3)
plt.plot(time_series[150000:150000+100000])
plt.axis('off')
plt.subplot(4,1,4)
plt.plot(time_series[200000:200000+100000])
plt.axis('off')
plt.savefig('figures/traces.pdf', dpi = 600)
plt.close()

##############################################################################
# Computing voltage dependent probability
##############################################################################
hist, data_matrix_subthreshold, thr_values, v, phi = \
compute_voltage_dependent_spike_probability(time_series, height = -30, window = 15, bin_length = 0.2)

##############################################################################
# Plotting histograms
##############################################################################
plt.figure()
fig, ax1 = plt.subplots()
ax1.hist(data_matrix_subthreshold, bins = hist['v_m'], alpha = 0.8, log=False, color='blue')
ax2 = ax1.twinx()
ax2.hist(thr_values, bins = hist['v_m'], alpha = 0.8, log=False, color='orange')
plt.savefig('figures/histograms.pdf', dpi = 600)
plt.close()

##############################################################################
# Fitting phi(v)
##############################################################################
x,y,x_fit, y_fit,r = exponential_fit(data_matrix_subthreshold, thr_values, v, phi)

##############################################################################
# Plotting phi(V)
##############################################################################
plt.figure()
plt.plot(x_fit, y_fit)
plt.plot(v, phi, 'r.')
plt.ylim([-0.01, 1+0.01])
plt.xlim([-60, -45])
plt.legend([r'$r^{2} = $'+str(r), 'Experimental data'])
plt.ylabel(r'$\phi(V)$')
plt.xlabel(r'$V$ [mV]')
plt.savefig('figures/spike_probability.pdf', dpi = 600)
plt.close()
