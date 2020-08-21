############################################################################################
# Here we define the method to compute the voltage dependent spike probability
############################################################################################

import numpy             as np
import scipy.signal

def compute_voltage_dependent_spike_probability(data, height = -30, window = 15, bin_length = 0.2):
	#############################################################################################
	# Function to compute the voltge dependent spike time probability
	# Inputs
	# > data : The electrofisiological recording (voltage time-series)
	# > height : The voltage value for which values above will be considered a peak
	# > window : Window size used to isolate the AP
	# > bin_length : The bin size used to construct the voltage and threshold histograms
	# > return_hist : Wheater to return or not histograms of voltage and thresholds
	# Outputs
	# > hist : The  histograms of voltage and thresholds
	# > data_matrix_subthreshold : Voltage time series without the spikes
	# > thr_values : Threshold values found
	# > v : Voltage values for which the voltge dependent spike time probability was computed
	# > phi : Voltge dependent spike time probability
 	#############################################################################################

	# First we detect the index of each peak (maximum value of each action potential) by detecting peaks
	index, _ = scipy.signal.find_peaks(data, height = height)  # peaks with height greater than -30 mV
	# Store the number of peaks (action potential )in the signal
	Npeaks = len(index)
	# Store threshold values
	thr_values = []
		
	# Store all values of the time series except the ones above the threshold value for the respective peak
	data_matrix_subthreshold = []

	# Finding thresholds
	#plt.figure()
	P_a = []
	for i in range(0,Npeaks-1):
		# Here we separete all action potentials, to do so we define a window around 
		# each peak we found above.
		# We are getting the action potential from the time t_peak - window to t_peak
		P = data[index[i]-window:index[i]].copy()
		# First derivative
		P1 = np.diff(P,1)
		# Second derivative
		P2 = np.diff(P,2)
		# Using maximum curvature to detect threshold (Method VII in Sekerli et. al., 2004)
		Kp = P2 * (1 + P1[:-1]**2)**-1.5
		# Get the point where Kp is maximum
		kp_max_idx = np.argmax(Kp) + 1
		# Append threshold found in thr_values
		thr_values.append(P[kp_max_idx].copy())
		#if i > 130 and i < 150:
		#	plt.plot(P, 'k')
		#	plt.plot(kp_max_idx, thr_values[-1], 'ro')
		# Overwriting P
		aux = data[index[i]-window:index[i+1]-window-1].copy()
		aux[aux > thr_values[i]] = np.nan
		# Append peak in data_matrix_subthreshold, excluding all values
		# greater than the found threshold
		data_matrix_subthreshold = np.concatenate((data_matrix_subthreshold, aux.copy()), axis = 0)

	v_m          = np.arange(-80,-30, bin_length)
	data_matrix_subthreshold = data_matrix_subthreshold[~np.isnan(data_matrix_subthreshold)] # Removing NaN

	n1, x1 = np.histogram(data_matrix_subthreshold, v_m)
	n2, x2 = np.histogram(thr_values, v_m)

	# Probability
	p = n2.astype(np.double) / n1.astype(np.double)

	v   = v_m[1:][~np.isnan(p)]
	phi = p[~np.isnan(p)]

	# Storing histograms
	hist = {'v_m': v_m, 'voltage': {}, 'threshold' : {}}
	hist['voltage']['count']   = n1
	hist['voltage']['v']   = x1
	hist['threshold']['count'] = n2
	hist['threshold']['v'] = x2
	return hist, data_matrix_subthreshold, thr_values, v, phi

