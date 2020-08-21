############################################################################################
# Here we define the method to fit the voltage dependent spike probability
############################################################################################

import numpy             as np
import scipy.optimize

def heaviside(x, x_th):
	###########################################################################
	# Defines the heaviseid funtion. If x is less or equal x_th returns False,
	# and True otherwise.
	###########################################################################
	return np.logical_not(x<=x_th)

def f(x, a, b, x_th):
	###########################################################################
	# Defines the exponential funtion used to fit the data.
	###########################################################################
	return (1/b) * np.exp((x-x_th)/a)

def exponential_fit(data, thresholds, x, y):
	###########################################################################
	# Performs the exponential fit
	###########################################################################
	
	data      = data
	Vth       = np.mean( thresholds )
	V         = x
	phi       = y

	v   = np.linspace(data.min(), data.max()+10, phi.shape[0])

	idx = (V>-54)*(V<-48) 
	x   = V[idx]
	y   = phi[idx]

	a, b = np.polyfit(x, np.log(y), 1)
	A, B = 1 / a, np.exp(-b-Vth*a)
	p0   = [A, B, Vth]
	p,_  = scipy.optimize.curve_fit(f, x, y, p0=p0) 


	p1 = f(v, p[0], p[1], p[2]) 

	r2 = np.corrcoef(phi,p1)[0,1]

	return x, y, v, p1, r2
'''
plt.plot(v, p1)
plt.plot(V, phi, 'r.')
plt.ylim([-0.01, 1+0.01])
plt.xlim([-60, -45])
plt.legend([r'$r^{2} = $'+str(r2), 'Experimental data'])



	Vth = np.mean( thresholds )

	idx = (x>-54)*(x<-48) 
	#idx = (x>-58)*(x<-46) 
	x   = x[idx]
	y   = y[idx]

	a, b = np.polyfit(x, np.log(y), 1)
	A, B = 1 / a, np.exp(-b-Vth*a)
	p0   = [A, B, Vth]
	p,_  = scipy.optimize.curve_fit(f, x, y, p0=p0) 


	x_fit   = np.linspace(data.min(), data.max()+10, y.shape[0])
	y_fit = f(x_fit, p[0], p[1], p[2]) 

	r2 = np.corrcoef(y,y_fit)[0,1]

	return x, y, x_fit, y_fit, r2
'''
