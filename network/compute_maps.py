from   brian2          import *
import pandas          as     pd
from   joblib          import Parallel, delayed
import multiprocessing
import sys
from   scipy.integrate import simps
from   scipy.stats     import ks_2samp
from   scipy.signal    import hilbert
from   tools           import *

def read_file(a, v_ext, g):
    filename = 'data/a_'+str(int(10*a))+'_vext_'+str(int(10*v_ext))+'_g_'+str(int(10*g))+'.npy'
    return np.load(filename, allow_pickle=True).item()

def create_spiketrain(i, times, Tsim, dt):    
    Nbins = int(Tsim / dt)
    idx      = (times[i]/dt/ms).astype(int)
    spike_train = np.zeros(Nbins)  
    spike_train[idx]=1 / dt
    return spike_train

def compute_freq(a, v_ext, g, N, Tsim):
    times = read_file(a, v_ext, g)
    return np.sum(np.fromiter((len(t) for t in times.values()), dtype=int)) / (N * Tsim * 1e-3)

def compute_cv(a, v_ext, g):
    from elephant.statistics import isi, cv
    times = read_file(a, v_ext, g)
    cv_list = [cv(isi(spiketrain)) for spiketrain in times.values()]
    return np.nanmean(cv_list)

def compute_Hspec(a, v_ext, g, N, Tsim, dt):
    Tsim = Tsim * 1e-3
    dt   = dt   * 1e-3
    Nbins = int(Tsim / dt)
    times = read_file(a, v_ext, g)
    df = 1.0 / Tsim
    xf = np.arange(0.0, 1.0/(2*dt)+df, df)
    sxx= np.zeros(xf.shape[0])
    for i in range(N):
        spike_train = np.zeros(Nbins)  
        idx         = (times[i]/dt/second).astype(int)
        spike_train[idx]=1 / dt
        yf = np.fft.rfft(spike_train,n=len(spike_train))
        sxx_aux = (np.multiply(yf,np.conjugate(yf))*(dt)*(dt))
        sxx += sxx_aux.real /Tsim
    sxxM2=(sxx/N).real
    sxxM2=sxxM2 / simps(sxxM2) 
    Hs = -np.sum(sxxM2[1:] * np.log(sxxM2[1:])) / np.log(len(xf)-1)
    return Hs

def computePLV(a, v_ext, g, N, Tsim, dt):
    times = read_file(a, v_ext, g)
    Nbins = int(Tsim / dt) 
    PLV  = np.zeros(N)
    for i in range(N):
        spk_1 = np.zeros(Nbins)  
        spk_2 = np.zeros(Nbins)  
        idx1, idx2 = (times[i]/dt/ms).astype(int), (times[i+1]/dt/ms).astype(int)
        spk_1[idx1]= 1 / dt
        spk_2[idx2]= 1 / dt
        h1 = hilbert(spk_1)
        h2 = hilbert(spk_2)
        theta1 = np.unwrap(np.angle(h1))
        theta2 = np.unwrap(np.angle(h2))
        complex_phase_diff = np.exp(np.complex(0,1)*(theta1 - theta2))
        PLV[i] = np.abs(np.sum(complex_phase_diff))/len(theta1)
    return np.nanmean(PLV)

# Simulation paramters
Tsim   = 10000 # ms
dt     = 0.1   # ms

# Network parameters
N      = 12500
a_values          = np.array([0.5, 1.2, 2.0, 5.0])
poisson_frequency = np.linspace(2.5, 40, 20)
exc_inh_ratio     = np.linspace(0, 7, 20)


F_map   = np.zeros([len(a_values), poisson_frequency.shape[0], exc_inh_ratio.shape[0]])

for j in range(len(a_values)):
    for i in range(poisson_frequency.shape[0]):
        print('a = ' + str(a_values[j]) + ' mV, v = ' + str(poisson_frequency[i]) + ' Hz')
        F_map[j,i,:]  = Parallel(n_jobs=40, backend='loky', max_nbytes=1e6)(delayed(compute_freq)(a_values[j], poisson_frequency[i], g, N, Tsim) for g in exc_inh_ratio)
np.save('maps/Fmap.npy', {'F':F_map})

CV_map  = np.zeros([len(a_values), poisson_frequency.shape[0], exc_inh_ratio.shape[0]])

for j in range(len(a_values)):
    for i in range(poisson_frequency.shape[0]):
        print('a = ' + str(a_values[j]) + ' mV, v = ' + str(poisson_frequency[i]) + ' Hz')
        CV_map[j,i,:] = Parallel(n_jobs=40, backend='loky', max_nbytes=1e6)(delayed(compute_cv)(a_values[j], poisson_frequency[i], g) for g in exc_inh_ratio)
np.save('maps/CVmap.npy', {'CV':CV_map})

HS_map  = np.zeros([len(a_values), poisson_frequency.shape[0], exc_inh_ratio.shape[0]])

for j in range(len(a_values)):
    for i in range(poisson_frequency.shape[0]):
        print('a = ' + str(a_values[j]) + ' mV, v = ' + str(poisson_frequency[i]) + ' Hz')
        HS_map[j,i,:] = Parallel(n_jobs=40, backend='loky', max_nbytes=1e6)(delayed(compute_Hspec)(a_values[j], poisson_frequency[i], g, N, Tsim, dt) for g in exc_inh_ratio)
np.save('maps/HSmap.npy', {'HS': HS_map})

PLV_map = np.zeros([len(a_values), poisson_frequency.shape[0], exc_inh_ratio.shape[0]])

for j in range(len(a_values)):
    for i in range(poisson_frequency.shape[0]):
        print('a = ' + str(a_values[j]) + ' mV, v = ' + str(poisson_frequency[i]) + ' Hz')
        PLV_map[j,i,:] = Parallel(n_jobs=40, backend='loky', max_nbytes=1e6)(delayed(computePLV)(a_values[j], poisson_frequency[i], g, 10000, Tsim, dt) for g in exc_inh_ratio)
np.save('maps/PLVmap.npy', {'PLV': PL_map})

