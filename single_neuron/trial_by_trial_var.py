import numpy               as     np
from   brian2              import *
from   scipy.signal        import find_peaks
from   joblib              import Parallel, delayed
import multiprocessing

############################################################################################
# Implementation of the single neuron in Brian 2
############################################################################################
def simulate_model(N=1000, Tsim = 1, dt = 0.1, Ia = 0, noise=0.0, a = 1.2, seed=0):
    start_scope()

    defaultclock.dt = dt * ms

    tau_m   = 10.0*ms		# membrane time constant
    tau_ref = 2.0*ms		# absolute refractory period
    Cm      = 250.0*pF		# membrane capacity
    v_r     = -65.0*mV		# reset potential
    v_th    = -51.3*mV		# fixed firing threshold
    
    np.random.seed(seed)

    a, b, V0 = a, 27.07, v_th
    stimulus = TimedArray(np.hstack([c for c in (noise*np.random.randn(int(Tsim*1000/0.1)))]), dt=0.1*ms)
    
    # Stochastic model model equations
    sn_model = '''
        dv/dt = (-v + v_r)/tau_m + Iext/Cm + stimulus(t)*(mV/ms) : volt (unless refractory)
        Iext : amp
        phi = (1/b) * exp((v-V0)/(a*mV)) : 1
        '''

    # Reset condition
    Neuron = NeuronGroup(N, sn_model, threshold='rand()<phi', reset='v=v_r', refractory=tau_ref, method ='linear')
    Neuron.Iext = Ia
    
    Neuron.v = v_r
    statemon = StateMonitor(Neuron, ['v'], record=True)
    spikemon = SpikeMonitor(Neuron)
    RateMon  = PopulationRateMonitor(Neuron)
    
    run(Tsim * second, report='stdout')

    return statemon.t/ms, statemon.v[0]/mV, spikemon, RateMon


#########################################################################################################################
# Trial-by-trial variability
#########################################################################################################################
bins = np.arange(0, 500, 1)

[t, v, n,_] = simulate_model(N = 1000, Tsim = .5, dt = 0.1, Ia=400.*pA, noise=0.0, a = 1.2)
plt.subplot2grid((3,1),(0,0), rowspan=2)
plot(n.t/ms, n.i, '.k', ms=2)
plt.ylim(0, 50)
plt.xlim(100, 500)
plt.xticks([])
ylabel('Neuron index')
plt.subplot2grid((3,1),(2,0), rowspan=1)
plt.hist(n.t/ms, bins)
plt.xlim(100, 500)
xlabel('Time (ms)')
plt.subplots_adjust(hspace = 0.1)
plt.savefig('figures/raster_no_noise.pdf', dpi=60)
plt.close()

[t, v, n,_] = simulate_model(N = 1000, Tsim = .5, dt = 0.1, Ia=400.*pA, noise=7.0, a = 1.2)
plt.subplot2grid((3,1),(0,0), rowspan=2)
plot(n.t/ms, n.i, '.k', ms=2)
plt.ylim(0, 50)
plt.xlim(100, 500)
plt.xticks([])
ylabel('Neuron index')
plt.subplot2grid((3,1),(2,0), rowspan=1)
plt.hist(n.t/ms, bins)
plt.xlim(100, 500)
xlabel('Time (ms)')
plt.subplots_adjust(hspace = 0.1)
plt.savefig('figures/raster_noise.pdf', dpi=60)
plt.close()

#########################################################################################################################
# Reliability deterministic vs stochastic
#########################################################################################################################
def reliability(atv, r, N):
	# Measures reliability based on the PSTH
    x   = atv[10000:] / N
    thr = r * x.max()
    peaks, properties = find_peaks(x, thr)
    return np.nanmean(x[peaks])

def run_trials(N, Tsim, delta, Ia, noise, seed):
	r = np.zeros(noise.shape[0])
	for i in range(noise.shape[0]):
		_,_, sm, rm = simulate_model(N = N, Tsim = Tsim/1000, dt = delta, Ia=Ia*pA, noise=noise[i], a = 1.2, seed=seed)
		atv, _      = np.histogram(sm.t/ms, np.arange(0, Tsim+delta,delta))
		r[i]        = reliability(atv, 0.7, N)
	return r

# Defines noise variace
noise  = np.linspace(0, 20, 10)
# Trials
Trials = 10
# Vector to store reliability value for each noise variance
#R      = np.zeros([Trials, noise.shape[0]])
# Number of trials
N      = 10000
# Simulation time
Tsim   = 10000
# Inegration time-step
delta  = 0.1
# DC current
Ia     = 200.

R = Parallel(n_jobs=-1, backend='loky', max_nbytes=1e6)(
		delayed(run_trials)
		(N, Tsim, delta, Ia, noise, T*100)  
		for T in range(Trials)
		)
R = np.squeeze(R)

plt.errorbar(noise, R.mean(axis=0), R.std(axis=0) / np.sqrt(Trials), fmt='s')
plt.ylabel('Reliability')
plt.xlabel('Noise variance [mV]')
plt.savefig('figures/reliability.pdf', dpi = 600)
plt.close()