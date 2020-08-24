import numpy               as     np
from   brian2              import *
from   scipy.signal        import find_peaks, hilbert
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
        Iext  = Ia * sin(2*pi*t*10*Hz)  : amp (constant over dt)
        phi = (1/b) * exp((v-V0)/(a*mV)) : 1
        '''

    # Reset condition
    Neuron = NeuronGroup(N, sn_model, threshold='rand()<phi', reset='v=v_r', refractory=tau_ref, method ='linear')
    
    Neuron.v = v_r
    statemon = StateMonitor(Neuron, ['v', 'Iext'], record=True)
    spikemon = SpikeMonitor(Neuron)
    RateMon  = PopulationRateMonitor(Neuron)
    
    run(Tsim * second, report='stdout')

    return statemon.t/ms, statemon.v[0]/mV, statemon.Iext[0]/pA, spikemon, RateMon

############################################################################################
# Function to compute CC between signal and the spike trains
############################################################################################
def compute_CC(N, Tsim, delta, Ia, noise, a, seed):
    _,_, I, sm, _ = simulate_model(N = N, Tsim = Tsim/1000, dt = delta, Ia=Ia*pA, noise=noise, a = a, seed=seed)
    times         = sm.spike_trains()
    cc            = np.zeros(N)
    for i in range(N):
        spk,_ = np.histogram(times[i]/ms, np.arange(0, Tsim+1, 1))
        cc[i] = np.corrcoef(I[::10], spk, int(spk.max()+1))[0,1]
    return np.nansum(cc) / N


########################################################################################################
# Measuring SR as a function of the level of intrinsic noise
########################################################################################################
delta             = 0.1
N, Tsim, Trials   = 1000, 1000, 10
a_values          = np.linspace(2.5, 10, 40)
I_values          = np.array([100.0, 300., 600.])
CC_single         = np.zeros([Trials, I_values.shape[0], a_values.shape[0]])

for T in range(Trials):
    for i in range(I_values.shape[0]):
        CC_single[T, i, :] =\
        Parallel(n_jobs=a_values.shape[0])(delayed(compute_CC)(N, Tsim, delta, I_values[i], 0, a, T*100) for a in a_values)


for i in range(I_values.shape[0]):
    plt.plot(a_values, np.nanmean(CC_single[:,i,:], axis=0) / np.nanmax(np.nanmean(CC_single[:,i,:], axis=0)), label='I = ' + str(I_values[i]/1000) + ' nA')
plt.legend()
plt.savefig('figures/SR_curves.pdf', dpi=600)
plt.close()

########################################################################################################
# Measuring SR as a function of the level of intrinsic and extrinsic noise
########################################################################################################
N, Tsim, Ia, Trials   = 1000, 1000, 300., 30
a_values          = np.linspace(0.01, 10, 40)
n_values          = np.linspace(0, 20, 20)
CC                = np.zeros([Trials, n_values.shape[0], a_values.shape[0]])

for T in range(Trials):
    for i in range(n_values.shape[0]):
        CC[T, i,:] = \
        Parallel(n_jobs=a_values.shape[0])(delayed(compute_CC)(N, Tsim, delta, Ia, n_values[i], a, T*100) for a in a_values)

CC = np.squeese(CC)

plt.imshow(CC.mean(axis=0),aspect='auto',cmap='jet',origin='lower', extent=[0.01,10,0,20])
plt.colorbar()
plt.ylabel('Noise variance [mV]')
plt.xlabel('Intrinsic stochasticity [mV]')
plt.savefig('figures/SR_map.pdf', dpi=300)
plt.close()

########################################################################################################
# Rasters plots and polar graphs
########################################################################################################
t,V, I, sm, rm = simulate_model(N=50, Tsim = 0.5, Ia = 300. * pA, noise=0, a = 2, seed=0)

plt.subplot2grid((4,1),(0,0), rowspan=2)
plt.plot(sm.t/ms, sm.i, '.k', ms=2.5)
plt.xticks([])
plt.subplot2grid((4,1),(2,0), rowspan=1)
idx = (sm.spike_trains()[0]/ms/0.1).astype(int)-1
V[idx] = -10
plt.plot(t, V)
plt.xticks([])
plt.subplots_adjust(hspace = 0.1)
plt.subplot2grid((4,1),(3,0), rowspan=1)
#plt.plot(t, I)
plt.plot(t, rm.smooth_rate(width=1*ms))
plt.savefig('figures/raster_below.pdf', dpi=150)
plt.close()

t,V, I, sm, rm = simulate_model(N=50, Tsim = 0.5, Ia = 300. * pA, noise=0, a = 5, seed=0)

plt.subplot2grid((4,1),(0,0), rowspan=2)
plt.plot(sm.t/ms, sm.i, '.k', ms=2.5)
plt.xticks([])
plt.subplot2grid((4,1),(2,0), rowspan=1)
idx = (sm.spike_trains()[0]/ms/0.1).astype(int)-1
V[idx] = -10
plt.plot(t, V)
plt.xticks([])
plt.subplots_adjust(hspace = 0.1)
plt.subplot2grid((4,1),(3,0), rowspan=1)
#plt.plot(t, I)
plt.plot(t, rm.smooth_rate(width=1*ms))
plt.savefig('figures/raster_in.pdf', dpi=150)
plt.close()

t,V, I, sm, rm = simulate_model(N=50, Tsim = 0.5, Ia = 300. * pA, noise=0, a = 10, seed=0)

plt.subplot2grid((4,1),(0,0), rowspan=2)
plt.plot(sm.t/ms, sm.i, '.k', ms=2.5)
plt.xticks([])
plt.subplot2grid((4,1),(2,0), rowspan=1)
idx = (sm.spike_trains()[0]/ms/0.1).astype(int)-1
V[idx] = -10
plt.plot(t, V)
plt.xticks([])
plt.subplots_adjust(hspace = 0.1)
plt.subplot2grid((4,1),(3,0), rowspan=1)
#plt.plot(t, I)
plt.plot(t, rm.smooth_rate(width=1*ms))
plt.savefig('figures/raster_above.pdf', dpi=150)
plt.close()

count = 1
for a in [2, 5, 10]:
    t, V, I, sm, rm = simulate_model(N=100, Tsim = 1, Ia = 300. * pA, noise=0, a = a, seed=0)  
    idx   = (sm.t/ms/0.1).astype(int)
    I_spk = I[idx]
    r     = rm.smooth_rate(width=1*ms)
    h_I   = hilbert(I)
    h_r   = hilbert(r)
    psi   = np.angle(h_I)-np.angle(h_r)

    plt.polar(psi[idx], I_spk, '.m')
    plt.ylim(-300,300)

    plt.savefig('figures/polar_a_'+str(a)+'.pdf', dpi=150, transparent=True)
    plt.close()
    #n, x  = np.histogram(I_spk, bins=np.linspace(-300, 300, 100))
    #plt.plot(x[1:], n)