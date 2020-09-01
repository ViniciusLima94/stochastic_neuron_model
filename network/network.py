from   brian2            import *
from   joblib            import Parallel, delayed
from   tools             import *
import multiprocessing
import sys

defaultclock.dt = 0.1 * ms

def brunel(N = 12500, a = 1.194, g = 6.0, v_ext = 20, Tsim = 5000, D = 1.5):
   
    start_scope()

    ##################################################################################
    # Network Parameters
    # N : Number of neurons in the Network
    # f : Fraction of excitatory neurons
    # Nexct : Number of excitatory neurons
    # Ninhb : Number of inhibitory neurons
    # g     : Excitation/Inhibition ratio
    # R     : Ratio between \ni_{thr} and \ni_{ext}
    # delay : Synaptic delay
    ##################################################################################
    f     = 0.8
    Nexct = int(f * N)
    Ninhb = int((1-f) * N)
    w_ex  = 0.1 * mV
    w_in  = -g * w_ex
    delay = D * ms
    Ce    = 1000
    Ci    = int( Ce / 4.0 )

    ##################################################################################
    # Neuron Parameters
    # tau_m : Membrane time constant
    # tau_ref : Refractory period
    # Vrest   : Resting potential
    # Vreset  : Reset potential
    # Vth     : Threshold
    # eqs     : Model equation
    # v_ext   : External input frequency
    ##################################################################################
    tau_m    = 20. * ms
    tau_ref  =  2. * ms
    Vrest    =  0. * mV
    Vreset   = 10. * mV
    Vth      = 20. * mV
    b, V0    = 27.07, Vth

    eqs = '''
      dV/dt = -(V - Vrest) / tau_m : volt (unless refractory)
      phi = (1/b) * exp((V-V0)/(a*mV)) : 1
    '''

    ##################################################################################
    # Creating Network of Spiking Neurons
    ##################################################################################
    Network = NeuronGroup(N, model = eqs, threshold = 'rand()<phi', reset = 'V = Vreset', method = 'euler', refractory = tau_ref)
    Network.V = 'rand()*(Vth-Vreset) + Vreset'
    PopEx = Network[:Nexct]
    PopIn = Network[Nexct:]

    ##################################################################################
    # Creating Connections
    ##################################################################################
    presyn_indices,postsyn_indices=fixed_indegree(Ce,(Nexct+Ninhb),Nexct)
    ExSyn = Synapses(PopEx, target = Network, on_pre = 'V += w_ex', delay = delay)
    ExSyn.connect(i=presyn_indices,j=postsyn_indices)
    presyn_indices,postsyn_indices=fixed_indegree(Ci,(Nexct+Ninhb),Ninhb)
    InSyn = Synapses(PopIn, target = Network, on_pre = 'V += w_in', delay = delay)
    InSyn.connect(i=presyn_indices,j=postsyn_indices)

    ##################################################################################
    # Poisson Background
    ##################################################################################
    PoissonBG = PoissonInput(target = Network, target_var = 'V', N = Ce, rate = v_ext * Hz, weight = w_ex)
   
    ##################################################################################
    # Devices
    ##################################################################################
    SpikeMon = SpikeMonitor(Network, record = True)

    ##################################################################################
    # Simulation
    ##################################################################################
    run(Tsim*ms)

    np.save('data/a_'+str(int(10*a))+'_vext_'+str(int(10*v_ext))+'_g_'+str(int(10*g))+'.npy', SpikeMon.spike_trains())

'''
a_idx             = int(float(sys.argv[-1]))
a_values          = np.array([0.5, 1.2, 2.0, 5.0])
poisson_frequency = np.linspace(2.5, 40, 20)
exc_inh_ratio     = np.linspace(0, 7, 20)
Parallel(n_jobs=-1)(
				   delayed(brunel)(N = 12500, a = a_values[a_idx], g = g, v_ext = v, Tsim = 10000, D = 1.5) 
	               for v in poisson_frequency for g in exc_inh_ratio
	               )
'''

idx = int(float(sys.argv[-1]))
par = np.load('net_params.npy', allow_pickle=True)

try:
    filename = 'data/a_'+str(int(10*par[idx, 0]))+'_vext_'+str(int(10*par[idx,1]))+'_g_'+str(int(10*par[idx,2]))+'.npy'
    np.load(filename, allow_pickle=True).item()
except:
    brunel(N = 12500, a = par[idx, 0], g = par[idx,2], v_ext = par[idx,1], Tsim = 10000, D = 1.5)