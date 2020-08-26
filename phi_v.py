import numpy as np
import matplotlib.pyplot as plt

def phi_v(a, b, Vth, V):
	return (1/b) * np.exp((V-Vth)/a)

a   = np.array([0.5, 1.2, 2])
b   = 27
Vth = 20 
V   = np.linspace(0, 30, 1000)

for i in range(a.shape[0]):
	phi = phi_v(a[i], b, Vth, V)
	plt.plot(V, phi)
plt.ylim([-0.01, 1])

