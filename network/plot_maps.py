import numpy             as np
import matplotlib.pyplot as plt

# Load maps
F   = np.load('maps/Fmap.npy', allow_pickle=True).item()['F']
CV  = np.load('maps/CVmap.npy', allow_pickle=True).item()['CV']
HS  = np.load('maps/HSmap.npy', allow_pickle=True).item()['HS']
PLV = np.load('maps/PLVmap.npy', allow_pickle=True).item()['PLV']
MAPS = [F, CV, HS, PLV]

# Network parameters
a_values          = np.array([0.5, 1.2, 2.0, 5.0])
poisson_frequency = np.linspace(2.5, 40, 20)
exc_inh_ratio     = np.linspace(0, 7, 20)


count = 1
for i in range(a_values.shape[0]):
	plt.subplot(4,4,count)
	plt.imshow(MAPS[i], aspect='auto', cmap='jet', origin = 'lower', interpolation='gaussian')

	if i == 0:
		plt.title('a = ' + str(a_values[i]) + ' mV')

	count += 1
plt.show()