import meep as mp
import meep.adjoint as mpa
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# --------------------------- #
# Design filter
# --------------------------- #
freq_min = 0.9
freq_max = 1.1
nfreq = 3
freqs = np.linspace(freq_min, freq_max, nfreq)

# --------------------------- #
# Run normal simulation
# --------------------------- #
cell = mp.Vector3(5, 5)

fcen = 1.0
gauss_src = mp.GaussianSource(frequency=fcen, fwidth=0.3*fcen)

sources = [mp.Source(gauss_src,
                     component=mp.Ey,
                     center=mp.Vector3(x=-1),
                     amplitude=1)]

pml_layers = [mp.PML(1.0)]
resolution = 10

sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    sources=sources,
                    resolution=resolution)

mon = sim.add_dft_fields([mp.Ey], freqs, center=mp.Vector3(), size=mp.Vector3(y=0.5))

# sim.plot2D()
# plt.show()

sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mp.Vector3(), 1e-9))

fields = np.zeros(nfreq, dtype=np.complex128)
# just store one spatial point for each freq
for f in range(nfreq):
    fields[f] = sim.get_dft_array(mon, mp.Ey, f)[2]

# build simple bandpass filter
dt = sim.fields.dt
fs = 1/dt
freqs_scipy = freqs * np.pi / (fs/2)
num_taps = 320
taps = signal.firwin(num_taps, [0.8*freq_min/(fs/2), freq_max/(fs/2)], pass_zero=False, window='boxcar')
w,h = signal.freqz(taps, worN=freqs_scipy)

# frequency domain calculation
desired_fields = h * fields

# --------------------------- #
# Run filtered simulation
# --------------------------- #
filtered_src = mpa.FilteredSource(fcen, freqs, h, dt, gauss_src)

sources = [mp.Source(filtered_src,
                     component=mp.Ey,
                     center=mp.Vector3(x=-1),
                     amplitude=1)]

sim.reset_meep()
sim.change_sources(sources)

mon = sim.add_dft_fields([mp.Ey], freqs, center=mp.Vector3(), size=mp.Vector3(y=0.5))

sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mp.Vector3(), 1e-9))

fields_filtered = np.zeros(nfreq, dtype=np.complex128)
# just store one spatial point for each freq
for f in range(nfreq):
    fields_filtered[f] = sim.get_dft_array(mon, mp.Ey, f)[2]

# --------------------------- #
# Compare results
# --------------------------- #
plt.figure()
plt.subplot(2, 2, 1)
plt.semilogy(freqs, np.abs(desired_fields)**2, 'o', markersize=3, label='Frequency Domain')
plt.semilogy(freqs, np.abs(fields_filtered)**2, '--', label='Time Domain')
plt.grid()
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.legend()

plt.subplot(2, 2, 2)
plt.scatter(range(len(desired_fields)), (np.abs(fields_filtered)**2 - np.abs(desired_fields)**2) / np.abs(desired_fields) * 100)
plt.xlabel('Points')
plt.ylabel('Relative Error (%)')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(freqs, np.unwrap(np.angle(desired_fields)), 'o', markersize=3, label='Frequency Domain')
plt.plot(freqs, np.unwrap(np.angle(fields_filtered)), '--', label='Time Domain')
plt.grid()
plt.xlabel('Frequency')
plt.ylabel('Angle')
plt.legend()

plt.subplot(2, 2, 4)
plt.scatter(range(len(desired_fields)), (np.unwrap(np.angle(fields_filtered)) - np.unwrap(np.angle(desired_fields))) / np.unwrap(np.angle(desired_fields)) * 100)
plt.xlabel('Points')
plt.ylabel('Relative Error (%)')
plt.grid(True)

plt.tight_layout()
plt.show()