import numpy as np
import meep as mp
import meep.adjoint as mpa
import matplotlib.pyplot as plt
from collections import namedtuple

Grid = namedtuple('Grid', ['x', 'y', 'z', 'w'])

EH_components = [[mp.Ey, mp.Ez, mp.Hy, mp.Hz], [mp.Ez, mp.Ex, mp.Hz, mp.Hx], [mp.Ex, mp.Ey, mp.Hx, mp.Hy]]
sign = [-1.0, -1.0, 1.0, 1.0]

np.random.seed(240)

def atleast_3d(*arys):
    from numpy import asanyarray, newaxis

    res = []
    for ary in arys:
        ary = asanyarray(ary)
        if ary.ndim == 0:
            result = ary.reshape(1, 1, 1)
        elif ary.ndim == 1:
            result = ary[:, newaxis, newaxis]
        elif ary.ndim == 2:
            result = ary[:, :, newaxis]
        else:
            result = ary
        res.append(result)
    if len(res) == 1:
        return res[0]
    else:
        return res

def fix_array_metadata(xyzw, center, size):
    """fixes for the perenially buggy get_array_metadata routine in core meep."""
    for d in range(0, 3):
        xyzw[d] = (center[d],) if size[d] == 0.0 and xyzw[d][0] != center[d] else xyzw[d]
    return xyzw

''' Define epsilon matrix '''
Air = 1.00
TiO2 = 2.51 ** 2

fcen = 1/1.55
frequencies = 1 / np.linspace(1.5, 1.6, 3)
nf = len(frequencies)

res = 10

dpml = 1.0
dpad = 0.5

npml = int(dpml * res)

nx = int(4.0 * res)
ny = int(4.0 * res)

design_region_x = int(1.0 * res)
design_region_y = int(1.0 * res)

eps = np.ones((nx, ny))
design_region = np.random.rand(design_region_x, design_region_y) * (TiO2 - Air) + Air

start = int((dpml + dpad) * res)
end_x = start + design_region_x
end_y = start + design_region_y

eps[start:end_x, start:end_y] = design_region

''' Define epsilon function '''
def dat2pos(r, data, nx, ny, res, val):
    decplace = 12

    x0 = nx / (2 * res)
    y0 = ny / (2 * res)
    jx = np.around((r.x + x0) * res, decplace)
    jy = np.around((r.y + y0) * res, decplace)
    ix = int(np.floor(jx))
    iy = int(np.floor(jy))

    if 0 <= ix < nx and 0 <= iy < ny:
        return data[ix, iy]
    else:
        return val

def eps_func(p):
    return dat2pos(p, eps, nx, ny, res, 1)

''' Define simulation '''
Lx = nx / res
Ly = ny / res
dpml = npml / res

cell = mp.Vector3(Lx, Ly)

src = mp.GaussianSource(frequency=fcen, fwidth=0.4*fcen)
sources = [mp.Source(src,
                     component=mp.Ey,
                     center=mp.Vector3(x=-0.8),
                     amplitude=10)]

f_center = mp.Vector3(x=0.75)
f_size = mp.Vector3(y=1.0)

sim = mp.Simulation(cell_size=cell,
                    boundary_layers=[mp.PML(dpml)],
                    epsilon_func=eps_func,
                    eps_averaging=False,
                    resolution=res,
                    sources=sources,
                    force_complex_fields=True)

# Add design region monitor
fwd_mon = sim.add_dft_fields([mp.Ex, mp.Ey], frequencies, where=mp.Volume(center=mp.Vector3(), size=cell), yee_grid=True)

# Add objective function monitor
f_mon = sim.add_flux(frequencies, mp.FluxRegion(center=f_center, size=f_size))
normal_direction = f_mon.normal_direction

# sim.plot2D()
# plt.show()

decay_by = 1e-8

''' Forward simulation '''
sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ey, mp.Vector3(), decay_by))

f_grid = Grid(*fix_array_metadata(sim.get_array_metadata(dft_cell=f_mon), f_center, f_size))

# Get forward simulation fields
d_E = np.zeros((nx, ny, 1, nf, 3), dtype=np.complex128)
for f in range(nf):
    for ic, c in enumerate([mp.Ex, mp.Ey, mp.Ez]):
        a = sim.get_dft_array(fwd_mon, c, f)
        d_E[:, :, :, f, ic] = atleast_3d(sim.get_dft_array(fwd_mon, c, f))

# Get fields at monitor
m_EH = np.zeros((len(f_grid.x) * len(f_grid.y) * len(f_grid.z), nf, 4), dtype=complex)
for f in range(nf):
    for ic, c in enumerate(EH_components[normal_direction]):
        m_EH[:, f, ic] = sim.get_dft_array(f_mon, c, f).flatten()

''' Calculate adjoint source '''
time_src = sim.sources[0].src
# T = 2 * time_src.cutoff * time_src.width
T = sim.meep_time()
dt = sim.fields.dt

dV = 1 / res ** 2

# an ugly way to calcuate the scaled dtft of the forward source
t_signal = np.array([time_src.swigobj.current(t, dt) for t in np.arange(0, T, dt)])  # time domain signal
fwd_dtft = np.matmul(np.exp(1j * 2 * np.pi * frequencies[:, np.newaxis] * np.arange(t_signal.size) * dt), t_signal) * dt / np.sqrt(2 * np.pi)  # dtft

# we need to compensate for the phase added by the time envelope at our freq of interest
src_center_dtft = np.matmul(np.exp(1j * 2 * np.pi * np.array([time_src.frequency])[:, np.newaxis] * np.arange(t_signal.size) * dt), t_signal) * dt / np.sqrt(2 * np.pi)
adj_src_phase = np.exp(1j * np.angle(src_center_dtft))

# TODO: Why the 'frequencies' is needed?
# iomega in FDTD
iomega = (1.0 - np.exp(-1j * (2 * np.pi * fcen) * dt)) * (1.0 / dt) * (frequencies/fcen)

scale = dV * iomega / adj_src_phase
amp = sign * scale[None, :, None] * atleast_3d(f_grid.w) * np.conj(m_EH)

src_list = []
non_zero_c = []

for ic, c in enumerate(EH_components[normal_direction]):
    if np.any(amp[:, :, ic]):
        # This component has value
        non_zero_c.append(c)
        src_list += [[mpa.FilteredSource(time_src.frequency, frequencies, amp[iv, :, ic], dt) for iv in
                      range(len(f_grid.x) * len(f_grid.y) * len(f_grid.z))]]

x, y, z = np.squeeze(np.meshgrid(f_grid.x, f_grid.y, f_grid.z))
adj_src = []
for ic, c in enumerate(non_zero_c):
    for ip, p in enumerate(zip(x, y, z)):
        adj_src.append(mp.Source(src_list[ic][ip],
                                 component=non_zero_c[1-ic],
                                 center=p,
                                 amplitude=1))

''' Adjoint simulation '''
sim.reset_meep()

# Change to adjoint source
sim.change_sources(adj_src)

# Add design region monitor
adj_mon = sim.add_dft_fields([mp.Ex, mp.Ey], frequencies, where=mp.Volume(center=mp.Vector3(), size=cell), yee_grid=True)

sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ey, mp.Vector3(), decay_by))

# Get design region fields
a_E = np.zeros((nx, ny, 1, nf, 3), dtype=np.complex128)
for f in range(nf):
    for ic, c in enumerate([mp.Ex, mp.Ey, mp.Ez]):
        a_E[:, :, :, f, ic] = atleast_3d(sim.get_dft_array(adj_mon, c, f))

# Calculate gradients
grad = np.squeeze(np.real(np.sum(np.multiply(a_E, d_E), axis=4)))
g_adj = grad[start:end_x, start:end_y, :].reshape(-1, nf)

# Compare with finite difference method
print('Finite difference Simulation')
g_fd = []
for i in range(1):
    for j in range(design_region_y):
        ''' 'plus' optimization '''
        design_region[i, j] += 1e-4
        eps[start:end_x, start:end_y] = design_region

        sim.reset_meep()
        sim.change_sources(sources)

        # Add objective function monitor
        fp_mon = sim.add_flux(frequencies, mp.FluxRegion(center=f_center, size=f_size))

        sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.X, mp.Vector3(), decay_by))

        # Calculate objective function
        obj_p = np.array(mp.get_fluxes(fp_mon))

        ''' 'minus' optimization '''
        design_region[i, j] -= 2 * 1e-4
        eps[start:end_x, start:end_y] = design_region

        sim.reset_meep()

        # Add objective function monitor
        fm_mon = sim.add_flux(frequencies, mp.FluxRegion(center=f_center, size=f_size))

        sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.X, mp.Vector3(), decay_by))

        # Calculate objective function
        obj_m = np.array(mp.get_fluxes(fm_mon))

        ''' Calculate gradients '''
        g_fd.append((obj_p - obj_m) / (2 * 1e-4))

        ''' Reset epsilon '''
        design_region[i, j] += 1e-4
        eps[start:end_x, start:end_y] = design_region

g_fd = np.array(g_fd)

for i in range(nf):
    m, b = np.polyfit(g_fd[:, i], g_adj[:10, i], 1)
    min_g = np.min(g_fd[:, i])
    max_g = np.max(g_fd[:, i])
    plt.figure(i)
    plt.plot([min_g, max_g], [min_g, max_g], label='y=x comparison')
    plt.plot([min_g, max_g], [m*min_g+b, m*max_g+b], '--', label='Best fit')
    plt.plot(g_fd[:, i], g_adj[:10, i], 'o',label='Adjoint comparison')
    a = g_adj[:10, i] / g_fd[:, i]
    plt.xlabel('Finite Difference Gradient')
    plt.ylabel('Adjoint Gradient')
    plt.title('Frequency: {}'.format(frequencies[i]))
    plt.legend()
    plt.grid(True)

plt.show()