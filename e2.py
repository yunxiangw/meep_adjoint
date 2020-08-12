import numpy as np
import fdtd
import matplotlib.pyplot as plt

np.random.seed(240)

# Define epsilon matrix
Air = 1.00
TiO2 = 2.51 ** 2

fcen = 1.0
df = 0.2

resolution = 40

dpml = 1.0
dpad = 0.5

npml = int(dpml * resolution)

nx = int(4.0 * resolution)
ny = int(4.0 * resolution)

design_region_x = int(1.0 * resolution)
design_region_y = int(1.0 * resolution)

eps = np.ones((nx, ny))
design_region = np.random.rand(design_region_x, design_region_y) * (TiO2 - Air) + Air

start = int((dpml + dpad) * resolution)
end = start + design_region_x

eps[start:end, start:end] = design_region

# Define current source matrix
jx = np.zeros((nx, ny), dtype=complex)
jy = np.zeros((nx, ny), dtype=complex)
jz = np.zeros((nx, ny), dtype=complex)
jx[int(nx / 2), ny - npml - 2] = 1 * resolution ** 2

# Define monitor point matrix
p = np.zeros((nx, ny))
p[npml+2:nx-npml-2, npml + 2] = 1

print('Forward simulation')
# Forward simulation
ex, ey, hz, ex_c, ey_c, hz_c, dt, time_src, T = fdtd.fdtd(nx=nx, ny=ny, npml=npml, res=resolution, fcen=fcen, jx=jx, jy=jy, jz=jz, eps=eps)

# Calculate objective function
# f = integral(E x H) = sum(weight*(EzHx - ExHz))
obj = np.real(np.sum(1/resolution * np.multiply(p, np.multiply(ex_c, np.conj(hz_c)))))
# f = E^2 = Ex^2 + Ey^2 + Ez^2
# obj = np.real(np.sum(np.multiply(p, np.multiply(np.conj(ex), ex)) + np.multiply(p, np.multiply(np.conj(ey), ey))))

# Calculate dtft of the forward source
y = np.array([time_src.swigobj.current(t, dt) for t in np.arange(0, T, dt)])  # time domain signal
# we need to compensate for the phase added by the time envelope at our freq of interest
src_center_dtft = np.matmul(np.exp(1j * 2 * np.pi * np.array([time_src.frequency])[:, np.newaxis] * np.arange(y.size) * dt), y) * dt / np.sqrt(2 * np.pi)
adj_src_phase = np.exp(1j * np.angle(src_center_dtft))

# Define adjoint source
# iomega in FDTD
amp = src_center_dtft * adj_src_phase
iomega = (1.0 - np.exp(-1j * (2 * np.pi * fcen) * dt)) * (1.0 / dt)

# source amp
# jx_adj = iomega * np.multiply(p, np.conj(ex)) / amp
jx_adj = iomega * 1/resolution * np.multiply(p, np.conj(hz_c)) / amp
# Transpose of the interpolation matrix (equally weighted to nearby yee grid point)
jx_adj[npml+2:nx-npml-2, npml + 3] = jx_adj[npml+2:nx-npml-2, npml + 2]
jx_adj /= 2

jy_adj = iomega * np.multiply(p, 0) / amp
# jy_adj = iomega * np.multiply(p, np.conj(ey)) / amp
jz_adj = iomega * np.multiply(p, 0) / amp

print('Adjoint simulation')
# Adjoint simulation
ex_adj, ey_adj, hz_adj, _, _, _, _, _, _ = fdtd.fdtd(nx=nx, ny=ny, npml=npml, res=resolution, fcen=fcen, jx=jx_adj, jy=jy_adj, jz=jz_adj, eps=eps)

# Calculate gradients
grad = 2. * np.real((np.multiply(ex_adj, ex)+np.multiply(ey_adj, ey)))
g_adj = grad[start:end, start:end].flatten()

# Compare results with finite difference method
print('Finite difference Simulation')
g_fd = []
for i in range(1):
    for j in range(design_region_y):
        # Update epsilon
        design_region[i, j] += 1e-4
        eps[start:end, start:end] = design_region

        # Forward simulation
        ex_p, ey_p, hz_p, exc_p, eyc_p, hzc_p, _, _, _ = fdtd.fdtd(nx=nx, ny=ny, npml=npml, res=resolution, fcen=fcen, jx=jx, jy=jy, jz=jz, eps=eps)

        # Calculate objective function
        obj_p = np.real(np.sum(1 / resolution * np.multiply(p, np.multiply(exc_p, np.conj(hzc_p)))))
        # obj_p = np.real(np.sum(np.multiply(p, np.multiply(np.conj(ex_p), ex_p)) + np.multiply(p, np.multiply(np.conj(ey_p), ey_p))))

        # Update epsilon
        design_region[i, j] -= 2 * 1e-4
        eps[start:end, start:end] = design_region

        # Forward simulation
        ex_m, ey_m, hz_m, exc_m, eyc_m, hzc_m, _, _, _ = fdtd.fdtd(nx=nx, ny=ny, npml=npml, res=resolution, fcen=fcen, jx=jx, jy=jy, jz=jz, eps=eps)

        # Calculate objective function
        obj_m = np.real(np.sum(1 / resolution * np.multiply(p, np.multiply(np.conj(exc_m), hzc_m))))
        # obj_m = np.real(np.sum(np.multiply(p, np.multiply(np.conj(ex_m), ex_m)) + np.multiply(p, np.multiply(np.conj(ey_m), ey_m))))

        # Calculate gradients
        g_fd.append((obj_m - obj_p) / (2 * 1e-4))

        # Reset epsilon
        design_region[i, j] += 1e-4
        eps[start:end, start:end] = design_region

(m, b) = np.polyfit(g_fd, g_adj[:40], 1)
min_g = np.min(g_fd)
max_g = np.max(g_fd)

plt.figure(1)
plt.plot([min_g, max_g], [min_g, max_g],label='y=x comparison')
plt.plot([min_g, max_g], [m*min_g+b, m*max_g+b], '--', label='Best fit')
plt.plot(g_fd, g_adj[:40], 'o',label='Adjoint comparison')
plt.xlabel('Finite Difference Gradient')
plt.ylabel('Adjoint Gradient')
plt.legend()
plt.grid(True)

plt.figure(2)
plt.scatter(range(len(g_fd)), (g_adj[:40] - g_fd) / g_fd * 100)
plt.xlabel('Points')
plt.ylabel('Relative Error (%)')
plt.grid(True)
plt.show()
