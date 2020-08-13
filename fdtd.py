import meep as mp
import numpy as np
import matplotlib.pyplot as plt

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

def fdtd(nx, ny, npml, res, fcen, jx, jy, jz, mz, eps):

    # Define epsilon and source function
    def eps_func(p):
        return dat2pos(p, eps, nx, ny, res, 1)

    def jx_func(p):
        value = dat2pos(p, jx, nx, ny, res, 0)
        # if value != 0:
        #     print('px: ({}, {}), value: {}'.format(p.x, p.y, value))
        return value

    def jy_func(p):
        value = dat2pos(p, jy, nx, ny, res, 0)
        # if value != 0:
        #     print('py: ({}, {}), value: {}'.format(p.x, p.y, value))
        return value

    def jz_func(p):
        return dat2pos(p, jz, nx, ny, res, 0)

    def mz_func(p):
        return dat2pos(p, mz, nx, ny, res, 0)

    # Define simulation region
    Lx = nx / res
    Ly = ny / res
    dpml = npml / res

    cell = mp.Vector3(Lx, Ly)

    sources = [mp.Source(mp.GaussianSource(fcen, fwidth=0.2),
                         component=mp.Ex,
                         center=mp.Vector3(0, 0, 0),
                         size=cell,
                         amp_func=jx_func),
               mp.Source(mp.GaussianSource(fcen, fwidth=0.2),
                         component=mp.Ey,
                         center=mp.Vector3(0, 0, 0),
                         size=cell,
                         amp_func=jy_func),
               mp.Source(mp.GaussianSource(fcen, fwidth=0.2),
                         component=mp.Ez,
                         center=mp.Vector3(0, 0, 0),
                         size=cell,
                         amp_func=jz_func),
               mp.Source(mp.GaussianSource(fcen, fwidth=0.2),
                         component=mp.Hz,
                         center=mp.Vector3(0, 0, 0),
                         size=cell,
                         amp_func=mz_func)]

    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=[mp.PML(dpml)],
                        epsilon_func=eps_func,
                        eps_averaging=False,
                        resolution=res,
                        sources=sources,
                        Courant=0.5,
                        force_complex_fields=True)

    # Add monitor
    dft_vol = mp.Volume(center=mp.Vector3(), size=cell)
    dft_yee = sim.add_dft_fields([mp.Ex, mp.Ey, mp.Hz], [fcen], where=dft_vol, yee_grid=True)
    dft_center = sim.add_dft_fields([mp.Ex, mp.Ey, mp.Hz], [fcen], where=dft_vol, yee_grid=False)

    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.X, mp.Vector3(), 1e-9))

    ex = sim.get_dft_array(dft_yee, mp.Ex, 0)
    ey = sim.get_dft_array(dft_yee, mp.Ey, 0)
    hz = sim.get_dft_array(dft_yee, mp.Hz, 0)

    ex_c = sim.get_dft_array(dft_center, mp.Ex, 0)
    ey_c = sim.get_dft_array(dft_center, mp.Ey, 0)
    hz_c = sim.get_dft_array(dft_center, mp.Hz, 0)

    dt = sim.fields.dt
    time_src = sim.sources[0].src
    T = sim.meep_time()

    return ex, ey, hz, ex_c, ey_c, hz_c, dt, time_src, T