import meep as mp
import numpy as np
import matplotlib.pyplot as plt

def dat2pos(r, data, nx, ny, du, val):
    decplace = 5

    x0 = np.around(nx * du / 2., decplace)
    y0 = np.around(ny * du / 2., decplace)
    rx = np.around(r.x, decplace)
    ry = np.around(r.y, decplace)
    jx = np.around((rx + x0) / du, decplace)
    jy = np.around((ry + y0) / du, decplace)
    ix = int(np.floor(jx))
    iy = int(np.floor(jy))

    if 0 <= ix < nx and 0 <= iy < ny:
        return data[ix, iy]
    else:
        return val

def fdtd(nx, ny, npml, res, fcen, df, p0, jx, jy, jz, eps):

    Lx = nx / res
    Ly = ny / res
    dpml = npml / npml

    cell = mp.Vector3(Lx, Ly)

    def eps_func(p):
        return dat2pos(p, eps, nx, ny, 1/res, 1)

    def jx_func(p):
        return dat2pos(p, jx, nx, ny, 1/res, 0)

    def jy_func(p):
        return dat2pos(p, jy, nx, ny, 1 / res, 0)

    def jz_func(p):
        return dat2pos(p, jz, nx, ny, 1 / res, 0)

    sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df),
                         component=mp.Ex,
                         center=mp.Vector3(0, 0, 0),
                         size=cell,
                         amp_func=jx_func),
               mp.Source(mp.GaussianSource(fcen, fwidth=df),
                         component=mp.Ey,
                         center=mp.Vector3(0, 0, 0),
                         size=cell,
                         amp_func=jy_func),
               mp.Source(mp.GaussianSource(fcen, fwidth=df),
                         component=mp.Ez,
                         center=mp.Vector3(0, 0, 0),
                         size=cell,
                         amp_func=jz_func)]

    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=[mp.PML(dpml)],
                        epsilon_func=eps_func,
                        eps_averaging=False,
                        resolution=res,
                        sources=sources,
                        Courant=0.5,
                        force_complex_fields=True)

    dft_vol = mp.Volume(center=mp.Vector3(), size=cell)
    dft_obj = sim.add_dft_fields([mp.Ex, mp.Ey, mp.Ez], [fcen], where=dft_vol, yee_grid=True)

    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.X, mp.Vector3(), 1e-9))

    ex = sim.get_dft_array(dft_obj, mp.Ex, 0)
    ey = sim.get_dft_array(dft_obj, mp.Ey, 0)
    ez = sim.get_dft_array(dft_obj, mp.Ez, 0)
    dt = sim.fields.dt

    time_src = sim.sources[0].src
    T = sim.meep_time()

    return ex, ey, ez, dt, time_src, T