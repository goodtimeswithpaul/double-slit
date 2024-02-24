import numpy as np
cimport numpy as np
cimport cython
import matplotlib.pyplot as plt

def simulate_finite_difference(U, Uprev, mask, boxsize, N, c, cmap, tEnd, plotRealTime):
    t = 0
    # === Initialize Mesh ===
    dx = boxsize / N
    dt = (np.sqrt(2)/2) * dx / c
    fac = dt**2 * c**2 / dx**2
    aX = 0   # x-axis
    aY = 1   # y-axis
    R = -1   # right
    L = 1    # left

    # === Initialize grid ===
    xlin = np.linspace(0.5 * dx, boxsize - 0.5*dx, N)

    # === Main simulation loop ===
    while t < tEnd:
        # === Compute Laplacian ===
        ULX = np.roll(U, L, axis=aX)
        URX = np.roll(U, R, axis=aX)
        ULY = np.roll(U, L, axis=aY)
        URY = np.roll(U, R, axis=aY)
        
        laplacian = ( ULX + ULY - 4*U + URX + URY )
        
        # === Update U ===
        Unew = 2*U - Uprev + fac * laplacian
        Uprev = 1.*U
        U = 1.*Unew
        
        # === Apply boudary conditions (Dirichlet/inflow) ===
        U[mask] = 0
        U[0,:] = np.sin(20*np.pi*t) * np.sin(np.pi*xlin)**2
        
        # === Update time ===
        t += dt
        
        if (plotRealTime) or t >= tEnd:
            plot_U(U, mask, cmap)

def plot_U(U, mask, cmap):
    plt.cla()
    Uplot = 1.*U
    Uplot[mask] = np.nan
    plt.imshow(Uplot.T, cmap=cmap)
    plt.clim(-3, 3)
    ax = plt.gca()
    ax.invert_yaxis()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)	
    ax.set_aspect('equal')	
    plt.pause(0.001)
