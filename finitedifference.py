import matplotlib.pyplot as plt
import numpy as np

"""
Create Your Own Finite Difference Wave Equation Simulation (With Python)
Philip Mocz (2023), @PMocz

Simulate the Wave Equation
with the finite difference method

"""

def get_mask(N):
	mask = np.zeros((N,N),dtype=bool)
	mask[0,:]  = True
	mask[-1,:] = True
	mask[:,0]  = True
	mask[:,-1] = True
	mask[int(N/4):int(N*9/32),:N-1] = True
	mask[1:N-1,int(N*5/16):int(N*3/8)] = False
	mask[1:N-1,int(N*5/8):int(N*11/16)] = False

	return mask

def get_initial_U(N, mask):
	U = np.zeros((N,N))
	U[mask] = 0
	return U

@profile
def compute_finite_difference(U, Uprev, mask, L, R, fac, aX, aY, xlin, tEnd, dt, plotRealTime):
	t = 0
	outputCount = 1
	while t < tEnd:
		# calculate laplacian 
		ULX = np.roll(U, L, axis=aX)
		URX = np.roll(U, R, axis=aX)
		ULY = np.roll(U, L, axis=aY)
		URY = np.roll(U, R, axis=aY)
		
		laplacian = ( ULX + ULY - 4*U + URX + URY )
		
		# update U
		Unew = 2*U - Uprev + fac * laplacian
		Uprev = 1.*U
		U = 1.*Unew
		
		# apply boundary conditions (Dirichlet/inflow)
		U[mask] = 0
		U[0,:] = np.sin(20*np.pi*t) * np.sin(np.pi*xlin)**2
		
		# update time
		t += dt
		
		# plot in real time
		# if (plotRealTime) or t >= tEnd:
		# 	plt.cla()
		# 	Uplot = 1.*U
		# 	Uplot[mask] = np.nan
		# 	plt.imshow(Uplot.T, cmap=cmap)
		# 	plt.clim(-3, 3)
		# 	ax = plt.gca()
		# 	ax.invert_yaxis()
		# 	ax.get_xaxis().set_visible(False)
		# 	ax.get_yaxis().set_visible(False)	
		# 	ax.set_aspect('equal')	
		# 	plt.pause(0.001)
		# 	outputCount += 1

def main():
	""" Finite Difference simulation """
	
	# Simulation parameters
	N              = 256   # resolution
	boxsize        = 3.    # box size
	c              = 1.    # wave speed
	t              = 0     # time
	tEnd           = 2.    # stop time
	plotRealTime   = False  # switch for plotting simulation in real time
	
	# Mesh
	dx = boxsize / N
	dt = (np.sqrt(2)/2) * dx / c
	aX = 0   # x-axis
	aY = 1   # y-axis
	R = -1   # right
	L = 1    # left
	fac = dt**2 * c**2 / dx**2

	xlin = np.linspace(0.5*dx, boxsize-0.5*dx, N)
	Y, X = np.meshgrid( xlin, xlin )
	
	# Generate Initial Conditions & mask
	mask = get_mask(N)
	U = get_initial_U(N, mask)
	Uprev = 1.*U

	# prep figure
	# fig = plt.figure(figsize=(6,6), dpi=80)
	# cmap = plt.cm.bwr
	# cmap.set_bad('gray')

	compute_finite_difference(U, Uprev, mask, L, R, fac, aX, aY, xlin, tEnd, dt, plotRealTime)
				
	# Save figure
	# plt.savefig('finitedifference.png',dpi=240)
	# plt.show()
	    
	return 0


if __name__== "__main__":
  main()

