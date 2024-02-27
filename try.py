import matplotlib.pyplot as plt
import matplotlib
import numpy as np
# from datetime import datetime	
import os
import time

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

# def save_output_figure():
# 	current_directory = os.getcwd()
# 	relative_path = 'output/'
# 	output_directory = os.path.join(current_directory, relative_path)
# 	os.makedirs(output_directory, exist_ok=True)

# 	now = datetime.now()
# 	current_time = now.strftime("%H%M%S")
# 	filename = 'finitedifference' + current_time +'.png'

# 	save_path = os.path.join(output_directory, filename)
# 	plt.close('all')
# 	matplotlib.use('Agg') 
# 	plt.savefig(save_path, dpi=240)

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


def main():
	""" Finite Difference simulation """
	
	# === Simulation parameters ===
	N              = 900   # Resolution
	boxsize        = 1.    # Size of the box
	c              = 1.    # Wave Speed
	tEnd           = 2.    # Simulation time
	plotRealTime   = False  # Set to True for real-time vizualisation
	
	# === Generate mask & initial conditions
	mask = get_mask(N)
	U = get_initial_U(N, mask)
	Uprev = 1.*U

	# === Prepare output figure ===
	fig = plt.figure(figsize=(6,6), dpi=80)
	cmap = plt.cm.bwr
	cmap.set_bad('gray')

	# === Run main function ===
	start = time.time()
	simulate_finite_difference(U, Uprev, mask, boxsize, N, c, cmap, tEnd, plotRealTime)
	print(time.time() - start)
				
	# === Save output figure ===
	# save_output_figure()
	    
	return 0


if __name__== "__main__":
  main()

