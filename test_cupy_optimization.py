import numpy as np
import pytest
import cupy as cp
from cupy import (roll)
from datetime import datetime
import os
"""
Create Your Own Finite Difference Wave Equation Simulation (With Python)
Philip Mocz (2023), @PMocz
Simulate the Wave Equation
with the finite difference method
"""


def run_original(input_N, input_tEnd):
	""" Finite Difference simulation """

	# Simulation parameters
	N              = input_N   # resolution
	boxsize        = 1.    # box size
	c              = 1.    # wave speed
	t              = 0     # time
	tEnd           = input_tEnd    # stop time
	plotRealTime   = True  # switch for plotting simulation in real time

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
	U = np.zeros((N,N))
	mask = np.zeros((N,N),dtype=bool)
	mask[0,:]  = True
	mask[-1,:] = True
	mask[:,0]  = True
	mask[:,-1] = True
	mask[int(N/4):int(N*9/32),:N-1]     = True
	mask[1:N-1,int(N*5/16):int(N*3/8)]  = False
	mask[1:N-1,int(N*5/8):int(N*11/16)] = False
	U[mask] = 0
	Uprev = 1.*U

	# prep figure
	outputCount = 1

	# Simulation Main Loop
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

	return U

def compare_print(original, optimized):
  if(not np.testing.assert_allclose(original, optimized, rtol=1e-5, atol=1e-5)):
    print("Does not match!!!")
    print(original)
    print(optimized)


def compare(original, optimized):
	np.testing.assert_allclose(original, optimized, rtol=1e-8, atol=1e-8)


def test_optimization_correctness():
  smallest = run_original(1, 1)
  smallest_cp = run_optimised(1, 1)
  compare(smallest, smallest_cp)

  ten = run_original(10, 2)
  forty_two = run_original(42, 2)
  hundred_and_one = run_original(101, 2)
  longer = run_original(37, 6)

  ten_cp = run_optimised(10, 2)
  forty_two_cp = run_optimised(42, 2)
  hundred_and_one_cp = run_optimised(101, 2)
  longer_cp = run_optimised(37, 6)

  compare(ten, ten_cp)
  compare(forty_two, forty_two_cp)
  compare(hundred_and_one, hundred_and_one_cp)
  compare(longer, longer_cp)

# -=-=-=- Optimized version -=-=-=-

def get_mask(N):
	mask = np.zeros((N,N),dtype=bool)
	mask[0,:]  = True
	mask[-1,:] = True
	mask[:,0]  = True
	mask[:,-1] = True
	mask[int(N/4):int(N*9/32),:N-1] = True
	mask[1:N-1,int(N*5/16):int(N*3/8)] = False
	mask[1:N-1,int(N*5/8):int(N*11/16)] = False
	#cp.cuda.Stream.null.synchronize()
	return mask

def get_initial_U(N, mask):
	U = cp.zeros((N,N))
	U[mask] = 0
	#cp.cuda.Stream.null.synchronize()
	return U

def simulate_finite_difference(U, Uprev, mask, boxsize, N, c, tEnd, plotRealTime):
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
	xlin = cp.linspace(0.5 * dx, boxsize - 0.5*dx, N)

	# === Main simulation loop ===
	while t < tEnd:
		# === Compute Laplacian ===
		ULX = cp.roll(U, L, axis=aX)
		URX = cp.roll(U, R, axis=aX)
		ULY = cp.roll(U, L, axis=aY)
		URY = cp.roll(U, R, axis=aY)
		#cp.cuda.Stream.null.synchronize()

		laplacian = ( ULX + ULY - 4*U + URX + URY )
		#cp.cuda.Stream.null.synchronize()


		# === Update U ===
		Unew = 2*U - Uprev + fac * laplacian
		#cp.cuda.Stream.null.synchronize()
		Uprev = 1.*U
		#cp.cuda.Stream.null.synchronize()
		U = 1.*Unew
		#cp.cuda.Stream.null.synchronize()

		# === Apply boudary conditions (Dirichlet/inflow) ===
		U[mask] = 0
		U[0,:] = cp.sin(20*np.pi*t) * cp.sin(np.pi*xlin)**2
		#cp.cuda.Stream.null.synchronize()

		# === Update time ===
		t += dt
	return U


def run_optimised(input_N, input_tEnd):
	""" Finite Difference simulation """

	# === Simulation parameters ===
	N              = input_N   # Resolution
	boxsize        = 1.    # Size of the box
	c              = 1.    # Wave Speed
	tEnd           = input_tEnd    # Simulation time
	plotRealTime   = True  # Set to True for real-time vizualisation

	# === Generate mask & initial conditions
	mask = get_mask(N)
	U = get_initial_U(N, mask)
	Uprev = 1.*U

	# === Run main function ===
	U_ret = cp.asnumpy(simulate_finite_difference(U, Uprev, mask, boxsize, N, c, tEnd, plotRealTime))

	return U_ret

if __name__== "__main__":
  test_optimization_correctness()