import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import os
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')
clear_screen()

# Constants (atomic units: ħ = m = 1)
hbar = 1
m = 1
#Initial Values
x0 = float(input("Enter the initial center,e.g, x0 = -20.\n x0 = "))        #Initial center
sigma = float(input("Enter the width, e.g., sigma = 2.\n sigma = "))        #Pulse Width
k0 = float(input("Enter the value of momentum, e.g., k0 = 5. \n k0 = "))    #Initial momentum
print("Enter the range of x-axis, e.g., x_min = -50, x_max = 50")
x_min, x_max = float(input("x_min = ")), float(input("x_max = "))           #Range of the x-axis
n = (int)((np.abs(x_min)+np.abs(x_max))/10)
# Spatial grid
N = pow(2,n) #No. of discrete points(resolution) used to sample the range from x_min to x_max, e.g., N= 1024
print("Resolution =",N)
x = np.linspace(x_min, x_max, N) # creates an array of N evenly spaced point from x_min to x_max
dx = x[1] - x[0] #spacing between adjacent point in the array

# Time grid
dt = 0.005
steps = 500

# Potential (free particle => V=0)
V = np.zeros(N) #defines free space

# Initial wave packet
psi0 = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)  #Initial wavefunction
psi0 /= np.sqrt(np.sum(np.abs(psi0)**2) * dx)  # Normalize

# Hamiltonian matrix using Crank-Nicolson Method
a = 1j * hbar * dt / (4 * m * dx**2)
main_diag = (1 + 2 * a) * np.ones(N) + 1j * dt / (2 * hbar) * V   #main-diagonal elements
off_diag = -a * np.ones(N - 1)      #off-diagonal elements

from scipy.sparse import diags 
from scipy.sparse.linalg import spsolve
#Sparse matrix A and B
A = diags([off_diag, main_diag, off_diag], [-1, 0, 1], format='csr') #Builds tridiagonal matrix A, which multiplies with the wavefunction at (n+1) time level
B = diags([-off_diag, (1 - 2 * a) * np.ones(N) - 1j * dt / (2 * hbar) * V, -off_diag], [-1, 0, 1], format='csr') # Builds matrix B for wavefunction at n time level
psi = psi0.copy() #copy the intial wavefunction to another mutable and independent variable
#Set limits of y-axis of probability density, real part and imaginary part
Prob0 = np.abs(psi)**2
Real0 = np.real(psi)
Imag0 = np.imag(psi)
Prob_ylim = 1.1*Prob0.max()
Real_ylim = 1.1*max(abs(Real0.min()), abs(Real0.max()))
Imag_ylim = 1.1*max(abs(Imag0.min()), abs(Imag0.max()))

# Time evolution
fig, ax = plt.subplots(3,1, figsize=(10, 10), sharex=True) #creates a figure and an array of 3 axes objects
#Probability density
line, = ax[0].plot(x, np.abs(psi)**2, label = 'Probability density', color='black') #Plots initial Probability density and stoes a reference line object
ax[0].set_ylabel('|ψ(x, t)|²') #label for the probability density axis
ax[0].set_ylim(0,Prob_ylim) #sets the vertical range of the the Probabaility density plot
#Real part
line_real, = ax[1].plot(x,np.real(psi), label = 'Real part', color='red') #Plots initial Real part
ax[1].set_ylabel('Re(ψ(x, t))') #label for the real part axis
ax[1].set_ylim(-Real_ylim,Real_ylim) #sets the vertical range of the the Real part plot
#Imaginary part
ax[2].set_ylabel('Im(ψ(x, t))') #label for the imaginary part axis
line_imag, = ax[2].plot(x,np.imag(psi), label = 'Imaginary part',color='blue') #Plots initial Imaginary part
ax[2].set_ylim(-Imag_ylim, Imag_ylim) #sets the vertical range of the the Imaginary part plot
ax[2].set_xlabel('Position (x)') #label for the x-axis

#Legends for all
ax[0].legend(loc='upper right')
ax[1].legend(loc='upper right')
ax[2].legend(loc='upper right')

fig.suptitle('1D Time Dependent Free-Particle Schrödinger Wave-Packet Evolution', fontsize=14) #Title of the whole figure
fig.tight_layout(rect=[0,0,1,1]) #Sets the layout/spacing between subplots and axes to prevent overlapping
def animate(frame): #defines the animate function
    global psi # declares the wavefunction as global variable to modify it inside animate()
    psi = spsolve(A, B @ psi)   #compute the RHS of the Crank-Nicolson equation and update the wavefunction to the next time step and store it back in ψ
    line.set_ydata(np.abs(psi)**2) #updates the values of |ψ(x, t)|²
    line_real.set_ydata(np.real(psi)) #updates the values of Re(ψ(x, t))
    line_imag.set_ydata(np.imag(psi)) #updates the values of Im(ψ(x, t))
    return line, line_real,line_imag

ani = animation.FuncAnimation(fig, animate, frames=steps, interval=20, blit=True) #starts animation and calls the object animate()
plt.show() #Show the simulation