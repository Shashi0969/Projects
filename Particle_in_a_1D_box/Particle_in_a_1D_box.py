import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.sparse import diags 
from scipy.sparse.linalg import spsolve
from mpl_toolkits.mplot3d import Axes3D

import os
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')
clear_screen()
# Start function to initiate wavefunction
def Start():
    L = 1e-8  # Length of the box in meters
    N = 1024  # Number of spatial slices

    hbar = 1.0545718e-34  # Reduced Planck's constant, J*s
    m = 9.10938356e-31  # Mass of electron, kg
    h = 1e-18  # Time-step in seconds

    # Initial wave function
    x = np.linspace(0, L, N)    # Spatial grid from 0 to L with N points
    x0 = L / 2  # Initial center of the wave packet
    sigma = 1e-10
    k0 = 5e10
    k = x[1]-x[0] #spacing between adjacent points in the array
    dx = k
    
    # Time grid
    dt = h
    steps = 2000

    # Initial wave packet
    psi0 = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)  # Initial wavefunction
    psi0 /= np.sqrt(np.sum(np.abs(psi0)**2) * dx)  # Normalize

    # Hamiltonian matrix using Crank-Nicolson Method
    a = 1j * hbar * dt / (4 * m * dx**2)
    main_diag = (1 + 2 * a) * np.ones(N)  # Main-diagonal elements
    off_diag = -a * np.ones(N - 1)        # Off-diagonal elements

    #Sparse matrix A and B
    A = diags([off_diag, main_diag, off_diag], [-1, 0, 1], format='csr') #Builds tridiagonal matrix A, which multiplies with the wavefunction at (n+1) time level
    B = diags([-off_diag, (1 - 2 * a) * np.ones(N) , -off_diag], [-1, 0, 1], format='csr') # Builds matrix B for wavefunction at n time level
    return x, psi0, A, B, steps
# Function for 2D plots at 2000th step of the evolution
def Plot2D():
    x, psi0, A, B, steps = Start()
    psi = psi0.copy()
    for step in range(steps):
        psi = spsolve(A, B @ psi)
        psi[0] = psi[-1] = 0

    #Set limits of y-axis of probability density, real part and imaginary part
    fig, ax = plt.subplots(3, 1, figsize=(14, 18), constrained_layout=True)
    line, = ax[0].plot(x, np.abs(psi)**2, label='Probability Density')
    line_real, = ax[1].plot(x, np.real(psi), label='Real Part')
    line_imag, = ax[2].plot(x, np.imag(psi), label='Imaginary Part')
    for a in ax:
        a.set_xlabel('Position x (m)')
        a.legend()
    ax[0].set_title('Probability Density at 2000th Step')
    ax[1].set_title('Real Part of the Wavefunction at 2000th Step')
    ax[2].set_title('Imaginary Part of the Wavefunction at 2000th Step')
    plt.show()
# Function for 3D plots at 2000th step of the evolution
def Plot3D():
    x, psi0, A, B, steps = Start()
    psi = psi0.copy()
    for step in range(steps):       
        psi = spsolve(A, B @ psi)
        psi[0] = psi[-1] = 0
    # 3D plot of the wavefunction at the final step
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, np.real(psi), np.imag(psi))
    ax.set_xlabel('Position x (m)')
    ax.set_ylabel('Real Part of Psi')
    ax.set_zlabel('Imaginary Part of Psi')
    ax.set_title('3D Visualization of the Wavefunction at Final Step')
    plt.show()
# Function for 2D plots at 500th, 1000th, 1500th and 2000th steps
def Plot2D_1():
    x, psi0, A, B, steps = Start()
    selected_steps = [500, 1000, 1500, 2000]
    evolution = {step: {} for step in selected_steps}
    psi = psi0.copy()
    for step in range(1, steps+1):
        psi = spsolve(A, B @ psi)
        psi[0] = psi[-1] = 0
        # Dictionary named evolution
        if step in selected_steps:
            evolution[step]['prob'] = np.abs(psi)**2
            evolution[step]['real'] = np.real(psi)
            evolution[step]['imag'] = np.imag(psi)
    # Visualization of the real, imaginary parts, and probability density of the wavefunction at selected steps
    fig, ax = plt.subplots(3, 1, figsize=(12, 18), constrained_layout=True)
    for step in selected_steps:
        ax[0].plot(x, evolution[step]['prob'], label=f'Probability Density at Time Step {step}')
        ax[1].plot(x, evolution[step]['real'], label=f'Real Part at Time Step {step}')
        ax[2].plot(x, evolution[step]['imag'], label=f'Imaginary Part at Time Step {step}')

    ax[0].set_title('Probability Density Over Time')
    ax[0].set_xlabel('Position x (m)')
    ax[0].set_ylabel('Probability Density')
    ax[0].legend(loc='upper left', fontsize=8)

    ax[1].set_title('Real Part of the Wavefunction Over Time')
    ax[1].set_xlabel('Position x (m)')
    ax[1].set_ylabel('Real Part of Psi')
    ax[1].legend(loc='upper left', fontsize=8)

    ax[2].set_title('Imaginary Part of the Wavefunction Over Time')
    ax[2].set_xlabel('Position x (m)')
    ax[2].set_ylabel('Imaginary Part of Psi')
    ax[2].legend(loc='upper left', fontsize=8)
    plt.show()
# Function for 3D plots at 500th, 1000th, 1500th and 2000th steps
def Plot3D_1():
    x, psi0, A, B, steps = Start()
    L = 1e-8
    selected_steps = [500, 1000, 1500, 2000]
    evolution = {step: {} for step in selected_steps}
    psi = psi0.copy()
    for step in range(1, steps+1):
        psi = spsolve(A, B @ psi)
        psi[0] = psi[-1] = 0
        # Dictionary named evolution
        if step in selected_steps:
            evolution[step]['prob'] = np.abs(psi)**2
            evolution[step]['real'] = np.real(psi)
            evolution[step]['imag'] = np.imag(psi)
    # 3D Visualization of the wavefunction at specified time steps
    fig = plt.figure(figsize=(14, 18), constrained_layout = True)
    for i, step in enumerate(selected_steps, 1):    #loops over the list and gives an index starting at 1
        ax = fig.add_subplot(2, 2, i, projection='3d')      # creates 3D figure with 2x2 grid
        ax.plot3D(x, evolution[step]['real'], evolution[step]['imag'], label=f'Time step {step}')
        ax.set_xlabel('Position x (m)')
        ax.set_ylabel('Real Part of Psi')
        ax.set_zlabel('Imaginary Part of Psi')
        ax.set_title(f'3D Visualization of the Wavefunction at Time Step {step}')
        ax.legend()
    plt.show()
# Function for 2D Animation of Time evolution of Probability density and Imaginary and Real part of wavefunction
def Animate2D():
    x, psi0, A, B, steps = Start()
    psi = psi0.copy() #copy the initial wavefunction to another mutable and independent variable
    Prob0 = np.abs(psi)**2
    Real0 = np.real(psi)
    Imag0 = np.imag(psi)
    Prob_ylim = 1.1*Prob0.max()
    Real_ylim = 1.1*max(abs(Real0.min()), abs(Real0.max()))
    Imag_ylim = 1.1*max(abs(Imag0.min()), abs(Imag0.max()))

    # Time evolution
    fig, ax = plt.subplots(3,1, figsize=(12, 17), sharex=True, constrained_layout = True) #creates a figure and an array of 3 axes objects
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
    ax[2].set_xlabel('Position x (m)') #label for the x-axis

    #Legends for all
    ax[0].legend(loc='upper right')
    ax[1].legend(loc='upper right')
    ax[2].legend(loc='upper right')

    fig.suptitle('1D Time Evolution of the wavefunction', fontsize=14) #Title of the whole figure
    def animate_2D(frame): #defines the animate function
        nonlocal psi # declares the wavefunction as global variable to modify it inside animate()
        psi = spsolve(A, B @ psi)   #compute the RHS of the Crank-Nicolson equation and update the wavefunction to the next time step and store it back in ψ
        psi[0] = psi[-1] = 0
        line.set_ydata(np.abs(psi)**2) #updates the values of |ψ(x, t)|²
        line_real.set_ydata(np.real(psi)) #updates the values of Re(ψ(x, t))
        line_imag.set_ydata(np.imag(psi)) #updates the values of Im(ψ(x, t))
        return line, line_real,line_imag
    
    # Call the animator
    ani_2D = animation.FuncAnimation(fig, animate_2D, frames=steps, interval=20, blit=True) #starts animation and calls the object animate_1()
    choice = int(input("Save Video? Enter '1' for Yes and '0' for No (show the animation) : "))
    if choice == 1:
        clear_screen()
        print("Saving the animation as a video file...")
        ani_2D.save('1D_TDSE_Particle_in_a_box (Animate2D()).mp4', writer='ffmpeg', fps=30) #saves the animation as a mp4 file
    else:
        print("Not saving the animation, just showing it.")
        plt.show() # Show the simulation
# Function for 3D Animation of Time evolution of Probability density and Imaginary and Real part of wavefunction
def Animate3D():
    x, psi0, A, B, steps = Start()
    L = 1e-8
    psi = psi0.copy() #copy the intial wavefunction to another mutable and independent variable
    Real0 = np.real(psi)
    Imag0 = np.imag(psi)
    Real_lim = 0.4*max(abs(Real0.min()), abs(Real0.max()))
    Imag_lim = 0.4*max(abs(Imag0.min()), abs(Imag0.max()))
    
    # Time evolution
    fig = plt.figure(figsize=(12,8), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d') #creates a figure and an array of 3 axes objects
    ax.set_xlabel('Position x (m)') #label for the x-axis
    ax.set_xlim(0,L) #sets the vertical range of the the Probabaility density plot
    line, = ax.plot(x, np.real(psi), np.imag(psi), lw=1.5)
    
    # Real part
    ax.set_ylabel('Re(ψ(x, t))') #label for the real part axis
    ax.set_ylim(-Real_lim,Real_lim) #sets the vertical range of the the Real part plot
    
    # Imaginary part
    ax.set_zlabel('Im(ψ(x, t))') #label for the imaginary part axis
    ax.set_zlim(-Imag_lim, Imag_lim) #sets the vertical range of the the Imaginary part plot
    ax.set_title('3D Time Evolution of the Wavefunction') #Title of the figure
    
    def animate_3D(frame): #defines the animate function
        nonlocal psi # declares the wavefunction as global variable to modify it inside animate()
        psi = spsolve(A, B @ psi)   #compute the RHS of the Crank-Nicolson equation and update the wavefunction to the next time step and store it back in ψ
        psi[0] = psi[-1] = 0
        line.set_data_3d(x, np.real(psi), np.imag(psi)) #updates the values of Im(ψ(x, t))
        return line
    
    # Call the animator
    ani_3D = animation.FuncAnimation(fig, animate_3D, frames=steps, interval=20, blit=False) #starts animation and calls the object animate()
    choice = int(input("Save Video? Enter '1' for Yes and '0' for No (show the animation) : "))
    if choice == 1:
        clear_screen()
        print("Saving the animation as a video file...")
        ani_3D.save('1D_TDSE_Particle_in_a_box (Animate3D()).mp4', writer='ffmpeg', fps=30) #saves the animation as a mp4 file
    else:
        print("Not saving the animation, just showing it.")
        plt.show() # Show the simulation
# Call the function to display the required simulation
while True:
    clear_screen()
    p = int(input("Enter '1' to start and '0' to terminate : "))
    if p==1:
        print("What should be done:\n1. 2D plot at 2000th step\n2. 3D plot at 2000th step\n3. 2D plots at 500th, 1000th, 1500th and 2000th steps\n4. 2D plots at 500th, 1000th, 1500th and 2000th steps\n5. 2D Animation\n6. 3D Animation")
        n = int(input("Enter your choice : "))
        if n==1:
            Plot2D()
        elif n==2:
            Plot3D()
        elif n==3:
            Plot2D_1()
        elif n==4:
            Plot3D_1()
        elif n==5:
            Animate2D()
        elif n==6:
            Animate3D()
        else:
            print("Wrong Choice")
    elif p==0:
        print("Terminated")
        import time
        time.sleep(2)
        import sys
        sys.exit()
    else:
        print("Wrong Choice")