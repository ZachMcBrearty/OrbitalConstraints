# PDE:
# C dT/dt - d/dx (D(1-x^2)*dT/dx) + I = S(1-A)
# sub x = sin(λ), d/dx = 1/cos(λ) d/dλ, sin^2+cos^2=1
# assume D constant, expand derivative
# C dT/dt - D(d^2 T / dλ^2 - tan(λ) dT/dλ) - S(1-A) + I = 0

# discretise time derivative by forward difference:
# dT/dt = (T(x, t_n+1) - T(x, t_n)) / Δt
# solve for T(x, t_n+1):

# T(x, t_n+1) = T(x, t_n) + Δt/C [D(d^2 T / dλ^2 - tan(λ) dT/dλ) + S(1-A) - I]

# discretise space derivatives using functions below
# central for most points
# forward and backward for boundaries

import numpy as np

from InsolationFunction import S

## derivatives ##
def forwarddifference(x, i, dx):
    return (x[i+1] - x[i  ]) / dx
def centraldifference(x, i, dx):
    # (x[i+1/2] - x[i-1/2]) / dx
    # let x[i+(-)1/2] = (x[i+(-)1] + x[i]) / 2
    # => (x[i+1] - x[i-1]) / (2*dx)
    return (x[i+1] - x[i-1]) / (2*dx)
def backwarddifference(x, i, dx):
    return (x[i  ] - x[i-1]) / dx

def forward2ndorder(x,i,dx):
    return (x[i+2] - 2*x[i+1] + x[i  ]) / dx / dx
def central2ndorder(x,i,dx):
    return (x[i+2] - 2*x[i  ] + x[i-2]) / (2*dx) / (2*dx)
def backward2ndorder(x,i,dx):
    return (x[i  ] - 2*x[i-1] + x[i-2]) / dx / dx

##  ##
spacedim = 100  # number of points in space
dlam = np.pi / (spacedim - 1)  # spacial separation from 2 units from -pi/2 to pi/2
lat = np.linspace(-np.pi/2+0.1, np.pi/2-0.1, spacedim)

# timedim may need to be split in to chunks which are then recorded
# e.g. do a year of evolution then write to file and overwrite the array
# numchunks = 50
# timedim_chunk = timedim / numchunks
timedim = 1000  # number of iterations
time = 1  # length of time the iterations should be over
dt = time / timedim  # timestep

Temp = np.ones((spacedim, timedim + 1))  # timedim_chunk))

# Temp[:, 0] = np.linspace(-1, 1, spacedim) * 100 + 250
# Temp[:, 0] = np.abs(np.sin(np.linspace(0, 1, spacedim)*np.pi))*50+300
Temp[:, 0] = np.exp(-5 * np.linspace(-1, 1, spacedim) ** 2) * 50 + 300

Capacity = np.ones_like(Temp)  # effective heat capacity
Ir_emission = np.ones_like(Temp) * 0  # IR emission function (Energy sink)
Source = np.zeros_like(Temp)  # Diurnally averaged insolation function (Energy Source)
Albedo = np.zeros_like(Temp)  # Albedo

Diffusion = np.ones_like(Temp) * 0.2  # diffusion coefficient (Lat)
# Diffusion[:, :] = np.array([np.linspace(0.1, 0.4, spacedim)]*(timedim+1)).T

for n in range(timedim):
    for m in range(spacedim):
        if m == 0 or m == 1:
            # forward difference for zero edge
            secondT = forward2ndorder(Temp[:, n], m, dlam)
            firstT = forwarddifference(Temp[:, n], m, dlam)
        elif m == spacedim - 2 or m == spacedim - 1:
            # backwards difference for end edge
            secondT = backward2ndorder(Temp[:, n], m, dlam)
            firstT = backwarddifference(Temp[:, n], m, dlam)
        else:
            # central difference for most cases
            secondT = central2ndorder(Temp[:, n], m, dlam)
            firstT = centraldifference(Temp[:, n], m, dlam)
        ### NOTE: tan(-pi/2) = inf (!)
        ### NOTE: tan(pi/2) = inf (!)
        # diff = D(d^2 T / dλ^2 - tan(λ) dT/dλ)
        diff_elem = Diffusion[m, n] * (secondT - np.tan(lat[m]) * firstT)
        # T(x, t_n+1) = T(x, t_n) + Δt/C [diff + S(1-A) - I]
        Temp[m, n + 1] = Temp[m, n] + dt / Capacity[m, n] * (
            diff_elem - Ir_emission[m, n] + Source[m, n] * (1 - Albedo[m, n])
        )

import matplotlib.pyplot as plt

for n in range(0, 21, 5):
    plt.plot(lat, Temp[:, n], label=f"t={n}")

# plt.axvline(dx*20-1)
plt.ylabel("Temperature")
plt.xlabel(r"$x = $sin$(\lambda)$")
plt.legend()
plt.show()
