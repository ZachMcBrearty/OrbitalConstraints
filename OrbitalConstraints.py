# PDE:
# C dT/dt - d/dx (D(1-x^2)*dT/dx) + I = S(1-A)
# discretise time derivative by forward difference:
# dT/dt = (T(x, t_n+1) - T(x, t_n)) / Δt
# solve for T(x, t_n+1):

# T(x, t_n+1) = T(x, t_n) + Δt/C [d/dx (D(1-x^2)*dT/dx) - I + S(1-A)]

# d/dx (D(1-x^2)*dT/dx)
# expand outer derivative
# d(D(1-x^2))/dx * dT/dx + D(1-x^2)* d^2(T)/ dx^2
# diff = dD/dx (1-x^2) * dT/dx + D*-2x*dT/dx + D(1-x^2)* d^2(T)/ dx^2

# discretise space derivatives using derivative functions below
# Forward and backward for edge cases
# central otherwise

# All together:
# T(x_m, t_n+1) = T(x_m, t_n) + Δt / C(x_m, t_n)
# * (diff - I + S(1-A) )

import numpy as np
import numpy.typing as npt

from InsolationFunction import S
from HeatCapacity import C

floatarr = npt.NDArray[np.float64]


## derivatives ##
def forwarddifference(x: list[float] | floatarr, i: int, dx: float) -> float:
    return (x[i + 1] - x[i]) / dx


def centraldifference(x: list[float] | floatarr, i: int, dx: float) -> float:
    # (x[i+1/2] - x[i-1/2]) / dx
    # let x[i+(-)1/2] = (x[i+(-)1] + x[i]) / 2
    # => (x[i+1] - x[i-1]) / (2*dx)
    return (x[i + 1] - x[i - 1]) / (2 * dx)


def backwarddifference(x: list[float] | floatarr, i: int, dx: float) -> float:
    return (x[i] - x[i - 1]) / dx


def forward2ndorder(x: list[float] | floatarr, i: int, dx: float) -> float:
    return (x[i + 2] - 2 * x[i + 1] + x[i]) / dx / dx


def central2ndorder(x: list[float] | floatarr, i: int, dx: float) -> float:
    return (x[i + 2] - 2 * x[i] + x[i - 2]) / (2 * dx) / (2 * dx)


def backward2ndorder(x: list[float] | floatarr, i: int, dx: float) -> float:
    return (x[i] - 2 * x[i - 1] + x[i - 2]) / dx / dx


##  ##
spacedim = 200  # number of points in space
dx = 2 / (spacedim - 1)  # spacial separation from 2 units from -1 to 1
xs = np.linspace(-1, 1, spacedim)

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

Diffusion = np.ones_like(Temp) * 0.3  # diffusion coefficient (Lat)
# Diffusion[:, :] = np.array([np.linspace(0.1, 0.4, spacedim)]*(timedim+1)).T

for n in range(timedim):
    for m in range(spacedim):
        if m == 0 or m == 1:
            # forward difference for zero edge
            secondT = forward2ndorder(Temp[:, n], m, dx)
            firstT = forwarddifference(Temp[:, n], m, dx)
            firstD = forwarddifference(Diffusion[:, n], m, dx)
        elif m == spacedim - 2 or m == spacedim - 1:
            # backwards difference for end edge
            secondT = backward2ndorder(Temp[:, n], m, dx)
            firstT = backwarddifference(Temp[:, n], m, dx)
            firstD = backwarddifference(Diffusion[:, n], m, dx)
        else:
            # central difference for most cases
            secondT = central2ndorder(Temp[:, n], m, dx)
            firstT = centraldifference(Temp[:, n], m, dx)
            firstD = centraldifference(Diffusion[:, n], m, dx)
        # diff = (dD/dx (1-x^2) + D*-2x)*dT/dx + D(1-x^2)* d^2(T)/ dx^2
        diff_elem = (
            firstD * (1 - xs[m] ** 2) - 2 * Diffusion[m, n] * xs[m]
        ) * firstT + Diffusion[m, n] * (1 - xs[m] ** 2) * secondT
        # T(x_m, t_n+1) = T(x_m, t_n) + Δt / C(x_m, t_n)
        # * (diff - I + S(1-A) )
        Temp[m, n + 1] = Temp[m, n] + dt / Capacity[m, n] * (
            diff_elem - Ir_emission[m, n] + Source[m, n] * (1 - Albedo[m, n])
        )

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    for n in range(0, 501, 50):
        plt.plot(xs, Temp[:, n], label=f"t={n}")

    # plt.axvline(dx*20-1)
    plt.ylabel("Temperature")
    plt.xlabel(r"$x = $sin$(\lambda)$")
    plt.legend()
    plt.show()
