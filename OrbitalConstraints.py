# PDE:
# C dT/dt - d/dx (D(1-x^2)*dT/dx) + I = S(1-A)
# discretise time derivative by forward difference:
# dT/dt = (T(x, t_n+1) - T(x, t_n)) / Δt
# solve for T(x, t_n+1):

# T(x, t_n+1) = T(x, t_n) + Δt/C [d/dx (D(1-x^2)*dT/dx) - I + S(1-A)]

# d/dx (D(1-x^2)*dT/dx)
# expand outer derivative
# d(D(1-x^2))/dx * dT/dx + D(1-x^2)* d^2(T)/ dx^2

# discretise space derivatives by central difference:
# dT/dx = (T(x_m+1, t_n) - T(x_m-1, t_n)) / 2Δx
# d^2 T / dx^2 = (T(x_m+2, t_n) - 2*T(x_m, t_n) + T(x_m-2, t_n)) / 4 Δx^2
# dD/dx = (D(x_m+1, t_n) - D(x_m-1, t_n)) / 2Δx

# All together:
# T(x_m, t_n+1) = T(x_m, t_n) + Δt / C(x_m, t_n)
# * ((T(x_m+2)-2*T(x_m)+T(x_m-2))*D(x_m)(1-x_n^2)/4Δx^2
#     + (T(x_m+1) - T(x_m-1)) * ((1-x^2_m)*(D(x_m+1)-D(x_m-1))/2Δx - 2x_m*D(x_m)) / 2Δx)
# - I + S(1-A) )
# i.e. a mess
# Boundaries: wrap back / reflect to same latitude

import numpy as np

spacedim = 200  # number of points in space
dx = 2 / (spacedim - 1)  # spacial separation from 2 units from -1 to 1

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
        if m == 0:
            # forward difference for zero edge
            # (T(x_m+2) - 2T(x_m+1) + T(x_m)) * D(x_m) * (1-x_m^2) / Δx^2
            # + (T(x_m+1) - T(x_m) / Δx) * ((1-x_m^2) * (D(x_m+1) - D(x_m)) / Δx - 2*x_m*D(x_m))
            # with m = 0 this reduces to:
            diff_elem = (Temp[1, n] - Temp[0, n]) / dx * (2 * Diffusion[0, n])
        elif m == 1:
            # (T(x_m+2) - 2T(x_m+1) + T(x_m)) * D(x_m) * (1-x_m^2) / Δx^2
            # + (T(x_m+1) - T(x_m) / Δx) * ((1-x_m^2) * (D(x_m+1) - D(x_m)) / Δx - 2*x_m*D(x_m))
            # x_m = dx-1 i.e. one dx increment after the start at -1
            diff_elem = (Temp[3, n] - 2 * Temp[2, n] + Temp[1, n]) * Diffusion[1, n] * (
                1 - (dx - 1) ** 2
            ) / dx**2 + ((Temp[2, n] - Temp[1, n]) / dx) * (
                (1 - (dx - 1) ** 2) * (Diffusion[2, n] - Diffusion[1, n]) / dx
                - 2 * (dx - 1) * Diffusion[1, n]
            )
        elif m == spacedim - 2:  # mirror of m == 1, ie m = -2 in python arrays
            # backwards difference for end edge
            # (T(x_m) - 2*T(x_m-1) + T(x_m-2)) * D(x_m) * (1-x_m^2) / Δx^2
            # + (T(x_m) - T(x_m-1)) / Δx * ((1-x_m^2) * (D(x_m) - D(x_m-1)) / Δx - 2*x_m*D(x_m))
            # x_m = 1-dx i.e. one dx increment before the end at 1
            diff_elem = (Temp[-2, n] - 2 * Temp[-3, n] + Temp[-4, n]) * Diffusion[
                -2, n
            ] * (1 - (1 - dx) ** 2) / dx**2 + ((Temp[-2, n] - Temp[-3, n]) / dx) * (
                (1 - (1 - dx) ** 2) * (Diffusion[-2, n] - Diffusion[-3, n]) / dx
                - 2 * (1 - dx) * Diffusion[-2, n]
            )
        elif m == spacedim - 1:  # mirror of m == 0 ie m = -1 in python arrays
            # (T(x_m) - 2*T(x_m-1) + T(x_m-2)) * D(x_m) * (1-x_m^2) / Δx^2
            # + (T(x_m) - T(x_m-1)) / Δx * ((1-x_m^2) * (D(x_m) - D(x_m-1)) / Δx - 2*x_m*D(x_m))
            # x_m = 1
            diff_elem = (Temp[-2, n] - Temp[-1, n]) / dx * (2 * Diffusion[-1, n])
        else:
            # x_m = dx*m - 1 i.e. m steps along from -1
            diff_elem = (Temp[m + 2, n] - 2 * Temp[m, n] + Temp[m - 2, n]) * Diffusion[
                m, n
            ] * (1 - (dx * m - 1) ** 2) / (4 * dx**2) + (
                Temp[m + 1, n] - Temp[m - 1, n]
            ) * (
                (1 - (dx * m - 1) ** 2)
                * (Diffusion[m + 1, n] - Diffusion[m - 1, n])
                / (2 * dx)
                - 2 * (dx * m - 1) * Diffusion[m, n]
            ) / (
                2 * dx
            )
        Temp[m, n + 1] = Temp[m, n] + dt / Capacity[m, n] * (
            diff_elem - Ir_emission[m, n] + Source[m, n] * (1 - Albedo[m, n])
        )

import matplotlib.pyplot as plt

for n in range(5):
    plt.plot(np.linspace(-1, 1, spacedim), Temp[:, 12 * n], label=f"t={12*n}")

# plt.axvline(dx*20-1)
plt.ylabel("Temperature")
plt.xlabel(r"$x = $sin$(\lambda)$")
plt.legend()
plt.show()
