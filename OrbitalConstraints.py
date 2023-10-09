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
import matplotlib.pyplot as plt

from InsolationFunction import S
from HeatCapacity import C, f_o, f_i
from IRandAlbedo import A_1, A_2, A_2, I_1, I_2, I_3

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
    return (x[i + 2] - 2 * x[i + 1] + x[i]) / dx**2


def central2ndorder(x: list[float] | floatarr, i: int, dx: float) -> float:
    return (x[i + 2] - 2 * x[i] + x[i - 2]) / (2 * dx) ** 2


def backward2ndorder(x: list[float] | floatarr, i: int, dx: float) -> float:
    return (x[i] - 2 * x[i - 1] + x[i - 2]) / dx**2


##  ##
def climate_model_in_lat(spacedim=200, time=1):
    dlam = np.pi / (spacedim - 1)  # spacial separation in -pi/2 to pi/2
    lats = np.linspace(-1, 1, spacedim) * np.pi / 2

    # timedim may need to be split in to chunks which are then recorded
    # e.g. do a year of evolution then write to file and overwrite the array
    dt = 1 / 365  # 1 day timestep
    timedim = int(np.ceil(time / dt))

    Temp = np.ones((spacedim, timedim + 1))
    Temp[:, 0] = Temp[:, 0] * 350
    # Temp[:, 0] = np.linspace(-1, 1, spacedim) * 100 + 250
    # Temp[:, 0] = np.abs(np.sin(np.linspace(0, 1, spacedim)*np.pi))*50+300
    # Temp[:, 0] = np.exp(-5 * np.linspace(-1, 1, spacedim) ** 2) * 50 + 300

    Capacity = np.zeros_like(Temp)  # effective heat capacity
    Ir_emission = np.zeros_like(Temp)  # IR emission function (Energy sink)
    Source = np.zeros_like(
        Temp
    )  # Diurnally averaged insolation function (Energy Source)
    Albedo = np.zeros_like(Temp)  # Albedo

    D_0 = 0.58  # J s^-1 m^-2 K^-1
    omega_0 = 7.27 * 10**-5  # rad s^-1
    omega = omega_0
    D = D_0 * (omega_0 / omega) ** 2

    Diffusion = np.ones_like(Temp) * D  # diffusion coefficient (Lat)

    secondT = np.zeros(spacedim)
    firstT = np.zeros(spacedim)
    firstD = np.zeros(spacedim)

    yeartosecond = 365.25 * 24 * 3600  # s / yr

    for n in range(timedim):
        for m in range(spacedim):
            if m == 0:
                secondT[m] = forward2ndorder(Temp[:, n], m, dlam)
                firstT[m] = 0
                firstD[m] = forwarddifference(Diffusion[:, n], m, dlam)
            elif m == 1:
                # forward difference for zero edge
                secondT[m] = forward2ndorder(Temp[:, n], m, dlam)
                firstT[m] = forwarddifference(Temp[:, n], m, dlam)
                firstD[m] = forwarddifference(Diffusion[:, n], m, dlam)
            elif m == spacedim - 2:
                # backwards difference for end edge
                secondT[m] = backward2ndorder(Temp[:, n], m, dlam)
                firstT[m] = backwarddifference(Temp[:, n], m, dlam)
                firstD[m] = backwarddifference(Diffusion[:, n], m, dlam)
            elif m == spacedim - 1:
                secondT[m] = backward2ndorder(Temp[:, n], m, dlam)
                firstT[m] = 0
                firstD[m] = backwarddifference(Diffusion[:, n], m, dlam)
            else:
                # central difference for most cases
                secondT[m] = central2ndorder(Temp[:, n], m, dlam)
                firstT[m] = centraldifference(Temp[:, n], m, dlam)
                firstD[m] = centraldifference(Diffusion[:, n], m, dlam)

        # diff = (dD/dx (1-x^2) + D*-2x)*dT/dx + D(1-x^2)* d^2(T)/ dx^2
        diff_elem = (
            firstD - 2 * Diffusion[:, n] * np.tan(lats[:])
        ) * firstT + secondT * Diffusion[:, n]
        # T(x_m, t_n+1) = T(x_m, t_n) + Δt / C(x_m, t_n)
        # * (diff - I + S(1-A) )
        Capacity[:, n] = C(f_o(lats), f_i(Temp[:, n]))
        Ir_emission[:, n] = I_1(Temp[:, n])
        Source[:, n] = S(1, lats, dt * n, np.deg2rad(23.5))
        Albedo[:, n] = A_1(Temp[:, n])
        Temp[:, n + 1] = Temp[:, n] + yeartosecond * dt / Capacity[:, n] * (
            diff_elem - Ir_emission[:, n] + Source[:, n] * (1 - Albedo[:, n])
        )

    for n in range(0, 10):
        plt.plot(lats, Temp[:, n], label=f"t={dt * n :.3f}")

    plt.ylabel("Temperature")
    plt.xlabel(r"$x = $sin$(\lambda)$")
    plt.legend()
    plt.show()


def climate_model_in_x(spacedim=200, time=1):
    dx = 2 / (spacedim - 1)  # spacial separation from 2 units from -1 to 1
    xs = np.linspace(-1, 1, spacedim)
    lats = np.arcsin(xs)

    # timedim may need to be split in to chunks which are then recorded
    # e.g. do a year of evolution then write to file and overwrite the array
    dt = 1 / 365  # 1 day timestep
    timedim = int(np.ceil(time / dt))

    Temp = np.ones((spacedim, timedim + 1))
    Temp[:, 0] = Temp[:, 0] * 350
    # Temp[:, 0] = np.linspace(-1, 1, spacedim) * 100 + 250
    # Temp[:, 0] = np.abs(np.sin(np.linspace(0, 1, spacedim)*np.pi))*50+300
    # Temp[:, 0] = np.exp(-5 * np.linspace(-1, 1, spacedim) ** 2) * 50 + 300

    Capacity = np.zeros_like(Temp)  # effective heat capacity
    Ir_emission = np.zeros_like(Temp)  # IR emission function (Energy sink)
    Source = np.zeros_like(
        Temp
    )  # Diurnally averaged insolation function (Energy Source)
    Albedo = np.zeros_like(Temp)  # Albedo

    D_0 = 0.58  # J s^-1 m^-2 K^-1
    omega_0 = 7.27 * 10**-5  # rad s^-1
    omega = omega_0
    D = D_0 * (omega_0 / omega) ** 2

    Diffusion = np.ones_like(Temp) * D  # diffusion coefficient (Lat)

    secondT = np.zeros(spacedim)
    firstT = np.zeros(spacedim)
    firstD = np.zeros(spacedim)

    yeartosecond = 365.25 * 24 * 3600  # s / yr
    for n in range(timedim):
        for m in range(spacedim):
            if m == 0 or m == 1:
                # forward difference for zero edge
                secondT[m] = forward2ndorder(Temp[:, n], m, dx)
                firstT[m] = forwarddifference(Temp[:, n], m, dx)
                firstD[m] = forwarddifference(Diffusion[:, n], m, dx)
            elif m == spacedim - 2 or m == spacedim - 1:
                # backwards difference for end edge
                secondT[m] = backward2ndorder(Temp[:, n], m, dx)
                firstT[m] = backwarddifference(Temp[:, n], m, dx)
                firstD[m] = backwarddifference(Diffusion[:, n], m, dx)
            else:
                # central difference for most cases
                secondT[m] = central2ndorder(Temp[:, n], m, dx)
                firstT[m] = centraldifference(Temp[:, n], m, dx)
                firstD[m] = centraldifference(Diffusion[:, n], m, dx)
        # diff = (dD/dx (1-x^2) + D*-2x)*dT/dx + D(1-x^2)* d^2(T)/ dx^2
        diff_elem = (
            firstD * (1 - xs**2) - 2 * Diffusion[:, n] * xs
        ) * firstT + Diffusion[:, n] * (1 - xs**2) * secondT
        # T(x_m, t_n+1) = T(x_m, t_n) + Δt / C(x_m, t_n)
        # * (diff - I + S(1-A) )
        Capacity[:, n] = C(f_o(lats), f_i(Temp[:, n]))
        Ir_emission[:, n] = I_1(Temp[:, n])
        Source[:, n] = S(1, lats, dt * n, np.deg2rad(0))
        Albedo[:, n] = A_1(Temp[:, n])

        Temp[:, n + 1] = Temp[:, n] + yeartosecond * dt / Capacity[:, n] * (
            diff_elem - Ir_emission[:, n] + Source[:, n] * (1 - Albedo[:, n])
        )
    # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1)
    # for n in range(0, 5, 1):
    #     ax1.plot(
    #         xs[:10],
    #         Temp[:10, n],
    #         label=f"t={dt * n :.3f}",
    #     )
    #     ax2.plot(xs[:10], -Ir_emission[:10, n] + Source[:10, n] * (1 - Albedo[:10, n]))
    #     ax3.plot(
    #         xs[:10],
    #         (Temp[:10, n + 1] - Temp[:10, n]) * Capacity[:10, n] / (yeartosecond * dt)
    #         - (-Ir_emission[:10, n] + Source[:10, n] * (1 - Albedo[:10, n])),
    #     )
    #     ax4.plot(xs[:10], -Ir_emission[:10, n])
    #     ax5.plot(xs[:10], Source[:10, n] * (1 - Albedo[:10, n]))
    # ax1.set_ylabel("Temp")
    # ax2.set_ylabel("-I + S(1-A)")
    # ax3.set_ylabel("Diff_elem")
    # ax4.set_ylabel("-I")
    # ax5.set_ylabel("S(1-A)")
    for n in range(0, 5, 1):
        plt.plot(xs, Temp[:, n], label=f"t={dt*n}")
    plt.ylabel("Temperature")
    plt.xlabel(r"$x = $sin$(\lambda)$")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    # climate_model_in_lat(200, 5)
    climate_model_in_x(200, 5)
