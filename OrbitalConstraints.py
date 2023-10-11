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


def forwardbackward_pole(x, dx):
    """Used for pole at the start of the array, i.e. i = 0
    Assumes dx/dt = 0 at the pole"""
    return (x[1] - x[0]) / dx**2


def backwardforward_pole(x, dx):
    """Used for pole at the end of the array, i.e. i = len(x)-1
    Assumes dx/dt = 0 at the pole"""
    return (x[-2] - x[-1]) / dx**2


def centralbackward_edge(x, dx):
    """Used for one along from the start of the array, i.e. i = 1"""
    return (x[2] - x[1]) / (2 * dx**2)


def centralforward_edge(x, dx):
    """Used for one along from the end of the array, i.e. i = len(x)-2"""
    return (x[-2] - x[-3]) / (2 * dx**2)


def forward2ndorder(x: list[float] | floatarr, i: int, dx: float) -> float:
    return (x[i + 2] - 2 * x[i + 1] + x[i]) / dx**2


def central2ndorder(x: list[float] | floatarr, i: int, dx: float) -> float:
    return (x[i + 2] - 2 * x[i] + x[i - 2]) / (2 * dx) ** 2


def backward2ndorder(x: list[float] | floatarr, i: int, dx: float) -> float:
    return (x[i] - 2 * x[i - 1] + x[i - 2]) / dx**2


##  ##
def climate_model_in_lat(spacedim=200, time=1):
    dlam = np.pi / (spacedim - 1)  # spacial separation in -pi/2 to pi/2
    lats = np.linspace(-1, 1, spacedim) * (np.pi / 2)
    degs = np.rad2deg(lats)

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
                secondT[0] = forwardbackward_pole(Temp[:, n], dlam)
                firstT[0] = 0
                firstD[0] = forwarddifference(Diffusion[:, n], 0, dlam)
            elif m == 1:
                # forward difference for zero edge
                secondT[1] = centralbackward_edge(Temp[:, n], dlam)
                firstT[1] = centraldifference(Temp[:, n], 1, dlam)
                firstD[1] = centraldifference(Diffusion[:, n], 1, dlam)
            elif m == spacedim - 2:
                # backwards difference for end edge
                secondT[-2] = centralforward_edge(Temp[:, n], dlam)
                firstT[-2] = centraldifference(Temp[:, n], -2, dlam)
                firstD[-2] = centraldifference(Diffusion[:, n], -2, dlam)
            elif m == spacedim - 1:
                secondT[-1] = backwardforward_pole(Temp[:, n], dlam)
                firstT[-1] = 0
                firstD[-1] = backwarddifference(Diffusion[:, n], -1, dlam)
            else:
                # central difference for most cases
                secondT[m] = central2ndorder(Temp[:, n], m, dlam)
                firstT[m] = centraldifference(Temp[:, n], m, dlam)
                firstD[m] = centraldifference(Diffusion[:, n], m, dlam)

        # diff = (dD/dx (1-x^2) + D*-2x)*dT/dx + D(1-x^2)* d^2(T)/ dx^2
        diff_elem = (
            firstD - Diffusion[:, n] * np.tan(lats[:])
        ) * firstT + secondT * Diffusion[:, n]
        # T(x_m, t_n+1) = T(x_m, t_n) + Δt / C(x_m, t_n)
        # * (diff - I + S(1-A) )
        Capacity[:, n] = C(f_o(lats), f_i(Temp[:, n]), Temp[:, n])
        Ir_emission[:, n] = I_2(Temp[:, n])
        Source[:, n] = S(1, lats, dt * n, np.deg2rad(23.5))
        Albedo[:, n] = A_2(Temp[:, n])
        Temp[:, n + 1] = Temp[:, n] + yeartosecond * dt / Capacity[:, n] * (
            diff_elem - Ir_emission[:, n] + Source[:, n] * (1 - Albedo[:, n])
        )

    # fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)

    # q1 = 0
    # q2 = len(degs)
    # for n in range(0, 100, 10):
    #     ax1.plot(
    #         degs[q1:q2],
    #         Temp[q1:q2, n],
    #         label=f"t={dt * n :.3f}",
    #     )
    #     ax2.plot(
    #         degs[q1:q2],
    #         -Ir_emission[q1:q2, n] + Source[q1:q2, n] * (1 - Albedo[q1:q2, n]),
    #     )
    #     ax3.plot(
    #         degs[q1:q2],
    #         (Temp[q1:q2, n + 1] - Temp[q1:q2, n])
    #         * Capacity[q1:q2, n]
    #         / (yeartosecond * dt)
    #         - (-Ir_emission[q1:q2, n] + Source[q1:q2, n] * (1 - Albedo[q1:q2, n])),
    #     )
    #     ax4.plot(degs[q1:q2], -Ir_emission[q1:q2, n])
    #     ax5.plot(degs[q1:q2], Source[q1:q2, n] * (1 - Albedo[q1:q2, n]))
    #     ax6.plot(degs[q1:q2], 1 / Capacity[q1:q2, n])
    # ax1.set_ylabel("Temp")
    # ax2.set_ylabel("-I + S(1-A)")
    # ax3.set_ylabel("Diff_elem")
    # ax4.set_ylabel("-I")
    # ax5.set_ylabel("S(1-A)")
    # ax6.set_ylabel("1/Capacity")

    for n in range(0, timedim + 1, timedim // 10):
        plt.plot(degs, Temp[:, n], label=f"t={dt * n :.3f}")
    plt.axhline(273, ls="--", label=r"0$^\circ$C")
    plt.ylabel("Temperature")
    plt.xlabel(r"$\lambda$")
    plt.legend()
    plt.show()


def climate_model_in_x(spacedim=200, time=1):
    dx = 2 / (spacedim - 1)  # spacial separation from 2 units from -1 to 1
    xs = np.linspace(-1, 1, spacedim)
    lats = np.arcsin(xs)
    degs = np.rad2deg(lats)

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
                # forward then backward for zero edge
                secondT[0] = forward2ndorder(Temp[:, n], 0, dx)
                firstT[0] = forwarddifference(Temp[:, n], 0, dx)
                firstD[0] = forwarddifference(Diffusion[:, n], 0, dx)
            elif m == 1:
                # forward difference for zero edge
                secondT[1] = centralbackward_edge(Temp[:, n], dx)
                firstT[1] = centraldifference(Temp[:, n], 1, dx)
                firstD[1] = centraldifference(Diffusion[:, n], 1, dx)
            elif m == spacedim - 2:
                # backwards difference for end edge
                secondT[-2] = centralforward_edge(Temp[:, n], dx)
                firstT[-2] = centraldifference(Temp[:, n], -2, dx)
                firstD[-2] = centraldifference(Diffusion[:, n], -2, dx)
            elif m == spacedim - 1:
                # backwards then forwards difference for end edge
                secondT[-1] = backwardforward_pole(Temp[:, n], dx)
                firstT[-1] = backwarddifference(Temp[:, n], -1, dx)
                firstD[-1] = backwarddifference(Diffusion[:, n], -1, dx)
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
        Capacity[:, n] = C(f_o(lats), f_i(Temp[:, n]), Temp[:, n])
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
    for n in range(0, 365 * 50 + 1, 365 * 5):
        plt.plot(degs, Temp[:, n], label=f"t={dt*n}")
    plt.ylabel("Temperature")
    # plt.xlabel(r"$x = $sin$(\lambda)$")
    plt.xlabel(r"latitude, $\lambda$")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    climate_model_in_lat(120, 20)
    # climate_model_in_x(120, 100)
