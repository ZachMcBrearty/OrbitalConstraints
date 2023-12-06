# PDE:
# C dT/dt - d/dx (D(1-x^2)*dT/dx) + I = S(1-A)
# discretise time derivative by forward difference:
# dT/dt = (T(x, t_n+1) - T(x, t_n)) / Δt
# solve for T(x, t_n+1):

# T(x, t_n+1) = T(x, t_n) + Δt/C [d/dx (D(1-x^2)*dT/dx) - I + S(1-A)]

# d/dx (D(1-x^2)*dT/dx)
# expand outer derivative
# d(D(1-x^2))/dx * dT/dx + D(1-x^2)* d^2(T)/ dx^2
# define the differential element as
# (in x) diff_elem = dD/dx (1-x^2) * dT/dx + D*-2x*dT/dx + D(1-x^2)* d^2(T)/ dx^2
# and in latitude (x = sin(λ))
# d/dx = 1/cos(λ)d/dλ , d^2/dx^2 = sin(λ) / cos^3(λ) d/dλ + 1/cos^2(λ) d^2/dλ^2
# diff_elem = dT/dλ * (dD/dλ - D tanλ) + d^2 T/dλ^2

# discretise space derivatives using derivative functions below
# for the most part use central and central^2 discretisation
# edges are more complicated
# in λ we can apply the boundary condition: dT/dλ = 0 for λ=+/- π/2
# and use a forward-backward (backward-forward) for the south (north) poles
# one in from the poles (called edges here) we use a central backward (forward)
# for the south (north) edge

# in x we simply use forward^2 and backward^2 at the poles and edges.

# All together:
# T(x_m, t_n+1) = T(x_m, t_n) + Δt / C(x_m, t_n)
# * (diff_elem - I + S(1-A))

import numpy as np
import numpy.typing as npt

from Constants import *
from InsolationFunction import S
from HeatCapacity import get_C_func, f_o, f_i
from IRandAlbedo import A_1, A_2, A_3, I_1, I_2, I_3
from plotting import colourplot, complexplotdata
from filemanagement import write_to_file, load_config
from tidalheating import get_viscoheating, fixed_Q_tidal_heating
from orbital_model import orbital_model_explicit as orbital_model

skip = 60


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


def forwardbackward_pole(T: floatarr, dx: float) -> float:
    """Used for pole at the start of the array, i.e. i = 0
    Assumes dT/dx = 0 at the pole"""
    # Forward: d^2T/dx^2 = (dT/dx|i=1 - dT/dx|i=0) / dx
    # dT/dx|i=0 == 0
    # Backward: d^2T/dx^2 = (T(i=1) - T(i=0)) / dx^2
    return (T[1] - T[0]) / dx**2


def backwardforward_pole(x: floatarr, dx: float) -> float:
    """Used for pole at the end of the array, i.e. i = len(x)-1
    Assumes dx/dt = 0 at the pole"""
    # Backward: d^2T/dx^2 = (dT/dx|i=i_max - dT/dx|i=i_max-1) / dx
    # dT/dx|i=i_max == 0
    # Forward: d^2T/dx^2 = -(T(i=i_max) - T(i=i_max-1)) / dx^2
    # => d^2T/dx^2 = (T(i=i_max-1) - T(i=i_max)) / dx^2
    # in python the final entry is -1 and last to final is -2
    return (x[-2] - x[-1]) / dx**2


def centralbackward_edge(x: floatarr, dx: float) -> float:
    """Used for one along from the start of the array, i.e. i = 1"""
    # Central: d^2T/dx^2 = (dT/dx|i=2 - dT/dx|i=0) / 2dx
    # dT/dx|i=0 == 0
    # Backward: d^2T/dx^2 = (T(i=2) - T(i=1)) / 2dx^2
    return (x[2] - x[1]) / (2 * dx**2)


def centralcentral_firstedge(x: floatarr, dx: float) -> float:
    """Used for one along from the start of the array, i.e. i = 1"""
    # Central: d^2T/dx^2 = (dT/dx|i=2 - dT/dx|i=0) / 2dx
    # dT/dx|i=0 == 0
    # Central: d^2T/dx^2 = (T(i=3) - T(i=1)) / 4dx^2
    return (x[3] - x[1]) / (4 * dx**2)


def centralcentral_secondedge(x: floatarr, dx: float) -> float:
    """Used for one along from the end of the array, i.e. i = len(x)-2"""
    # Central: d^2T/dx^2 = (dT/dx|i=i_max - dT/dx|i=i_max-2) / 2dx
    # dT/dx|i=i_max == 0
    # Central: d^2T/dx^2 = -(T(i=i_max-1) - T(i=i_max-3)) / 4dx^2
    return (x[-4] - x[-2]) / (4 * dx**2)


def centralforward_edge(x: floatarr, dx: float) -> float:
    """Used for one along from the end of the array, i.e. i = len(x)-2"""
    # Central: d^2T/dx^2 = (dT/dx|i=i_max - dT/dx|i=i_max-2) / 2dx
    # dT/dx|i=i_max == 0
    # Forward: d^2T/dx^2 = -(T(i=i_max-1) - T(i=i_max-2)) / 2dx^2
    # => d^2T/dx^2 = (T(i=i_max-2) - T(i=i_max-1)) / 2dx^2
    return (x[-3] - x[-2]) / (2 * dx**2)


def forward2ndorder(x: list[float] | floatarr, i: int, dx: float) -> float:
    return (x[i + 2] - 2 * x[i + 1] + x[i]) / dx**2


def central2ndorder(x: list[float] | floatarr, i: int, dx: float) -> float:
    return (x[i + 2] - 2 * x[i] + x[i - 2]) / (2 * dx) ** 2


def backward2ndorder(x: list[float] | floatarr, i: int, dx: float) -> float:
    return (x[i] - 2 * x[i - 1] + x[i - 2]) / dx**2


##  ##
def climate_model_in_lat(
    config: CONF_PARSER_TYPE,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    # number of spatial nodes
    spacedim = config.getint(section="PDE", option="spacedim")

    dlam = np.pi / (spacedim - 1)  # spacial separation in -pi/2 to pi/2
    lats = np.linspace(-1, 1, spacedim) * (np.pi / 2)
    coslats = np.cos(lats)
    degs = np.rad2deg(lats)

    # simulation length in years
    time = config.getfloat(section="PDE", option="time")
    # simulation timestep in years
    dt = config.getfloat(section="PDE", option="timestep") / 365
    # timedim may need to be split in to chunks which are then recorded
    # e.g. do a year of evolution then write to file and overwrite the array
    timedim = int(np.ceil(time / dt))

    Temp = np.ones((spacedim, timedim + 1))
    Temp[:, 0] = Temp[:, 0] * config.getfloat("PDE", "starttemp")

    Capacity = np.zeros_like(Temp)  # effective heat capacity
    Ir_emission = np.zeros_like(Temp)  # IR emission function (Energy sink)
    Source = np.zeros_like(
        Temp
    )  # Diurnally averaged insolation function (Energy Source)
    Albedo = np.zeros_like(Temp)  # Albedo

    D_0 = config.getfloat("PLANET", "D_0")  # J s^-1 m^-2 K^-1
    p = config.getfloat("PLANET", "p") / 101  # kPa
    c_p = config.getfloat("PLANET", "c") / 1  # 10^3 g^-1 K^-1
    m = config.getfloat("PLANET", "m") / 28
    omega = config.getfloat("PLANET", "omega") / 1  # 7.27 * 10**-5 rad s^-1
    D = D_0 * p * c_p * m**-2 * omega**-2

    Diffusion = np.ones_like(Temp) * D  # diffusion coefficient (Lat)

    secondT = np.zeros(spacedim)
    firstT = np.zeros(spacedim)
    firstD = np.zeros(spacedim)

    a = config.getfloat("ORBIT", "gassemimajoraxis")  # semimajoraxis
    e = config.getfloat("ORBIT", "gaseccentricity")  # eccentricity
    axtilt = np.deg2rad(config.getfloat("PLANET", "obliquity"))  # obliquity

    if (frac := config.get("PLANET", "landfractype")) in [
        "earthlike",
        "earth",
        "earth-like",
        "earth like",
    ]:
        F_o = f_o(lats)  # earth like
    elif frac.startswith("uniform"):
        # uniform
        if (q := float(frac.split(":")[1])) > 1 or q < 0:
            raise ValueError(f"Uniform land-ocean fraction must be 0<f<1, got: {q}")
        F_o = np.ones_like(lats) * q
    else:
        raise ValueError(f"Unknown land-ocean fraction type, got: {frac}")

    orbits = orbital_model(config, 24)
    M_gas = config.getfloat("ORBIT", "gasgiantmass") * MASS["jupiter"]
    gas_rad = config.getfloat("ORBIT", "gasgiantradius") * RADIUS["jupiter"]
    moon_rad = config.getfloat("ORBIT", "moonradius") * RADIUS["luna"]
    moon_a = config.getfloat("ORBIT", "moonsemimajoraxis") * AU
    moon_ecc = config.getfloat("ORBIT", "mooneccentricity")

    heating_dist = coslats * dlam / (8 * np.pi * moon_rad**2)
    # shearmod = config.getfloat("TIDALHEATING", "shearmod") * 2e10  # Nm^-2
    # Q = config.getfloat("TIDALHEATING", "Q")
    gas_albedo = config.getfloat("TIDALHEATING", "gasalbedo")

    # moon_density = M_moon / (4 / 3 * np.pi * moon_rad**3)

    C = get_C_func(spacedim)

    T_surf = np.sum(Temp[:, 0] * coslats * dlam) / 2  # type: float
    tidal_heating_value = get_viscoheating(config, T_surf)
    # print(tidal_heating_value, tidal_heating_value_fixed)
    heatings = tidal_heating_value * heating_dist
    # print(heatings, tidal_heating_value_fixed * heating_dist)
    for n in range(timedim):
        for m in range(spacedim):
            if m == 0:
                secondT[0] = forwardbackward_pole(Temp[:, n], dlam)
                firstT[0] = 0
                firstD[0] = forwarddifference(Diffusion[:, n], 0, dlam)
            elif m == 1:
                # forward difference for zero edge
                # secondT[1] = centralbackward_edge(Temp[:, n], dlam)
                secondT[1] = centralcentral_firstedge(Temp[:, n], dlam)
                firstT[1] = centraldifference(Temp[:, n], 1, dlam)
                firstD[1] = centraldifference(Diffusion[:, n], 1, dlam)
            elif m == spacedim - 2:
                # backwards difference for end edge
                # secondT[-2] = centralforward_edge(Temp[:, n], dlam)
                secondT[-2] = centralcentral_secondedge(Temp[:, n], dlam)
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

        diff_elem = (
            firstD - Diffusion[:, n] * np.tan(lats[:])
        ) * firstT + secondT * Diffusion[:, n]
        # T(x_m, t_n+1) = T(x_m, t_n) + Δt / C(x_m, t_n)
        # * (diff - I + S(1-A) )
        # f_o_point7 = np.ones_like(lats) * 0.7
        Capacity[:, n] = C(F_o, f_i(Temp[:, n]), Temp[:, n])
        Ir_emission[:, n] = I_2(Temp[:, n])
        # r = dist(a, e, dt * n)
        # eclip = 0
        # star, gas, moon, eclipsed = next(orbits)
        # eclip += eclipsed
        # for _ in range(12 - 1):
        #     star, gas, moon, eclipsed = next(orbits)
        #     eclip += eclipsed
        # eclip /= 12
        # if eclip < 1:
        #     print(f"Eclipsed at {dt*n}")

        Source[:, n] = S(
            a, lats, dt * n, axtilt, e, gas_albedo, gas_rad, moon_a
        )  # * eclip
        Albedo[:, n] = A_2(Temp[:, n])

        if n != 0 and n % skip == 0:
            T_surf = np.sum(np.sum(Temp[:, n - skip : n]) * coslats * dlam) / 2 / skip
            tidal_heating_value = get_viscoheating(config, T_surf)
            heatings = tidal_heating_value * heating_dist
            # print(diff_elem[spacedim // 2])
            # print(Ir_emission[spacedim // 2, n])
            # print(Albedo[spacedim // 2, n])
            # print(Source[spacedim // 2, n] * (1 - Albedo[spacedim // 2, n]))
            # print(heatings[spacedim // 2])
            # print()
        # if 100 * 365 <= n < 101 * 365:
        #     diff_elem += 25
        Temp[:, n + 1] = Temp[:, n] + YEARTOSECOND * dt / Capacity[:, n] * (
            diff_elem - Ir_emission[:, n] + Source[:, n] * (1 - Albedo[:, n]) + heatings
        )
        if np.any(Temp[:, n + 1] > 1000):
            Temp[:, n + 2 :] = -1
            break
        # if 100 * 365 <= n < 101 * 365:
        #     Temp[len(Temp) // 2 + 0, n + 1] = 200
        #     Temp[len(Temp) // 2 - 1, n + 1] = 200
        #     Temp[len(Temp) // 2 + 1, n + 1] = 200
        #     Temp[len(Temp) // 2 - 2, n + 1] = 200
    times = np.linspace(0, time, timedim)

    if config.getboolean("FILEMANAGEMENT", "save"):
        write_to_file(times, Temp, degs, config.get("FILEMANAGEMENT", "save_name"))

    if config.getboolean("FILEMANAGEMENT", "plot"):
        # complexplotdata(degs, Temp, dt, Ir_emission, Source, Albedo, Capacity)
        colourplot(degs, Temp, times)

    return degs, Temp, times


if __name__ == "__main__":
    # import cProfile
    # from pstats import SortKey, Stats

    config = load_config("config.ini", "OrbitalConstraints")
    # with cProfile.Profile() as p:
    #     q = orbital_model_explicit(config, 24)
    #     # next(q)
    #     for x in range(100000):
    #         next(q)
    # Stats(p).strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(100)

    climate_model_in_lat(config)

    # times, temps, degs = read_file(config.get("FILEMANAGEMENT", "save_name") + ".npz")
    # dt = times[1] - times[0]

    # plotdata(degs, temps, dt, 0, 365 * 1, 10)
    # yearavgplot(degs, temps, dt, 0, None, 20)
    # colourplot(degs, temps, times, None, None, 5)
