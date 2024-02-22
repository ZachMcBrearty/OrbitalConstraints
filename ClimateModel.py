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
from derivatives import *
from InsolationFunction import S_planet, S_moon
from HeatCapacity import get_C_func, f_o, f_i
from IRandAlbedo import A_1, A_2, A_3, I_1, I_2, I_3
from plotting import colourplot, complexplotdata, yearavgplot
from filemanagement import write_to_file, load_config
from tidalheating import get_visco_func
from orbital_model import orbital_model_explicit as orbital_model

skip = 60


##  ##
def climate_model_moon(
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

    Diffusion = D  # diffusion coefficient (Lat)

    secondT = np.zeros(spacedim)
    firstT = np.zeros(spacedim)

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
        if (q := config.getfloat("PLANET", "landfrac")) > 1 or q < 0:
            raise ValueError(f"Uniform land-ocean fraction must be 0<f<1, got: {q}")
        F_o = np.ones_like(lats) * q
    else:
        raise ValueError(f"Unknown land-ocean fraction type, got: {frac}")

    gas_mass = config.getfloat("ORBIT", "gasgiantmass") * MASS["jupiter"]
    gas_rad = config.getfloat("ORBIT", "gasgiantradius") * RADIUS["jupiter"]

    moon_rad = config.getfloat("ORBIT", "moonradius") * RADIUS["luna"]
    moon_a = config.getfloat("ORBIT", "moonsemimajoraxis") * AU
    moon_ecc = config.getfloat("ORBIT", "mooneccentricity")
    moon_dens = config.getfloat("ORBIT", "moondensity")

    heating_dist = coslats * dlam / 2
    gas_albedo = config.getfloat("TIDALHEATING", "gasalbedo")

    C = get_C_func(spacedim)

    T_surf = np.sum(Temp[:, 0] * coslats * dlam, dtype=float) / 2
    visco_func = get_visco_func(gas_mass, moon_rad, moon_a, moon_ecc, moon_dens, B=25)
    tidal_heating_value = visco_func(T_surf)
    heatings = tidal_heating_value * heating_dist

    # fraction of light let through, i.e. 1-ε where ε is the eclipsing fraction
    eclip = 1 - np.arcsin(gas_rad / moon_a) / np.pi
    if eclip > 1:
        eclip = 1
    if eclip < 0:
        eclip = 0
    for n in range(timedim):
        secondT[0] = forwardbackward_pole(Temp[:, n], dlam)
        firstT[0] = 0

        # forward difference for zero edge
        # secondT[1] = centralbackward_edge(Temp[:, n], dlam)
        secondT[1] = centralcentral_firstedge(Temp[:, n], dlam)
        firstT[1] = centraldifference(Temp[:, n], 1, dlam)

        # backwards difference for end edge
        # secondT[-2] = centralforward_edge(Temp[:, n], dlam)
        secondT[-2] = centralcentral_secondedge(Temp[:, n], dlam)
        firstT[-2] = centraldifference(Temp[:, n], -2, dlam)

        secondT[-1] = backwardforward_pole(Temp[:, n], dlam)
        firstT[-1] = 0

        for m in range(2, spacedim - 2):
            # central difference for most cases
            secondT[m] = central2ndorder(Temp[:, n], m, dlam)
            firstT[m] = centraldifference(Temp[:, n], m, dlam)

        diff_elem = Diffusion * (secondT - np.tan(lats[:]) * firstT)
        # T(x_m, t_n+1) = T(x_m, t_n) + Δt / C(x_m, t_n)
        # * (diff - I + S(1-A) )
        Capacity[:, n] = C(F_o, f_i(Temp[:, n]), Temp[:, n])
        Ir_emission[:, n] = I_2(Temp[:, n])

        Source[:, n] = (
            S_moon(a, lats, dt * n, axtilt, e, gas_albedo, gas_rad, moon_a, moon_ecc)
            * eclip
        )
        Albedo[:, n] = A_2(Temp[:, n])

        if n != 0 and n % skip == 0:
            T_surf = (
                np.sum(
                    np.sum(Temp[:, n - skip : n], dtype=float) * coslats * dlam,
                    dtype=float,
                )
                / 2
                / skip
            )
            tidal_heating_value = visco_func(T_surf)
            heatings = tidal_heating_value * heating_dist
        Temp[:, n + 1] = Temp[:, n] + YEARTOSECOND * dt / Capacity[:, n] * (
            diff_elem - Ir_emission[:, n] + Source[:, n] * (1 - Albedo[:, n]) + heatings
        )
        if np.any(Temp[:, n + 1] > 1000):
            Temp[:, n + 2 :] = -1
            break
    times = np.linspace(0, time, timedim)

    if config.getboolean("FILEMANAGEMENT", "save"):
        write_to_file(times, Temp, degs, config.get("FILEMANAGEMENT", "save_name"))

    if config.getboolean("FILEMANAGEMENT", "plot"):
        # complexplotdata(degs, Temp, dt, Ir_emission, Source, Albedo, Capacity)
        colourplot(degs, Temp, times)

    return degs, Temp, times


def climate_model_planet(
    config: CONF_PARSER_TYPE,
) -> tuple[NDArray, NDArray, NDArray]:
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

    # derivatives of the T and D variables
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
        if (q := config.getfloat("PLANET", "landfrac")) > 1 or q < 0:
            raise ValueError(f"Uniform land-ocean fraction must be 0<f<1, got: {q}")
        F_o = np.ones_like(lats) * q
    else:
        raise ValueError(f"Unknown land-ocean fraction type, got: {frac}")

    C = get_C_func(spacedim)

    for n in range(timedim):
        secondT[0] = forwardbackward_pole(Temp[:, n], dlam)
        firstT[0] = 0
        firstD[0] = forwarddifference(Diffusion[:, n], 0, dlam)

        # forward difference for zero edge
        # secondT[1] = centralbackward_edge(Temp[:, n], dlam)
        secondT[1] = centralcentral_firstedge(Temp[:, n], dlam)
        firstT[1] = centraldifference(Temp[:, n], 1, dlam)
        firstD[1] = centraldifference(Diffusion[:, n], 1, dlam)

        # backwards difference for end edge
        # secondT[-2] = centralforward_edge(Temp[:, n], dlam)
        secondT[-2] = centralcentral_secondedge(Temp[:, n], dlam)
        firstT[-2] = centraldifference(Temp[:, n], -2, dlam)
        firstD[-2] = centraldifference(Diffusion[:, n], -2, dlam)

        secondT[-1] = backwardforward_pole(Temp[:, n], dlam)
        firstT[-1] = 0
        firstD[-1] = backwarddifference(Diffusion[:, n], -1, dlam)
        for m in range(2, spacedim - 2):
            # central difference for most cases
            secondT[m] = central2ndorder(Temp[:, n], m, dlam)
            firstT[m] = centraldifference(Temp[:, n], m, dlam)
            firstD[m] = centraldifference(Diffusion[:, n], m, dlam)

        diff_elem = (
            firstD - Diffusion[:, n] * np.tan(lats[:])
        ) * firstT + secondT * Diffusion[:, n]
        # T(x_m, t_n+1) = T(x_m, t_n) + Δt / C(x_m, t_n)
        # * (diff - I + S(1-A) )
        Capacity[:, n] = C(F_o, f_i(Temp[:, n]), Temp[:, n])

        Ir_emission[:, n] = I_2(Temp[:, n])

        Source[:, n] = S_planet(a, lats, dt * n, axtilt, e)  # * eclip
        Albedo[:, n] = A_2(Temp[:, n])

        Temp[:, n + 1] = Temp[:, n] + YEARTOSECOND * dt / Capacity[:, n] * (
            diff_elem - Ir_emission[:, n] + Source[:, n] * (1 - Albedo[:, n])
        )
        # if the model reaches extreme temperatures above the threshold,
        # stop and return all -1s for the rest of the data
        if np.any(Temp[:, n + 1] > 1000):
            Temp[:, n + 2 :] = -1
            break
    times = np.linspace(0, time, timedim)

    if config.getboolean("FILEMANAGEMENT", "save"):
        write_to_file(times, Temp, degs, config.get("FILEMANAGEMENT", "save_name"))

    if config.getboolean("FILEMANAGEMENT", "plot"):
        # complexplotdata(degs, Temp, dt, Ir_emission, Source, Albedo, Capacity)
        colourplot(degs, Temp, times)

    return degs, Temp, times


def run_climate_model(conf: CONF_PARSER_TYPE):
    type_ = conf.get("PDE", "type").lower().replace(" ", "").replace("-", "")
    if type_ in [
        "planet",
        "planetary",
        "earth",
        "earthlike",
    ]:
        return climate_model_planet(conf)
    elif type_ in ["moon", "tidalheating", "gas", "gasgiant"]:
        return climate_model_moon(conf)
    else:
        raise ValueError(
            f"Type argument of config file incorrect, expected 'planet' or 'moon', got {type_}"
        )


if __name__ == "__main__":
    config = load_config("config.ini", "OrbitalConstraints")
    degs, temps, times = run_climate_model(config)
    dt = (times[1] - times[0]) * 365
    yearavgplot(degs, temps, dt, 150, 155, 10)
    colourplot(degs, temps, times, None, None, 5)
