import numpy as np

from Constants import *
from derivatives import *
from tidalheating import get_visco_func
from InsolationFunction import *


def depth_model(config: CONF_PARSER_TYPE):
    spacedim = config.getint(section="PDE", option="spacedim")

    depth_frac = config.getfloat("PDE", "depth_frac")
    moon_radius = config.getfloat("MOON", "moon_radius") * RADIUS["earth"]
    dr = depth_frac * moon_radius / (spacedim - 1)
    layers = np.linspace(moon_radius * (1 - depth_frac), moon_radius, spacedim)
    layers_sq = layers**2
    two_on_layers = 2 / layers

    # simulation length in years
    time = config.getfloat(section="PDE", option="time")
    # simulation timestep in years
    dt = config.getfloat(section="PDE", option="timestep") / 365
    # timedim may need to be split in to chunks which are then recorded
    # e.g. do a year of evolution then write to file and overwrite the array
    timedim = int(np.ceil(time / dt))

    Temp = np.ones((spacedim, timedim + 1))
    Temp[:, 0] = Temp[:, 0] * config.getfloat("PDE", "ocean_temp_start")
    # Temp[0, :] -> ocean floor
    # Temp[-1, :] -> top of ocean/iceplate

    Heat_capacity = np.zeros(
        spacedim
    )  # for each layer -> function of depth (i.e. pressure)
    Diffusion = np.zeros(
        spacedim
    )  # diffusion for each layer -> function of temperature, depth?
    Ir_emission = 0  # Emitted from top (-1) layer only
    Insolation = 0  # Absorbed at top (-1) layer only
    Albedo = 0  # function of top (-1) layer temperature
    Tidal_heating = 0  # function of bottom (0) layer of temperature

    gas_mass = config.getfloat("GASGIANT", "gas_mass") * MASS["jupiter"]
    gas_a = config.getfloat("GASGIANT", "gas_semimajor_axis") * AU
    gas_ecc = config.getfloat("GASGIANT", "gas_eccentricity")

    moon_rad = config.getfloat("ORBIT", "moonradius") * RADIUS["luna"]
    moon_a = config.getfloat("ORBIT", "moonsemimajoraxis") * AU
    moon_ecc = config.getfloat("ORBIT", "mooneccentricity")
    moon_dens = config.getfloat("ORBIT", "moondensity")

    # C = get_C_func(spacedim)

    secondT = np.zeros(spacedim)
    firstT = np.zeros(spacedim)

    T_surf = Temp[0, 0]
    visco_func = get_visco_func(gas_mass, moon_rad, moon_a, moon_ecc, moon_dens, B=25)
    Tidal_heating = visco_func(T_surf) * 4 * np.pi * layers_sq[0]
    for n in range(timedim):
        secondT[0] = forwardbackward_pole(Temp[:, n], dr)
        firstT[0] = 0

        # forward difference for zero edge
        # secondT[1] = centralbackward_edge(Temp[:, n], dlam)
        secondT[1] = centralcentral_firstedge(Temp[:, n], dr)
        firstT[1] = centraldifference(Temp[:, n], 1, dr)

        # backwards difference for end edge
        # secondT[-2] = centralforward_edge(Temp[:, n], dlam)
        secondT[-2] = centralcentral_secondedge(Temp[:, n], dr)
        firstT[-2] = centraldifference(Temp[:, n], -2, dr)

        secondT[-1] = backwardforward_pole(Temp[:, n], dr)
        firstT[-1] = 0

        for m in range(2, spacedim - 2):
            # central difference for most cases
            secondT[m] = central2ndorder(Temp[:, n], m, dr)
            firstT[m] = centraldifference(Temp[:, n], m, dr)

        diff_elem = Diffusion * layers_sq * (secondT + two_on_layers * firstT)

        T_surf = Temp[0, n]
        Tidal_heating = visco_func(T_surf)
        diff_elem[0] += Tidal_heating * 4 * np.pi * layers_sq[0]

        Insolation = 1360 / (gas_a**2 * (1 - gas_ecc**2) ** (1 / 2))
        Albedo = 0.525 - 0.245 * np.tanh((Temp[-1, n] - 268) / 5)
        Ir_emission = STEPH_BOLTZ * Temp[-1] ** 4
        diff_elem[-1] += Insolation * (1 - Albedo) - Ir_emission

        # T(x_m, t_n+1) = T(x_m, t_n) + Δt / C(x_m, t_n)
        # * (diff - I + S(1-A) )
        Heat_capacity = Depth_Capacity(Temp[:, n], layers)

        Temp[:, n + 1] = Temp[:, n] + YEARTOSECOND * dt / Heat_capacity * (diff_elem)

        # surface melted -> not good
        # but this decreases with pressure?
        # so is not necessarily 273 K
        if Temp[0, n + 1] > 273:
            Temp[:, n + 2 :] = -1
            break


def Depth_Capacity(temps, layers):
    C_ref = 5.25 * 10**6

    f_i = 1 - np.exp((temps - 273) / 10, dtype=float)
    f_i[f_i < 0] = 0.0
    f_i[f_i > 1] = 1.0

    C_o = 40 * C_ref  # depth dependent?
    C_i = np.zeros_like(layers)
    C_i[temps >= 263] = 9.2 * C_ref  # depth dependent?
    # T < 263 K
    C_i[temps < 263] = 2.0 * C_ref  # depth dependent?

    return (1 - f_i) * C_o + f_i * C_i
