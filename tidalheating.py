from typing import Iterable
import numpy as np

from Constants import *

visc_0 = 1.6e5  # Pa.s
T_s = 1600  # K
T_l = 2000  # K
frac = 0.5
T_b = T_s + frac * (T_l - T_s)  # K
activation_energy = 300e3  # J.mol^−1


def visc(T: float, B_) -> float:
    """T: temperature, K
    returns Viscosity, Pa.s"""
    if T < T_s:
        return visc_0 * np.exp(activation_energy / (GAS_CONST * T))
    elif T_s <= T < T_b:
        phi = (T - T_s) / (T_l - T_s)
        return visc_0 * np.exp(activation_energy / (GAS_CONST * T)) * np.exp(-B_ * phi)
    elif T_b <= T < T_l:
        phi = (T - T_s) / (T_l - T_s)
        return 1e-7 * np.exp(40_000 / T) * (1.35 * phi - 0.35) ** (-5 / 2)
    else:  # T_l < T
        return 1e-7 * np.exp(40_000 / T)


def shear_mod(T: float) -> float:
    """T: temperature, K
    returns: shear modulus, mu, Pa"""
    if T < T_s:
        return 50e9
    elif T_s <= T < T_b:
        mu_1 = 8.2e4
        mu_2 = -40.6
        return 10 ** (mu_1 / T + mu_2)
    else:
        return 1e-7


def delta(Ra_: float, d: float = 3e6, a_2: float = 1, Ra_c: float = 1100) -> float:
    """Ra_: Rayleigh Number,
    d: mantle thickness, m
    a_2: Flow geometry constant, ~1
    Ra_c: Critical Rayleigh Number

    returns: Conducting Boundary Layer, m"""
    return d / (2 * a_2) * (Ra_ / Ra_c) ** (-1 / 4)


def Ra(
    R: float,
    q_BL: float,
    eta: float,
    C_p: float = 1260,
    rho: float = 5e3,
    d: float = 3e6,
    alpha: float = 1e-4,
    k_therm: float = 2,
) -> float:
    """R: radius of the moon, m
    q_BL: conduction through surface, W m^-2
    eta: viscosity, Pa.s
    C_p: heat capacity, J kg^-1 K^-1
    rho: density of moon, kg m^-3
    d: mantle thickness, m
    alpha: thermal expansivity, K^-1
    k_therm: thermal conductivity, W m^-1 K^-1

    returns: Ra, Rayleigh Number"""
    g = G * rho * 4 / 3 * np.pi * R  # m s^-2
    kappa = k_therm / (rho * C_p)  # m^3 s^-1
    # J = kg m^2 s^-2
    # K^-1 m s^-2 kg m^-3 m^4 W m^-2 / (Pa s m^3 s^-1 W m^-1 K^-1)
    return alpha * g * rho * d**4 * q_BL / (eta * kappa * k_therm)


def q_BL(T_mantle: float, T_surf: float, delta_: float, k_therm: float = 2) -> float:
    """T_mantle: mantle temperature, K
    T_surf: surface temperature, K
    delta_: convective vigour, m
    k_therm: thermal conductivity, W m ^-1 K^-1

    returns: heat conduction through surface, W m^-2"""
    # W m^-1 K^-1 * K / m = W m^-2
    return k_therm * (T_mantle - T_surf) / (delta_)


def conv_cooling(
    T_man: float, T_surf: float = 300, B_: float = 10, R_m=RADIUS["earth"], dens_m=5000
) -> float:
    """returns: cooling flux, W.m^-2"""
    delt = 30e3  # m
    q_BL_ = 0.0
    q_BL_n = 0.0
    while True:
        q_BL_n = q_BL(T_man, T_surf, delt)
        if (q_BL_n - q_BL_) < 1e-10:
            break
        else:
            q_BL_ = q_BL_n
        Ra_ = Ra(R_m, q_BL_, visc(T_man, B_), rho=dens_m)
        delt = delta(Ra_)
    return q_BL_n


def viscoelastic_tidal_heating(
    T: float, dens_m: float, R_m: float, M_p: float, e: float, a: float, B_=10
) -> float:
    """T: mantle temperature, K
    dens_m: density of the moon, kg m^-3
    R_m: radius of the moon, m
    M_p: mass of the planet, kg
    e: moon eccentricity,
    a: moon semimajor axis, m

    returns: viscoelastic tidal heating rate, W
    """
    shearmod_ = shear_mod(T)  # Pa
    rho_g_Rm = dens_m**2 * G * 4 / 3 * np.pi * R_m**2
    visc_orbit_freq = visc(T, B_) * np.sqrt(G * M_p / a**3)

    minus_im_k_2_num = 57 * visc_orbit_freq
    minus_im_k_2_denom = (
        4
        * rho_g_Rm
        * (
            1
            + (1 + 19 * shearmod_ / (2 * rho_g_Rm)) ** 2
            * (visc_orbit_freq / shearmod_) ** 2
        )
    )
    minus_im_k_2 = minus_im_k_2_num / minus_im_k_2_denom
    return (
        21
        / (2)
        * minus_im_k_2
        * G ** (3 / 2)
        * M_p ** (5 / 2)
        * R_m**5
        * e**2
        / a ** (15 / 2)
    )


def fixed_Q_tidal_heating(
    dens_m: float,
    M_m: float,
    R_m: float,
    shearmod: float,
    Q: float,
    M_p: float,
    e: float,
    a_m: float,
) -> float:
    """returns: heating, W"""
    dens_m_times_grav_m_times_R_m = dens_m * G * M_m / R_m
    k_2 = 3 / (2 + 19 * shearmod / dens_m_times_grav_m_times_R_m)
    return (
        21
        / 2
        * k_2
        / Q
        * G ** (3 / 2)
        * M_p ** (5 / 2)
        * R_m**5
        * e**2
        / a_m ** (15 / 2)
    )


def get_visco_func(M_gas, moon_rad, moon_a, moon_ecc, moon_density, B: float = 25):
    """M_gas: mass of the gas giant, Jupiter masses
    moon_rad: moon radius, m
    moon_a: moon semimajor axis, m
    moon_ecc: eccentricity
    moon_density: kg m^-3"""
    temps = np.arange(1000, 3000, 1)
    visco_fluxes = np.array(
        [
            viscoelastic_tidal_heating(
                t, moon_density, moon_rad, M_gas, moon_ecc, moon_a
            )
            / (4 * np.pi * moon_rad**2)
            for t in temps
        ]
    )

    def get_viscoheating(T_surf: float) -> float:
        """returns: heating flux, W m^-2"""
        conv_cool_fluxes = np.array(
            [conv_cooling(t, T_surf, B, moon_rad) for t in temps]
        )
        if np.all(visco_fluxes > conv_cool_fluxes):

            return np.max(visco_fluxes)
        elif np.all(visco_fluxes < conv_cool_fluxes):
            # print("NO HEATING")
            return 0.0
        else:
            # they must cross somewhere!
            # find when they cross by finding where visco[i] < conv[i] after visco[i-1] > conv[i-1]
            for i in range(0, len(temps) - 1):
                if (
                    visco_fluxes[i + 1] < conv_cool_fluxes[i + 1]
                    and visco_fluxes[i] > conv_cool_fluxes[i]
                ):
                    # print("CROSSES")
                    return (visco_fluxes[i + 1] + visco_fluxes[i]) / 2
            # print("NOT FOUND")
            return 0.0

    return get_viscoheating


def plot_vary_eccentricity(conf: CONF_PARSER_TYPE, T_surf, e_range: Iterable[float]):
    import matplotlib.pyplot as plt

    viscos = []
    fixedQs = []
    M_gas = conf.getfloat("ORBIT", "gasgiantmass") * MASS["jupiter"]  # kg
    moon_density = conf.getfloat("ORBIT", "moondensity")  # kg
    moon_rad = conf.getfloat("ORBIT", "moonradius") * RADIUS["luna"]  # m
    moon_a = conf.getfloat("ORBIT", "moonsemimajoraxis") * AU  # m
    moon_ecc = conf.getfloat("ORBIT", "mooneccentricity")

    M_moon = moon_density * 4 * np.pi / 3 * moon_rad**3

    visc_func = get_visco_func(M_gas, moon_rad, moon_a, moon_ecc, moon_density)

    shearmod = conf.getfloat("TIDALHEATING", "shearmod") * 2e10  # Nm^-2
    Q = conf.getfloat("TIDALHEATING", "Q")
    for e in e_range:
        conf["ORBIT"]["mooneccentricity"] = str(e)
        viscos.append(visc_func(T_surf))
        moon_ecc = e
        fixedQs.append(
            fixed_Q_tidal_heating(
                moon_density, M_moon, moon_rad, shearmod, Q, M_gas, moon_ecc, moon_a
            )
        )
    plt.title(f"Semimajoraxis: {moon_a / AU} AU")
    plt.plot(e_range, viscos, label="Visco")
    plt.plot(e_range, fixedQs, label="fixedQ")
    plt.xlabel("Eccentricity")
    plt.ylabel("Heat rate, W")
    plt.legend()
    plt.yscale("log")
    plt.show()


def plot_vary_semimajoraxis(conf: CONF_PARSER_TYPE, T_surf, a_range: Iterable[float]):
    import matplotlib.pyplot as plt

    viscos = []
    fixedQs = []
    M_gas = conf.getfloat("ORBIT", "gasgiantmass") * MASS["jupiter"]  # kg
    moon_density = conf.getfloat("ORBIT", "moondensity")  # kg
    moon_rad = conf.getfloat("ORBIT", "moonradius") * RADIUS["luna"]  # m
    moon_a = conf.getfloat("ORBIT", "moonsemimajoraxis") * AU  # m
    moon_ecc = conf.getfloat("ORBIT", "mooneccentricity")

    M_moon = moon_density * 4 * np.pi / 3 * moon_rad**3

    visc_func = get_visco_func(M_gas, moon_rad, moon_a, moon_ecc, moon_density)

    shearmod = conf.getfloat("TIDALHEATING", "shearmod") * 2e10  # Nm^-2
    Q = conf.getfloat("TIDALHEATING", "Q")
    moon_density = M_moon / (4 / 3 * np.pi * moon_rad**3)  # kg.m^-3
    for a in a_range:
        conf["ORBIT"]["moonsemimajoraxis"] = str(a)
        viscos.append(visc_func(T_surf))
        moon_a = a * AU
        fixedQs.append(
            fixed_Q_tidal_heating(
                moon_density, M_moon, moon_rad, shearmod, Q, M_gas, moon_ecc, moon_a
            )
        )
    plt.title(f"Eccentricity: {moon_ecc}")
    plt.plot(a_range, viscos, label="Visco")
    plt.plot(a_range, fixedQs, label="fixedQ")
    plt.axvline(421_800_000 / AU, 0, 1, label="Io")
    plt.axvline(671_100_000 / AU, 0, 1, label="Europa")
    plt.axvline(1_070_400_000 / AU, 0, 1, label="Ganymede")
    plt.axvline(1_882_700_000 / AU, 0, 1, label="Callisto")
    plt.xlabel("Semimajoraxis, AU")
    plt.ylabel("Heat rate, W")
    plt.legend()
    plt.yscale("log")
    plt.show()


def roche_limit(R_m, rho_m, M_p):
    """returns: Roche limit, AU"""
    M_m = 4 * np.pi / 3 * R_m**3 * rho_m
    return R_m * (2 * M_p / M_m) ** (1 / 3) / AU


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    temperatures = np.linspace(1000, 2500, 1000)
    heats = [
        viscoelastic_tidal_heating(
            t, 4000, RADIUS["earth"], MASS["jupiter"], 0.01, 0.001 * AU, 15
        )
        / (4 * np.pi * RADIUS["earth"] ** 2)
        for t in temperatures
    ]
    cools = [conv_cooling(t, 300, B_=15, dens_m=4000) for t in temperatures]
    plt.plot(temperatures, heats, label="heat")
    plt.plot(temperatures, cools, label="cool")
    plt.yscale("log")
    plt.legend()
    plt.show()

    viscfunc = get_visco_func(
        MASS["jupiter"], RADIUS["earth"], 0.001 * AU, 0.01, 4000, 15
    )
    print(viscfunc(300))

    # from filemanagement import load_config

    # config = load_config("config.ini", "OrbitalConstraints")
    # plot_vary_eccentricity(config, 300, np.linspace(0.0001, 0.1, 100))
    # config = load_config("config.ini", "OrbitalConstraints")
    # plot_vary_semimajoraxis(config, 300, np.linspace(0.00001, 0.02, 100))
    # for q in np.linspace(0.004, 0.01, 100):
    # viscfunc = get_visco_func(MASS["jupiter"], 1821e3, 421700e3, q, 3528)
    # print(q, viscfunc(110))
    # print(1e14 / (4 * np.pi * (1821e3) ** 2))
