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


def get_viscoheating(config, T_surf: float, B: float = 25, rtol: float = 0.1) -> float:
    M_gas = config.getfloat("ORBIT", "gasgiantmass") * MASS["jupiter"]  # kg
    moon_rad = config.getfloat("ORBIT", "moonradius") * RADIUS["luna"]  # m
    moon_a = config.getfloat("ORBIT", "moonsemimajoraxis") * AU  # m
    moon_ecc = config.getfloat("ORBIT", "mooneccentricity")
    moon_density = config.getfloat("ORBIT", "moondensity")  # kg m^-3

    temps = np.arange(1400, 2200, 1)
    visco_fluxes = np.array(
        [
            viscoelastic_tidal_heating(
                t, moon_density, moon_rad, M_gas, moon_ecc, moon_a
            )
            for t in temps
        ]
    )
    conv_cool_fluxes = (
        np.array([conv_cooling(t, T_surf, B, moon_rad) for t in temps])
        * 4
        * np.pi
        * moon_rad**2
    )
    if np.all(visco_fluxes > conv_cool_fluxes):
        return np.max(visco_fluxes)
    elif np.all(visco_fluxes < conv_cool_fluxes):
        return 0.0
    else:
        # they must cross somewhere!
        # find when they cross by finding where visco[i] < conv[i] after visco[i-1] > conv[i-1]
        for i in range(0, len(temps) - 1):
            if (
                visco_fluxes[i + 1] < conv_cool_fluxes[i + 1]
                and visco_fluxes[i] > conv_cool_fluxes[i]
            ):
                return visco_fluxes[i + 1]


def plot_vary_eccentricity(conf: CONF_PARSER_TYPE, T_surf, e_range: Iterable[float]):
    import matplotlib.pyplot as plt

    viscos = []
    fixedQs = []
    M_gas = config.getfloat("ORBIT", "gasgiantmass") * MASS["jupiter"]  # kg
    M_moon = config.getfloat("ORBIT", "moonmass") * MASS["luna"]  # kg
    moon_rad = config.getfloat("ORBIT", "moonradius") * RADIUS["luna"]  # m
    moon_a = config.getfloat("ORBIT", "moonsemimajoraxis") * AU  # m
    moon_ecc = config.getfloat("ORBIT", "mooneccentricity")

    shearmod = config.getfloat("TIDALHEATING", "shearmod") * 2e10  # Nm^-2
    Q = config.getfloat("TIDALHEATING", "Q")
    moon_density = M_moon / (4 / 3 * np.pi * moon_rad**3)  # kg.m^-3
    for e in e_range:
        conf["ORBIT"]["mooneccentricity"] = str(e)
        viscos.append(get_viscoheating(conf, T_surf))
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
    M_gas = config.getfloat("ORBIT", "gasgiantmass") * MASS["jupiter"]  # kg
    M_moon = config.getfloat("ORBIT", "moonmass") * MASS["luna"]  # kg
    moon_rad = config.getfloat("ORBIT", "moonradius") * RADIUS["luna"]  # m
    moon_a = config.getfloat("ORBIT", "moonsemimajoraxis") * AU  # m
    moon_ecc = config.getfloat("ORBIT", "mooneccentricity")

    shearmod = config.getfloat("TIDALHEATING", "shearmod") * 2e10  # Nm^-2
    Q = config.getfloat("TIDALHEATING", "Q")
    moon_density = M_moon / (4 / 3 * np.pi * moon_rad**3)  # kg.m^-3
    for a in a_range:
        conf["ORBIT"]["moonsemimajoraxis"] = str(a)
        viscos.append(get_viscoheating(conf, T_surf))
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


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    from filemanagement import load_config

    # config = load_config("config.ini", "OrbitalConstraints")
    # plot_vary_eccentricity(config, 300, np.linspace(0.0001, 0.1, 100))
    config = load_config("config.ini", "OrbitalConstraints")
    plot_vary_semimajoraxis(config, 300, np.linspace(0.00001, 0.02, 100))
    # M_gas = config.getfloat("ORBIT", "gasgiantmass") * MASS["jupiter"]  # kg
    # M_moon = config.getfloat("ORBIT", "moonmass") * MASS["luna"]  # kg
    # moon_rad = config.getfloat("ORBIT", "moonradius") * RADIUS["luna"]  # m
    # moon_a = config.getfloat("ORBIT", "moonsemimajoraxis") * AU  # m
    # moon_ecc = config.getfloat("ORBIT", "mooneccentricity")

    # shearmod = config.getfloat("TIDALHEATING", "shearmod") * 2e10  # Nm^-2
    # Q = config.getfloat("TIDALHEATING", "Q")
    # moon_density = M_moon / (4 / 3 * np.pi * moon_rad**3)  # kg.m^-3
    # fixed_Q = fixed_Q_tidal_heating(
    #     moon_density, M_moon, moon_rad, shearmod, Q, M_gas, moon_ecc, moon_a
    # )
    # temps = np.linspace(100, 500, 401)
    # visc_heat = [get_viscoheating(config, T_surf) for T_surf in temps]
    # plt.plot(temps, visc_heat)
    # plt.axhline(fixed_Q, 0, 1)
    # plt.xlabel("Surface Temperature")
    # plt.ylabel("Surface flux, W m$^{-2}$")
    # plt.yscale("log")
    # plt.show()

    # M_gas = config.getfloat("ORBIT", "gasgiantmass") * MASS["jupiter"]  # kg
    # M_moon = config.getfloat("ORBIT", "moonmass") * MASS["luna"]  # kg
    # moon_rad = config.getfloat("ORBIT", "moonradius") * RADIUS["luna"]  # m
    # moon_a = config.getfloat("ORBIT", "moonsemimajoraxis") * AU  # m
    # moon_ecc = config.getfloat("ORBIT", "mooneccentricity")

    # shearmod = config.getfloat("TIDALHEATING", "shearmod") * 2e10  # Nm^-2
    # Q = config.getfloat("TIDALHEATING", "Q")

    # moon_density = M_moon / (4 / 3 * np.pi * moon_rad**3)  # kg.m^-3

    # ts = np.linspace(1400, 2000, 600, endpoint=False)  #
    # # viscs = [[visc(t, B_) for t in ts] for B_ in np.linspace(10, 40, 30, endpoint=True)]
    # # shearmods = [shear_mod(t) for t in ts]
    # thss = []
    # qs = np.logspace(-5, -1, 6, endpoint=True)
    # # qs = np.linspace(0.1, 10, 6, endpoint=True)
    # for x in qs:
    #     thss.append(
    #         np.array(
    #             [
    #                 viscoelastic_tidal_heating(
    #                     t, moon_density, moon_rad, M_gas, x, moon_a
    #                 )
    #                 for t in ts
    #             ]
    #         )
    #     )

    # ths = np.array(
    #     [
    #         viscoelastic_tidal_heating(
    #             t, moon_density, moon_rad, M_gas, moon_ecc, moon_a
    #         )
    #         for t in ts
    #     ]
    # )
    # for i, B_ in enumerate(np.linspace(10, 40, 5, endpoint=True)):
    #     plt.plot(ts, viscs[i], label=f"Viscosity(T, {B_})")
    # plt.plot(ts, shearmods, label=f"Shearmodulus(T)")

    # for ths, q in zip(thss, qs):
    #     plt.plot(ts, ths, label=f"Viscoelastic Tidal heating, e={q}")
    # # for ths, q in zip(thss, qs):
    # # plt.plot(ts, ths, label=f"Viscoelastic Tidal heating")
    # B_ = 25
    # surf_temp = 300
    # # for B_ in np.linspace(10, 40, 5, endpoint=True):
    # # for surf_temp in np.linspace(100, 500, 6):
    # conv = (
    #     np.array([conv_cooling(t, surf_temp, B_) for t in ts])
    #     * 4
    #     * np.pi
    #     * moon_rad**2
    # )
    # # plt.plot(ts, conv / 1e12, label=r"Conv cooling, $B_ =" + str(B_) + "$")
    # plt.plot(ts, conv, "--", ms=2, label=r"Conv cooling")
    # for q in qs:
    #     fixed_Q = fixed_Q_tidal_heating(
    #         moon_density, M_moon, moon_rad, shearmod, Q, M_gas, q, moon_a
    #     )
    #     plt.plot(ts, [fixed_Q] * len(ts), ls="-.", label=f"Fixed-Q, e={q}")
    # plt.xlabel("Mantle Temperature, K")
    # plt.ylabel("Heat flux, W")
    # plt.yscale("log")
    # plt.legend()
    # plt.show()

    # print(ts[np.isclose(ths, conv, rtol=0.1)])
    # print(q := ths[np.isclose(ths, conv, rtol=0.1)])
    # #
    # spacedim = 60

    # dlam = np.pi / (spacedim - 1)  # spacial separation in -pi/2 to pi/2
    # lats = np.linspace(-1, 1, spacedim) * (np.pi / 2)
    # degs = np.rad2deg(lats)

    # # semi = np.logspace(-3, 0, 50, True, 10) * AU
    # config = load_config("config.ini", "OrbitalConstraints")
    # M_gas = config.getfloat("ORBIT", "gasgiantmass") * MASS["jupiter"]
    # M_moon = config.getfloat("ORBIT", "moonmass") * MASS["luna"]
    # moon_rad = config.getfloat("ORBIT", "moonradius") * RADIUS["luna"]
    # moon_a = config.getfloat("ORBIT", "moonsemimajoraxis") * AU
    # moon_ecc = config.getfloat("ORBIT", "mooneccentricity")

    # shearmod = config.getfloat("TIDALHEATING", "shearmod") * 2e10  # Nm^-2
    # Q = config.getfloat("TIDALHEATING", "Q")

    # moon_density = M_moon / (4 / 3 * np.pi * moon_rad**3)
    # # fixQvals = fixed_Q_tidal_heating(
    # #     moon_density, M_moon, moon_rad, shearmod, Q, M_gas, moon_ecc, semi
    # # )
    # fixQvals = fixed_Q_tidal_heating(
    #     moon_density, M_moon, moon_rad, shearmod, Q, M_gas, moon_ecc, moon_a
    # )
    # heatings = fixQvals * ((dlam / (2 * moon_rad)) * np.cos(lats))
    # plt.plot(degs, heatings, label="fixed-Q values")
    # plt.title(f"Eccentricity: {moon_ecc}")
    # # plt.yscale("log")
    # # plt.xscale("log")
    # plt.legend()
    # # plt.xlabel("Semimajor axis, au")
    # plt.show()
