import numpy as np

from Constants import *

visc_0 = 1.6e5  # Pa.s
T_s = 1600  # K
T_l = 2000  # K
frac = 0.5
T_b = T_s + frac * (T_l - T_s)  # K
B = 10  # 10 - 40
activation_energy = 300  # kJ.mol^−1


def q_bl_iter(T, T_mantle, T_surf, M_m, R_m, dens_m):
    k_therm = 2e3  # W K^-1
    Ra_c = 1100
    a_2 = 1
    alpha = 1e-4
    heat_cap = 1260  # J kg^-1 K^-1
    d = 3.000e6  # m
    g = G * M_m / R_m**2
    kappa = k_therm / dens_m / heat_cap

    delta_T = 30_000  # m
    q_bl = k_therm * (T_mantle - T_surf) / delta_T

    Ra = alpha * g * dens_m * d**4 * q_bl / (visc(T) * kappa * k_therm)
    delta_T = d / (2 * a_2) * (Ra / Ra_c) ** (-1 / 4)


def visc(T: float) -> float:
    """T: temperature, K
    returns Viscosity, Pa.s"""
    if T < T_s:
        return visc_0 * np.exp(activation_energy / (GAS_CONST * T))
    elif T_s < T < T_b:
        phi = (T - T_s) / (T_l - T_s)
        return visc_0 * np.exp(activation_energy / (GAS_CONST * T)) * np.exp(-B * phi)
    elif T_b < T < T_l:
        phi = (T - T_s) / (T_l - T_s)
        return 1e-7 * np.exp(40_000 / T) * (1.35 * phi - 0.35) ** (-5 / 2)
    else:  # T_l < T
        return 1e-7 * np.exp(40_000 / T)


def shear_mod(T: float) -> float:
    """T: temperature, K
    returns: shear modulus, mu, Pa"""
    if T < T_s:
        return 50e9
    elif T_s < T < T_b:
        mu_1 = 8.2e4
        mu_2 = -40.6
        return 10 ** (mu_1 / T + mu_2)
    else:
        return 1e-7


def tidal_heating(T, M_p, M_m, a_m, e_m, R_m, dens_m):
    """T: equilibrium temperature established from q_bl"""
    visc_T = visc(T)
    shear_mod_T = shear_mod(T)
    n_m = np.sqrt(G * M_p / a_m**3)
    dens_m_times_grav_m_times_R_m = dens_m * G * M_m / R_m
    minus_Im_k_2 = (
        57
        * visc_T
        * n_m
        / (
            4
            * dens_m_times_grav_m_times_R_m
            * (
                1
                + (1 + 19 * shear_mod_T / (2 * dens_m_times_grav_m_times_R_m)) ** 2
                * (visc_T * n_m / shear_mod_T) ** 2
            )
        )
    )
    return 21 / 2 * minus_Im_k_2 * (R_m * n_m) ** 5 * e_m**2 / G


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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from filemanagement import load_config

    #
    spacedim = 60

    dlam = np.pi / (spacedim - 1)  # spacial separation in -pi/2 to pi/2
    lats = np.linspace(-1, 1, spacedim) * (np.pi / 2)
    degs = np.rad2deg(lats)

    # semi = np.logspace(-3, 0, 50, True, 10) * AU
    config = load_config("config.ini", "OrbitalConstraints")
    M_gas = config.getfloat("ORBIT", "gasgiantmass") * MASS["jupiter"]
    M_moon = config.getfloat("ORBIT", "moonmass") * MASS["luna"]
    moon_rad = config.getfloat("ORBIT", "moonradius") * RADIUS["luna"]
    moon_a = config.getfloat("ORBIT", "moonsemimajoraxis") * AU
    moon_ecc = config.getfloat("ORBIT", "mooneccentricity")

    shearmod = config.getfloat("TIDALHEATING", "shearmod") * 2e10  # Nm^-2
    Q = config.getfloat("TIDALHEATING", "Q")

    moon_density = M_moon / (4 / 3 * np.pi * moon_rad**3)
    # fixQvals = fixed_Q_tidal_heating(
    #     moon_density, M_moon, moon_rad, shearmod, Q, M_gas, moon_ecc, semi
    # )
    fixQvals = fixed_Q_tidal_heating(
        moon_density, M_moon, moon_rad, shearmod, Q, M_gas, moon_ecc, moon_a
    )
    heatings = fixQvals * ((dlam / (2 * moon_rad)) * np.cos(lats))
    plt.plot(degs, heatings, label="fixed-Q values")
    plt.title(f"Eccentricity: {moon_ecc}")
    # plt.yscale("log")
    # plt.xscale("log")
    plt.legend()
    # plt.xlabel("Semimajor axis, au")
    plt.show()
