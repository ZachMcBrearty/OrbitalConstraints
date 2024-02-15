from configparser import ConfigParser
from typing import TypeAlias

from numpy import floating
from numpy.typing import NDArray

floatarr = NDArray[floating]
CONF_PARSER_TYPE: TypeAlias = ConfigParser

YEARTOSECOND = 365.25 * 24 * 3600  # s / yr

spacedim_name = "Number of spatial nodes, S"
spacedim_unit = None

timestep_unit = r"Timestep, $\Delta t$"
timestep_unit = "days"

temp_name = "Temperature, T"
global_conv_temp_name = r"Equilibrium temperature, $\langle$T$\rangle$"
temp_unit = "K"

omega_name = r"Rotation Rate, $\Omega$"
omega_unit = "Earth days$^{-1}$"

obliquity_name = r"Obliquity, $\delta$"
obliquity_unit = r"$^{\circ}$"

aplt_name = "Planet semimajoraxis, $a$"
agas_name = "Gas giant semimajoraxis, $a_{gas}$"
amoon_name = "Moon semimajoraxis, $a_{moon}$"
a_unit = "au"

eplt_name = "Planet eccentricity, $e$"
egas_name = "Gas giant eccentricity, $e_{gas}$"
emoon_name = "Moon eccentricity, $e_{moon}$"
e_unit = None

landfrac_name = "Uniform ocean fraction"
landfrac_unit = None

gas_mass_unit = r"M$_{Jupiter}$"

moon_density_unit = r"kg m$^{-3}$"
moon_rad_unit = r"R$_{moon}$"

GAS_CONST = 8.314_462_618  # J mol^-1 K^-1
STEPH_BOLTZ = 5.670_374_419e-8  # W m^-2 K^-4
G = 6.67e-11  # N m^2 kg^-2 = m^3 s^-2 kg^-1

MASS = {  # kg
    "solar": 1.988e30,
    "jupiter": 1.89819e27,
    "earth": 5.9723e24,
    "luna": 7.346e22,
}

AU = 1.5e11  # m

RADIUS = {  # m
    "solar": 6.957e8,
    "jupiter": 6.9911e7,
    "earth": 6.371e6,
    "luna": 1.7374e6,
}

DISTANCE = {  # m
    "mercury": 57.91e9,  # 0.4au
    "venus": 108.21e9,  # 0.72 au
    "earth": 149.6e9,  # 1.0 au
    "mars": 227.92e9,  # 1.5 au
    "asteroidbelt": 414.012e9,  # Ceres, 2.7 au
    "jupiter": 778.57e9,  # 5.2 au
    "saturn": 1433.53e9,  # 9.5 au
    "uranus": 2872.46e9,  # 19 au
    "neptune": 4495.06e9,  # 30 au
    "kuiperbelt": 5906.38e9,  # Pluto, 39 au
    "luna": 384.400e6,  # 0.0026 au
    "io": 421.800e6,  # 0.0028 au
    "europa": 671.100e6,  # 0.0045 au
    "ganymede": 1.070400e9,  # 0.0071 au
    "callisto": 1.882700e9,  # 0.013 au
}

EARTH_CURRENT_ECCENTRICITY = 0.0167
EARTH_MAX_ECCENTRICITY = 0.0679

EARTH_MIN_OBLIQUITY = 22
EARTH_MAX_OBLIQUITY = 25

if __name__ == "__main__":
    for _dist in DISTANCE:
        print(_dist, DISTANCE[_dist], "m", DISTANCE[_dist] / AU, "AU")
    # for _rad in RADIUS:
    #     print(_rad, RADIUS[_rad], "m", RADIUS[_rad] / AU, "AU")
