from configparser import ConfigParser
from typing import TypeAlias

from numpy import floating
from numpy.typing import NDArray

floatarr = NDArray[floating]
CONF_PARSER_TYPE: TypeAlias = ConfigParser

YEARTOSECOND = 365.25 * 24 * 3600  # s / yr

spacedim_unit = None
timestep_unit = "days"
temp_unit = "K"
omega_unit = "days$^{-1}$"
obliquity_unit = r"$^{\circ}$"
a_unit = "au"
e_unit = None

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
