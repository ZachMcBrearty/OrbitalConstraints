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

from configparser import ConfigParser
from typing import Literal

import numpy as np
import numpy.typing as npt

from Constants import *
from InsolationFunction import S
from HeatCapacity import get_C_func, f_o, f_i
from IRandAlbedo import A_1, A_2, A_3, I_1, I_2, I_3
from plotting import colourplot, complexplotdata
from filemanagement import write_to_file, load_config
from tidalheating import tidal_heating, fixed_Q_tidal_heating


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


def getexp(conf, sec, val):
    q = conf.get(sec, val)
    a, b = q.split("e")
    return float(a) * pow(10, float(b))


def r(pos):
    return np.sqrt(np.sum(pos**2))


def line(x, p1, p2):
    # y - y2 = m(x - x2); m = (y2-y1) / (x2 - x1)
    return p2[1] + (p2[1] - p1[1]) * (x - p2[0]) / (p2[0] - p1[0])


def check_eclipse(starpos, gaspos, moonpos, star_rad, gas_rad) -> float:
    # if the planet is infront of the gas giant do nothing
    # i.e. the planet cannot block the light for the moon
    if r(moonpos - starpos) < r(gaspos - starpos):
        return 1.0
    # find unit vector from star to gas giant
    star_to_gas_dir = (gaspos - starpos) / r(gaspos - starpos)
    # rotate 90deg left and right to get edges of the star and planet
    perp_up = np.array([-star_to_gas_dir[1], star_to_gas_dir[0]])
    perp_down = np.array([star_to_gas_dir[1], -star_to_gas_dir[0]])
    star_up = starpos + perp_up * star_rad
    star_down = starpos + perp_down * star_rad
    gas_up = gaspos + perp_up * gas_rad
    gas_down = gaspos + perp_down * gas_rad

    # logic is reversed since the system is "upside-down"
    if moonpos[0] < 0:
        # Umbra:
        # lines from edges of star to same edges to planet
        if (moonpos[1] > line(moonpos[0], star_up, gas_up)) and (
            moonpos[1] < line(moonpos[0], star_down, gas_down)
        ):
            q = 0.0
        # Penumbra:
        # lines from edges of star to opposite edges of planet
        elif (moonpos[1] > line(moonpos[0], star_down, gas_up)) and (
            moonpos[1] < line(moonpos[0], star_up, gas_down)
        ):
            q = 0.5
        # unblocked
        else:
            q = 1.0
    else:
        if (moonpos[1] < line(moonpos[0], star_up, gas_up)) and (
            moonpos[1] > line(moonpos[0], star_down, gas_down)
        ):
            q = 0.0
        elif (moonpos[1] < line(moonpos[0], star_down, gas_up)) and (
            moonpos[1] > line(moonpos[0], star_up, gas_down)
        ):
            q = 0.5
        else:
            q = 1.0
    return q


def r_explicit(x, y):
    return np.sqrt(x * x + y * y)


def line_explicit(x, p1_x, p1_y, p2_x, p2_y):
    # y - y2 = m(x - x2); m = (y2-y1) / (x2 - x1)
    return p2_y + (p2_y - p1_y) * (x - p2_x) / (p2_x - p1_x)


def check_eclipse_explicit(
    starpos: tuple[float, float],
    gaspos: tuple[float, float],
    moonpos: tuple[float, float],
    star_rad: float,
    gas_rad: float,
) -> float:
    # if the planet is infront of the gas giant do nothing
    # i.e. the planet cannot block the light for the moon
    gas_to_star_dist = r_explicit(gaspos[0] - starpos[0], gaspos[1] - starpos[1])
    if r_explicit(moonpos[0] - starpos[0], moonpos[1] - starpos[1]) < gas_to_star_dist:
        return 1.0
    # find unit vector from star to gas giant
    star_to_gas_dir_x = (gaspos[0] - starpos[0]) / gas_to_star_dist
    star_to_gas_dir_y = (gaspos[1] - starpos[1]) / gas_to_star_dist
    # rotate 90deg left and right to get edges of the star and planet
    perp_up_x = -star_to_gas_dir_y
    perp_up_y = star_to_gas_dir_x
    perp_down_x = star_to_gas_dir_y
    perp_down_y = -star_to_gas_dir_x

    star_up_x = starpos[0] + perp_up_x * star_rad
    star_up_y = starpos[1] + perp_up_y * star_rad
    star_down_x = starpos[0] + perp_down_x * star_rad
    star_down_y = starpos[1] + perp_down_y * star_rad

    gas_up_x = gaspos[0] + perp_up_x * gas_rad
    gas_up_y = gaspos[1] + perp_up_y * gas_rad
    gas_down_x = gaspos[0] + perp_down_x * gas_rad
    gas_down_y = gaspos[1] + perp_down_y * gas_rad

    # logic is reversed since the system is "upside-down"
    if moonpos[0] < 0:
        # Umbra:
        # lines from edges of star to same edges to planet
        if (
            moonpos[1]
            > line_explicit(moonpos[0], star_up_x, star_up_y, gas_up_x, gas_up_y)
        ) and (
            moonpos[1]
            < line_explicit(
                moonpos[0], star_down_x, star_down_y, gas_down_x, gas_down_y
            )
        ):
            q = 0.0
        # Penumbra:
        # lines from edges of star to opposite edges of planet
        elif (
            moonpos[1]
            > line_explicit(moonpos[0], star_down_x, star_down_y, gas_up_x, gas_up_y)
        ) and (
            moonpos[1]
            < line_explicit(moonpos[0], star_up_x, star_up_y, gas_down_x, gas_down_y)
        ):
            q = 0.5
        # unblocked
        else:
            q = 1.0
    else:
        if (
            moonpos[1]
            < line_explicit(moonpos[0], star_up_x, star_up_y, gas_up_x, gas_up_y)
        ) and (
            moonpos[1]
            > line_explicit(
                moonpos[0], star_down_x, star_down_y, gas_down_x, gas_down_y
            )
        ):
            q = 0.0
        elif (
            moonpos[1]
            < line_explicit(moonpos[0], star_down_x, star_down_y, gas_up_x, gas_up_y)
        ) and (
            moonpos[1]
            > line_explicit(moonpos[0], star_up_x, star_up_y, gas_down_x, gas_down_y)
        ):
            q = 0.5
        else:
            q = 1.0
    return q


def orbital_model(conf: ConfigParser, dt_steps=10):
    G_prime = G * (24 * 60 * 60) ** 2  # s^-2 -> days^-2
    star = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=float)
    gas = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=float)
    moon = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=float)

    dt = conf.getfloat("PDE", "timestep") / dt_steps  # days
    M_star = conf.getfloat("ORBIT", "starmass") * MASS["solar"]
    r_star = conf.getfloat("ORBIT", "starradius") * RADIUS["solar"]
    M_gas = conf.getfloat("ORBIT", "gasgiantmass") * MASS["jupiter"]
    r_gas = conf.getfloat("ORBIT", "gasgiantradius") * RADIUS["jupiter"]
    M_moon = conf.getfloat("ORBIT", "moonmass") * MASS["luna"]

    gas_a = conf.getfloat("ORBIT", "gassemimajoraxis") * AU
    gas_ecc = conf.getfloat("ORBIT", "gasgianteccentricity")
    moon_a = conf.getfloat("ORBIT", "moonsemimajoraxis") * AU
    moon_ecc = conf.getfloat("ORBIT", "mooneccentricity")

    gas[0][0] = gas_a * (1 + gas_ecc)
    moon[0][0] = gas[0][0] + moon_a * (1 + moon_ecc)

    # v^2 = GM(2/r - 1/a) = GM/a
    # use vis-viva equation to put giant (and moon) on orbit around the star
    gas[1][1] = np.sqrt(G_prime * M_star * (2 / gas[0][0] - 1 / gas_a))
    # then use vis-viva to put moon on orbit around giant
    moon[1][1] = gas[1][1] + np.sqrt(
        G_prime * M_gas * (2 / (moon_a * (1 + moon_ecc)) - 1 / moon_a)
    )
    # then use conservation of momentum to set the star on the opposite orbit
    # so center of momentum doesnt move
    star[1][1] = -gas[1][1] * M_gas / M_star - moon[1][1] * M_moon / M_star  #
    epss = 0
    # setup Leapfrog (ie put velocities back 1/2 timestep)
    stargas = np.sum((gas[0] - star[0]) ** 2 + epss) ** (-3 / 2)
    starmoon = np.sum((moon[0] - star[0]) ** 2 + epss) ** (-3 / 2)
    moongas = np.sum((gas[0] - moon[0]) ** 2 + epss) ** (-3 / 2)

    star[2] = -(
        G_prime * M_gas * (star[0] - gas[0]) * stargas
        + G_prime * M_moon * (star[0] - moon[0]) * starmoon
    )
    gas[2] = -(
        +G_prime * M_star * (gas[0] - star[0]) * stargas
        + G_prime * M_moon * (gas[0] - moon[0]) * moongas
    )
    moon[2] = -(
        +G_prime * M_gas * (moon[0] - gas[0]) * moongas
        + G_prime * M_star * (moon[0] - star[0]) * starmoon
    )
    leapfrog_update_matrix = np.array([[1, 0, 0], [0, 1, -dt / 2], [0, 0, 0]])
    star = leapfrog_update_matrix @ star
    gas = leapfrog_update_matrix @ gas
    moon = leapfrog_update_matrix @ moon

    update_matrix = np.array([[1, dt, dt**2], [0, 1, dt], [0, 0, 0]])
    # eclipsed = check_eclipse(star[0], gas[0], moon[0], r_star, r_gas)
    eclipsed = 0

    yield star[0], gas[0], moon[0], eclipsed
    while True:
        stargas = np.sum((gas[0] - star[0]) ** 2 + epss) ** (-3 / 2)
        starmoon = np.sum((moon[0] - star[0]) ** 2 + epss) ** (-3 / 2)
        moongas = np.sum((gas[0] - moon[0]) ** 2 + epss) ** (-3 / 2)

        star[2] = -(
            G_prime * M_gas * (star[0] - gas[0]) * stargas
            + G_prime * M_moon * (star[0] - moon[0]) * starmoon
        )
        gas[2] = -(
            +G_prime * M_star * (gas[0] - star[0]) * stargas
            + G_prime * M_moon * (gas[0] - moon[0]) * moongas
        )
        moon[2] = -(
            +G_prime * M_gas * (moon[0] - gas[0]) * moongas
            + G_prime * M_star * (moon[0] - star[0]) * starmoon
        )

        star = update_matrix @ star
        gas = update_matrix @ gas
        moon = update_matrix @ moon
        # eclipsed = check_eclipse(star[0], gas[0], moon[0], r_star, r_gas)
        yield star[0], gas[0], moon[0], eclipsed


def orbital_model_explicit(conf: ConfigParser, dt_steps=10):
    G_prime = G * (24 * 60 * 60) ** 2  # s^-2 -> days^-2
    star_x, star_y, star_vx, star_vy, star_ax, star_ay = 0, 0, 0, 0, 0, 0
    gas_x, gas_y, gas_vx, gas_vy, gas_ax, gas_ay = 0, 0, 0, 0, 0, 0
    moon_x, moon_y, moon_vx, moon_vy, moon_ax, moon_ay = 0, 0, 0, 0, 0, 0

    dt = conf.getfloat("PDE", "timestep") / dt_steps  # days
    M_star = conf.getfloat("ORBIT", "starmass") * MASS["solar"]
    r_star = conf.getfloat("ORBIT", "starradius") * RADIUS["solar"]
    M_gas = conf.getfloat("ORBIT", "gasgiantmass") * MASS["jupiter"]
    r_gas = conf.getfloat("ORBIT", "gasgiantradius") * RADIUS["jupiter"]
    M_moon = conf.getfloat("ORBIT", "moonmass") * MASS["luna"]

    gas_a = conf.getfloat("ORBIT", "gassemimajoraxis") * AU
    gas_ecc = conf.getfloat("ORBIT", "gasgianteccentricity")
    moon_a = conf.getfloat("ORBIT", "moonsemimajoraxis") * AU
    moon_ecc = conf.getfloat("ORBIT", "mooneccentricity")

    gas_x = gas_a * (1 + gas_ecc)
    moon_x = gas_x + moon_a * (1 + moon_ecc)

    # v^2 = GM(2/r - 1/a) = GM/a
    # use vis-viva equation to put giant (and moon) on orbit around the star
    gas_vy = np.sqrt(G_prime * M_star * (2 / gas_x - 1 / gas_a))
    # then use vis-viva to put moon on orbit around giant
    moon_vy = gas_vy + np.sqrt(
        G_prime * M_gas * (2 / (moon_a * (1 + moon_ecc)) - 1 / moon_a)
    )
    # then use conservation of momentum to set the star on the opposite orbit
    # so center of momentum doesnt move
    star_vy = -gas_vy * M_gas / M_star - moon_vy * M_moon / M_star  #
    # setup Leapfrog (ie put velocities back 1/2 timestep)
    stargas = ((gas_x - star_x) ** 2 + (gas_y - star_y) ** 2) ** (-3 / 2)
    starmoon = ((moon_x - star_x) ** 2 + (moon_y - star_y) ** 2) ** (-3 / 2)
    moongas = ((moon_x - gas_x) ** 2 + (moon_y - gas_y) ** 2) ** (-3 / 2)

    star_ax = -(
        G_prime * M_gas * (star_x - gas_x) * stargas
        + G_prime * M_moon * (star_x - moon_x) * starmoon
    )
    star_ay = -(
        G_prime * M_gas * (star_y - gas_y) * stargas
        + G_prime * M_moon * (star_y - moon_y) * starmoon
    )
    gas_ax = -(
        G_prime * M_star * (gas_x - star_x) * stargas
        + G_prime * M_moon * (gas_x - moon_x) * moongas
    )
    gas_ay = -(
        G_prime * M_gas * (gas_y - star_y) * stargas
        + G_prime * M_moon * (gas_y - moon_y) * moongas
    )
    moon_ax = -(
        G_prime * M_gas * (moon_x - gas_x) * moongas
        + G_prime * M_star * (moon_x - star_x) * starmoon
    )
    moon_ay = -(
        G_prime * M_gas * (moon_y - gas_y) * moongas
        + G_prime * M_star * (moon_y - star_y) * starmoon
    )
    # leapfrog_update_matrix = np.array([[1, -dt / 2, 0], [0, 1, -dt / 2], [0, 0, 0]])
    star_vx -= dt / 2 * star_ax
    star_vy -= dt / 2 * star_ay
    gas_vx -= dt / 2 * gas_ax
    gas_vy -= dt / 2 * gas_ay
    moon_vx -= dt / 2 * moon_ax
    moon_vy -= dt / 2 * moon_ay

    # update_matrix = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 0]])
    eclipsed = check_eclipse_explicit(
        (star_x, star_y), (gas_x, gas_y), (moon_x, moon_y), r_star, r_gas
    )

    # eclipsed = 0

    yield (star_x, star_y), (gas_x, gas_y), (moon_x, moon_y), eclipsed
    while True:
        stargas = ((gas_x - star_x) ** 2 + (gas_y - star_y) ** 2) ** (-3 / 2)
        starmoon = ((moon_x - star_x) ** 2 + (moon_y - star_y) ** 2) ** (-3 / 2)
        moongas = ((moon_x - gas_x) ** 2 + (moon_y - gas_y) ** 2) ** (-3 / 2)
        star_ax = -(
            G_prime * M_gas * (star_x - gas_x) * stargas
            + G_prime * M_moon * (star_x - moon_x) * starmoon
        )
        star_ay = -(
            G_prime * M_gas * (star_y - gas_y) * stargas
            + G_prime * M_moon * (star_y - moon_y) * starmoon
        )
        gas_ax = -(
            G_prime * M_star * (gas_x - star_x) * stargas
            + G_prime * M_moon * (gas_x - moon_x) * moongas
        )
        gas_ay = -(
            G_prime * M_gas * (gas_y - star_y) * stargas
            + G_prime * M_moon * (gas_y - moon_y) * moongas
        )
        moon_ax = -(
            G_prime * M_gas * (moon_x - gas_x) * moongas
            + G_prime * M_star * (moon_x - star_x) * starmoon
        )
        moon_ay = -(
            G_prime * M_gas * (moon_y - gas_y) * moongas
            + G_prime * M_star * (moon_y - star_y) * starmoon
        )

        star_x += dt * star_vx + dt**2 * star_ax
        star_y += dt * star_vy + dt**2 * star_ay
        gas_x += dt * gas_vx + dt**2 * gas_ax
        gas_y += dt * gas_vy + dt**2 * gas_ay
        moon_x += dt * moon_vx + dt**2 * moon_ax
        moon_y += dt * moon_vy + dt**2 * moon_ay

        star_vx += dt * star_ax
        star_vy += dt * star_ay
        gas_vx += dt * gas_ax
        gas_vy += dt * gas_ay
        moon_vx += dt * moon_ax
        moon_vy += dt * moon_ay

        eclipsed = check_eclipse_explicit(
            (star_x, star_y), (gas_x, gas_y), (moon_x, moon_y), r_star, r_gas
        )
        yield (star_x, star_y), (gas_x, gas_y), (moon_x, moon_y), eclipsed


##  ##
def climate_model_in_lat(
    config: ConfigParser,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    # number of spatial nodes
    spacedim = config.getint(section="PDE", option="spacedim")

    dlam = np.pi / (spacedim - 1)  # spacial separation in -pi/2 to pi/2
    lats = np.linspace(-1, 1, spacedim) * (np.pi / 2)
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
    e = config.getfloat("ORBIT", "gasgianteccentricity")  # eccentricity
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
    M_moon = config.getfloat("ORBIT", "moonmass") * MASS["luna"]
    moon_rad = config.getfloat("ORBIT", "moonradius") * RADIUS["luna"]
    moon_a = config.getfloat("ORBIT", "moonsemimajoraxis") * AU
    moon_ecc = config.getfloat("ORBIT", "mooneccentricity")

    shearmod = config.getfloat("TIDALHEATING", "shearmod") * 2e10  # Nm^-2
    Q = config.getfloat("TIDALHEATING", "Q")
    gas_albedo = config.getfloat("TIDALHEATING", "gasalbedo")

    moon_density = M_moon / (4 / 3 * np.pi * moon_rad**3)
    tidal_heating_value = fixed_Q_tidal_heating(
        moon_density, M_moon, moon_rad, shearmod, Q, M_gas, moon_ecc, moon_a
    )
    # heat flux proportional to area
    heatings = tidal_heating_value * ((dlam / (2 * moon_rad)) * np.cos(lats))

    C = get_C_func(spacedim)
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
        # if 100 * 365 <= n < 101 * 365:
        #     diff_elem += 25
        Temp[:, n + 1] = Temp[:, n] + YEARTOSECOND * dt / Capacity[:, n] * (
            diff_elem - Ir_emission[:, n] + Source[:, n] * (1 - Albedo[:, n]) + heatings
        )
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
